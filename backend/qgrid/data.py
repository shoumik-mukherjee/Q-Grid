"""Real-data pipeline: raw files in data/raw/ -> 15-min series in data/processed/*.parquet.

Run:  python -m qgrid.data
Outputs:
  data/processed/pv.parquet         per-building PV output (kW), 15-min, 2023
  data/processed/load.parquet       per-building load (kW), 15-min, 2023
  data/processed/price.parquet      IEX DAM MCP (Rs/kWh), 15-min, 2023
  data/processed/weather.parquet    GHI (W/m2), T2M (C), 15-min
  data/processed/meta.json          CO2 factor, test days, eval days, mapping details
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
SLOT = "15min"
SLOTS_PER_DAY = 96


def load_config(path: Path | None = None) -> dict:
    """config.yaml, or the file named by QGRID_CONFIG (another campus, see scripts/new_campus.py)."""
    import os
    p = path or (ROOT / os.environ["QGRID_CONFIG"] if os.environ.get("QGRID_CONFIG") else ROOT / "config.yaml")
    return yaml.safe_load(Path(p).read_text())


def year_index(year: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq=SLOT, inclusive="left")


# ----------------------------------------------------------------------------- NASA POWER / PV
def load_weather(year: int) -> pd.DataFrame:
    """Hourly NASA POWER (local solar time) -> 15-min by repeating each hourly mean 4x
    (energy-preserving; NASA hourly values are hour averages)."""
    raw = json.loads((RAW / "nasa_power" / f"bengaluru_{year}_hourly.json").read_text())
    p = raw["properties"]["parameter"]
    idx = pd.to_datetime(list(p["T2M"].keys()), format="%Y%m%d%H")
    hourly = pd.DataFrame(
        {"ghi_wm2": list(p["ALLSKY_SFC_SW_DWN"].values()), "t2m_c": list(p["T2M"].values())},
        index=idx,
    ).astype(float)
    hourly[hourly < -900] = np.nan  # NASA fill value
    hourly = hourly.interpolate(limit=3)
    out = hourly.reindex(year_index(year), method="ffill")
    out["ghi_wm2"] = out["ghi_wm2"].clip(lower=0)
    return out


def pv_output_kw(weather: pd.DataFrame, kwp: float, pv_model: dict) -> pd.Series:
    """PVWatts-style: P = kWp * G/1000 * (1 + gamma*(Tcell-25)) * derate,
    Tcell = Tamb + (NOCT-20)/800 * G."""
    g = weather["ghi_wm2"].to_numpy()
    tcell = weather["t2m_c"].to_numpy() + (pv_model["noct_c"] - 20.0) / 800.0 * g
    p = kwp * g / 1000.0 * (1 + pv_model["gamma_per_c"] * (tcell - 25.0)) * pv_model["system_derate"]
    return pd.Series(np.clip(p, 0, kwp), index=weather.index)


# ----------------------------------------------------------------------------- Ausgrid
def _read_ausgrid_csv() -> pd.DataFrame:
    """Return the 2012-13 Ausgrid half-hourly table (one row per customer/category/date)."""
    d = RAW / "ausgrid"
    csvs = sorted(d.glob("*2012*2013*.csv")) or sorted(d.glob("*.csv"))
    if csvs:
        return _parse_ausgrid(csvs[0].read_bytes())
    for zpath in sorted(d.glob("*.zip")):
        found = _find_in_zip(zipfile.ZipFile(zpath))
        if found is not None:
            return _parse_ausgrid(found)
    raise FileNotFoundError("No Ausgrid 2012-13 CSV found in data/raw/ausgrid (zip or csv)")


def _find_in_zip(z: zipfile.ZipFile) -> bytes | None:
    names = z.namelist()
    for n in names:  # direct CSV
        if n.lower().endswith(".csv") and "2012" in n and "2013" in n:
            return z.read(n)
    for n in names:  # nested zip (the mirror bundles the yearly zips)
        if n.lower().endswith(".zip") and "2012" in n and "2013" in n:
            return _find_in_zip(zipfile.ZipFile(io.BytesIO(z.read(n))))
    return None


def _parse_ausgrid(blob: bytes) -> pd.DataFrame:
    text = blob.decode("latin-1")
    lines = text.splitlines()
    # first line is a free-text title in the original files; header starts with "Customer"
    start = next(i for i, l in enumerate(lines[:5]) if l.lower().startswith("customer"))
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
    df.columns = [c.strip() for c in df.columns]
    return df


def ausgrid_customer_loads(df: pd.DataFrame) -> pd.DataFrame:
    """Customer x half-hour load (kW) = (GC + CL) kWh / 0.5h. Index: timestamps, cols: customer."""
    cat_col = next(c for c in df.columns if c.lower().startswith("consumption category"))
    date_col = next(c for c in df.columns if c.lower() == "date")
    hh_cols = [c for c in df.columns if re.fullmatch(r"\d{1,2}:\d{2}", c)]
    assert len(hh_cols) == 48, f"expected 48 half-hour columns, got {len(hh_cols)}"
    df = df[df[cat_col].isin(["GC", "CL"])].copy()
    df["date"] = pd.to_datetime(df[date_col], dayfirst=True, format="mixed")
    g = df.groupby(["Customer", "date"])[hh_cols].sum()  # GC + CL
    long = g.stack().rename("kwh").reset_index().rename(columns={"level_2": "hh"})
    # column "0:30" is the half hour ending 00:30 -> start 00:00; "0:00" is ending midnight
    hh_end = pd.to_timedelta(long["hh"].str.replace(r"^(\d+):(\d+)$", r"\1:\2:00", regex=True))
    hh_end = hh_end.where(hh_end > pd.Timedelta(0), pd.Timedelta(hours=24))
    long["ts"] = long["date"] + hh_end - pd.Timedelta(minutes=30)
    wide = long.pivot_table(index="ts", columns="Customer", values="kwh", aggfunc="sum")
    return (wide / 0.5).sort_index()


def cluster_profiles(cust_kw: pd.DataFrame, k: int = 5, seed: int = 0):
    """k-means on each customer's normalised mean daily profile (48 values)."""
    from scipy.cluster.vq import kmeans2

    daily = cust_kw.groupby(cust_kw.index.time).mean().T  # customers x 48
    daily = daily.dropna()
    norm = daily.div(daily.sum(axis=1), axis=0).to_numpy()
    rng = np.random.default_rng(seed)
    centroids, labels = kmeans2(norm, k, seed=rng, minit="++")
    return daily.index.to_numpy(), labels, centroids


def daytime_share(profile48: np.ndarray) -> float:
    return float(profile48[18:34].sum() / profile48.sum())  # 09:00-17:00


def map_dates(target_year: int, src_start: str = "2012-07-01", src_end: str = "2013-06-30"):
    """Map each target-year date to a source date with the same month/day (Jul-Dec from 2012,
    Jan-Jun from 2013), then shift by up to +-3 days so the weekday matches."""
    s0, s1 = pd.Timestamp(src_start), pd.Timestamp(src_end)
    out = {}
    for d in pd.date_range(f"{target_year}-01-01", f"{target_year}-12-31", freq="D"):
        y = 2012 if d.month >= 7 else 2013
        day = min(d.day, 28) if d.month == 2 else d.day
        base = pd.Timestamp(year=y, month=d.month, day=day)
        shift = (d.weekday() - base.weekday() + 3) % 7 - 3
        src = base + pd.Timedelta(days=shift)
        if src < s0:
            src += pd.Timedelta(days=7)
        if src > s1:
            src -= pd.Timedelta(days=7)
        out[d] = src
    return out


def ausgrid_daily_shape(target_year: int) -> pd.Series:
    """Real residential shape: mean kW over all 300 Ausgrid homes, per 2023 date mapped (weekday-aligned)
    from Jul-2012..Jun-2013, at 15-min resolution, normalised to a mean of 1 over the year."""
    cust = ausgrid_customer_loads(_read_ausgrid_csv())
    mean = cust.mean(axis=1)
    mean = mean.reindex(pd.date_range(mean.index[0], mean.index[-1] + pd.Timedelta("15min"), freq=SLOT)).ffill()
    idx = year_index(target_year)
    out = np.empty(len(idx))
    fallback = mean.groupby(mean.index.time).mean().to_numpy()
    for i, (d, src) in enumerate(map_dates(target_year).items()):
        day = mean.loc[src: src + pd.Timedelta(hours=23, minutes=45)].to_numpy()
        out[i * SLOTS_PER_DAY:(i + 1) * SLOTS_PER_DAY] = day if len(day) == SLOTS_PER_DAY and not np.isnan(day).any() else fallback
    return pd.Series(out / out.mean(), index=idx)


def _ranges(spec) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    return [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in spec]


def day_types(cfg: dict) -> pd.Series:
    """Day type per date: fest / exam / vacation / holiday (Sat, Sun) / teaching.
    An optional data/raw/rvce_calendar.csv (date, day_type) overrides the config windows."""
    year = cfg["site"]["year"]
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    cal = cfg["calendar"]
    out = pd.Series("teaching", index=days)
    out[days.weekday >= 5] = "holiday"
    for key, label in (("vacation", "vacation"), ("exams", "exam"), ("fest", "fest")):
        for a, b in _ranges(cal.get(key, [])):
            out[(days >= a) & (days <= b)] = label
    f = RAW / "rvce_calendar.csv"
    if f.exists():
        c = pd.read_csv(f, comment="#", parse_dates=["date"])
        out.loc[c["date"]] = c["day_type"].to_numpy()
    return out


def load_profiles() -> dict:
    return yaml.safe_load((ROOT / "configs" / "profiles.yaml").read_text())


def lab_schedule() -> pd.DataFrame:
    f = RAW / "lab_schedule.csv"
    return pd.read_csv(f, comment="#") if f.exists() else pd.DataFrame(
        columns=["zone", "day", "start", "end", "kw", "critical", "name"])


def build_zone_loads(cfg: dict, layout: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    """Per-zone load and critical load (kW, 15-min, full year) and campus water demand (m3/slot).
    Annual kWh = floor area x EPI (ASSUMPTION, PENDING RVCE bills); shape = timetable template by day type
    (loads.source = bdg2: real BDG2 education-campus meter shapes per zone, see qgrid/bdg2.py; otherwise templates and the
    Ausgrid residential shape for hostels). If data/raw/rvce_bills.csv exists, totals are scaled to it."""
    year = cfg["site"]["year"]
    idx = year_index(year)
    prof = load_profiles()
    dtypes = day_types(cfg)
    dt_slot = dtypes.reindex(idx.normalize()).to_numpy()
    hours = idx.hour.to_numpy()
    zc = cfg["campus"]["zones"]
    hostel_shape = None
    loads, crits, info = {}, {}, {}
    sched = lab_schedule()
    dow = np.array(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])[idx.weekday]
    from . import bdg2
    bmap = cfg.get("bdg2", {}).get("zones", {}) if cfg.get("loads", {}).get("source") == "bdg2" else {}
    use_bdg2 = bool(bmap) and bdg2.available()
    for z, spec in zc.items():
        ptype = spec.get("profile", "flat")
        if use_bdg2 and z in bmap:
            shape, binfo = bdg2.year_shape(z, bmap[z]["primary"], bmap[z]["subs"], idx, dt_slot)
            ptype = f"bdg2:{bmap[z]['primary']}/{'|'.join(bmap[z]['subs'])} ({binfo['buildings']} buildings)"
        elif ptype == "hostel":
            if hostel_shape is None:
                hostel_shape = ausgrid_daily_shape(year).to_numpy()
            mult = np.array([prof["hostel"].get(d, 1.0) for d in dt_slot])
            shape = hostel_shape * mult
        else:
            tmpl = prof[ptype]
            shape = np.empty(len(idx))
            for d in set(dt_slot):
                arr = np.array(tmpl.get(d, tmpl["teaching"]), float)
                m = dt_slot == d
                shape[m] = arr[hours[m]]
        floor = layout["zones"][z]["floor_m2"]
        annual = floor * float(spec.get("epi_kwh_m2_yr", 0.0))
        tot = shape.sum() * 0.25
        load = shape * (annual / tot) if tot > 0 else np.zeros(len(idx))
        crit = load * float(spec.get("critical_frac", 0.0))
        # lab schedule (TEMPLATE until faculty provide it): teaching days only
        for _, r in sched[sched.zone == z].iterrows():
            m = (dow == r["day"]) & (dt_slot == "teaching") & _window_mask(idx, r["start"], r["end"])
            load = load + m * float(r["kw"])
            if str(r["critical"]).strip().lower().startswith("y"):
                crit = crit + m * float(r["kw"])
        if "fest_load" in spec:
            fl = spec["fest_load"]
            m = (dt_slot == "fest") & _window_mask(idx, fl["start"], fl["end"])
            load = load + m * float(fl["kw"])
        loads[z], crits[z] = load, np.minimum(crit, load)
        info[z] = {"annual_kwh": round(float(load.sum() * 0.25)), "peak_kw": round(float(load.max()), 1),
                   "floor_m2": round(floor), "epi_kwh_m2_yr": spec.get("epi_kwh_m2_yr"), "profile": ptype}
    load_df, crit_df = pd.DataFrame(loads, index=idx), pd.DataFrame(crits, index=idx)
    bills = RAW / "rvce_bills.csv"
    if bills.exists():   # columns: month (YYYY-MM), kwh ; scale campus total month by month
        b = pd.read_csv(bills, comment="#")
        for _, r in b.iterrows():
            m = load_df.index.strftime("%Y-%m") == r["month"]
            cur = load_df[m].to_numpy().sum() * 0.25
            if cur > 0:
                load_df.loc[m] *= float(r["kwh"]) / cur
                crit_df.loc[m] *= float(r["kwh"]) / cur
        info["_scaling"] = "scaled month by month to data/raw/rvce_bills.csv"
    else:
        info["_scaling"] = "PENDING RVCE DATA: floor area x EPI (ASSUMPTION)"
    wd = prof["water_demand"]
    shape = np.array(wd["shape"], float)[hours] * np.array([wd["multipliers"].get(d, 1.0) for d in dt_slot])
    water = np.zeros(len(idx))
    for z, spec in zc.items():
        if "tank" in spec:
            t = spec["tank"]
            daily = shape.reshape(-1, SLOTS_PER_DAY).sum(1, keepdims=True)
            base = np.array(wd["shape"], float)[np.arange(24)].sum() * 4
            water = (shape * t["daily_demand_m3"] / base).ravel()
    return load_df, crit_df, pd.Series(water, index=idx), {"zones": info, "day_types": dtypes.value_counts().to_dict()}


# ----------------------------------------------------------------------------- IEX
def _find_col(cols, *keys):
    for c in cols:
        lc = str(c).lower()
        if all(k in lc for k in keys):
            return c
    return None


def parse_iex_file(path: Path) -> pd.DataFrame:
    """Tolerant parser for IEX DAM market-snapshot exports -> DataFrame[ts, mcp_rs_mwh]."""
    if path.suffix.lower() == ".csv":
        sheets = {"csv": pd.read_csv(path, header=None)}
    else:
        sheets = pd.read_excel(path, sheet_name=None, header=None)
    frames = []
    for raw in sheets.values():
        # locate header row: the first row containing both 'date' and 'mcp'
        hdr = None
        for i in range(min(30, len(raw))):
            row = " ".join(str(x).lower() for x in raw.iloc[i].tolist())
            if "date" in row and "mcp" in row:
                hdr = i
                break
        if hdr is None:
            continue
        df = raw.iloc[hdr + 1:].copy()
        df.columns = [str(c).strip() for c in raw.iloc[hdr]]
        dcol = _find_col(df.columns, "date")
        tcol = _find_col(df.columns, "time", "block") or _find_col(df.columns, "time")
        mcol = _find_col(df.columns, "mcp")
        df = df[[dcol, tcol, mcol]].copy()
        df[dcol] = df[dcol].ffill()  # exports often merge the date cell across a day's blocks
        df = df.dropna()
        date = pd.to_datetime(df[dcol], dayfirst=True, format="mixed", errors="coerce")
        start = df[tcol].astype(str).str.extract(r"(\d{1,2}:\d{2})")[0]
        ts = date + pd.to_timedelta(start + ":00", errors="coerce")
        price = pd.to_numeric(df[mcol].astype(str).str.replace(",", ""), errors="coerce")
        frames.append(pd.DataFrame({"ts": ts, "mcp_rs_mwh": price}).dropna())
    if not frames:
        raise ValueError(f"Could not find a Date/MCP table in {path.name}")
    return pd.concat(frames)


def _window_mask(idx: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    """[start, end) by time of day; wraps midnight when end <= start. '24:00' allowed."""
    mins = idx.hour * 60 + idx.minute
    a = int(start[:2]) * 60 + int(start[3:])
    b = int(end[:2]) * 60 + int(end[3:])
    return (mins >= a) & (mins < b) if a < b else (mins >= a) | (mins < b)


def tod_adder(idx: pd.DatetimeIndex, windows: list[dict]) -> np.ndarray:
    """Sum of ToD adders (Rs/kWh) per slot. Each window is {start, end, adder[, months]}: [start, end) by time
    of day (wraps midnight), applied only in the listed calendar months (1-12) of the slot if `months` is given,
    otherwise all year. Season-dependent ToD (e.g. KERC 2023 HT: Dec-Jun only) is one window per season."""
    out = np.zeros(len(idx))
    for w in windows:
        m = _window_mask(idx, w["start"], w["end"])
        if w.get("months") is not None:
            m = m & np.isin(idx.month, [int(x) for x in w["months"]])
        out += float(w["adder"]) * m
    return out


def iex_series(year: int) -> pd.Series | None:
    """Real IEX DAM MCP (Rs/kWh) if data/raw/iex covers the year (<=3 days gap-filled), else None."""
    d = RAW / "iex"
    files = sorted(p for p in d.glob("*") if p.suffix.lower() in {".xlsx", ".xls", ".csv"}) if d.exists() else []
    if not files:
        return None
    df = pd.concat([parse_iex_file(p) for p in files]).drop_duplicates("ts").set_index("ts").sort_index()
    s = df["mcp_rs_mwh"].reindex(year_index(year))
    if s.isna().sum() > 4 * 24 * 3:
        print(f"[iex] only {int(s.notna().sum())} of {len(s)} slots present -> treating IEX as PENDING")
        return None
    return s.interpolate(limit_direction="both") / 1000.0


# ----------------------------------------------------------------------------- price scenario A (date-dependent tariff)
def tariff_period(cfg: dict, day) -> dict:
    """The price-scenario-A tariff period in force on `day`: {name, energy_slabs, tod, demand_rs_per_kva_month,
    in_range}. Periods are [from, to) by date (config.yaml prices.A.periods). A date outside every period (the
    benchmark sites' 2017/2019 years) gets prices.A.out_of_range_period with in_range=False. Configs without
    `periods` (configs/v1_generic.yaml) keep the old flat keys: import_base_rs_kwh, tod, tariff_v2 demand charge."""
    a = cfg["prices"]["A"]
    if "periods" not in a:
        dem = cfg.get("tariff_v2", {}).get("demand_rs_per_kva_month", 0.0)
        return {"name": "flat", "energy_slabs": [{"rs_kwh": float(a["import_base_rs_kwh"])}], "tod": a["tod"],
                "demand_rs_per_kva_month": float(dem), "in_range": True}
    try:
        d = pd.Timestamp(day).normalize()
    except (ValueError, TypeError):           # SYNTHETIC unit-test days carry no date
        d = None
    for p in (a["periods"] if d is not None else []):
        if pd.Timestamp(p["from"]) <= d < pd.Timestamp(p["to"]):
            return {**p, "in_range": True}
    p = next(p for p in a["periods"] if p["name"] == a["out_of_range_period"])
    return {**p, "in_range": False}


def demand_rate_month(cfg: dict, day) -> float:
    """Demand charge Rs/kVA/month in force on `day` (date-dependent, see tariff_period)."""
    return float(tariff_period(cfg, day)["demand_rs_per_kva_month"])


def slab_energy_rate(slabs: list[dict], cum_kwh: np.ndarray, e_kwh: np.ndarray) -> np.ndarray:
    """Rs/kWh for slots whose energy e_kwh is billed after cum_kwh already consumed in the billing month.
    slabs = [{first_kwh_per_month: X, rs_kwh: r1}, ..., {rs_kwh: rn}] as written in the KERC schedule ("for the first
    one lakh units ... for the balance units"). A slot straddling a slab boundary gets the energy-weighted mix of the
    two rates, so sum(rate * e) over the month equals the slab bill exactly. e_kwh = 0 -> the marginal rate at cum_kwh."""
    rates = np.full(len(cum_kwh), float(slabs[-1]["rs_kwh"]))
    lo = 0.0
    billed = np.zeros(len(cum_kwh))                   # Rs for e_kwh across the slabs
    for s in slabs:
        hi = lo + float(s["first_kwh_per_month"]) if "first_kwh_per_month" in s else np.inf
        part = np.clip(np.minimum(cum_kwh + e_kwh, hi) - np.maximum(cum_kwh, lo), 0.0, None)
        billed += part * float(s["rs_kwh"])
        marginal = (cum_kwh >= lo) & (cum_kwh < hi)
        rates = np.where((e_kwh <= 0) & marginal, float(s["rs_kwh"]), rates)
        lo = hi
    out = np.where(e_kwh > 0, billed / np.where(e_kwh > 0, e_kwh, 1.0), rates)
    return np.round(out, 9)                            # drop float noise from the cumulative sums (< 1e-9 Rs/kWh)


def price_a_import(idx: pd.DatetimeIndex, cfg: dict, grid_import_kw: np.ndarray | None = None) -> np.ndarray:
    """Scenario-A import price (Rs/kWh) per slot: the energy charge of the tariff period in force on the slot's date
    (two-slab charges use the campus's cumulative grid import in the calendar month, grid_import_kw) plus that
    period's ToD adders."""
    out = np.zeros(len(idx))
    dates = idx.normalize()
    keys = pd.Series([tariff_period(cfg, d)["name"] for d in dates.unique()], index=dates.unique())
    pname = keys.reindex(dates).to_numpy()
    for name in pd.unique(pname):
        sel = pname == name
        per = tariff_period(cfg, idx[sel][0])
        sub = idx[sel]
        slabs = per["energy_slabs"]
        if len(slabs) == 1:
            base = np.full(sel.sum(), float(slabs[0]["rs_kwh"]))
        else:
            if grid_import_kw is None:
                raise ValueError(f"tariff period {name!r} has a slab energy charge: grid_import_kw is required")
            e = np.clip(np.asarray(grid_import_kw, float)[sel], 0.0, None) * (int(cfg["site"]["slot_minutes"]) / 60.0)
            month = sub.to_period("M")
            cum = pd.Series(e).groupby(np.asarray(month)).cumsum().to_numpy() - e      # kWh before this slot
            base = slab_energy_rate(slabs, cum, e)
        out[sel] = base + tod_adder(sub, per["tod"])
    return out


def build_prices(cfg: dict, grid_import_kw: np.ndarray | None = None) -> tuple[pd.DataFrame, dict]:
    """Columns A_import, A_export, B_import, B_export (Rs/kWh). Returns (prices, status).
    grid_import_kw: campus grid import per slot, needed only for tariff periods with slab energy charges."""
    year = cfg["site"]["year"]
    idx = year_index(year)
    pc = cfg["prices"]
    status = {}

    a = pc["A"]
    imp = price_a_import(idx, cfg, grid_import_kw)
    out = pd.DataFrame({"A_import": imp, "A_export": float(a["export_rs_kwh"])}, index=idx)
    status["A"] = {"status": a["status"], "source": a["source"],
                   "note": a["standin_note"] if a["status"] != "REAL" else "",
                   **({"periods": [{k: p[k] for k in ("name", "from", "to")} for p in a["periods"]]} if "periods" in a else {})}

    b = pc["B"]
    real = iex_series(year)
    if real is not None:
        out["B_import"] = real.to_numpy()
        status["B"] = {"status": "REAL", "source": b["source"], "note": ""}
    else:
        prof = np.zeros(len(idx))
        for w in b["standin_profile_rs_kwh"]:
            prof = np.where(_window_mask(idx, w["start"], w["end"]), w["price"], prof)
        out["B_import"] = prof
        status["B"] = {"status": "PENDING", "source": b["source"], "note": b["standin_note"]}
    out["B_export"] = out["B_import"] * float(b["export_price_factor"])
    return out, status


# ----------------------------------------------------------------------------- CEA
def load_co2_factor(cfg: dict) -> dict:
    f = RAW / "cea" / "emission_factor.yaml"
    if f.exists() and list((RAW / "cea").glob("*.pdf")):
        d = yaml.safe_load(f.read_text())
        v = float(d["weighted_average_tco2_per_mwh"])
        assert 0.3 < v < 1.5
        return {"status": "REAL", "tco2_per_mwh": v, **d}
    c = cfg["co2"]
    return {"status": "PENDING", "tco2_per_mwh": float(c["standin_tco2_per_mwh"]), "note": c["standin_note"]}


# ----------------------------------------------------------------------------- scenarios & eval days
def pick_scenarios(cfg: dict, pv_total: pd.Series, dtypes: pd.Series) -> dict:
    daily_pv = pv_total.resample("D").sum() / 4.0
    monsoon = daily_pv[(daily_pv.index.month >= 6) & (daily_pv.index.month <= 9)]
    out = {}
    for k, v in cfg["scenarios"].items():
        d = str(monsoon.idxmin().date()) if v["date"] == "auto" else v["date"]
        out[k] = {"date": d, "day_type": dtypes.get(pd.Timestamp(d), "teaching"), "note": v["note"],
                  "outage": v.get("outage")}
    days = pd.Series(daily_pv.index)
    n = cfg["test_days"]["n_eval_days"]
    step = len(days) / n
    used = {v["date"] for v in out.values()}
    ev = []
    for i in range(n):
        d = days.iloc[int(step * i + step / 2)]
        while str(d.date()) in used:
            d += pd.Timedelta(days=1)
        ev.append(str(d.date()))
    return {"scenarios": out, "eval_days": ev, "eval_criteria": f"{n} days evenly spaced across the year"}


# ----------------------------------------------------------------------------- main
def build_all(cfg: dict | None = None) -> dict:
    from .campus import build_layout

    cfg = cfg or load_config()
    PROC.mkdir(parents=True, exist_ok=True)
    year = cfg["site"]["year"]
    layout = build_layout(cfg)
    weather = load_weather(year)
    weather.to_parquet(PROC / "weather.parquet")
    pv = pd.DataFrame({z: pv_output_kw(weather, layout["zones"][z]["pv_kwp"], cfg["pv_model"])
                       for z in cfg["campus"]["zones"]})
    pv.to_parquet(PROC / "pv.parquet")
    loads, crit, water, load_info = build_zone_loads(cfg, layout)
    loads.to_parquet(PROC / "load.parquet")
    crit.to_parquet(PROC / "crit.parquet")
    water.to_frame("m3").to_parquet(PROC / "water.parquet")
    # slab position for two-slab energy charges (FY2022-23): campus grid import without battery/DG dispatch
    # (status-quo basis, ASSUMPTION: billing month = calendar month), see config.yaml prices.A.periods
    sq_import = (loads.sum(axis=1) - pv.sum(axis=1)).clip(lower=0).to_numpy()
    prices, price_status = build_prices(cfg, sq_import)
    prices.to_parquet(PROC / "price.parquet")
    co2 = load_co2_factor(cfg)
    dtypes = day_types(cfg)
    days = pick_scenarios(cfg, pv.sum(axis=1), dtypes)
    meta = {"year": year, "campus": layout["campus"], "osm_sha256": layout["osm_sha256"], "co2": co2,
            "prices": price_status, "days": days, "loads": load_info,
            "data_status": {"nasa_power": "REAL", "osm_layout": "REAL",
                            **({"bdg2_load_shapes": "REAL"} if any(str(v.get("profile", "")).startswith("bdg2:")
                                                                  for v in load_info["zones"].values() if isinstance(v, dict))
                               else {"ausgrid_hostel_shape": "REAL"}),
                            "iex": price_status["B"]["status"], "tariff_A": price_status["A"]["status"],
                            "cea": co2["status"], "calendar": cfg["calendar"]["status"],
                            "campus_loads": "PENDING RVCE DATA" if not (RAW / "rvce_bills.csv").exists() else "SCALED TO BILLS",
                            "lab_schedule": "TEMPLATE" if "TEMPLATE" in (RAW / "lab_schedule.csv").read_text() else "REAL"}}
    (PROC / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return meta


def load_processed() -> dict:
    return {
        "pv": pd.read_parquet(PROC / "pv.parquet"),
        "load": pd.read_parquet(PROC / "load.parquet"),
        "crit": pd.read_parquet(PROC / "crit.parquet"),
        "water": pd.read_parquet(PROC / "water.parquet")["m3"],
        "campus": json.loads((PROC / "campus.json").read_text()),
        "price": pd.read_parquet(PROC / "price.parquet"),
        "weather": pd.read_parquet(PROC / "weather.parquet"),
        "meta": json.loads((PROC / "meta.json").read_text()),
    }


def day_slice(df: pd.DataFrame | pd.Series, day: str):
    d = pd.Timestamp(day)
    return df.loc[d: d + pd.Timedelta(hours=23, minutes=45)]


if __name__ == "__main__":
    print(json.dumps(build_all(), indent=2, default=str))
