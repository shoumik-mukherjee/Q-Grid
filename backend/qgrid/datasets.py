"""Benchmark datasets: every campus becomes the same structures the methods use (a Campus and a `proc` dict shaped
like data.load_processed()), so each method runs unchanged on each dataset.

  rvce            RVCE model (OSM layout, NASA POWER 2023, BDG2 load shapes scaled by floor area x EPI)
  bdg2_fox|bull|cockatoo   BDG2 university sites, 2017: zones = groups of the site's own metered buildings (real kW)
  cubems          CU-BEMS, Chamchuri 5 office building (Bangkok), 2019: zones = its 7 floors (real kW)

For the non-RVCE sites the physical assets are not in the data, so one rule (ASSUMPTION) sizes them everywhere:
  PV kWp      30 % of the zone's annual energy / the site's annual PV yield per kWp (NASA POWER all-sky)
  battery     2 h of the zone's mean load (kWh), C-rate 0.5
  diesel      one set at the zone with the largest critical load, 50 % of its peak
  critical    dormitories 10 %, every other zone 5 % of load
  cables      zones in a chain plus two seeded cross-links; rating 25 % of the median zone peak
  tariff      the stand-in tariff (price scenario A) on local clock time, PENDING like RVCE's
Outages (all datasets, labelled SYNTHETIC): each evaluation day has a seeded 20 % chance of one unplanned grid outage
of 1-3 h starting between 08:00 and 20:00. Official BESCOM planned-cut schedules are PENDING.
"""
from __future__ import annotations

import copy
import json
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as qdata
from .model import T, Battery, Building, Campus, DG, load_campus

RAW = qdata.RAW


@dataclass(frozen=True)
class Spec:
    key: str
    label: str
    kind: str
    year: int
    site: str = ""
    lat: float = 0.0
    lon: float = 0.0
    licence: str = ""


SPECS = {
    "rvce": Spec("rvce", "RVCE, Bengaluru (case study)", "rvce", 2023, licence="see data/SOURCES.md"),
    "bdg2_fox": Spec("bdg2_fox", "BDG2 university site 'Fox' (Arizona)", "bdg2", 2017, "Fox", 33.424425, -111.928140, "CC BY-SA 4.0"),
    "bdg2_bull": Spec("bdg2_bull", "BDG2 university site 'Bull' (Texas)", "bdg2", 2017, "Bull", 30.2672, -97.7431, "CC BY-SA 4.0"),
    "bdg2_cockatoo": Spec("bdg2_cockatoo", "BDG2 university site 'Cockatoo' (New York)", "bdg2", 2017, "Cockatoo", 42.459837, -76.485292, "CC BY-SA 4.0"),
    "cubems": Spec("cubems", "CU-BEMS office building, Bangkok (7 floors)", "cubems", 2019, "Chamchuri 5", 13.7367, 100.5332, "CC BY 4.0"),
}

GROUPS = [   # BDG2 zone groups (primary use, sub-uses or None for all)
    ("academic", "Education", {"College Classroom", "Academic", "Classroom", "Education", "Student Union"}),
    ("labs", "Education", {"College Laboratory", "Research"}),
    ("dorms", "Lodging/residential", None),
    ("library", "Public services", {"Library"}),
    ("offices", "Office", None),
    ("other", None, None),
]


def _weather(spec: Spec) -> tuple[pd.DataFrame, np.ndarray]:
    """15-min weather for the site/year (GHI, T2M) and clear-sky GHI."""
    p = json.loads((RAW / "nasa_power" / f"{spec.key}_{spec.year}_hourly.json").read_text())["properties"]["parameter"]
    idx_h = pd.to_datetime(list(p["T2M"].keys()), format="%Y%m%d%H")
    h = pd.DataFrame({"ghi_wm2": list(p["ALLSKY_SFC_SW_DWN"].values()), "t2m_c": list(p["T2M"].values()),
                      "cs": list(p["CLRSKY_SFC_SW_DWN"].values())}, index=idx_h).clip(lower=-50)
    h[["ghi_wm2", "cs"]] = h[["ghi_wm2", "cs"]].clip(lower=0)
    idx = qdata.year_index(spec.year)
    w = h.reindex(idx, method="ffill").fillna(0.0)
    return w[["ghi_wm2", "t2m_c"]], w["cs"].to_numpy()


def _day_types(idx: pd.DatetimeIndex, office: bool) -> np.ndarray:
    from .bdg2 import _is_break
    d = idx.normalize()
    out = np.where(d.weekday >= 5, "holiday", "teaching")
    if not office:
        brk = np.array([_is_break(x) for x in d])
        out = np.where((out == "teaching") & brk, "vacation", out)
    return out


def _bdg2_zones(spec: Spec) -> tuple[pd.DataFrame, dict]:
    from .bdg2 import _meters
    el, md = _meters()
    md = md[(md.site_id == spec.site) & md.index.isin(el.columns)]
    x = el[md.index]
    x = x[(x.index.year == spec.year)]
    ok = [b for b in x.columns if x[b].notna().mean() >= 0.9 and x[b].mean() > 1.0]
    md = md.loc[ok]
    zones, info = {}, {}
    used = set()
    for name, prim, subs in GROUPS:
        if prim is None:
            ids = [b for b in md.index if b not in used]
        else:
            m = md.primaryspaceusage == prim
            if subs is not None:
                m &= md.sub_primaryspaceusage.isin(subs)
            ids = [b for b in md.index[m] if b not in used]
        if not ids:
            continue
        used.update(ids)
        xx = x[ids].clip(upper=x[ids].quantile(0.999), axis=1)          # meter spikes (as qgrid/bdg2.py)
        s = xx.interpolate(limit=24).bfill().ffill().sum(axis=1)
        zones[name] = s
        info[name] = {"buildings": len(ids), "mean_kw": round(float(s.mean()), 1)}
    df = pd.DataFrame(zones)
    return df, info


def _cubems_zones(spec: Spec) -> tuple[pd.DataFrame, dict]:
    zones, info = {}, {}
    for f in range(1, 8):
        zp = RAW / "cubems" / f"2019Floor{f}.csv.zip"
        with zipfile.ZipFile(zp) as z:
            name = [n for n in z.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX")][0]
            df = pd.read_csv(z.open(name))
        df.index = pd.to_datetime(df.pop("Date"), errors="coerce", format="mixed")
        df = df[df.index.notna()]
        kw = df[[c for c in df.columns if c.endswith("(kW)")]].clip(lower=0).sum(axis=1, min_count=1)
        kw = kw.clip(upper=kw.quantile(0.999))
        hourly = kw.resample("1h").mean()
        zones[f"floor{f}"] = hourly
        info[f"floor{f}"] = {"columns": int(sum(c.endswith("(kW)") for c in df.columns)), "mean_kw": round(float(kw.mean()), 1)}
    df = pd.DataFrame(zones)
    full = pd.date_range(f"{spec.year}-01-01", f"{spec.year}-12-31 23:00", freq="1h")
    df = df.reindex(full).interpolate(limit=24)
    # long gaps: fill with the same hour of the same weekday in the nearest available week
    df = df.fillna(df.groupby([df.index.weekday, df.index.hour]).transform("median"))
    return df, info


def _outages(days: list[str], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for d in days:
        if rng.random() < 0.20:
            a = int(rng.integers(8 * 4, 20 * 4 + 1)); dur = int(rng.integers(4, 13))
            out[d] = [f"{a // 4:02d}:{a % 4 * 15:02d}", f"{(a + dur) // 4:02d}:{(a + dur) % 4 * 15:02d}"]
    return out


def _eval_days(year: int, seed: int, n: int = 30) -> list[str]:
    days = pd.date_range(f"{year}-01-10", f"{year}-12-20", freq="D")
    rng = np.random.default_rng(seed)
    step = len(days) // n
    return [str(days[k * step + int(rng.integers(0, step))].date()) for k in range(n)]


@lru_cache(maxsize=8)
def build(key: str) -> tuple[Campus, dict]:
    spec = SPECS[key]
    seed = sum(map(ord, key))
    if spec.kind == "rvce":
        campus = load_campus()
        proc = qdata.load_processed()
        proc["meta"] = copy.deepcopy(proc["meta"])
        days = proc["meta"]["days"]["eval_days"]
        proc["meta"]["days"]["outages"] = _outages(days, seed)
        proc["meta"]["dataset"] = {"key": key, "label": spec.label}
        return campus, proc
    w, cs_ghi = _weather(spec)
    idx = qdata.year_index(spec.year)
    cfg = qdata.load_config()
    if spec.kind == "bdg2":
        hourly, zinfo = _bdg2_zones(spec)
    else:
        hourly, zinfo = _cubems_zones(spec)
    hourly.index = pd.DatetimeIndex(hourly.index)
    load = hourly.reindex(idx, method="ffill").bfill()
    pv1 = qdata.pv_output_kw(w, 1.0, cfg["pv_model"]).to_numpy()
    wcs = w.copy(); wcs["ghi_wm2"] = cs_ghi
    cs1 = np.maximum(qdata.pv_output_kw(wcs, 1.0, cfg["pv_model"]).to_numpy(), pv1)
    yield_kwh = pv1.sum() * 0.25
    bd = cfg["battery_defaults"]; dsl = cfg["diesel"]
    crit_frac = {z: (0.10 if z == "dorms" else 0.05) for z in load.columns}
    dg_zone = max(load.columns, key=lambda z: crit_frac[z] * load[z].mean())
    blds = []
    for z in load.columns:
        ann = load[z].sum() * 0.25
        kwp = round(0.30 * ann / yield_kwh, 1)
        kwh = round(2 * load[z].mean(), 0)
        bat = Battery(kwh=float(kwh), kw=float(round(kwh * 0.5, 0)), **bd)
        dg = DG(float(round(0.5 * load[z].quantile(0.99), 0)), 0.3, 4, dsl["fuel_l_per_kwh_rated"], dsl["fuel_l_per_kwh_out"],
                dsl["price_rs_per_l"], dsl["co2_kg_per_l"]) if z == dg_zone else None
        blds.append(Building(z, float(kwp), crit_frac[z], bat, [], dg, None))
    n = len(blds)
    rng = np.random.default_rng(seed)
    lines = [(k, k + 1) for k in range(n - 1)]
    cand = [(a, b) for a in range(n) for b in range(a + 2, n)]
    for k in rng.choice(len(cand), size=min(2, len(cand)), replace=False):
        lines.append(cand[k])
    step = float(max(10.0, round(0.25 * np.median(load.quantile(0.99).to_numpy()) / 10) * 10))
    layout = {"zones": {b.name: {"pv_kwp": b.pv_kwp} for b in blds}, "lines": [[blds[a].name, blds[b].name] for a, b in lines]}
    campus = Campus(blds, lines, step, cfg, layout)
    names = campus.names
    pv = pd.DataFrame({b.name: b.pv_kwp * pv1 for b in blds}, index=idx)
    crit = pd.DataFrame({z: load[z] * crit_frac[z] for z in names}, index=idx)
    water = pd.Series(np.zeros(len(idx)), index=idx)
    # stand-in tariff A on local clock time (same rates as RVCE's stand-in)
    pa = cfg["prices"]["A"]
    imp = np.full(len(idx), pa["import_base_rs_kwh"])
    for r in pa["tod"]:
        imp = imp + pa_adder(idx, r)
    price = pd.DataFrame({"A_import": imp, "A_export": np.full(len(idx), pa["export_rs_kwh"])}, index=idx)
    price["B_import"], price["B_export"] = price.A_import, price.A_export
    days = _eval_days(spec.year, seed)
    dtypes = _day_types(idx, office=spec.kind == "cubems")
    meta = {"year": spec.year, "co2": {"tco2_per_mwh": cfg["co2"]["standin_tco2_per_mwh"], "status": "PENDING"},
            "prices": {"A": {"status": "PENDING"}, "B": {"status": "PENDING"}},
            "days": {"eval_days": days, "scenarios": {}, "outages": _outages(days, seed)},
            "data_status": {"loads": f"REAL ({spec.label}, {spec.licence})", "weather": "REAL (NASA POWER)",
                            "assets": "ASSUMPTION (sizing rule, qgrid/datasets.py)", "tariff": "PENDING stand-in",
                            "outages": "SYNTHETIC seeded"},
            "dataset": {"key": key, "label": spec.label, "zones": zinfo, "lines": layout["lines"], "step_kw": step,
                        "pv_kwp": {b.name: b.pv_kwp for b in blds}, "battery_kwh": {b.name: b.battery.kwh for b in blds},
                        "dg": {dg_zone: blds[names.index(dg_zone)].dg.kw}}}
    proc = {"pv": pv, "load": load[names], "crit": crit, "water": water, "price": price,
            "weather": w, "meta": meta, "campus": layout, "per_kwp_act": pv1, "per_kwp_cs": cs1, "day_type": dtypes}
    return campus, proc


def pa_adder(idx: pd.DatetimeIndex, r: dict) -> np.ndarray:
    a = int(r["start"][:2]) * 4 + int(r["start"][3:]) // 15
    b = int(r["end"][:2]) * 4 + int(r["end"][3:]) // 15
    slot = idx.hour * 4 + idx.minute // 15
    m = (slot >= a) & (slot < b) if a < b else (slot >= a) | (slot < b)
    if r.get("months") is not None:              # season-dependent ToD (same rule as qgrid.data.tod_adder)
        m = m & np.isin(idx.month, [int(x) for x in r["months"]])
    return np.where(m, r["adder"], 0.0)
