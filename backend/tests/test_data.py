"""Data-pipeline unit tests. All inputs here are SYNTHETIC (unit tests only, never headline results)."""
import numpy as np
import pandas as pd
import pytest

from qgrid import data


def test_pv_model_stc_and_night():
    idx = pd.date_range("2023-01-01", periods=3, freq="15min")
    w = pd.DataFrame({"ghi_wm2": [0.0, 1000.0, 500.0], "t2m_c": [20.0, 25.0 - 31.25, 25.0]}, index=idx)
    pm = {"noct_c": 45.0, "gamma_per_c": -0.004, "system_derate": 0.86}
    p = data.pv_output_kw(w, 100.0, pm).to_numpy()
    assert p[0] == 0
    assert p[1] == pytest.approx(86.0)          # Tcell = 25 C exactly -> no derating
    assert 0 < p[2] < 43.0                      # hotter cell -> derated below linear


def test_map_dates_weekday_and_range():
    m = data.map_dates(2023)
    assert len(m) == 365
    for d, s in m.items():
        assert d.weekday() == s.weekday()
        assert pd.Timestamp("2012-07-01") <= s <= pd.Timestamp("2013-06-30")


def _synthetic_ausgrid_csv(n_cust=12, days=3):
    rng = np.random.default_rng(0)
    cols = [f"{h}:{m:02d}" for h in range(24) for m in (0, 30)][1:] + ["0:00"]
    rows = []
    for c in range(1, n_cust + 1):
        for d in pd.date_range("2012-07-01", periods=days):
            for cat in ("GC", "CL", "GG"):
                rows.append([c, 2.0, 2000, cat, d.strftime("%d/%m/%Y")] + list(rng.random(48).round(3)))
    hdr = ["Customer", "Generator Capacity", "Postcode", "Consumption Category", "date"] + cols
    body = pd.DataFrame(rows, columns=hdr).to_csv(index=False)
    return ("Solar home half-hour data - SYNTHETIC test file\n" + body).encode()


def test_ausgrid_parse_and_cluster():
    df = data._parse_ausgrid(_synthetic_ausgrid_csv())
    kw = data.ausgrid_customer_loads(df)
    assert kw.shape == (3 * 48, 12)
    assert kw.index[0] == pd.Timestamp("2012-07-01 00:00")
    # GC + CL only, kWh/0.5h -> kW
    raw = df[(df.Customer == 1) & (df["Consumption Category"].isin(["GC", "CL"])) & (df.date == "01/07/2012")]
    assert kw.iloc[0, 0] == pytest.approx(raw["0:30"].sum() / 0.5)
    ids, labels, cents = data.cluster_profiles(kw, k=3)
    assert len(ids) == 12 and cents.shape == (3, 48)


def test_iex_parser(tmp_path):
    rows = [["IEX DAM Market Snapshot (SYNTHETIC)", None, None], [None, None, None],
            ["Date", "Time Block", "MCP (Rs/MWh) *"]]
    for i in range(96):
        h, m = divmod(i * 15, 60)
        rows.append(["01-01-2023" if i == 0 else None, f"{h:02d}:{m:02d} - {h:02d}:{m + 14:02d}", f"{3000 + i:,}"])
    p = tmp_path / "x.xlsx"
    pd.DataFrame(rows).to_excel(p, header=False, index=False)
    out = data.parse_iex_file(p)
    assert len(out) == 96
    assert out.iloc[0].ts == pd.Timestamp("2023-01-01 00:00")
    assert out.iloc[-1].mcp_rs_mwh == 3095


def test_window_mask_wraps_midnight():
    idx = pd.date_range("2023-01-01", periods=96, freq="15min")
    m = data._window_mask(idx, "22:00", "06:00")
    assert m.sum() == 32 and m[0] and not m[40] and m[95]


def test_price_scenarios_standin_shapes():
    cfg = data.load_config()
    prices, status = data.build_prices(cfg, np.full(365 * 96, 1000.0))   # SYNTHETIC campus import (slab tariff)
    assert {"A_import", "A_export", "B_import", "B_export"} <= set(prices.columns)
    assert len(prices) == 365 * 96
    for k in ("A", "B"):
        assert status[k]["status"] in ("REAL", "PENDING")
        if status[k]["status"] == "PENDING":
            assert "STAND-IN" in status[k]["note"]


DEC_JUN = [12, 1, 2, 3, 4, 5, 6]


def test_tod_adder_season_dependent():
    """SYNTHETIC windows: Dec-Jun 18-22 h +1.00, 22-06 h -1.00; Jul-Nov no ToD; windows without months = all year."""
    win = [{"start": "18:00", "end": "22:00", "adder": 1.00, "months": DEC_JUN},
           {"start": "22:00", "end": "06:00", "adder": -1.00, "months": DEC_JUN}]
    idx = pd.DatetimeIndex(["2023-01-10 07:00", "2023-01-10 18:00", "2023-01-10 21:45", "2023-01-10 22:00",
                            "2023-01-10 05:45", "2023-06-30 19:00", "2023-07-01 19:00", "2023-08-15 23:00",
                            "2023-11-30 19:00", "2023-12-01 19:00", "2023-12-31 02:00"])
    got = data.tod_adder(idx, win)
    assert got.tolist() == [0.0, 1.0, 1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0, -1.0]
    allyear = data.tod_adder(idx[[6, 7]], [{"start": "18:00", "end": "22:00", "adder": 1.0}])
    assert allyear.tolist() == [1.0, 0.0]                 # no `months` key -> every month


def test_slab_energy_rate_bills_exactly():
    """SYNTHETIC two-slab charge: first 100 kWh at 1.00, balance at 2.00 per billing month."""
    slabs = [{"first_kwh_per_month": 100, "rs_kwh": 1.0}, {"rs_kwh": 2.0}]
    e = np.array([60.0, 60.0, 60.0, 0.0])
    cum = np.array([0.0, 60.0, 120.0, 180.0])
    r = data.slab_energy_rate(slabs, cum, e)
    assert r[0] == pytest.approx(1.0)
    assert r[1] == pytest.approx((40 * 1.0 + 20 * 2.0) / 60)    # straddles the slab boundary
    assert r[2] == pytest.approx(2.0)
    assert r[3] == pytest.approx(2.0)                          # no energy -> marginal rate
    assert float((r * e).sum()) == pytest.approx(100 * 1.0 + 80 * 2.0)   # equals the slab bill


def test_price_scenario_a_date_dependent_kerc():
    """Either side of 2023-04-01: FY2022-23 (two-slab 825/865 paise, Rs 265/kVA) vs FY2023-24 (850 paise, Rs 350/kVA).
    Grid import is SYNTHETIC (constant 2,000 kW = 500 kWh per slot -> the 1 lakh slab is used up after 200 slots)."""
    cfg = data.load_config()
    if cfg["prices"]["A"]["status"] != "REAL":
        pytest.skip("price scenario A is not REAL")
    idx = data.year_index(2023)
    s = pd.Series(data.price_a_import(idx, cfg, np.full(len(idx), 2000.0)), index=idx)
    # FY2022-23 (before 2023-04-01)
    assert s["2023-03-01 00:00"] == pytest.approx(8.25 - 1.0)   # first lakh units, Dec-Jun night rebate
    assert s["2023-03-31 08:00"] == pytest.approx(8.65)         # balance units, no morning peak
    assert s["2023-03-31 19:00"] == pytest.approx(8.65 + 1.0)   # balance units + evening peak
    assert data.demand_rate_month(cfg, "2023-03-31") == 265.0
    # FY2023-24 (from 2023-04-01)
    assert s["2023-04-01 08:00"] == pytest.approx(8.50)
    assert s["2023-04-01 19:00"] == pytest.approx(8.50 + 1.0)
    assert s["2023-04-01 23:00"] == pytest.approx(8.50 - 1.0)
    assert s["2023-08-10 19:00"] == pytest.approx(8.50)         # Jul-Nov: no ToD
    assert data.demand_rate_month(cfg, "2023-04-01") == 350.0
    # slab charge needs the campus import
    with pytest.raises(ValueError):
        data.price_a_import(idx, cfg)
    # benchmark sites (2017/2019) are outside every period -> FY2023-24 for comparability
    p = data.tariff_period(cfg, "2017-06-01")
    assert p["name"] == cfg["prices"]["A"]["out_of_range_period"] and not p["in_range"]
    s17 = data.price_a_import(data.year_index(2017), cfg)
    assert set(np.round(np.unique(s17), 2)) == {7.5, 8.5, 9.5}
