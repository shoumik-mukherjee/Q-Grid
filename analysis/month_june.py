"""June 2023 day by day (no outages): status quo vs Q-Grid (scorecard design 1111), for the monthly HT bill and the
monthly-peak analysis. Saves per-day settlement components and the per-slot campus import of both methods.

  python analysis/month_june.py [workers]   -> analysis/june_days.json

Run on 2026-11-27 (6 workers) with the config of commit 6045f1c (500 kWp PV, SSR floor areas, 980 kVA diesel); the JSON
is that run's output. The monthly bill (billing demand = max(monthly MD, 0.85 x contract demand), Rs 350/kVA, KERC
Tariff Order 2023 p374) and the peak-shaving estimate were computed from it in the session report."""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT), str(ROOT / "scripts")]
SCR = Path(__file__).resolve().parent
CACHE = ROOT / "results" / "analysis"            # forecaster cache (*.pkl, not committed)
_C = {}


def ctx():
    if not _C:
        from qgrid.ablation.forecast import GBMForecaster
        from qgrid.datasets import build
        from qgrid.v2.horizon import YearData
        campus, proc = build("rvce")
        yd = YearData(campus, proc, "A", forecast=False)
        meta = proc["meta"]
        hold = meta["days"]["eval_days"] + [v["date"] for v in meta["days"]["scenarios"].values()]
        fc = GBMForecaster(campus, proc, yd, hold, cache=CACHE / "gbm_rvce.pkl")
        _C.update(campus=campus, proc=proc, yd=yd, fc=fc)
    return _C


def run(day: str) -> dict:
    from baselines import b0_status_quo
    from qgrid.ablation.engine import ACfg, run_day
    from qgrid.model import DT, check_feasibility, make_day, net_power
    from qgrid.v2.tariff import settle_v2
    c = ctx()
    campus, proc, yd, fc = c["campus"], c["proc"], c["yd"], c["fc"]
    dd = make_day(campus, proc, day, "A", "eval", None)
    out = {"day": day, "price_imp": dd.price_imp.tolist(), "load_kw": dd.load.sum(axis=0).tolist(), "pv_kw": dd.pv.sum(axis=0).tolist()}
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for name, fn in [("status_quo", lambda: b0_status_quo.solve(campus, dd)),
                         ("qgrid", lambda: run_day(campus, yd, day, None, ACfg(True, True, True, True), forecaster=fc)[0])]:
            s = fn()
            m = settle_v2(campus, dd, s)
            net = net_power(campus, dd, s)
            imp = np.where(dd.grid[None, :], np.maximum(net, 0), 0.0).sum(axis=0)       # campus import per slot (kW)
            out[name] = {k: float(m[k]) for k in ("energy_cost_rs", "fuel_cost_rs", "terminal_adj_rs", "wear_rs", "voll_rs",
                                                 "demand_charge_rs", "import_kwh", "export_kwh", "peak_kva")}
            out[name]["feasible"] = bool(check_feasibility(campus, s, dd)["feasible"])
            out[name]["import_kw"] = imp.tolist()
            out[name]["battery_throughput_kwh"] = float(np.abs(s.p_bat).sum() * DT)
    return out


if __name__ == "__main__":
    import pandas as pd
    CACHE.mkdir(parents=True, exist_ok=True)
    outdir = SCR
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    days = [str(d.date()) for d in pd.date_range("2023-06-01", "2023-06-30")]
    ctx()                                   # train / load the forecaster once before forking workers
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(run, days))
    (outdir / "june_days.json").write_text(json.dumps(res))
    print("wrote", outdir / "june_days.json", len(res), "days")
