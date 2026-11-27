"""Fair comparison (analysis/fair_compare.py): status quo vs Q-Grid (scorecard design 1111) on the current config,
full objective breakdown and a step-by-step attribution of Q-Grid's saving.
Attribution: start from the status-quo schedule and swap in Q-Grid's decisions one group at a time
(flex start times -> pumps -> tie-line flows -> batteries + diesel); each step is re-settled with the same settlement
(energy, fuel, end-of-day storage value, demand charge, wear, lost load). Order-dependent; interactions land in the
later steps.

  python analysis/fair_compare.py   -> analysis/fair_compare.json

Run on 2026-11-27 with the config of the commit that added this file (500 kWp PV, SSR floor areas, 980 kVA diesel,
KERC 2022/2023 tariff by date); the JSON is that run's output."""
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT), str(ROOT / "scripts")]
SCR = Path(__file__).resolve().parent
CACHE = ROOT / "results" / "analysis"            # forecaster cache (*.pkl, not committed)
CACHE.mkdir(parents=True, exist_ok=True)

from baselines import b0_status_quo  # noqa: E402
from qgrid.ablation.engine import ACfg, run_day  # noqa: E402
from qgrid.ablation.forecast import GBMForecaster  # noqa: E402
from qgrid.datasets import build  # noqa: E402
from qgrid.model import DT, check_feasibility, make_day, net_power, flex_load, pump_load  # noqa: E402
from qgrid.v2.horizon import YearData  # noqa: E402
from qgrid.v2.tariff import settle_v2  # noqa: E402

campus, proc = build("rvce")
yd = YearData(campus, proc, "A", forecast=False)
meta = proc["meta"]
hold = meta["days"]["eval_days"] + [v["date"] for v in meta["days"]["scenarios"].values()]
fc = GBMForecaster(campus, proc, yd, hold, cache=CACHE / "gbm_rvce.pkl")
hs = meta["days"]["scenarios"][campus.cfg["headline_scenario"]]
DAYS = [("headline outage_fest", hs["date"], hs.get("outage")),
        ("teaching Dec-Jun (ToD)", "2023-04-26", None),
        ("teaching Jul-Nov (no ToD)", "2023-09-13", None)]
KEYS = ["energy_cost_rs", "fuel_cost_rs", "terminal_adj_rs", "demand_charge_rs", "wear_rs", "voll_rs"]


def parts(dd, s):
    m = settle_v2(campus, dd, s)
    out = {k: float(m[k]) for k in KEYS}
    out["total_rs"] = sum(out.values())
    out["check_objective_incl_demand_rs"] = float(m["objective_incl_demand_rs"])
    for k in ("import_kwh", "export_kwh", "peak_kva", "critical_downtime_h", "dg_litres"):
        out[k] = float(m[k])
    return out


res = {}
with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
    for label, day, outage in DAYS:
        dd = make_day(campus, proc, day, "A", "eval", outage)
        s0 = b0_status_quo.solve(campus, dd)
        sq, _ = run_day(campus, yd, day, outage, ACfg(True, True, True, True), forecaster=fc)
        r = {"day": day, "outage": outage, "wear_rate_rs_per_kwh": float(dd.wear),
             "price_min_max": [float(dd.price_imp.min()), float(dd.price_imp.max())], "export_price": float(dd.price_exp.max()),
             "feasible": [bool(check_feasibility(campus, s0, dd)["feasible"]), bool(check_feasibility(campus, sq, dd)["feasible"])],
             "status_quo": parts(dd, s0), "qgrid": parts(dd, sq)}
        # step-by-step attribution
        steps, cur = [], copy.deepcopy(s0)
        prev = parts(dd, cur)
        for name, fields in [("load shifting (flexible loads)", ["flex_start"]), ("pump scheduling (water tank)", ["pump_on"]),
                             ("sharing between buildings (tie-lines)", ["flows"]),
                             ("batteries + diesel dispatch", ["p_bat", "dg_kw"])]:
            for f in fields:
                setattr(cur, f, copy.deepcopy(getattr(sq, f)))
            now = parts(dd, cur)
            steps.append({"step": name, **{k: now[k] - prev[k] for k in KEYS + ["total_rs", "peak_kva", "import_kwh"]}})
            prev = now
        r["steps"] = steps
        # battery detail (Q-Grid schedule): where the charging energy came from, prices it bought and sold at
        pre = net_power(campus, dd, sq) - sq.p_bat                     # building net before its battery
        ch = np.maximum(sq.p_bat, 0)
        dis = np.maximum(-sq.p_bat, 0)
        from_surplus = np.minimum(ch, np.maximum(-pre, 0))
        g = dd.grid[None, :]
        price = np.broadcast_to(dd.price_imp, ch.shape)
        r["battery"] = {"charge_kwh": float(ch.sum() * DT), "discharge_kwh": float(dis.sum() * DT),
                        "charge_from_own_solar_surplus_kwh": float(from_surplus.sum() * DT),
                        "charge_from_grid_kwh": float(((ch - from_surplus) * g).sum() * DT),
                        "discharge_during_outage_kwh": float((dis * ~g).sum() * DT),
                        "avg_import_price_when_charging": float((ch * price).sum() / max(ch.sum(), 1e-9)),
                        "avg_import_price_when_discharging": float((dis * price).sum() / max(dis.sum(), 1e-9)),
                        "throughput_kwh": float(np.abs(sq.p_bat).sum() * DT),
                        "wear_rs": float(dd.wear * np.abs(sq.p_bat).sum() * DT),
                        "status_quo_throughput_kwh": float(np.abs(s0.p_bat).sum() * DT)}
        # flex shifts
        r["flex_moves"] = [(campus.names[i], k, int(s0.flex_start[i].get(k, -1)), int(v))
                           for i in range(campus.n) for k, v in sq.flex_start[i].items() if v != s0.flex_start[i].get(k)]
        res[label] = r
        print(label, "done", flush=True)
(SCR / "fair_compare.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print("wrote")
