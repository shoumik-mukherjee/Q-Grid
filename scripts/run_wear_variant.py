"""Design change under test: Q-Grid with battery wear priced in each building's objective.

  python scripts/run_wear_variant.py [--workers=2]   -> results/impact/wear_variant.csv

Same RVCE evaluation days, forecasts, outages and settlement as the scorecard (scripts/run_bench_v3.py); the only
change is ACfg.wear_rs_per_kwh = battery cost / (2 x cycle life x depth of discharge), at the mid-point assumptions in
config.yaml -> impact (the same rate the impact page uses for battery wear). Not part of the scorecard."""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT), str(ROOT / "scripts")]


def wear_rate() -> float:
    im = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))["impact"]
    mid = lambda v: 0.5 * (v[0] + v[1])   # noqa: E731
    return mid(im["battery_capex_rs_per_kwh"]) / (2 * mid(im["battery_cycle_life"]) * float(im["battery_dod"]))


def one(day: str, rate: float) -> dict:
    import run_bench_v3 as rb
    from qgrid.ablation.engine import ACfg, run_day
    from qgrid.model import check_feasibility, make_day
    from qgrid.v2.tariff import settle_v2
    c = rb.ctx("rvce")
    campus, proc, yd, fc = c["campus"], c["proc"], c["yd"], c["fc"]
    outage = proc["meta"]["days"]["outages"].get(day)
    dd = make_day(campus, proc, day, "A", "eval", outage)
    t0 = time.perf_counter()
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sched, st = run_day(campus, yd, day, outage, ACfg(True, True, True, True, wear_rs_per_kwh=rate), forecaster=fc)
    m = settle_v2(campus, dd, sched)
    x = rb.extra_metrics(campus, dd, sched, m)
    return {"dataset": "rvce", "day": day, "method": "qgrid_wear", "wear_rs_per_kwh": rate, "feasible": check_feasibility(campus, sched, dd)["feasible"],
            **{k: m[k] for k in ("objective_rs", "cost_rs", "cost_incl_demand_rs", "critical_downtime_h", "critical_unserved_kwh", "unserved_kwh",
                                 "co2_kg", "dg_litres", "peak_kva")}, **x, "runtime_s": time.perf_counter() - t0}


def main(workers: int):
    import run_bench_v3 as rb
    rate = wear_rate()
    days = rb.ctx("rvce")["proc"]["meta"]["days"]["eval_days"][:30]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, d, rate): d for d in days}
        for n, f in enumerate(as_completed(futs), 1):
            r = f.result(); rows.append(r)
            print(f"[{n}/{len(days)}] {r['day']} bill {r['cost_incl_demand_rs']:,.0f} cycles {r['battery_cycles']:.2f}", flush=True)
    out = ROOT / "results" / "impact"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "wear_variant.csv", index=False)
    print(f"wrote {out / 'wear_variant.csv'} (wear Rs {rate:.2f}/kWh)")


if __name__ == "__main__":
    from qgrid.parallel import workers as _workers
    main(_workers(next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--workers=")), None), default=2))
