"""Benchmark v3: structured, fair, multi-dataset.

Fairness protocol
  - identical data, constraints, value of lost load and objective for every method on a dataset-day
  - every non-oracle method sees the same realistic forecasts (gradient-boosted trees, qgrid/ablation/forecast.py):
    day-ahead methods (GA, PSO, the earlier prototype) get the forecast issued at 00:00, MPC methods the rolling one;
    reactive methods (status quo, rule-based) use only the current measurement
  - the only oracle is the reference "best possible plan" (central MILP, perfect information), shown separately
  - same machine and process pool, fixed seeds; certified gap from proven lower bounds (qgrid/ablation/bounds.py)
  - comms-failure methods share one seeded 3 h campus network outage per day
  - battery wear (config battery_wear) is priced in every optimiser's objective and in the scored objective; the
    earlier design rows (Benders, first prototype) predate it and do not price it

  python scripts/run_bench_v3.py [--workers=8] [--days=30] [--qgrid=1111] [--datasets=rvce,cubems] [--resume]
      -> results/bench_v3/runs.csv, bounds.csv, run_info.json
  --resume keeps the rows already in runs.csv / bounds.csv and runs only the missing (dataset, day, method) and
  (dataset, day) bound tasks. Without --resume the script refuses to overwrite a non-empty runs.csv.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT), str(ROOT / "scripts")]
OUT = ROOT / "results" / "bench_v3"
DATASETS = ["rvce", "bdg2_fox", "bdg2_bull", "bdg2_cockatoo", "cubems"]
MAIN = ["status_quo", "rule_based", "ga", "pso", "per_building_mpc", "central_mpc", "central_mpc_comms_fail",
        "qgrid", "qgrid_comms_fail", "optimum"]
DESIGN = ["design_prototype_1h_binary", "design_earlier_prototype", "design_benders_exact", "design_benders_qaoa",
          "design_benders_qiea"]                                   # RVCE only, shown under Verify -> Design choices
WITHOUT_COORDINATOR = {"status_quo": True, "rule_based": True, "ga": True, "pso": True, "per_building_mpc": True,
                       "central_mpc": False, "central_mpc_comms_fail": False, "qgrid": True, "qgrid_comms_fail": True,
                       "optimum": False}
QGRID = "1111"
_C: dict = {}


def ctx(ds: str):
    if ds not in _C:
        from qgrid.ablation.forecast import GBMForecaster
        from qgrid.datasets import build
        from qgrid.v2.horizon import YearData
        campus, proc = build(ds)
        yd = YearData(campus, proc, "A", forecast=False)
        meta = proc["meta"]
        hold = meta["days"]["eval_days"] + [v["date"] for v in meta["days"].get("scenarios", {}).values()]
        fc = GBMForecaster(campus, proc, yd, hold, cache=OUT / f"gbm_{ds}.pkl")
        _C[ds] = {"campus": campus, "proc": proc, "yd": yd, "fc": fc}
    return _C[ds]


def comms_window(ds: str, day: str) -> tuple[int, int]:
    rng = np.random.default_rng(sum(map(ord, ds + day)))
    a = int(rng.integers(6 * 4, 18 * 4 + 1)) // 4 * 4                 # hour-aligned, 06:00-18:00
    return a, a + 11


def forecast_day(c, dd, day: str):
    """DayData with solar / load / critical load replaced by the forecast issued at 00:00 of the day."""
    yd, fc = c["yd"], c["fc"]
    d0 = yd.day_start(day)
    pv = yd.kwp[:, None] * fc.solar_per_kwp(d0, 96)[None, :]
    load = np.vstack([fc.load(i, d0, 96) for i in range(c["campus"].n)])
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(dd.load > 1e-9, dd.crit / dd.load, 0.0)
    return replace(dd, pv=pv, load=load, crit=np.minimum(load * share, load))


class _FcYear:
    """YearData stand-in whose horizon windows carry the rolling forecast (for the Benders design variants)."""
    def __init__(self, yd, fc):
        self.__dict__.update(yd.__dict__)
        self._fc = fc

    def day_start(self, day):
        return int(np.searchsorted(self.index, pd.Timestamp(day)))

    def window(self, day0, t0, H, outage):
        return self._fc.window(day0, t0, H, outage)


def extra_metrics(campus, dd, sched, m) -> dict:
    from qgrid.model import DT, flex_load, pump_load
    econ = campus.cfg.get("economics", {})
    capex = float(econ.get("battery_capex_rs_per_kwh", 25000))
    life = float(econ.get("battery_cycle_life", 4000))          # ASSUMPTION (config.yaml economics)
    dod = 0.75
    thr = float(np.abs(sched.p_bat).sum() * DT)                 # kWh charged + discharged
    cap = sum(b.battery.kwh for b in campus.buildings) or 1.0
    net = m["net_kw"]
    demand = dd.load + flex_load(campus, sched) + pump_load(campus, sched)
    deficit = np.where(dd.grid[None, :], 0.0, np.maximum(net, 0))
    shed_c = np.maximum(deficit - np.maximum(demand - dd.crit, 0), 0)
    crit_tot = dd.crit.sum(axis=0)
    big = (shed_c.sum(axis=0) > 0.01 * np.maximum(crit_tot, 1e-9))
    # degradation: capex per kWh of capacity spread over (cycle life x 2 x depth of discharge) kWh of throughput
    return {"battery_cycles": thr / (2 * cap), "battery_degradation_rs": thr * capex / (2 * life * dod),
            "critical_downtime_1pct_h": float(big.sum() * DT)}


def run_one(ds: str, day: str, method: str, return_sched: bool = False) -> dict:
    from baselines import b0_status_quo, b1_rule_based, b2_ga_pso, b4_central_milp
    from qgrid.ablation.engine import ACfg, run_day
    from qgrid.admm import run_day as v1_run_day
    from qgrid.model import check_feasibility, make_day
    from qgrid.v2.mpc import V2Config, run_day_v2
    from qgrid.v2.tariff import settle_v2
    c = ctx(ds)
    campus, proc, yd, fc = c["campus"], c["proc"], c["yd"], c["fc"]
    outage = proc["meta"]["days"]["outages"].get(day)
    dd = make_day(campus, proc, day, "A", "eval", outage)
    win = comms_window(ds, day)
    q = [ch == "1" for ch in QGRID]
    t0 = time.perf_counter()
    st = {}
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        if method == "status_quo":
            sched = b0_status_quo.solve(campus, dd)
        elif method == "rule_based":
            sched = b1_rule_based.solve(campus, dd)
        elif method in ("ga", "pso"):
            sched = b2_ga_pso.solve(campus, forecast_day(c, dd, day), method.upper())
        elif method == "optimum":
            sched = b4_central_milp.solve(campus, dd)
        elif method == "per_building_mpc":
            sched, st = run_day(campus, yd, day, outage, ACfg(True, True, True, True, share=False), forecaster=fc)
        elif method == "central_mpc":
            sched, st = run_day(campus, yd, day, outage, ACfg(True, True, True, True, central=True), forecaster=fc)
        elif method == "central_mpc_comms_fail":
            sched, st = run_day(campus, yd, day, outage, ACfg(True, True, True, True, central=True, comms_fail=win), forecaster=fc)
        elif method == "qgrid":
            sched, st = run_day(campus, yd, day, outage, ACfg(*q), forecaster=fc)
        elif method == "qgrid_comms_fail":
            sched, st = run_day(campus, yd, day, outage, ACfg(*q, comms_fail=win), forecaster=fc)
        elif method == "design_prototype_1h_binary":
            sched, st = run_day(campus, yd, day, outage, ACfg(False, False, False, True), forecaster=fc)
        elif method == "design_earlier_prototype":
            sched, s1 = v1_run_day(campus, forecast_day(c, dd, day), method="classical")
            st = {"admm_converged_rate": s1.summary()["admm_convergence_rate"]}
        elif method.startswith("design_benders"):
            solver = method.split("_")[-1]
            cfg = V2Config.from_cfg(campus.cfg, solver=solver)
            sched, log = run_day_v2(campus, _FcYear(yd, fc), day, outage, cfg)
            s2 = log.summary()
            st = {"admm_converged_rate": s2["admm_converged_rate"], "plan_time_s": s2["replan_time_s"],
                  "master_opt_share": s2["master_opt_share"]}
        else:
            raise ValueError(method)
    rt = time.perf_counter() - t0
    if return_sched:
        return {"sched": sched, "dd": dd, "campus": campus}
    m = settle_v2(campus, dd, sched)
    f = check_feasibility(campus, sched, dd)
    x = extra_metrics(campus, dd, sched, m)
    decisions = 24 if method not in ("status_quo", "rule_based", "optimum", "ga", "pso", "design_earlier_prototype") else 1
    return {"dataset": ds, "day": day, "method": method, "outage": "-".join(outage) if outage else "",
            "comms_window": f"{win[0]}-{win[1]}" if "comms" in method else "", "feasible": f["feasible"],
            **{k: m[k] for k in ("objective_rs", "cost_rs", "demand_charge_rs", "cost_incl_demand_rs", "peak_kva",
                                 "critical_downtime_h", "critical_unserved_kwh", "unserved_kwh", "import_kwh", "co2_kg",
                                 "dg_litres", "spill_kwh", "voll_rs", "wear_rs")}, **x,
            "objective_incl_demand_rs": m["objective_incl_demand_rs"], "runtime_s": rt,
            "runtime_per_decision_s": st.get("plan_time_s", rt / decisions), "messages": st.get("messages", 0),
            "admm_converged_rate": st.get("admm_converged_rate", np.nan), "master_opt_share": st.get("master_opt_share", np.nan),
            "fallback_slots": st.get("fallback_slots", 0), "mip_timelimit": st.get("mip_timelimit", 0),
            "dg_guard_used": st.get("dg_guard_used", 0), "no_incumbent_dg_off": st.get("no_incumbent_dg_off", 0),
            "rule_fallback_hours": st.get("rule_fallback_hours", 0), "works_without_coordinator": WITHOUT_COORDINATOR.get(method, True)}


def run_bound(ds: str, day: str) -> dict:
    from qgrid.ablation.bounds import day_bounds
    c = ctx(ds)
    outage = c["proc"]["meta"]["days"]["outages"].get(day)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return {"dataset": ds, "day": day, **day_bounds(c["campus"], c["yd"], day, outage)}


def _init(qg):
    global QGRID
    QGRID = qg


def _existing(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size < 3:
        return []
    try:
        return pd.read_csv(path).to_dict("records")
    except pd.errors.EmptyDataError:
        return []


def main(workers: int, ndays: int, qgrid: str, datasets: list[str], resume: bool):
    OUT.mkdir(parents=True, exist_ok=True)
    old_rows, old_bounds = _existing(OUT / "runs.csv"), _existing(OUT / "bounds.csv")
    if old_rows and not resume:
        sys.exit(f"{OUT / 'runs.csv'} has {len(old_rows)} rows; pass --resume to keep them and run only the missing "
                 "tasks, or move the file away first. Nothing was overwritten.")
    if not resume:
        old_bounds = []
    done_runs = {(r["dataset"], str(r["day"]), r["method"]) for r in old_rows}
    done_bounds = {(r["dataset"], str(r["day"])) for r in old_bounds}
    tasks = []
    for ds in datasets:
        c = ctx(ds)                                   # trains and caches each dataset's forecaster up front
        days = c["proc"]["meta"]["days"]["eval_days"][:ndays]
        for d in days:
            tasks += [("run", ds, d, m) for m in MAIN] + [("bound", ds, d, "")]
            if ds == "rvce":
                tasks += [("run", ds, d, m) for m in DESIGN]
    cost = {"design_benders_qaoa": 9, "design_benders_qiea": 6, "design_benders_exact": 4, "design_earlier_prototype": 3,
            "qgrid": 2, "qgrid_comms_fail": 2, "central_mpc": 2, "central_mpc_comms_fail": 2}
    n_all = len(tasks)
    tasks = [t for t in tasks if not ((t[0] == "run" and (t[1], t[2], t[3]) in done_runs) or
                                      (t[0] == "bound" and (t[1], t[2]) in done_bounds))]
    print(f"{len(tasks)} of {n_all} tasks to run ({len(old_rows)} runs and {len(old_bounds)} bounds kept)", flush=True)
    tasks.sort(key=lambda t: -cost.get(t[3], 1))
    rows, bounds = list(old_rows), list(old_bounds)
    t_all = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(qgrid,)) as ex:
        futs = {(ex.submit(run_one, ds, d, m) if k == "run" else ex.submit(run_bound, ds, d)): (k, ds, d, m)
                for k, ds, d, m in tasks}
        for n, fu in enumerate(as_completed(futs), 1):
            k, ds, d, m = futs[fu]
            try:
                r = fu.result()
            except Exception as e:
                print(f"FAILED {k} {ds} {d} {m}: {e!r}", flush=True)
                continue
            (rows if k == "run" else bounds).append(r)
            if k == "run":
                print(f"[{n}/{len(futs)}] {ds:13s} {d} {m:26s} {r['objective_rs']:14,.0f} {r['runtime_s']:6.1f}s", flush=True)
            if n % 10 == 0:                           # checkpoint (kept rows + new rows)
                pd.DataFrame(rows).to_csv(OUT / "runs.csv", index=False)
                pd.DataFrame(bounds).to_csv(OUT / "bounds.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "runs.csv", index=False)
    pd.DataFrame(bounds).to_csv(OUT / "bounds.csv", index=False)
    json.dump({"runtime_s": time.time() - t_all, "workers": workers, "days_per_dataset": ndays, "qgrid_combo": qgrid,
               "resumed": resume, "datasets_run": datasets,
               "datasets": sorted({r["dataset"] for r in rows}, key=DATASETS.index), "methods": MAIN, "design_choices": DESIGN}, open(OUT / "run_info.json", "w", encoding="utf-8"), indent=1)
    print(f"done in {time.time() - t_all:.0f} s", flush=True)


if __name__ == "__main__":
    from qgrid.parallel import workers as _workers
    w = _workers(next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--workers=")), None))
    nd = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--days=")), 30)
    qg = next((a.split("=")[1] for a in sys.argv if a.startswith("--qgrid=")), QGRID)
    dsa = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--datasets=")), None)
    dss = DATASETS if not dsa else [d.strip() for d in dsa.split(",") if d.strip()]
    bad = [d for d in dss if d not in DATASETS]
    if bad:
        sys.exit(f"unknown dataset(s) {bad}; choose from {DATASETS}")
    QGRID = qg
    main(w, nd, qg, dss, "--resume" in sys.argv)
