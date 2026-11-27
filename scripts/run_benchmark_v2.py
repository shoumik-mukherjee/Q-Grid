"""Benchmark v2: identical data, days and constraints for every method; tariff v2 (energy ToD + demand charge
on the campus peak kVA; rates and their status from config.yaml prices.A / tariff_v2); every schedule settled by model.settle() via settle_v2().

Methods
  B0 B1 B2-GA B2-PSO B3      existing baselines (status quo, rule-based, GA, PSO, per-zone MILP)
  B4                         central MILP, energy tariff only (v1 optimum)
  B4-DC                      central MILP with the demand charge  -> "optimum" for gap_to_opt_pct
  V1                         old Q-Grid: binary consensus-ADMM + QAOA local QUBOs (verified tiers)
  V2-QAOA / V2-QIEA / V2-SA  new hybrid: rolling 24 h MPC, Benders with QAOA / QIEA / annealing master
  V2-EXACT                   new hybrid with the exact (MILP) master
  V2-EXACT-FC, V2-QAOA-FC    same with the solar forecast instead of actual solar
Named scenarios run every method; the 30-day evaluation set runs B0, B3, B4-DC, V1 (classical local tiers)
and V2-EXACT. Price scenario A only (the headline tariff).
Outputs: results/v2/benchmark_v2.csv, benchmark_v2_summary.csv, forecast_error.csv, logs/*.json,
schedules/*.json, qaoa_blocks.json, convergence.csv.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT)]
OUT = ROOT / "results" / "v2"

SCEN_METHODS = ["B0", "B1", "B2-GA", "B2-PSO", "B3", "B4", "B4-DC", "V1", "V2-QAOA", "V2-QIEA", "V2-SA",
                "V2-EXACT", "V2-EXACT-FC", "V2-QAOA-FC"]
EVAL_METHODS = ["B0", "B3", "B4-DC", "V1", "V2-EXACT"]
HEAVY = {"V2-QAOA", "V2-QIEA", "V2-SA", "V2-QAOA-FC"}


def _sched_json(campus, day, sched, m):
    from qgrid.model import soc_trajectory, tank_trajectory
    return {"p_bat": np.round(sched.p_bat, 2).tolist(),
            "soc": [np.round(soc_trajectory(b.battery, sched.p_bat[i]) / max(b.battery.kwh, 1e-9), 4).tolist()
                    for i, b in enumerate(campus.buildings)],
            "flows": np.round(sched.flows, 2).tolist(), "dg_kw": np.round(sched.dg_kw, 2).tolist(),
            "pump_on": sched.pump_on.astype(int).tolist(), "flex_start": sched.flex_start,
            "tank_m3": {b.name: np.round(tank_trajectory(b.tank, sched.pump_on[i], day.water), 2).tolist()
                        for i, b in enumerate(campus.buildings) if b.tank},
            "net_kw": np.round(m["net_kw"], 2).tolist(),
            "metrics": {k: v for k, v in m.items() if k != "net_kw"}}


def run_one(day_str: str, key: str, method: str) -> dict:
    from baselines import b0_status_quo, b1_rule_based, b2_ga_pso, b3_mpc_milp, b4_central_milp
    from qgrid import data as qdata
    from qgrid.admm import run_day
    from qgrid.model import check_feasibility, load_campus, make_day
    from qgrid.solvers import STATS, reset_stats
    from qgrid.v2.horizon import YearData
    from qgrid.v2.mpc import V2Config, run_day_v2
    from qgrid.v2.tariff import settle_v2

    campus = load_campus()
    proc = qdata.load_processed()
    sc = proc["meta"]["days"]["scenarios"].get(key, {})
    outage = sc.get("outage")
    day = make_day(campus, proc, day_str, "A", key, outage)
    is_s = key != "eval"
    reset_stats()
    t0 = time.perf_counter()
    log = st = None
    if method.startswith("V2"):
        solver = method.split("-")[1].lower()
        fc = method.endswith("-FC")
        cfg = V2Config.from_cfg(campus.cfg, solver=solver, forecast=fc,
                                record_slots=(72, 76) if (key == campus.cfg["headline_scenario"] and solver == "qaoa"
                                                          and not fc) else ())
        yd = YearData(campus, proc, "A", forecast=fc)
        sched, log = run_day_v2(campus, yd, day_str, outage, cfg)
    elif method == "V1":
        sched, st = run_day(campus, day, method="qaoa" if is_s else "classical")
    else:
        fn = {"B0": lambda: b0_status_quo.solve(campus, day), "B1": lambda: b1_rule_based.solve(campus, day),
              "B2-GA": lambda: b2_ga_pso.solve(campus, day, "GA"), "B2-PSO": lambda: b2_ga_pso.solve(campus, day, "PSO"),
              "B3": lambda: b3_mpc_milp.solve(campus, day), "B4": lambda: b4_central_milp.solve(campus, day),
              "B4-DC": lambda: b4_central_milp.solve_dc(campus, day)}[method]
        sched = fn()
    rt = time.perf_counter() - t0
    m = settle_v2(campus, day, sched)
    f = check_feasibility(campus, sched, day)
    row = {"method": method, "scenario": key, "day": day_str, "day_type": day.day_type, "price_scenario": "A",
           "price_status": day.status["price_scenario"], "tariff_v2_status": m["tariff_v2_status"],
           **{k: m[k] for k in ("cost_rs", "objective_rs", "cost_incl_demand_rs", "objective_incl_demand_rs",
                                 "demand_charge_rs", "peak_kva", "co2_kg", "dg_hours", "critical_downtime_h",
                                 "critical_unserved_kwh", "unserved_kwh", "import_kwh", "export_kwh")},
           "runtime_s": rt, "replan_time_s": sched.info.get("replan_time_s", rt if method in ("B3", "B4", "B4-DC") else np.nan),
           "feasible": f["feasible"], "milp_status": str(sched.info.get("milp_status", ""))}
    extra = {}
    if log is not None:
        row.update(log.summary())
        row["solar"] = "forecast" if method.endswith("-FC") else "actual"
        if is_s:
            extra["log"] = {"steps": [{k: v for k, v in s_.items()} for s_ in log.steps]}
            extra["blocks"] = log.blocks
    if st is not None:
        s_ = st.summary()
        row.update({"admm_iters_mean": s_["admm_iters_mean"], "admm_converged_rate": s_["admm_convergence_rate"],
                    "admm_oscillations": s_["admm_oscillations"], "replan_time_s": s_["replan_time_s"],
                    "qaoa_found_opt_rate": s_.get("qaoa_found_opt_rate", np.nan),
                    **{f"tier_{k}": v for k, v in s_["tiers"].items()}})
    if is_s:
        extra["schedule"] = _sched_json(campus, day, sched, m)
    return {"row": row, "extra": extra, "key": key, "method": method, "day": day_str}


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def summarise(df: pd.DataFrame) -> None:
    """gap to the demand-charge-aware central optimum on the same day; per-set means -> CSVs."""
    opt = df[df.method == "B4-DC"].set_index(["scenario", "day"])["objective_incl_demand_rs"]
    ref = [opt.get((sc_, d_), np.nan) for sc_, d_ in zip(df.scenario, df.day)]
    df["gap_to_opt_pct"] = 100 * (df["objective_incl_demand_rs"] - ref) / np.abs(ref)
    df.to_csv(OUT / "benchmark_v2.csv", index=False)
    num = [c for c in df.columns if c not in ("method", "scenario", "day", "day_type", "price_scenario", "price_status",
                                              "tariff_v2_status", "milp_status", "solar", "error", "feasible")]
    sm = []
    for (setname, sub) in (("scenarios", df[df.scenario != "eval"]), ("eval30_mean", df[df.scenario == "eval"])):
        if len(sub):
            g = sub.groupby("method")[num].mean(numeric_only=True).reset_index()
            g.insert(0, "day_set", setname)
            g["n_days"] = sub.groupby("method").size().values
            g["feasible_all"] = sub.groupby("method")["feasible"].apply(lambda s_: bool(s_.fillna(False).all())).values
            sm.append(g)
    pd.concat(sm).to_csv(OUT / "benchmark_v2_summary.csv", index=False)


def main(workers: int | None = None, only: str | None = None):
    from qgrid import data as qdata
    from qgrid.model import load_campus
    from qgrid.v2.horizon import YearData, forecast_error

    t_all = time.time()
    proc = qdata.load_processed()
    meta = proc["meta"]
    campus = load_campus()
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    (OUT / "schedules").mkdir(parents=True, exist_ok=True)
    tasks = [(v["date"], k, m) for k, v in meta["days"]["scenarios"].items() for m in SCEN_METHODS]
    if only != "scenarios":
        tasks += [(d, "eval", m) for d in meta["days"]["eval_days"] for m in EVAL_METHODS]
    keep = []
    if only == "retry":                     # re-run only the tasks whose earlier attempt failed, keep the rest
        old = pd.read_csv(OUT / "benchmark_v2.csv")
        bad = old[old["error"].notna()] if "error" in old else old.iloc[0:0]
        todo = {(r.day, r.scenario, r.method) for r in bad.itertuples()}
        tasks = [t for t in tasks if t in todo]
        keep = old[old["error"].isna()].drop(columns=[c for c in ("error", "gap_to_opt_pct") if c in old]).to_dict("records") \
            if "error" in old else old.to_dict("records")
        blocks = json.loads((OUT / "qaoa_blocks.json").read_text(encoding="utf-8"))["blocks"] if (OUT / "qaoa_blocks.json").exists() else []
    tasks.sort(key=lambda t: (t[2] not in HEAVY, t[1] == "eval"))      # longest first
    from qgrid.parallel import workers as _workers
    workers = _workers(workers)
    print(f"{len(tasks)} tasks on {workers} workers", flush=True)
    rows = list(keep)
    if only != "retry":
        blocks = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_one, *t): t for t in tasks}
        for k, fu in enumerate(as_completed(futs), 1):
            t = futs[fu]
            try:
                r = fu.result()
            except Exception as e:                                   # record the failure, never invent a row
                print(f"FAILED {t}: {e!r}", flush=True)
                rows.append({"method": t[2], "scenario": t[1], "day": t[0], "error": repr(e)})
                continue
            rows.append(r["row"])
            if r["extra"].get("log"):
                (OUT / "logs" / f"{r['key']}_{r['method']}.json").write_text(json.dumps(_clean(r["extra"]["log"])), encoding="utf-8")
            if r["extra"].get("schedule"):
                (OUT / "schedules" / f"{r['key']}_{r['method']}.json").write_text(json.dumps(_clean(r["extra"]["schedule"])), encoding="utf-8")
            blocks += r["extra"].get("blocks", [])
            print(f"[{k}/{len(tasks)}] {t[1]:12s} {t[2]:12s} {r['row']['objective_incl_demand_rs']:12,.0f} "
                  f"{r['row']['runtime_s']:7.1f}s", flush=True)
    df = pd.DataFrame(rows)
    summarise(df)
    (OUT / "qaoa_blocks.json").write_text(json.dumps(_clean({"label": "QAOA on a local simulator: verified against exact "
                                                                       "solutions, no quantum advantage claimed",
                                                              "blocks": blocks})), encoding="utf-8")
    # solar forecast error on the named days
    yd = YearData(campus, proc, "A", forecast=True)
    fe = pd.DataFrame([{"scenario": k, **forecast_error(yd, v["date"])} for k, v in meta["days"]["scenarios"].items()])
    fe.to_csv(OUT / "forecast_error.csv", index=False)
    # convergence: old binary consensus-ADMM vs ADMM on the convex LP subproblem
    conv = df[df.method.isin(["V1", "V2-EXACT", "V2-QAOA"]) & (df.scenario != "eval")]
    conv[["method", "scenario", "admm_converged_rate", "admm_iters_mean"]].to_csv(OUT / "convergence.csv", index=False)
    (OUT / "run_info.json").write_text(json.dumps({"runtime_s": time.time() - t_all, "workers": workers,
                                                    "tasks": len(tasks), "price_scenario": "A",
                                                    "tariff_v2_status": campus.cfg["tariff_v2"]["status"]}), encoding="utf-8")
    print(f"done in {time.time() - t_all:.0f} s", flush=True)


if __name__ == "__main__" and "--resummarise" in sys.argv:
    d_ = pd.read_csv(OUT / "benchmark_v2.csv")
    summarise(d_.drop(columns=[c for c in ("error",) if c in d_ and d_[c].isna().all()]))
    sys.exit(0)

if __name__ == "__main__":
    only = "scenarios" if "--scenarios-only" in sys.argv else ("retry" if "--retry-failed" in sys.argv else None)
    w = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--workers=")), None)
    main(w, only)
