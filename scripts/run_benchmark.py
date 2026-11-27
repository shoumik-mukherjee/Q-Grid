"""Benchmark v2 (RVCE campus) — all methods on identical data, days, constraints and horizon.

Methods: B0 status quo, B1 rule-based, B2-GA, B2-PSO, B3 per-zone MPC-MILP, B4 central MILP (optimum within
0.1 % MIP gap), B5 ADMM + exact local solver, OURS ADMM + QAOA (verified tiers), OURS-QIEA ADMM + QIEA,
C-QIEA centralised campus QUBO solved by QIEA.
QAOA in the loop runs only on the 6 named scenarios; on the 30-day evaluation set the OURS row uses the
classical local tiers (SA -> exact) and is labelled "OURS (classical local tiers)".
Outputs: results/benchmark.csv, benchmark_summary.csv, robustness.csv, headline.json, economics.json,
search_space.json, results/plots/*.png, results/replay/admm_<scenario>_<price>.json, results/schedules/*.json.
"""
from __future__ import annotations

import copy
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

SCEN_METHODS = ["B0", "B1", "B2-GA", "B2-PSO", "B3", "B4", "B5", "OURS", "OURS-QIEA", "C-QIEA"]
EVAL_METHODS = ["B0", "B1", "B2-GA", "B2-PSO", "B3", "B4", "B5", "OURS", "C-QIEA", "B5-nobattery"]


def _schedule_json(campus, day, sched, m):
    from qgrid.model import net_power, soc_trajectory, tank_trajectory
    return {"p_bat": np.round(sched.p_bat, 2).tolist(),
            "soc": [np.round(soc_trajectory(b.battery, sched.p_bat[i]) / max(b.battery.kwh, 1e-9), 4).tolist()
                    for i, b in enumerate(campus.buildings)],
            "flows": np.round(sched.flows, 2).tolist(), "dg_kw": np.round(sched.dg_kw, 2).tolist(),
            "pump_on": sched.pump_on.astype(int).tolist(), "flex_start": sched.flex_start,
            "tank_m3": {b.name: np.round(tank_trajectory(b.tank, sched.pump_on[i], day.water), 2).tolist()
                        for i, b in enumerate(campus.buildings) if b.tank},
            "net_kw": np.round(net_power(campus, day, sched), 2).tolist(),
            "metrics": {k: v for k, v in m.items() if k != "net_kw"}}


def run_task(day_str: str, key: str, scen: str) -> dict:
    from baselines import b0_status_quo, b1_rule_based, b2_ga_pso, b3_mpc_milp, b4_central_milp, c_qiea
    from qgrid import data as qdata
    from qgrid.admm import run_day
    from qgrid.model import check_feasibility, load_campus, make_day, settle
    from qgrid.solvers import STATS, reset_stats

    campus = load_campus()
    proc = qdata.load_processed()
    meta = proc["meta"]
    sc = meta["days"]["scenarios"].get(key, {})
    day = make_day(campus, proc, day_str, scen, key, sc.get("outage"))
    is_s = key != "eval"
    nobat = copy.deepcopy(campus)
    for b in nobat.buildings:
        b.battery.kw = 0.0
        b.battery.kwh = 0.0
    solvers = {
        "B0": lambda: (b0_status_quo.solve(campus, day), None),
        "B1": lambda: (b1_rule_based.solve(campus, day), None),
        "B2-GA": lambda: (b2_ga_pso.solve(campus, day, "GA"), None),
        "B2-PSO": lambda: (b2_ga_pso.solve(campus, day, "PSO"), None),
        "B3": lambda: (b3_mpc_milp.solve(campus, day), None),
        "B4": lambda: (b4_central_milp.solve(campus, day), None),
        "B5": lambda: run_day(campus, day, method="exact"),
        "OURS": lambda: run_day(campus, day, method="qaoa" if is_s else "classical", log=is_s),
        "OURS-QIEA": lambda: run_day(campus, day, method="qiea"),
        "C-QIEA": lambda: (c_qiea.solve(campus, day), None),
        "B5-nobattery": lambda: run_day(nobat, day, method="exact"),
    }
    rows, extra = [], {"schedules": {}, "admm_log": None}
    for name in (SCEN_METHODS if is_s else EVAL_METHODS):
        reset_stats()
        t0 = time.perf_counter()
        sched, st = solvers[name]()
        rt = time.perf_counter() - t0
        cmp_campus = nobat if name == "B5-nobattery" else campus
        m = settle(cmp_campus, day, sched)
        f = check_feasibility(cmp_campus, sched, day)
        row = {"method": name, "price_scenario": scen, "scenario": key, "day": day_str, "day_type": day.day_type,
               "price_status": day.status["price_scenario"], "local_solver": "",
               **{k: m[k] for k in ("cost_rs", "objective_rs", "energy_cost_rs", "fuel_cost_rs", "voll_rs", "co2_kg",
                                     "dg_hours", "dg_litres", "critical_downtime_h", "critical_unserved_kwh",
                                     "unserved_kwh", "import_kwh", "export_kwh", "self_consumption_pct",
                                     "peak_import_kw", "terminal_adj_rs")},
               "runtime_s": rt, "replan_time_s": sched.info.get("replan_time_s", np.nan),
               "feasible": f["feasible"], "feasibility_rate": f["feasibility_rate"],
               "milp_status": str(sched.info.get("milp_status", "")),
               "qaoa_solves": STATS["qaoa_solves"], "qaoa_time_s": STATS["qaoa_time_s"], "cache_hits": STATS["cache_hits"]}
        if name in ("B3", "B4"):
            row["replan_time_s"] = rt
        if st is not None:
            s = st.summary()
            row.update({"admm_iters": s["admm_iters_mean"], "admm_convergence_rate": s["admm_convergence_rate"],
                        "admm_oscillations": s["admm_oscillations"], "line_disagreements": s["line_disagreements"],
                        "critical_guard_cuts": s["critical_guard_cuts"], "replan_time_s": s["replan_time_s"],
                        **{f"tier_{k}": v for k, v in s["tiers"].items()}})
            for k in ("qaoa_p_opt", "qaoa_approx_ratio", "qaoa_found_opt_rate", "qaoa_feasible_rate"):
                row[k] = s.get(k, np.nan)
            row["local_solver"] = {"B5": "exact", "OURS": "QAOA->QIEA->SA->exact" if is_s else "SA->exact (classical tiers)",
                                   "OURS-QIEA": "QIEA->SA->exact", "B5-nobattery": "exact"}.get(name, "")
            if name == "OURS" and is_s:
                extra["admm_log"] = {"iterations": st.log, "slots": st.slots, "lines": campus.lines, "zones": campus.names}
        rows.append(row)
        if is_s and name in ("B0", "B3", "B4", "B5", "OURS", "OURS-QIEA", "C-QIEA"):
            extra["schedules"][name] = _schedule_json(campus, day, sched, m)
    if is_s and scen == "A":
        k = campus.names.index("heavy_labs")
        ks = 74 if sc.get("outage") else 48
        sched, st = run_day(campus, day, method="classical", kill=(k, ks))
        extra["kill_inprocess"] = {"scenario": key, "day": day_str, "price_scenario": scen, "killed": "heavy_labs",
                                   "kill_slot": ks, "feasible": check_feasibility(campus, sched, day)["feasible"],
                                   "critical_downtime_h": settle(campus, day, sched)["critical_downtime_h"],
                                   **{k2: v for k2, v in st.kill.items() if k2 != "building"}}
    return {"rows": rows, "extra": extra, "day": day_str, "key": key, "scen": scen}


def main(workers: int | None = None):
    from qgrid import QUANTUM_LABEL
    from qgrid import data as qdata
    from qgrid.campus import search_space
    from qgrid.model import load_campus
    from qgrid.qubo import MAX_QUBITS

    t_all = time.time()
    proc = qdata.load_processed()
    meta = proc["meta"]
    campus = load_campus()
    cfg = campus.cfg
    scen_list = cfg["prices"]["scenarios"]
    tasks = [(v["date"], k, s) for s in scen_list for k, v in meta["days"]["scenarios"].items()]
    tasks += [(d, "eval", s) for s in scen_list for d in meta["days"]["eval_days"]]
    from qgrid.parallel import workers as _workers
    workers = _workers(workers)
    print(f"{len(tasks)} tasks on {workers} workers", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_task, *t): t for t in tasks}
        for n, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            o = next(x for x in r["rows"] if x["method"] == "OURS")
            print(f"[{n}/{len(tasks)}] {r['scen']} {r['day']} {r['key']:<12} OURS Rs {o['cost_rs']:,.0f} "
                  f"crit {o['critical_downtime_h']:.2f}h ({o['runtime_s']:.0f}s) t={time.time() - t_all:.0f}s", flush=True)

    df = pd.DataFrame([row for r in results for row in r["rows"]])
    b4 = df[df.method == "B4"].set_index(["price_scenario", "day", "scenario"])["objective_rs"]
    df["gap_to_B4_pct"] = [100 * (o - b4[(p, d, k)]) / abs(b4[(p, d, k)])
                           for o, p, d, k in zip(df.objective_rs, df.price_scenario, df.day, df.scenario)]
    sem = float(cfg["economics"]["semester_days"])
    df["cost_rs_semester"] = df.cost_rs * sem
    df["co2_t_year"] = df.co2_kg * 365 / 1000

    # networked kill (real processes over websockets) on the headline scenario, OURS and B5
    sys.path.insert(0, str(ROOT / "scripts"))
    from kill_node import run as net_run
    hs = cfg["headline_scenario"]
    hdate = meta["days"]["scenarios"][hs]["date"]
    rob = []
    for meth, label in (("qaoa", "OURS"), ("exact", "B5")):
        r = net_run(hdate, "A", meth, kill="heavy_labs", kill_slot=74, scenario_key=hs)
        rob.append({"mode": "networked", "method": label, "scenario": hs, "price_scenario": "A", "day": hdate,
                    "killed": "heavy_labs", "kill_slot": 74, "feasible": r["feasible"],
                    "tie_line_flows_consistent": r["tie_line_flows_consistent"], "recovery_time_s": r["recovery_time_s"],
                    "critical_downtime_h": r["critical_downtime_h"], "wall_time_s": r["wall_time_s"]})
        df.loc[(df.method == label) & (df.scenario == hs) & (df.price_scenario == "A"), "recovery_time_s"] = r["recovery_time_s"]
        print(f"[net-kill] {label}: recovery {r['recovery_time_s']}s feasible={r['feasible']}", flush=True)
    for r in results:
        if "kill_inprocess" in r["extra"]:
            rob.append({"mode": "in-process", "method": "OURS (classical tiers)", **r["extra"]["kill_inprocess"]})
    pd.DataFrame(rob).to_csv(ROOT / "results" / "robustness.csv", index=False)

    df = df.sort_values(["price_scenario", "scenario", "day", "method"])
    df.to_csv(ROOT / "results" / "benchmark.csv", index=False)
    num = ["cost_rs", "objective_rs", "cost_rs_semester", "co2_kg", "co2_t_year", "dg_hours", "dg_litres",
           "critical_downtime_h", "critical_unserved_kwh", "gap_to_B4_pct", "self_consumption_pct", "peak_import_kw",
           "replan_time_s", "runtime_s", "feasibility_rate", "qaoa_p_opt", "qaoa_approx_ratio", "admm_iters",
           "admm_convergence_rate", "admm_oscillations"]
    ev = df[df.scenario == "eval"].groupby(["price_scenario", "method"])[num].mean().reset_index()
    ev.insert(2, "day_set", "eval30_mean")
    sd = df[df.scenario != "eval"].copy()
    sd["day_set"] = sd.scenario
    summ = pd.concat([ev, sd[["price_scenario", "method", "day_set"] + num]])
    status = {s: meta["prices"][s]["status"] for s in scen_list}
    summ.insert(0, "price_status", summ.price_scenario.map(status))
    summ.to_csv(ROOT / "results" / "benchmark_summary.csv", index=False)

    # headline card: outage during fest evening, scenario A, status quo vs Q-Grid
    h = df[(df.scenario == hs) & (df.price_scenario == "A")].set_index("method")
    card = {"scenario": hs, "day": hdate, "price_scenario": "A", "price_status": status["A"],
            "note": meta["days"]["scenarios"][hs]["note"], "label": QUANTUM_LABEL,
            "before": {k: float(h.loc["B0", k]) for k in ("cost_rs", "dg_hours", "dg_litres", "critical_downtime_h",
                                                           "critical_unserved_kwh", "unserved_kwh")},
            "after": {k: float(h.loc["OURS", k]) for k in ("cost_rs", "dg_hours", "dg_litres", "critical_downtime_h",
                                                            "critical_unserved_kwh", "unserved_kwh", "replan_time_s")},
            "optimum_B4": {k: float(h.loc["B4", k]) for k in ("cost_rs", "critical_downtime_h", "critical_unserved_kwh")},
            "source": "results/benchmark.csv rows method in {B0, OURS, B4}, scenario=" + hs + ", price_scenario=A"}
    (ROOT / "results" / "headline.json").write_text(json.dumps(card, indent=2), encoding="utf-8")

    # battery payback (ASSUMPTION capex) from B5 with vs without batteries on the 30-day set
    eco = {}
    for s in scen_list:
        e = df[(df.scenario == "eval") & (df.price_scenario == s)].pivot(index="day", columns="method", values="cost_rs")
        saving_day = float((e["B5-nobattery"] - e["B5"]).mean())
        kwh = sum(b.battery.kwh for b in campus.buildings)
        capex = kwh * float(cfg["economics"]["battery_capex_rs_per_kwh"])
        eco[s] = {"price_status": status[s], "battery_kwh_total": kwh, "capex_rs_ASSUMPTION": capex,
                  "battery_saving_rs_per_day": saving_day, "annual_saving_rs": saving_day * 365,
                  "simple_payback_years": capex / (saving_day * 365) if saving_day > 0 else None,
                  "method": "mean daily cost B5 without batteries minus B5 with batteries, 30-day set, x365"}
    (ROOT / "results" / "economics.json").write_text(json.dumps(eco, indent=2), encoding="utf-8")

    counts = {}
    for i, b in enumerate(campus.buildings):
        counts[b.name] = min(MAX_QUBITS, (2 * cfg["admm"]["horizon_slots"] if b.battery.present else 0)
                             + 2 * len(campus.incident(i)) + (1 if b.dg else 0) + (1 if b.tank else 0) + len(b.flex))
    full = {b.name: (2 if b.battery.present else 0) + (1 if b.dg else 0) + (1 if b.tank else 0) + len(b.flex)
            for b in campus.buildings}
    full["_tie_lines"] = 2 * len(campus.lines)
    (ROOT / "results" / "search_space.json").write_text(json.dumps(
        {"centralised_full_day": search_space(full), "local_qubo_qubits": counts,
         "note": "centralised: per slot one charge + one discharge bit per battery, DG, pump, job-start bits and "
                 "2 bits per tie-line; local QUBOs: horizon of 3 slots, capped at 16 qubits"}, indent=2), encoding="utf-8")

    rdir, sdir = ROOT / "results" / "replay", ROOT / "results" / "schedules"
    sdir.mkdir(exist_ok=True)
    for r in results:
        if r["key"] != "eval":
            if r["extra"]["admm_log"]:
                (rdir / f"admm_{r['key']}_{r['scen']}.json").write_text(
                    json.dumps({"label": QUANTUM_LABEL, "day": r["day"], **r["extra"]["admm_log"]}, default=float), encoding="utf-8")
            (sdir / f"{r['key']}_{r['scen']}.json").write_text(
                json.dumps({"day": r["day"], "scenario": r["key"], "price_scenario": r["scen"], **r["extra"]["schedules"]},
                           default=float), encoding="utf-8")
    make_plots(df, status)
    print(f"done in {time.time() - t_all:.0f}s")
    print(ev.pivot_table(index="method", columns="price_scenario", values=["cost_rs", "gap_to_B4_pct"]).round(2))
    print(json.dumps(card, indent=1))


def make_plots(df, status):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from qgrid import QUANTUM_LABEL

    order = ["B0", "B1", "B2-GA", "B2-PSO", "B3", "C-QIEA", "B5", "OURS-QIEA", "OURS", "B4"]
    for kind, sel in (("eval", df.scenario == "eval"), ("headline", df.scenario == "outage_fest")):
        d = df[sel]
        fig, axs = plt.subplots(1, 2, figsize=(15, 4.8))
        for ax, s in zip(axs, ("A", "B")):
            g = d[d.price_scenario == s].groupby("method").objective_rs.mean().reindex([m for m in order if m in set(d.method)])
            ax.bar(g.index, g.values, color=["#9aa5b1"] * (len(g) - 3) + ["#5b8def", "#2a9d8f", "#264653"][-min(3, len(g)):])
            ax.axhline(g.get("B4", np.nan), color="#264653", ls="--", lw=1, label="B4 optimum")
            for k, v in enumerate(g.values):
                ax.text(k, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=7)
            tag = "" if status[s] == "REAL" else " [STAND-IN prices, PENDING]"
            ax.set_title(f"{'30-day mean' if kind == 'eval' else 'Outage during fest'} — scenario {s}: Rs/day incl. VOLL{tag}",
                         fontsize=9, color="k" if status[s] == "REAL" else "#b00020")
            ax.tick_params(axis="x", labelsize=7, rotation=30); ax.legend(fontsize=8)
        fig.suptitle(QUANTUM_LABEL + " · QIEA = quantum-inspired, classical CPU", fontsize=8)
        fig.tight_layout()
        fig.savefig(ROOT / "results" / "plots" / f"benchmark_{kind}.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    main()
