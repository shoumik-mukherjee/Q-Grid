"""Replay export (offline demo data): results/replay/scenario_<key>.json (+ copies in frontend/public/replay/).

Per scenario (price scenario A = headline; B summarised): campus geometry (OSM), per-slot sun position, PV,
load, critical load, prices, grid status; status quo (B0) vs Q-Grid (OURS, ADMM + QAOA) schedules with SoC,
tie-line flows, DG, pumps, tanks, grid import, cumulative Rs and CO2; ADMM iterations; QAOA distribution
reference; deterministic explanations (hourly); benchmark rows; headline card; "solve it live" instances
(headline scenario); provenance (git commit, SHA-256 of inputs, seeds). Every number comes from the pipeline.
"""
from __future__ import annotations

import hashlib
import json
import os
import math
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT)]
RDIR = ROOT / "results" / "replay"
PUB = ROOT / "frontend" / "public" / "replay"


def clean(o):
    """Strict JSON: NaN / inf -> null, numpy scalars -> Python numbers (browsers reject NaN in JSON)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def dump(o) -> str:
    return json.dumps(clean(o), allow_nan=False, default=str)


def sha(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def sun_position(date: str, lat: float) -> dict:
    """Solar elevation / azimuth (degrees) per 15-min slot in local solar time (NASA POWER LST convention)."""
    n = pd.Timestamp(date).dayofyear
    decl = math.radians(23.45) * math.sin(math.radians(360 / 365 * (284 + n)))
    phi = math.radians(lat)
    el, az = [], []
    for t in range(96):
        h = math.radians(15 * (t / 4 + 0.125 - 12))
        s = math.sin(phi) * math.sin(decl) + math.cos(phi) * math.cos(decl) * math.cos(h)
        e = math.asin(max(-1, min(1, s)))
        c = (math.sin(decl) - math.sin(e) * math.sin(phi)) / max(math.cos(e) * math.cos(phi), 1e-9)
        a = math.degrees(math.acos(max(-1, min(1, c))))
        el.append(round(math.degrees(e), 2))
        az.append(round(a if h < 0 else 360 - a, 2))
    return {"elevation_deg": el, "azimuth_deg": az}


def build_scenario(key: str) -> dict:
    from qgrid import QUANTUM_LABEL
    from qgrid import data as qdata
    from qgrid import qubo as Q
    from qgrid.admm import run_day
    from qgrid.agent import BuildingAgent
    from qgrid.model import DT, load_campus, make_day, settle, soc_trajectory, tank_trajectory
    from qgrid.solvers import reset_stats, solve_local
    from qgrid.solvers.exact import approx_ratio
    from qgrid.solvers.qaoa import run_qaoa

    campus = load_campus()
    proc = qdata.load_processed()
    meta = proc["meta"]
    sc = meta["days"]["scenarios"][key]
    day = make_day(campus, proc, sc["date"], "A", key, sc.get("outage"))
    sched_file = ROOT / "results" / "schedules" / f"{key}_A.json"
    bench = pd.read_csv(ROOT / "results" / "benchmark.csv")
    rows = bench[bench.scenario == key]
    reset_stats()
    s_q, st = run_day(campus, day, method="qaoa", log=True, explain_slots=set(range(0, 96, 4)))
    m_q = settle(campus, day, s_q)
    sq = json.loads(sched_file.read_text())["B0"] if sched_file.exists() else None
    bench_ours = rows[(rows.method == "OURS") & (rows.price_scenario == "A")]
    consistent = bool(len(bench_ours) and abs(float(bench_ours.cost_rs.iloc[0]) - m_q["cost_rs"]) < 1e-3)

    from qgrid.model import Schedule, flex_load, pump_load

    def run_block(p_bat, flows, dg_kw, pump_on, flex_start, metrics, net):
        net = np.asarray(net)
        sch = Schedule(np.asarray(p_bat, float), [dict(x) for x in flex_start], np.asarray(flows, float),
                       np.asarray(dg_kw, float), np.asarray(pump_on, float))
        demand = day.load + flex_load(campus, sch) + pump_load(campus, sch)
        deficit = np.where(day.grid[None, :], 0.0, np.maximum(net, 0))
        crit_short = np.maximum(deficit - np.maximum(demand - day.crit, 0), 0)
        g = day.grid[None, :]
        imp = np.where(g, np.maximum(net, 0), 0).sum(0)
        exp = np.where(g, np.maximum(-net, 0), 0).sum(0)
        cost_t = (imp * day.price_imp - exp * day.price_exp) * DT
        fuel_t = np.zeros(96)
        co2_t = imp * DT * day.ef
        for i, b in enumerate(campus.buildings):
            if b.dg:
                on = np.asarray(dg_kw[i]) > 1e-9
                L = b.dg.litres(on, np.asarray(dg_kw[i]))
                fuel_t += L * b.dg.price
                co2_t += L * b.dg.co2_per_l
        return {"p_bat": np.round(p_bat, 2).tolist(),
                "soc": [np.round(soc_trajectory(b.battery, np.asarray(p_bat[i])) / max(b.battery.kwh, 1e-9), 4).tolist()
                        for i, b in enumerate(campus.buildings)],
                "flows": np.round(flows, 2).tolist(), "dg_kw": np.round(dg_kw, 2).tolist(),
                "pump_on": np.asarray(pump_on).astype(int).tolist(), "flex_start": flex_start,
                "tank_m3": {b.name: np.round(tank_trajectory(b.tank, np.asarray(pump_on[i]), day.water), 2).tolist()
                            for i, b in enumerate(campus.buildings) if b.tank},
                "net_kw": np.round(net, 2).tolist(), "grid_import_kw": np.round(imp, 2).tolist(),
                "demand_kw": np.round(demand, 2).tolist(), "crit_unserved_kw": np.round(crit_short, 3).tolist(),
                "grid_export_kw": np.round(exp, 2).tolist(),
                "cum_cost_rs": np.round(np.cumsum(cost_t + fuel_t), 1).tolist(), "cum_co2_kg": np.round(np.cumsum(co2_t), 1).tolist(),
                "metrics": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics.items() if k != "net_kw"}}

    def problem_numbers():
        """Status-quo 'problem' figures (B0). Rs lost: per zone, surplus exported during the day that is bought
        back later: reusable = min(exported kWh, later-imported kWh); loss = reusable x (mean later import price -
        mean export price). Same definition as scripts/problem_day.py (v1)."""
        if not sq:
            return None
        net = np.array(sq["net_kw"])
        lost = reuse_t = 0.0
        for z in range(net.shape[0]):
            g = day.grid
            exp_ = np.where(g, np.maximum(-net[z], 0), 0) * DT
            imp_ = np.where(g, np.maximum(net[z], 0), 0) * DT
            if exp_.sum() <= 0:
                continue
            last = int(np.nonzero(exp_)[0].max())
            later = imp_[last + 1:]
            reuse = min(exp_.sum(), later.sum())
            if reuse <= 0:
                continue
            pe_ = (exp_ * day.price_exp).sum() / exp_.sum()
            pi_ = (later * day.price_imp[last + 1:]).sum() / later.sum()
            lost += reuse * max(pi_ - pe_, 0.0)
            reuse_t += reuse
        m = sq["metrics"]
        return {"rs_lost": round(lost, 1), "reusable_kwh": round(reuse_t, 1), "co2_kg": round(m["co2_kg"], 1),
                "dg_hours": m["dg_hours"], "cost_rs": round(m["cost_rs"], 1), "critical_downtime_h": m["critical_downtime_h"],
                "definition": "Rs lost = sum over zones of min(kWh exported in the day, kWh imported later) x (mean later import price - mean export price)"}

    runs = {"qgrid": run_block(s_q.p_bat, s_q.flows, s_q.dg_kw, s_q.pump_on, s_q.flex_start, m_q, m_q["net_kw"])}
    if sq:
        runs["status_quo"] = run_block(np.array(sq["p_bat"]), np.array(sq["flows"]), np.array(sq["dg_kw"]),
                                       np.array(sq["pump_on"]), sq["flex_start"], sq["metrics"], np.array(sq["net_kw"]))
    # optimiser v2 run of the same day (QAOA master), if the v2 benchmark has been run
    v2s = ROOT / "results" / "v2" / "schedules" / f"{key}_V2-QAOA.json"
    if v2s.exists():
        from qgrid.v2.tariff import settle_v2
        j = json.loads(v2s.read_text())
        sch2 = Schedule(np.asarray(j["p_bat"], float), [dict(x) for x in j["flex_start"]], np.asarray(j["flows"], float),
                        np.asarray(j["dg_kw"], float), np.asarray(j["pump_on"], float))
        m2 = settle_v2(campus, day, sch2)
        runs["qgrid_v2"] = run_block(sch2.p_bat, sch2.flows, sch2.dg_kw, sch2.pump_on, j["flex_start"], m2, m2["net_kw"])
    # compact ADMM log: per slot iterations with residuals and proposals
    admm = {}
    for rec in st.log:
        admm.setdefault(rec["t"], []).append({"k": rec["k"], "r_primal": rec["r_primal"], "r_dual": round(rec["r_dual"], 3),
                                              "agreed": rec["agreed"], "proposals": rec["proposals"]})
    out = {
        "label": QUANTUM_LABEL, "qiea_label": "QIEA = quantum-inspired evolutionary algorithm, classical CPU (not quantum computing)",
        "scenario": {"key": key, **sc}, "price_scenario": "A", "price_status": meta["prices"]["A"]["status"],
        "data_status": meta["data_status"],
        "campus": {"name": campus.layout["campus"], "credit": campus.layout["credit"], "outline": campus.layout["outline"],
                   "roads": __import__("qgrid.campus", fromlist=["fetch_roads"]).fetch_roads(campus.cfg),
                   "zones": [{"name": b.name, "cx": campus.layout["zones"][b.name]["cx"], "cy": campus.layout["zones"][b.name]["cy"],
                              "pv_kwp": b.pv_kwp, "battery_kwh": b.battery.kwh, "battery_kw": b.battery.kw,
                              "dg_kw": b.dg.kw if b.dg else 0, "tank_m3": b.tank.capacity if b.tank else 0,
                              "critical_frac": b.critical_frac, "source": campus.layout["zones"][b.name]["source"],
                              "flex": [{"name": j.name, "kw": j.kw, "slots": j.slots} for j in b.flex],
                              "buildings": [{"name": x["name"], "levels": x["levels"], "polygon": x["polygon"]}
                                            for x in campus.layout["zones"][b.name]["buildings"]]}
                             for b in campus.buildings],
                   "lines": [[campus.names[a], campus.names[b]] for a, b in campus.lines], "line_step_kw": campus.step_kw},
        "slots": {"t": [f"{t // 4:02d}:{t % 4 * 15:02d}" for t in range(96)], "sun": sun_position(sc["date"], campus.cfg["site"]["lat"]),
                  "pv_kw": np.round(day.pv, 2).tolist(), "load_kw": np.round(day.load, 2).tolist(),
                  "crit_kw": np.round(day.crit, 2).tolist(), "price_import": np.round(day.price_imp, 3).tolist(),
                  "price_export": np.round(day.price_exp, 3).tolist(), "grid_up": day.grid.astype(int).tolist(),
                  "co2_kg_per_kwh": day.ef},
        "runs": runs, "replay_matches_benchmark": consistent, "problem": problem_numbers(),
        "admm": {"iterations": admm, "summary": st.summary(), "lines": [[campus.names[a], campus.names[b]] for a, b in campus.lines]},
        "explanations": {str(t): {z: [e["text"] for e in ex if not e["var"].startswith("line")] for z, ex in d.items()}
                         for t, d in st.explanations.items()},
        "decisions": {str(t): {z: [{"var": e["var"], "decision": e["decision"], "impact_rs": round(e["delta_total_rs"], 2),
                                    "top_group": max(e["deltas_rs"], key=lambda g: abs(e["deltas_rs"][g])) if e["deltas_rs"] else None,
                                    "text": e["text"]} for e in ex if not e["var"].startswith("line")] for z, ex in d.items()}
                      for t, d in st.explanations.items()},
        "qaoa_file": f"qaoa_{key}.json",
        "benchmark": rows.drop(columns=[c for c in rows.columns if rows[c].isna().all()]).to_dict(orient="records"),
    }
    if key == campus.cfg["headline_scenario"]:
        hj = ROOT / "results" / "headline.json"
        out["headline"] = json.loads(hj.read_text()) if hj.exists() else None
    # worked examples (QAOA measured with Qiskit Aer + exact check) for every scenario, so each day in the demo's
    # scenario picker shows pipeline results
    if True:
        live = []
        soc = runs["qgrid"]["soc"]
        for i, b in enumerate(campus.buildings):
            for t in (24, 48, 76, 88):
                ag = BuildingAgent(campus, day, i, campus.cfg["admm"]["horizon_slots"])
                if b.battery.present:
                    ag.e = soc[i][t] * b.battery.kwh
                if b.tank:
                    ag.level = runs["qgrid"]["tank_m3"][b.name][t]
                for jn, s0 in s_q.flex_start[i].items():
                    if s0 < t:
                        j = next(j for j in b.flex if j.name == jn)
                        ag.started[jn] = s0
                        ag.running += j.profile(s0)
                lp = ag.local_problem(t, ag.incident, v={l: 0.0 for l, _ in ag.incident}, rho=0.05)
                qubos = Q.build(lp)
                en = Q.enumerate_all(lp, qubos)
                total, *_ = Q.auto_tune(lp, qubos, enum=en)
                import time as _t
                t0 = _t.perf_counter(); ex_sol = solve_local(lp, "exact", use_cache=False); t_ex = _t.perf_counter() - t0
                t0 = _t.perf_counter(); qr = run_qaoa(total, p=2, shots=256, seed=1234, backend="aer"); t_q = _t.perf_counter() - t0
                from qgrid.solvers.exact import solve_exact
                exr = solve_exact(lp, qubos, enum=en)
                feas = Q.feasible_mask(lp, qr.samples)
                best = qr.samples[feas][np.argmin(qubos[0].energy(qr.samples[feas]))] if feas.any() else None
                bits = lambda x: "".join(str(int(v)) for v in x)  # noqa: E731
                order = np.argsort(-qr.probs)[:64]
                live.append({"zone": b.name, "slot": t, "time": f"{t // 4:02d}:{t % 4 * 15:02d}", "n_qubits": lp.n,
                             "var_names": lp.var_names(), "qubo_matrix": np.round(total.M, 3).tolist(),
                             "exact": {"bits": bits(ex_sol.x), "objective": round(ex_sol.objective, 3), "time_s": round(t_ex, 4)},
                             "qaoa": {"best_bits": bits(best) if best is not None else None,
                                      "best_objective": round(float(qubos[0].energy(best)[0]), 3) if best is not None else None,
                                      "feasible": bool(feas.any()),
                                      "approx_ratio": round(approx_ratio(float(qubos[0].energy(best)[0]), exr), 4) if best is not None else 0,
                                      "p_opt": round(float(qr.probs[exr.opt_mask].sum()), 5), "time_s": round(t_q, 4),
                                      "top": [{"bits": bits(en["X"][k]), "p": round(float(qr.probs[k]), 5),
                                               "feasible": bool(en["feas"][k]), "optimal": bool(exr.opt_mask[k])} for k in order],
                                      # the 256 measured shots (Qiskit Aer), grouped by bitstring: the demo's histogram
                                      "backend": "qiskit-aer AerSimulator (ideal, noiseless), 256 shots, seed 1234",
                                      "shots": shot_table(qr.samples, lp, qubos, exr)},
                             "explanation": [e["text"] for e in __import__("qgrid.explain", fromlist=["explain"]).explain(lp, ex_sol.x, b.name)
                                             if not e["var"].startswith("line")]})
        out["live_solve"] = live
    return out


def shot_table(samples, lp, qubos, exr):
    """Distinct measured bitstrings with their counts, true objective (Rs), feasibility and optimality."""
    from qgrid import qubo as Q
    keys, cnt = np.unique(samples.astype(int), axis=0, return_counts=True)
    feas = Q.feasible_mask(lp, keys.astype(float))
    obj = qubos[0].energy(keys.astype(float))
    n = keys.shape[1]
    opt = exr.opt_mask[(keys * (1 << np.arange(n))).sum(1)]
    rows = [{"bits": "".join(map(str, k)), "count": int(c), "feasible": bool(f), "optimal": bool(o), "objective": round(float(e), 3)}
            for k, c, f, o, e in zip(keys, cnt, feas, opt, obj)]
    return sorted(rows, key=lambda r: (-r["count"], r["bits"]))


V2_COLS = ["method", "objective_incl_demand_rs", "cost_incl_demand_rs", "cost_rs", "objective_rs", "demand_charge_rs",
           "peak_kva", "co2_kg", "dg_hours", "critical_downtime_h", "critical_unserved_kwh", "gap_to_opt_pct", "runtime_s",
           "replan_time_s", "benders_iters_mean", "benders_gap_mean_pct", "benders_converged_rate", "master_opt_share",
           "qaoa_block_opt_share", "qaoa_block_solves", "admm_converged_rate", "admm_iters_mean", "feasible", "solar"]


def attach_qgrid(campus, meta, outs: dict) -> None:
    """Current Q-Grid design (24 h, continuous dispatch, LP coordination, realistic forecasts) on every named day:
    its run becomes runs.qgrid (the earlier prototype's run is kept as runs.earlier), and the headline day gets the
    worked example qgrid_story."""
    from qgrid import data as qdata
    from qgrid.ablation.engine import ACfg, run_day
    from qgrid.ablation.forecast import GBMForecaster
    from qgrid.ablation.story import qgrid_story
    from qgrid.model import Schedule, make_day
    from qgrid.v2.horizon import YearData
    from qgrid.v2.tariff import settle_v2
    proc = qdata.load_processed()
    yd = YearData(campus, proc, "A", forecast=False)
    hold = meta["days"]["eval_days"] + [v["date"] for v in meta["days"]["scenarios"].values()]
    fc = GBMForecaster(campus, proc, yd, hold, cache=ROOT / "results" / "bench_v3" / "gbm_rvce.pkl")
    qp = ROOT / "results" / "quantum_placement" / "summary.json"
    placement = json.loads(qp.read_text()) if qp.exists() else {}
    hs = campus.cfg["headline_scenario"]
    for k, o in outs.items():
        sc = meta["days"]["scenarios"][k]
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            if k == hs:
                sched, story = qgrid_story(campus, yd, sc["date"], sc.get("outage"), fc, slot=72, placement=placement)
                o["qgrid_story"] = story
            else:
                sched, _ = run_day(campus, yd, sc["date"], sc.get("outage"), ACfg(True, True, True, True), forecaster=fc)
        day = make_day(campus, proc, sc["date"], "A", k, sc.get("outage"))
        m = settle_v2(campus, day, sched)
        o["runs"]["earlier"] = o["runs"]["qgrid"]
        o["runs"]["qgrid"] = run_block_ext(campus, day, sched, m)
        print(f"{k}: current Q-Grid objective Rs {m['objective_rs']:,.0f}, critical downtime {m['critical_downtime_h']:.2f} h", flush=True)


def run_block_ext(campus, day, sched, metrics):
    """Replay run block for a schedule computed outside build_scenario (same fields as run_block)."""
    from qgrid.model import DT, flex_load, pump_load, soc_trajectory, tank_trajectory
    net = np.asarray(metrics["net_kw"])
    demand = day.load + flex_load(campus, sched) + pump_load(campus, sched)
    deficit = np.where(day.grid[None, :], 0.0, np.maximum(net, 0))
    crit_short = np.maximum(deficit - np.maximum(demand - day.crit, 0), 0)
    g = day.grid[None, :]
    imp = np.where(g, np.maximum(net, 0), 0).sum(0); exp = np.where(g, np.maximum(-net, 0), 0).sum(0)
    cost_t = (imp * day.price_imp - exp * day.price_exp) * DT
    fuel_t = np.zeros(96); co2_t = imp * DT * day.ef
    for i, b in enumerate(campus.buildings):
        if b.dg:
            on = np.asarray(sched.dg_kw[i]) > 1e-9
            L = b.dg.litres(on, np.asarray(sched.dg_kw[i]))
            fuel_t += L * b.dg.price; co2_t += L * b.dg.co2_per_l
    return {"p_bat": np.round(sched.p_bat, 2).tolist(),
            "soc": [np.round(soc_trajectory(b.battery, sched.p_bat[i]) / max(b.battery.kwh, 1e-9), 4).tolist() for i, b in enumerate(campus.buildings)],
            "flows": np.round(sched.flows, 2).tolist(), "dg_kw": np.round(sched.dg_kw, 2).tolist(),
            "pump_on": np.asarray(sched.pump_on).astype(int).tolist(), "flex_start": sched.flex_start,
            "tank_m3": {b.name: np.round(tank_trajectory(b.tank, np.asarray(sched.pump_on[i]), day.water), 2).tolist()
                        for i, b in enumerate(campus.buildings) if b.tank},
            "net_kw": np.round(net, 2).tolist(), "grid_import_kw": np.round(imp, 2).tolist(), "grid_export_kw": np.round(exp, 2).tolist(),
            "demand_kw": np.round(demand, 2).tolist(), "crit_unserved_kw": np.round(crit_short, 3).tolist(),
            "cum_cost_rs": np.round(np.cumsum(cost_t + fuel_t), 1).tolist(), "cum_co2_kg": np.round(np.cumsum(co2_t), 1).tolist(),
            "metrics": {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in metrics.items() if kk != "net_kw"}}


def attach_v2(campus, meta, outs: dict) -> None:
    """Optimiser v2 results (results/v2, from scripts/run_benchmark_v2.py) and the headline worked example."""
    d = ROOT / "results" / "v2"
    if not (d / "benchmark_v2.csv").exists():
        return
    import pandas as pd
    from qgrid import QUANTUM_LABEL
    from qgrid import data as qdata
    b = pd.read_csv(d / "benchmark_v2.csv")
    s = pd.read_csv(d / "benchmark_v2_summary.csv")
    cols = [c for c in V2_COLS if c in b.columns]
    summ = s[[c for c in ["day_set", "n_days"] + V2_COLS if c in s.columns]].to_dict(orient="records")
    tv = campus.cfg["tariff_v2"]
    hs = campus.cfg["headline_scenario"]
    for k, o in outs.items():
        rows = b[b.scenario == k][cols].to_dict(orient="records")
        o["v2"] = {"rows": rows, "summary": summ, "tariff_status": tv["status"],
                   "demand_rs_per_kva_month": qdata.demand_rate_month(campus.cfg, meta["days"]["scenarios"][k]["date"]),
                   "days_per_month": tv["days_per_month"],
                   "power_factor": tv["power_factor"], "label": QUANTUM_LABEL}
        if "qgrid_v2" in o["runs"]:
            row = b[(b.scenario == k) & (b.method == "V2-QAOA")]
            got = o["runs"]["qgrid_v2"]["metrics"]
            # stored v2 schedules are rounded to 0.01 kW, so the re-settled objective may differ by a few rupees
            ref = float(row.iloc[0].objective_incl_demand_rs) if len(row) else float("nan")
            o["v2"]["replay_matches_benchmark"] = bool(len(row) and abs(ref - got["objective_incl_demand_rs"]) <= 1e-4 * abs(ref))
    sched = d / "schedules" / f"{hs}_V2-QAOA.json"
    if sched.exists() and hs in outs:
        from qgrid import data as qdata
        from qgrid.v2.explain import explain_slot
        from qgrid.v2.horizon import YearData
        sc = meta["days"]["scenarios"][hs]
        proc = qdata.load_processed()
        yd = YearData(campus, proc, "A", forecast=False)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            ex = explain_slot(campus, yd, sc["date"], sc.get("outage"), json.loads(sched.read_text()), t0=72,
                              zone="pumping_station", kind="pump")
        outs[hs]["v2"]["example"] = ex
        print(f"v2 worked example: {ex['time']}, {len(ex['benders'])} Benders iterations, "
              f"block {ex['block']['n_qubits']} qubits, QAOA = exact: {ex['block']['qaoa']['best_bits'] == ex['block']['exact']['bits']}")


def main():
    from qgrid import data as qdata
    from qgrid.model import load_campus
    campus = load_campus()
    meta = qdata.load_processed()["meta"]
    keys = list(meta["days"]["scenarios"])
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    prov = {"git_commit": commit, "seeds": {"qaoa": campus.cfg["qaoa"]["seed"]},
            "files": {p: sha(ROOT / p) for p in ["data/raw/nasa_power/bengaluru_2023_hourly.json",
                                                  "data/raw/ausgrid/Ausgrid_solar_home_data.zip", "data/raw/osm_rvce.json",
                                                  "config.yaml", "configs/profiles.yaml", "results/benchmark.csv",
                                                  "results/qaoa_params.json"]}}
    with ProcessPoolExecutor(max_workers=min(6, len(keys))) as ex:
        outs = dict(zip(keys, ex.map(build_scenario, keys)))
    RDIR.mkdir(parents=True, exist_ok=True)
    index = {"scenarios": [], "headline": campus.cfg["headline_scenario"], "provenance": prov,
             "label": outs[keys[0]]["label"], "data_status": meta["data_status"]}
    if os.environ.get("QGRID_ATTACH_BENDERS") == "1":   # earlier hybrid-Benders example (old data), off by default
        attach_v2(campus, meta, outs)
    attach_qgrid(campus, meta, outs)
    for k, o in outs.items():
        o["provenance"] = prov
        (RDIR / f"scenario_{k}.json").write_text(dump(o))
        index["scenarios"].append({"key": k, "date": o["scenario"]["date"], "note": o["scenario"]["note"],
                                   "replay_matches_benchmark": o["replay_matches_benchmark"]})
        print(f"{k}: replay matches benchmark: {o['replay_matches_benchmark']}")
    for f in ["benchmark_summary.csv", "headline.json", "economics.json", "search_space.json", "robustness.csv",
              "qaoa_scaling.csv", "problem_day.json"]:
        p = ROOT / "results" / f
        if p.exists():
            index.setdefault("files", []).append(f)
            shutil.copy(p, RDIR / f)
    # offline sandbox / stress-test cases (one factor at a time, C-QIEA what-if) and parsed operator rules
    from qgrid.api import WhatIf, whatif
    from qgrid.rules import load_rules
    hs = campus.cfg["headline_scenario"]
    cases = {"base": {}, "kill_heavy_labs": {"kill_zone": "heavy_labs"}, "tariff_spike": {"tariff_scale": 2.0},
             "fest": {"scenario": "fest"}, "outage_3": {"outage_hours": 3}}
    for k, vals in {"battery_scale": [0, 0.5, 1.5, 2], "pv_scale": [0.5, 0.75, 1.25, 1.5], "outage_hours": [0, 1, 5],
                    "dg_cost_scale": [1.5, 2]}.items():
        for v in vals:
            cases[f"{k}_{v}"] = {k: v}
    cases["battery_scale_1"] = cases["pv_scale_1"] = cases["dg_cost_scale_1"] = {}
    sb = {"label": "WHAT-IF (C-QIEA, quantum-inspired classical solver, one factor at a time) — not headline results",
          "scenario": hs, "cases": {}}
    for name, params in cases.items():
        sb["cases"][name] = whatif(WhatIf(**{"scenario": hs, **params}))
    (RDIR / "sandbox.json").write_text(dump(sb))
    (RDIR / "rules.json").write_text(dump([r.model_dump() for r in load_rules(campus)]))
    index["files"] += ["sandbox.json", "rules.json"]
    (RDIR / "index.json").write_text(dump(index))
    PUB.mkdir(parents=True, exist_ok=True)
    for p in RDIR.glob("*.json"):   # validate every replay file is strict JSON before publishing
        json.loads(p.read_text(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"{p.name}: {c}")))
    for p in RDIR.glob("*"):
        if p.suffix in (".json", ".csv"):
            shutil.copy(p, PUB / p.name)
    print("replay written to", RDIR.relative_to(ROOT), "and", PUB.relative_to(ROOT))


if __name__ == "__main__":
    main()
