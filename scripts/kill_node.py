"""Networked ADMM: one OS process per building (websockets, no coordinator), optionally crashing one.

    python scripts/kill_node.py --day 2023-04-13 --scenario A --kill lab_block --kill-slot 48

Collects every node's record, checks that both ends of each tie-line applied the same flow, rebuilds the
campus schedule, audits feasibility and settles cost. Recovery time = wall time of the kill slot at the
surviving nodes (includes the 3 s silence detection and re-convergence).
Writes results/net/run_<day>_<scenario>[_kill-<b>].json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT)]
from qgrid import data as qdata  # noqa: E402
from qgrid.model import T, Schedule, check_feasibility, load_campus, make_day, settle  # noqa: E402


def run(day: str, scenario: str = "A", method: str = "qaoa", kill: str | None = None, kill_slot: int = 48,
        slots: int = T, port_base: int = 8765, scenario_key: str = "") -> dict:
    campus = load_campus()
    outdir = ROOT / "results" / "net"
    outdir.mkdir(parents=True, exist_ok=True)
    procs, outs = [], {}
    t0 = time.time()
    for b in campus.names:
        out = outdir / f"node_{b}.json"
        out.unlink(missing_ok=True)
        outs[b] = out
        cmd = [sys.executable, "-m", "qgrid.node", "--building", b, "--day", day, "--scenario", scenario,
               "--method", method, "--port-base", str(port_base), "--slots", str(slots), "--out", str(out),
               "--scenario-key", scenario_key]
        if kill == b:
            cmd += ["--die-at-slot", str(kill_slot)]
        procs.append(subprocess.Popen(cmd, cwd=ROOT, env={**__import__("os").environ,
                                                          "PYTHONPATH": f"{ROOT / 'backend'}:{ROOT}"}))
    for p in procs:
        p.wait(timeout=1800)
    wall = time.time() - t0

    recs = {b: json.loads(o.read_text(encoding="utf-8")) for b, o in outs.items() if o.exists()}
    sched = Schedule.empty(campus)
    consistent = True
    for i, b in enumerate(campus.names):
        if b in recs:
            r = recs[b]
            n = len(r["p_bat"])
            sched.p_bat[i, :n] = r["p_bat"]
            sched.dg_kw[i, :n] = r.get("dg_kw", [0.0] * n)
            sched.pump_on[i, :n] = r.get("pump_on", [0.0] * n)
            sched.flex_start[i] = dict(r["flex_start"])
        # a crashed node's battery stays idle from the crash on (its record ends at the crash -> zeros)
    for l, (a, b) in enumerate(campus.lines):
        fa = recs.get(campus.names[a], {}).get("flows", {}).get(str(l))
        fb = recs.get(campus.names[b], {}).get("flows", {}).get(str(l))
        if fa is not None and fb is not None:
            n = min(len(fa), len(fb))
            if not np.allclose(fa[:n], fb[:n]):
                consistent = False
            sched.flows[l, :n] = fa[:n]
    proc = qdata.load_processed()
    sc = proc["meta"]["days"]["scenarios"].get(scenario_key, {})
    dd = make_day(campus, proc, day, scenario, scenario_key, sc.get("outage"))
    # crashed zone after the crash: its controller is offline -> safe default (float-switch pump, DG/UPS only in outage)
    if kill:
        from qgrid.agent import BuildingAgent
        i = campus.names.index(kill)
        ag = BuildingAgent(campus, dd, i)
        r = recs.get(kill)
        for t in range(T):
            if r is not None and t < len(r["p_bat"]):
                ag.e = float(ag.bat.step(ag.e, r["p_bat"][t])) if ag.bat.present else 0.0
                if ag.tank is not None:
                    ag.level += (ag.tank.pump_m3 if r["pump_on"][t] else 0.0) - ag.water[t]
                for j in ag.pending_jobs():
                    if j.name in r["flex_start"] and r["flex_start"][j.name] <= t:
                        ag.started[j.name] = r["flex_start"][j.name]
                        ag.running += j.profile(ag.started[j.name])
                continue
            c = ag.apply_safe_default(t)
            sched.p_bat[i, t], sched.dg_kw[i, t], sched.pump_on[i, t] = c["p_bat"], c["dg_kw"], float(c["pump_on"])
        sched.flex_start[i] = dict(ag.started)
    m = settle(campus, dd, sched)
    feas = check_feasibility(campus, sched, dd)
    surv = [b for b in campus.names if b != kill and b in recs]
    rec_time = None
    if kill:
        times = [recs[b]["slots"][kill_slot]["time_s"] for b in surv if len(recs[b]["slots"]) > kill_slot]
        rec_time = max(times) if times else None
    res = {
        "day": day, "scenario": scenario, "method": method, "kill": kill, "kill_slot": kill_slot if kill else None,
        "slots": slots, "wall_time_s": round(wall, 2), "cost_rs": m["cost_rs"], "co2_kg": m["co2_kg"],
        "critical_downtime_h": m["critical_downtime_h"], "scenario_key": scenario_key,
        "feasible": feas["feasible"], "violations": feas["violations"], "feasibility_rate": feas["feasibility_rate"],
        "tie_line_flows_consistent": consistent,
        "recovery_time_s": rec_time,
        "dead_neighbours": {b: recs[b]["dead_neighbours"] for b in surv if recs[b]["dead_neighbours"]},
        "rounds_per_slot_mean": float(np.mean([s["rounds"] for b in surv for s in recs[b]["slots"]])),
        "slot_time_s_mean": float(np.mean([s["time_s"] for b in surv for s in recs[b]["slots"]])),
        "price_status": dd.status["price_scenario"],
    }
    tag = f"run_{day}_{scenario}" + (f"_kill-{kill}" if kill else "")
    (outdir / f"{tag}.json").write_text(json.dumps({**res, "nodes": recs}, default=str), encoding="utf-8")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default=None)
    ap.add_argument("--scenario", default="A")
    ap.add_argument("--method", default="qaoa")
    ap.add_argument("--kill", default=None)
    ap.add_argument("--kill-slot", type=int, default=48)
    ap.add_argument("--slots", type=int, default=T)
    ap.add_argument("--scenario-key", default="")
    a = ap.parse_args()
    meta = qdata.load_processed()["meta"]
    key = a.scenario_key or meta.get("headline_scenario", "outage_fest")
    day = a.day or meta["days"]["scenarios"][key]["date"]
    print(json.dumps(run(day, a.scenario, a.method, a.kill, a.kill_slot, a.slots, scenario_key=key), indent=2, default=str))
