"""Quantum placement experiment: does a QAOA incumbent help exact branch and bound on the commitment problem?

Instances: every building's commitment MILP at every hourly decision of the chosen design (A+B+C, perfect forecast)
on a sample of evaluation days plus the outage day, captured from a real run (engine hook).
For each instance:
  1. cold:  HiGHS branch and bound (0.01 % gap) -> optimum, nodes, time
  2. QAOA incumbent: LP relaxation; the <= 16 most fractional binaries form a block; block QUBO = LP reduced costs +
     quadratic penalties for the constraint rows that involve only block variables; QAOA (p = 2, trained angles,
     256 shots, numpy statevector) samples the block; the block is also brute-forced (verification); the 8 most
     frequent valid samples are completed (other binaries rounded from the LP) and scored exactly by the LP with all
     binaries fixed; the best becomes the incumbent. If the LP relaxation is already integral, no QAOA is needed.
  3. warm: the same branch and bound started from the QAOA incumbent -> nodes, time
Outputs results/quantum_placement.csv and a summary JSON. QAOA on a local simulator, verified against exact search,
no quantum advantage claimed.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT)]
OUT = ROOT / "results" / "quantum_placement"


def qaoa_incumbent(m, c, rng, max_q=16):
    from qgrid.qubo import QUBO
    from qgrid.solvers.qaoa import load_params, run_qaoa
    t0 = time.perf_counter()
    lp = m.solve(c, relax=True)
    x = lp["x"]
    ib = np.nonzero(m.integ)[0]
    frac = np.abs(x[ib] - np.round(x[ib]))
    rec = {"lp_integral": bool(frac.max() < 1e-6), "block_q": 0, "qaoa_block_opt": None}
    base = np.round(x[ib])
    if rec["lp_integral"]:
        rec["inc_time_s"] = time.perf_counter() - t0
        return x, rec
    order = np.argsort(-frac)[:max_q]
    B = ib[order[frac[order] > 1e-6]]
    nb = len(B)
    rec["block_q"] = nb
    rc = lp["col_dual"][B]                              # reduced cost: objective change per unit increase
    Mq = np.diag(rc.astype(float))
    const = 0.0
    # rows touching only block (and fixed-integer) columns: quadratic penalty on violation of equality/<= rows
    A = m.A.tocsr()
    pen = 4 * (np.abs(rc).max() + 1.0)
    inB = np.zeros(m.ncol, bool); inB[B] = True
    for r in range(A.shape[0]):
        cols = A.indices[A.indptr[r]:A.indptr[r + 1]]
        if len(cols) == 0 or not inB[cols].all():
            continue
        a = np.zeros(nb); pos = {cc: k for k, cc in enumerate(B)}
        for cc, v in zip(cols, A.data[A.indptr[r]:A.indptr[r + 1]]):
            a[pos[cc]] = v
        lo, hi = m.rlo[r], m.rhi[r]
        tgt = hi if np.isfinite(hi) else lo                # (a z - tgt)^2 for equalities; one-sided rows only if violated at rounding
        if np.isfinite(lo) and np.isfinite(hi) and abs(lo - hi) < 1e-9 or (np.isfinite(hi) and a @ np.round(x[B]) > hi):
            Mq += pen * np.outer(a, a); Mq[np.diag_indices(nb)] += -2 * pen * tgt * a; const += pen * tgt * tgt
    q = QUBO(const, Mq)
    qr = run_qaoa(q, p=2, shots=256, seed=int(rng.integers(1 << 30)), params=load_params(2))
    Z = ((np.arange(1 << nb)[:, None] >> np.arange(nb)[None, :]) & 1).astype(float)
    E_all = q.energy(Z)
    keys, cnt = np.unique(qr.samples.astype(int), axis=0, return_counts=True)
    E_s = q.energy(keys.astype(float))
    rec["qaoa_block_opt"] = bool(E_s.min() <= E_all.min() + 1e-9 * max(1.0, abs(E_all.min())))
    best, best_obj = None, np.inf
    for k in np.argsort(-cnt)[:8]:
        full = base.copy()
        full[np.searchsorted(ib, B)] = keys[k]
        try:
            s = m.solve(c, fix_int=full)
        except RuntimeError:
            continue
        if s["obj"] < best_obj:
            best_obj, best = s["obj"], s["x"]
    if best is None:                                    # fall back to the rounded LP
        try:
            best = m.solve(c, fix_int=base)["x"]
        except RuntimeError:
            best = None
    rec["inc_time_s"] = time.perf_counter() - t0
    return best, rec


def main(days_n=4):
    from qgrid import data as qdata
    from qgrid.ablation.engine import ACfg, run_day
    from qgrid.model import load_campus
    from qgrid.v2.horizon import YearData
    campus = load_campus()
    proc = qdata.load_processed()
    yd = YearData(campus, proc, "A", forecast=False)
    meta = proc["meta"]
    hs = meta["days"]["scenarios"][campus.cfg["headline_scenario"]]
    days = [(d, None) for d in meta["days"]["eval_days"][::max(1, 30 // days_n)][:days_n]] + [(hs["date"], hs["outage"])]
    rng = np.random.default_rng(7)
    rows = []

    def hook(t0, i, m, c):
        m2 = m                                          # same model object; solves below do not change its data
        cold_t = time.perf_counter()
        old_tl, old_gap = m2.time_limit, m2.mip_gap
        m2.time_limit, m2.mip_gap = 30.0, 1e-4
        m2._h = None                                   # fresh solver: no basis or cuts carried between runs
        cold_t = time.perf_counter()
        cold = m2.solve(c); cold_t = time.perf_counter() - cold_t
        inc, rec = qaoa_incumbent(m2, c, rng)
        m2._h = None
        warm_t = time.perf_counter()
        warm = m2.solve(c, start=inc) if inc is not None else cold
        warm_t = time.perf_counter() - warm_t
        m2._h = None
        m2.time_limit, m2.mip_gap = old_tl, old_gap
        inc_obj = float(m2.cost0 @ inc + c @ inc) if inc is not None else np.nan
        opt = cold["obj"]
        rows.append({"day": cur["day"], "t0": t0, "zone": campus.names[i], "n_binaries": int(m2.integ.sum()),
                     "cold_nodes": cold["nodes"], "cold_time_s": cold_t, "warm_nodes": warm["nodes"], "warm_time_s": warm_t,
                     "opt": opt, "incumbent_obj": inc_obj,
                     "incumbent_optimal": bool(np.isfinite(inc_obj) and inc_obj <= opt + 1e-4 * max(1.0, abs(opt))), **rec})

    cur = {}
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for d, o in days:
            cur["day"] = d
            run_day(campus, yd, d, o, ACfg(True, True, True, False), hook=hook)
            print(d, len(rows), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "quantum_placement.csv", index=False)
    q = df[~df.lp_integral]
    summ = {"instances": len(df), "lp_relaxation_integral_pct": 100 * float(df.lp_integral.mean()),
            "cold_root_node_pct": 100 * float((df.cold_nodes <= 1).mean()),
            "qaoa_used": int(len(q)), "qaoa_block_qubits_mean": float(q.block_q.mean()) if len(q) else None,
            "qaoa_block_opt_pct": 100 * float(q.qaoa_block_opt.astype(float).mean()) if len(q) else None,
            "incumbent_optimal_pct_qaoa_instances": 100 * float(q.incumbent_optimal.mean()) if len(q) else None,
            "incumbent_optimal_pct_all": 100 * float(df.incumbent_optimal.mean()),
            "nodes_cold_mean": float(df.cold_nodes.mean()), "nodes_warm_mean": float(df.warm_nodes.mean()),
            "time_cold_mean_s": float(df.cold_time_s.mean()), "time_warm_mean_s": float(df.warm_time_s.mean()),
            "time_incumbent_mean_s": float(df.inc_time_s.mean()),
            "time_warm_plus_incumbent_mean_s": float((df.warm_time_s + df.inc_time_s).mean()),
            "timing_note": "cold and warm each on a fresh HiGHS model (includes model build); run alongside other jobs",
            "nodes_cold_mean_qaoa_instances": float(q.cold_nodes.mean()) if len(q) else None,
            "nodes_warm_mean_qaoa_instances": float(q.warm_nodes.mean()) if len(q) else None,
            "time_cold_mean_qaoa_instances_s": float(q.cold_time_s.mean()) if len(q) else None,
            "time_warm_mean_qaoa_instances_s": float(q.warm_time_s.mean()) if len(q) else None,
            "days": [d for d, _ in days],
            "label": "QAOA on a local simulator: verified against exact solutions, no quantum advantage claimed"}
    json.dump(summ, open(OUT / "summary.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main(int(next((a.split("=")[1] for a in sys.argv if a.startswith("--days=")), 4)))
