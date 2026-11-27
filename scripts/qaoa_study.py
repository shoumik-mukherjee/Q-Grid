"""QAOA study on real-data local QUBOs.

1. Distributions for the 3D demo: one instance per test day (hostel, 12:00, scenario A), simulated with
   qiskit-aer (reference simulator), p = 2 trained parameters -> results/replay/qaoa_<key>.json
2. Scaling study: qubits (3..15, by truncating horizon/lines) x p = 1..3 ->
   results/qaoa_scaling.csv, results/plots/qaoa_scaling.png
   - trained (transferred) parameters on every instance (numpy statevector)
   - per-instance optimised parameters for n <= 10 (COBYLA), to show the parameter-transfer gap
   - Aer wall time per circuit for timing comparison
QAOA on a local simulator: verified against exact solutions, no quantum advantage claimed.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT)]
from qgrid import QUANTUM_LABEL  # noqa: E402
from qgrid import data as qdata  # noqa: E402
from qgrid import qubo as Q  # noqa: E402
from qgrid.agent import BuildingAgent  # noqa: E402
from qgrid.instances import sample_instances  # noqa: E402
from qgrid.model import load_campus, make_day  # noqa: E402
from qgrid.solvers.exact import approx_ratio, solve_exact  # noqa: E402
from qgrid.solvers.qaoa import load_params, optimise_params, probabilities_aer, run_qaoa  # noqa: E402


def export_distributions(proc, zone: str = "hostels", slot: int = 76):
    """One QUBO per scenario (hostels zone, 19:00, scenario A: battery, DG, jobs, tie-lines), simulated with
    qiskit-aer; saves the full distribution, QUBO matrix, term groups and exact optimum for the demo."""
    campus = load_campus()
    scen = proc["meta"]["days"]["scenarios"]
    i = campus.names.index(zone)
    out = {}
    for key, sc in scen.items():
        day = make_day(campus, proc, sc["date"], "A", key, sc.get("outage"))
        ag = BuildingAgent(campus, day, i, campus.cfg["admm"]["horizon_slots"])
        lp = ag.local_problem(slot, ag.incident, v={l: 0.0 for l, _ in ag.incident}, rho=0.05)
        qubos = Q.build(lp)
        groups, _, _ = Q.build_groups(lp)
        en = Q.enumerate_all(lp, qubos)
        total, A_x, A_s, _ = Q.auto_tune(lp, qubos, enum=en)
        ex = solve_exact(lp, qubos, enum=en)
        g, b = load_params(2)
        t0 = time.time()
        r = run_qaoa(total, p=2, shots=1024, seed=1234, backend="aer", params=(g, b))
        t_aer = time.time() - t0
        X = en["X"]
        order = np.argsort(-r.probs)
        bits = lambda k: "".join(str(int(v)) for v in X[k])  # noqa: E731
        top = [{"bits": bits(k), "p": round(float(r.probs[k]), 6), "energy": round(float(en["total"][k]), 3),
                "objective": round(float(en["ob"][k]), 3), "feasible": bool(en["feas"][k]),
                "optimal": bool(ex.opt_mask[k])} for k in order[:256]]
        opt_idx = int(np.argmax(ex.opt_mask))
        rec = {
            "label": QUANTUM_LABEL, "scenario": key, "day": sc["date"], "zone": zone, "slot": slot,
            "price_scenario": "A", "price_status": day.status["price_scenario"],
            "n_qubits": lp.n, "var_names": lp.var_names(), "p": 2,
            "gammas": np.round(g, 6).tolist(), "betas": np.round(b, 6).tolist(),
            "qubo_matrix": np.round(total.M, 4).tolist(), "qubo_const": round(float(total.const), 4),
            "groups": {k: {"M": np.round(v.M, 4).tolist(), "const": round(float(v.const), 4)} for k, v in groups.items()},
            "penalties": {"A_x": A_x, "A_s": A_s},
            "probs": np.round(r.probs, 7).tolist(), "energies": np.round(en["total"], 3).tolist(),
            "feasible": en["feas"].astype(int).tolist(), "top_states": top,
            # pure objective (Rs, no penalties) of every bitstring: the exact check and the demo's winner use this
            "objectives": np.round(en["ob"], 3).tolist(),
            "optimum": {"bits": bits(opt_idx), "objective": ex.objective,
                        "p_opt": round(float(r.probs[ex.opt_mask].sum()), 6)},
            "uniform_p": 2.0 ** -lp.n, "simulator": "qiskit-aer statevector (ideal, noiseless)",
            "aer_time_s": round(t_aer, 3),
        }
        (ROOT / "results" / "replay" / f"qaoa_{key}.json").write_text(json.dumps(rec), encoding="utf-8")
        out[key] = {"n": lp.n, "p_opt": rec["optimum"]["p_opt"], "uniform": rec["uniform_p"]}
        print(f"[dist] {key}: n={lp.n} P(opt)={rec['optimum']['p_opt']:.4f} (uniform {rec['uniform_p']:.6f})")
    return out


def truncate(lp: Q.LocalProblem, H: int, n_lines: int, jobs: bool, dg: bool = False) -> Q.LocalProblem:
    import dataclasses
    tk = lp.tank
    if tk is not None:
        tk = dataclasses.replace(tk, demand=tk.demand[:H])
    d = lp.dg
    if d is not None:
        d = dataclasses.replace(d, q=d.q[:H])
    lines = lp.lines[:n_lines]
    return dataclasses.replace(lp, r0=lp.r0[:H], pi=lp.pi[:H], pe=lp.pe[:H], grid=lp.grid[:H], noncrit=lp.noncrit[:H],
                               pc=lp.pc[:H] if lp.pc is not None else None, pd=lp.pd[:H] if lp.pd is not None else None,
                               lines=lines, v={l: lp.v.get(l, 0.0) for l, _ in lines},
                               jobs=lp.jobs if jobs else [], dg=d if dg else None, tank=tk)


def scaling_study(proc, n_per_size: int = 6, seed: int = 11):
    insts = [lp for _, lp in sample_instances(proc, proc["meta"]["days"]["eval_days"], 60, seed=seed)]
    configs = [(1, 0, False), (1, 1, False), (2, 1, False), (2, 2, False), (3, 1, False), (3, 2, False),
               (3, 2, True), (3, 3, True)]
    rows = []
    rng = np.random.default_rng(seed)
    for H, nl, fx in configs:
        pool = [truncate(lp, H, nl, fx) for lp in insts if lp.bat.present and len(lp.lines) >= nl and (not fx or lp.jobs)]
        pool = [pool[k] for k in rng.permutation(len(pool))[:n_per_size]]
        for lp in pool:
            qubos = Q.build(lp)
            en = Q.enumerate_all(lp, qubos)
            total, *_ = Q.auto_tune(lp, qubos, enum=en)
            ex = solve_exact(lp, qubos, enum=en)
            c, h, J = total.to_ising()
            sig = float(np.sqrt((h ** 2).sum() + (J ** 2).sum())) or 1.0
            E = (en["total"] - c) / sig
            for p in (1, 2, 3):
                for mode in ("trained", "optimised"):
                    if mode == "optimised" and lp.n > 10:
                        continue
                    t0 = time.perf_counter()
                    params = load_params(p) if mode == "trained" else optimise_params([E], p, maxiter=120)[:2]
                    t_opt = time.perf_counter() - t0
                    r = run_qaoa(total, p=p, shots=256, seed=seed, params=params, total_energies=en["total"])
                    feas = Q.feasible_mask(lp, r.samples)
                    best = float(qubos[0].energy(r.samples[feas]).min()) if feas.any() else np.nan
                    t1 = time.perf_counter()
                    if mode == "trained" and p == 2:
                        probabilities_aer(h / sig, J / sig, *params)
                        t_aer = time.perf_counter() - t1
                    else:
                        t_aer = np.nan
                    with np.errstate(all="ignore"):
                        e_mean = float(r.probs[ex.feasible] @ en["ob"][ex.feasible]) / max(r.probs[ex.feasible].sum(), 1e-12)
                    rows.append({
                        "n_qubits": lp.n, "p": p, "params": mode,
                        "p_opt": float(r.probs[ex.opt_mask].sum()), "uniform_p": 2.0 ** -lp.n,
                        "expected_approx_ratio": float(np.clip(
                            (ex.worst_feasible - e_mean) / max(ex.worst_feasible - ex.objective, 1e-12), -10, 1)),
                        "best_sample_approx_ratio": approx_ratio(best, ex) if feas.any() else np.nan,
                        "found_opt": bool(feas.any() and approx_ratio(best, ex) >= 1 - 1e-9),
                        "p_feasible": float(r.probs[ex.feasible].sum()),
                        "param_opt_time_s": t_opt if mode == "optimised" else 0.0,
                        "aer_time_s": t_aer,
                    })
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "qaoa_scaling.csv", index=False)
    agg = df.groupby(["params", "p", "n_qubits"]).mean(numeric_only=True).reset_index()

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    for (mode, p), g in agg.groupby(["params", "p"]):
        ls = "-" if mode == "trained" else "--"
        axs[0].semilogy(g.n_qubits, g.p_opt, ls, marker="o", label=f"p={p} {mode}")
        axs[1].plot(g.n_qubits, g.best_sample_approx_ratio, ls, marker="o", label=f"p={p} {mode}")
    u = agg.drop_duplicates("n_qubits")
    axs[0].semilogy(u.n_qubits, u.uniform_p, "k:", label="uniform 2^-n")
    axs[0].set(xlabel="qubits", ylabel="P(optimum)", title="Probability of the exact optimum")
    axs[1].set(xlabel="qubits", ylabel="approx. ratio (best of 256 shots)", title="Best-sample approximation ratio")
    t = agg[(agg.params == "trained") & (agg.p == 2)]
    axs[2].semilogy(t.n_qubits, t.aer_time_s, "o-", label="Aer statevector, p=2 (s/circuit)")
    axs[2].set(xlabel="qubits", ylabel="seconds", title="Simulation time")
    for a in axs:
        a.legend(fontsize=7); a.grid(alpha=.3)
    fig.suptitle(QUANTUM_LABEL, fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "plots" / "qaoa_scaling.png", dpi=130)
    print(agg[agg.params == "trained"].pivot_table(index="n_qubits", columns="p", values=["p_opt", "best_sample_approx_ratio"]).round(4))
    return df


if __name__ == "__main__":
    proc = qdata.load_processed()
    export_distributions(proc)
    if "--distributions-only" not in sys.argv:
        scaling_study(proc)
    print("saved results/qaoa_scaling.csv, results/plots/qaoa_scaling.png")
