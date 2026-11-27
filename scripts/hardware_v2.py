"""Optional real-hardware comparison for the v2 master (needs the team lead's tokens; never runs by default).

1. IBM Quantum: QAOA (p=2, trained angles) on a sample of real building master blocks recorded by the v2
   benchmark (results/v2/qaoa_blocks.json). Compared with Qiskit Aer (ideal simulator, same circuit and shots)
   and with exact search over the block.
2. D-Wave Leap: the whole-campus 24 h commitment QUBO at the first decision slot (all buildings' masters, the
   cut active at the incumbent, one-hot and tank penalties) -> results/v2/campus_qubo.json. Compared with
   simulated annealing and the exact MILP masters, all scored with the true master objective J.

Runs on hardware only if BOTH the token and the opt-in variable are set and the SDK is installed:
  IBM:    QGRID_IBM_TOKEN=... QGRID_ALLOW_IBM=1     (pip install qiskit-ibm-runtime)
  D-Wave: DWAVE_API_TOKEN=... QGRID_ALLOW_DWAVE=1   (pip install dwave-system)
Otherwise the platform is recorded as skipped with the reason. Hardware results are noisy and experimental.
Output: results/v2/hardware.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT)]
OUT = ROOT / "results" / "v2"
LABEL = "Hardware results are noisy and experimental; QAOA verified against exact search; no quantum advantage claimed."


def _ready(env_token: str, opt_in: str, module: str) -> tuple[bool, str]:
    if not os.environ.get(env_token):
        return False, f"no token ({env_token} not set)"
    if os.environ.get(opt_in) != "1":
        return False, f"not opted in ({opt_in}=1 not set)"
    try:
        __import__(module)
    except ImportError:
        return False, f"SDK missing ({module} not installed)"
    return True, "ready"


def _block_instances():
    p = OUT / "qaoa_blocks.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())["blocks"]


def _best_from_counts(counts: dict, n: int, J_all: np.ndarray) -> tuple[float, float]:
    """(best true-objective value among measured bitstrings, share of shots on an optimal assignment)."""
    best, hit, tot = np.inf, 0, 0
    jmin = J_all.min()
    for key, c in counts.items():
        bits = [int(ch) for ch in reversed(key.replace(" ", "")[-n:])]
        z = int(sum(b << i for i, b in enumerate(bits)))
        best = min(best, float(J_all[z]))
        hit += c * int(J_all[z] <= jmin + 1e-6 * max(1.0, abs(jmin)))
        tot += c
    return best, hit / max(tot, 1)


def ibm_and_aer(shots: int = 256, n_samples: int = 8) -> dict:
    from qgrid.qubo import QUBO
    from qgrid.solvers.qaoa import build_circuit, counts_aer, load_params
    inst = _block_instances()[:n_samples]
    g, b = load_params(2)
    ok, why = _ready("QGRID_IBM_TOKEN", "QGRID_ALLOW_IBM", "qiskit_ibm_runtime")
    rows = []
    backend = None
    if ok:
        from qiskit_ibm_runtime import QiskitRuntimeService
        svc = QiskitRuntimeService(channel="ibm_quantum_platform", token=os.environ["QGRID_IBM_TOKEN"])
        backend = svc.least_busy(operational=True, simulator=False)
    for k, r in enumerate(inst):
        M = np.array(r["qubo_M"]); n = len(M)
        q = QUBO(r["qubo_const"], M)
        c, h, J = q.to_ising()
        sig = float(np.sqrt((h ** 2).sum() + (J ** 2).sum())) or 1.0
        J_all = np.array(r["J_all"])
        aer = counts_aer(h / sig, J / sig, g, b, shots=shots, seed=1234 + k)
        best_a, popt_a = _best_from_counts(aer, n, J_all)
        row = {"instance": f"{r['zone']}@t0={r['t0']}", "block": r["block"], "n_qubits": n, "J_exact": float(J_all.min()),
               "aer_best_J": best_a, "aer_found_opt": bool(best_a <= J_all.min() + 1e-6 * max(1, abs(J_all.min()))),
               "aer_p_opt": popt_a, "ibm": "skipped: " + why}
        if ok:
            from qiskit import transpile
            from qiskit_ibm_runtime import SamplerV2
            qc = transpile(build_circuit(h / sig, J / sig, g, b, measure=True), backend, optimization_level=3)
            t0 = time.time()
            res = SamplerV2(mode=backend).run([qc], shots=shots).result()
            counts = res[0].data.meas.get_counts()
            best_h, popt_h = _best_from_counts(counts, n, J_all)
            row.update({"ibm": f"ran on {backend.name} (noisy)", "ibm_best_J": best_h, "ibm_p_opt": popt_h,
                        "ibm_found_opt": bool(best_h <= J_all.min() + 1e-6 * max(1, abs(J_all.min()))),
                        "ibm_depth": qc.depth(), "ibm_wall_s": time.time() - t0})
        rows.append(row)
    return {"platform": "IBM Quantum vs Qiskit Aer vs exact", "status": "ran" if ok else "skipped: " + why, "rows": rows}


def campus_qubo() -> dict:
    """Whole-campus 24 h commitment QUBO at the first decision slot of the headline day."""
    from qgrid import data as qdata
    from qgrid.model import load_campus
    from qgrid.solvers.anneal import anneal
    from qgrid.v2.admm_lp import run_admm
    from qgrid.v2.building import BState, BuildingLP, campus_max_kw, layout
    from qgrid.v2.horizon import YearData
    from qgrid.v2.master import Master
    from qgrid.v2.mpc import V2Config, _best_prices, _initial_y
    campus = load_campus()
    proc = qdata.load_processed()
    cfg = V2Config.from_cfg(campus.cfg)
    key = campus.cfg["headline_scenario"]
    sc = proc["meta"]["days"]["scenarios"][key]
    yd = YearData(campus, proc, "A", forecast=False)
    w = yd.window(yd.day_start(sc["date"]), 0, cfg.H, sc.get("outage"))
    tv, oc = campus.cfg["tariff_v2"], campus.cfg["outage_costs"]
    dem = {"voll_nc": oc["voll_noncritical_rs_kwh"], "voll_c": oc["voll_critical_rs_kwh"],
           "rate_day": qdata.demand_rate_month(campus.cfg, sc["date"]) / tv["days_per_month"], "pf": tv["power_factor"],
           "campus_max_kw": campus_max_kw(campus, w)}
    sts = [BState(e=b.battery.e0, level=b.tank.level0 if b.tank else 0.0, dg_on=False, dg_run=99) for b in campus.buildings]
    lays = [layout(campus, i, w, sts[i], 0, cfg.block, cfg.min_off) for i in range(campus.n)]
    blp = [BuildingLP(campus, i, w, sts[i], lays[i], dem, 0.0) for i in range(campus.n)]
    ys = [_initial_y(lays[i], w, i, campus, sts[i]) for i in range(campus.n)]
    res = run_admm(campus, blp, ys, cfg.rho, cfg.admm_iter, cfg.admm_tol)
    for i, b in enumerate(blp):
        f = np.concatenate([res.z[l][:cfg.H] for (l, _) in b.inc]) if b.inc else np.array([])
        b.recover(ys[i], f, b.get(res.x[i], "imp"))
    lam = _best_prices(campus, blp, res, ys, cfg.H)
    masters = [Master(l) for l in lays]
    for i, b in enumerate(blp):
        v, g, _ = b.lagrangian(ys[i], lam[i])
        masters[i].add_cut(v, g, ys[i])
    # block-diagonal campus QUBO over every free binary
    blocks, offs, n = [], [], 0
    for i, m in enumerate(masters):
        free = np.array([k for k in range(lays[i].ny) if k not in lays[i].fixed])
        q = m.block_qubo(ys[i], free)
        blocks.append((i, free, q)); offs.append(n); n += len(free)
    M = np.zeros((n, n)); const = 0.0
    for (i, free, q), o in zip(blocks, offs):
        M[o:o + len(free), o:o + len(free)] = q.M; const += q.const
    from qgrid.qubo import QUBO
    Q = QUBO(const, M)

    def J_total(xs):
        tot = 0.0
        for (i, free, q), o in zip(blocks, offs):
            y = ys[i].copy(); y[free] = xs[o:o + len(free)]
            tot += float(masters[i].J(y)[0])
        return tot

    exact = sum(m.exact()[1] for m in masters)
    t0 = time.time()
    sa = anneal(Q, n_chains=16, sweeps=200, seed=7)
    sa_best = min(J_total(x) for x in sa)
    out = {"n_binaries": n, "J_exact_milp": exact, "sa_best_J": sa_best, "sa_gap_pct": 100 * (sa_best - exact) / abs(exact),
           "sa_time_s": time.time() - t0, "note": "the master separates by building, so the campus QUBO is block-diagonal"}
    (OUT / "campus_qubo.json").write_text(json.dumps({"n": n, "const": const, "M_nonzero": [[int(a), int(b), float(M[a, b])]
                                                      for a, b in zip(*np.nonzero(M)) if a <= b]}))
    ok, why = _ready("DWAVE_API_TOKEN", "QGRID_ALLOW_DWAVE", "dwave.system")
    if ok:
        from dwave.system import LeapHybridSampler
        bqm = {(int(a), int(b)): float(M[a, b] + (M[b, a] if a != b else 0.0)) for a, b in zip(*np.nonzero(M)) if a <= b}
        t1 = time.time()
        ss = LeapHybridSampler(token=os.environ["DWAVE_API_TOKEN"]).sample_qubo(bqm)
        x = np.array([ss.first.sample[k] for k in range(n)], float)
        dj = J_total(x)
        out.update({"dwave": "ran on Leap hybrid solver (noisy, experimental)", "dwave_best_J": dj,
                    "dwave_gap_pct": 100 * (dj - exact) / abs(exact), "dwave_wall_s": time.time() - t1})
    else:
        out["dwave"] = "skipped: " + why
    return {"platform": "D-Wave Leap vs simulated annealing vs exact MILP masters", **out}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"label": LABEL, "ibm": ibm_and_aer(), "dwave": campus_qubo()}
    (OUT / "hardware.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({"ibm": rep["ibm"]["status"], "dwave": rep["dwave"]["dwave"],
                      "aer_found_opt": [r["aer_found_opt"] for r in rep["ibm"]["rows"]],
                      "sa_gap_pct": rep["dwave"]["sa_gap_pct"]}, indent=1))


if __name__ == "__main__":
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):   # spurious macOS Accelerate warnings
        main()
