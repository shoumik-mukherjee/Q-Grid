"""Pre-train QAOA (gamma, beta) for p = 1..3 on instances from TRAINING days (disjoint from the test
and evaluation days); saved to results/qaoa_params.json and reused for every solve (parameter transfer)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT)]
from qgrid import data as qdata  # noqa: E402
from qgrid import qubo as Q  # noqa: E402
from qgrid.instances import sample_instances, training_days  # noqa: E402
from qgrid.solvers.qaoa import PARAMS_FILE, ising_energies, normalised_ising, optimise_params  # noqa: E402


def main(n_inst: int = 16, seed: int = 7):
    proc = qdata.load_processed()
    days = training_days(proc["meta"])
    insts = sample_instances(proc, days, n_inst, seed=seed)
    cases = []
    for _, lp in insts:
        total, *_ = Q.auto_tune(lp)
        h, J, _ = normalised_ising(total)
        cases.append(ising_energies(h, J))
    out = {"training_days": days, "n_instances": n_inst, "seed": seed, "objective": "mean <H_C> (normalised Ising)",
           "params": {}}
    prev = None
    for p in (1, 2, 3):
        t0 = time.time()
        x0 = None
        if prev is not None:  # interpolate previous optimum as warm start
            g, b = prev
            k = np.linspace(0, 1, p)
            x0 = (np.interp(k, np.linspace(0, 1, p - 1), g), np.interp(k, np.linspace(0, 1, p - 1), b))
        g, b, f = optimise_params(cases, p, maxiter=150, x0=x0)
        prev = (g, b)
        out["params"][str(p)] = {"gammas": g.round(6).tolist(), "betas": b.round(6).tolist(),
                                 "mean_energy": round(f, 6), "train_time_s": round(time.time() - t0, 1)}
        print(f"p={p}: <H>={f:.4f}  gammas={np.round(g, 3)} betas={np.round(b, 3)}  ({time.time() - t0:.1f}s)")
    PARAMS_FILE.parent.mkdir(exist_ok=True)
    PARAMS_FILE.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("saved", PARAMS_FILE.relative_to(ROOT))


if __name__ == "__main__":
    main()
