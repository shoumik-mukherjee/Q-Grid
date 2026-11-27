"""make verify: re-run the headline subset and print PASS/FAIL per number against results/manifest.json.

  python scripts/verify.py          full: recompute B0, B4, B5, OURS on the headline scenario (needs data/processed)
  python scripts/verify.py --ci     fast (CI, no real data): file hashes of committed inputs + CSV/headline consistency
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT)]


def main(ci: bool = False) -> int:
    man = json.loads((ROOT / "results" / "manifest.json").read_text(encoding="utf-8"))
    ok = True
    if ci:
        for f in ("config.yaml", "configs/profiles.yaml", "data/processed/campus.json", "results/benchmark.csv", "results/headline.json"):
            p = ROOT / f
            if not p.exists() or man["files"].get(f) is None:
                continue
            good = hashlib.sha256(p.read_bytes()).hexdigest() == man["files"][f]
            ok &= good
            print(f"{'PASS' if good else 'FAIL'}  sha256 {f}")
        b = pd.read_csv(ROOT / "results" / "benchmark.csv")
        h = json.loads((ROOT / "results" / "headline.json").read_text(encoding="utf-8"))
        for c in man["headline_checks"]:
            r = b[(b.method == c["method"]) & (b.scenario == c["scenario"]) & (b.price_scenario == c["price_scenario"])].iloc[0]
            good = abs(float(r[c["metric"]]) - c["value"]) <= max(c["abs_tol"], c["rel_tol"] * abs(c["value"]))
            ok &= good
            print(f"{'PASS' if good else 'FAIL'}  csv {c['method']} {c['metric']} = {r[c['metric']]}")
        good = abs(h["after"]["cost_rs"] - float(b[(b.method == "OURS") & (b.scenario == h["scenario"]) & (b.price_scenario == "A")].cost_rs.iloc[0])) < 1e-6
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  headline.json matches benchmark.csv")
        return 0 if ok else 1

    from baselines import b0_status_quo, b4_central_milp
    from qgrid import data as qdata
    from qgrid.admm import run_day
    from qgrid.model import load_campus, make_day, settle
    from qgrid.solvers import reset_stats
    campus = load_campus()
    proc = qdata.load_processed()
    c0 = man["headline_checks"][0]
    sc = proc["meta"]["days"]["scenarios"][c0["scenario"]]
    day = make_day(campus, proc, sc["date"], "A", c0["scenario"], sc.get("outage"))
    runs = {}
    for m in sorted({c["method"] for c in man["headline_checks"]}):
        reset_stats()
        if m == "B0":
            s = b0_status_quo.solve(campus, day)
        elif m == "B4":
            s = b4_central_milp.solve(campus, day)
        else:
            s, _ = run_day(campus, day, method="exact" if m == "B5" else "qaoa")
        runs[m] = settle(campus, day, s)
    for c in man["headline_checks"]:
        v = runs[c["method"]][c["metric"]]
        good = abs(v - c["value"]) <= max(c["abs_tol"], c["rel_tol"] * abs(c["value"]))
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  {c['method']:5s} {c['metric']:22s} recomputed {v:,.4f}  manifest {c['value']:,.4f}")
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main("--ci" in sys.argv))
