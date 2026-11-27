"""results/manifest.json: git commit, SHA-256 of inputs/outputs, seeds, package versions, runtime and the
headline numbers that `make verify` re-computes (with tolerances)."""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILES = ["config.yaml", "configs/profiles.yaml", "data/raw/nasa_power/bengaluru_2023_hourly.json",
         "data/raw/ausgrid/Ausgrid_solar_home_data.zip", "data/raw/osm_rvce.json", "data/raw/lab_schedule.csv",
         "data/processed/campus.json", "results/qaoa_params.json", "results/benchmark.csv", "results/headline.json"]
PKGS = ["numpy", "scipy", "pandas", "qiskit", "qiskit-aer", "pulp", "highspy", "pymoo", "fastapi", "pydantic"]


def sha(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def main():
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    b = pd.read_csv(ROOT / "results" / "benchmark.csv")
    hs = cfg["headline_scenario"]
    checks = []
    for m in ("B0", "B4", "B5", "OURS"):
        r = b[(b.method == m) & (b.scenario == hs) & (b.price_scenario == "A")].iloc[0]
        tol = 0.002 if m == "B4" else 1e-6     # B4: MILP solved to a 0.1 % gap
        for k in ("cost_rs", "critical_downtime_h", "dg_hours"):
            checks.append({"method": m, "scenario": hs, "price_scenario": "A", "metric": k, "value": float(r[k]), "rel_tol": tol,
                           "abs_tol": 1e-6 if k != "cost_rs" else 0.01})
    run_info = ROOT / "results" / "run_info.json"
    man = {"git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
           "python": sys.version.split()[0], "platform": platform.platform(),
           "packages": {p: md.version(p) for p in PKGS if _has(p)},
           "seeds": {"qaoa": cfg["qaoa"]["seed"], "ga_pso": 1, "c_qiea": 7, "kmeans": 0},
           "files": {f: sha(ROOT / f) for f in FILES},
           "runtime": json.loads(run_info.read_text(encoding="utf-8")) if run_info.exists() else None,
           "headline_checks": checks}
    (ROOT / "results" / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(f"manifest: {len(checks)} headline checks, commit {man['git_commit'][:10]}")


def _has(p):
    try:
        md.version(p)
        return True
    except md.PackageNotFoundError:
        return False


if __name__ == "__main__":
    main()
