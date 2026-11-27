"""Offline copies of the live demo moments (same functions as the live API in qgrid/api.py):

  python scripts/make_live.py   -> results/replay/live.json (+ frontend/public/replay/live.json)

  remove   every building taken offline at the worked-example decision of the headline day (current design)
  qaoa     the worked example's QUBO block re-measured with three more seeds (QAOA, Aer simulator) vs exact search
  rules    the three preset faculty rules (one binding, one cheap, one rejected) priced on the headline day
Run after make replay (it reads the replay's current-design run)."""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT)]


def main():
    from qgrid import data as qd
    from qgrid.live import PRESETS, qaoa_block, remove_building, rule_effect
    from qgrid.model import load_campus, make_day
    from qgrid.v2.horizon import YearData
    campus = load_campus()
    proc = qd.load_processed()
    yd = YearData(campus, proc, "A", forecast=False)
    key = campus.cfg["headline_scenario"]
    sc = proc["meta"]["days"]["scenarios"][key]
    day = make_day(campus, proc, sc["date"], "A", key, sc.get("outage"))
    t0 = time.time()
    out = {"scenario": key, "generated_by": "scripts/make_live.py", "remove": {}, "qaoa": [], "rules": []}
    base = None
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for z in campus.names:
            r = remove_building(campus, yd, key, z, baseline=base)
            base = r["before"]
            out["remove"][z] = r
            print(f"remove {z:16s} others' critical shortfall {r['others']['crit_shed_after_kwh']:.3f} kWh, {r['compute_s']:.1f} s", flush=True)
    for seed in (2025, 7, 99):
        q = qaoa_block(key, seed=seed)
        out["qaoa"].append(q)
        print(f"qaoa seed {seed}: matches exact {q['qaoa']['matches_exact']}, {q['qaoa']['time_s']:.2f} s", flush=True)
    for t in PRESETS:
        r = rule_effect(campus, day, t)
        out["rules"].append(r)
        print(f"rule '{t}': ok={[x['ok'] for x in r['results']]} {r['time_s']:.1f} s", flush=True)
    rp = ROOT / "results" / "replay" / "live.json"
    rp.write_text(json.dumps(out, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    shutil.copy(rp, ROOT / "frontend" / "public" / "replay" / "live.json")
    print(f"wrote {rp} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
