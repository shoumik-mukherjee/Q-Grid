"""Energy-flow Sankey data for the explainer: Q-Grid's and the status quo's runs on each named day of the replay.

  python scripts/make_sankey.py   -> results/replay/sankey_<scenario>.json (+ frontend/public/replay)
Reads the replay's run blocks (results/replay/scenario_<key>.json) and settles them with qgrid/operate/flows.py."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT)]
LABEL = {"outage_fest": "Outage during the fest", "fest": "Fest", "teaching": "Teaching", "exam_week": "Exam week", "vacation": "Vacation", "monsoon": "Monsoon"}


def main():
    from qgrid import data as qd
    from qgrid.model import Schedule, load_campus, make_day
    from qgrid.operate.flows import energy_flows
    campus = load_campus(); proc = qd.load_processed()
    for f in sorted((ROOT / "results" / "replay").glob("scenario_*.json")):
        sc = json.loads(f.read_text(encoding="utf-8")); key = sc["scenario"]["key"]
        day = make_day(campus, proc, sc["scenario"]["date"], "A", key, sc["scenario"].get("outage"))
        out = {"day": sc["scenario"]["date"], "label": LABEL.get(key, key), "scenario": key}
        for name in ("qgrid", "status_quo"):
            r = sc["runs"].get(name)
            if not r:
                out[name] = None; continue
            s = Schedule.empty(campus)
            s.p_bat[:] = np.array(r["p_bat"]); s.dg_kw[:] = np.array(r["dg_kw"]); s.pump_on[:] = np.array(r["pump_on"]); s.flows[:] = np.array(r["flows"])
            s.flex_start = [dict(x) for x in r["flex_start"]]
            out[name] = energy_flows(campus, day, s)
        p = ROOT / "results" / "replay" / f"sankey_{key}.json"
        p.write_text(json.dumps(out), encoding="utf-8"); shutil.copy(p, ROOT / "frontend" / "public" / "replay" / p.name)
        print(f"{key}: solar {out['qgrid']['sources_kwh']['Solar']:,.0f} kWh, grid {out['qgrid']['sources_kwh']['Grid']:,.0f} kWh")


if __name__ == "__main__":
    main()
