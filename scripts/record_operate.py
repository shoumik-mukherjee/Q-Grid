"""Recorded Operate session for the offline / static demo: one full day through the same engine as live mode.

  python scripts/record_operate.py [--day=2023-06-15]   -> results/replay/operate_session.json (+ frontend/public/replay)

Historical day (actual solar from NASA POWER, the plan uses the solar forecast; BDG2-derived loads; tariff as labelled by the data status), with
scripted operator actions so every Operate screen has something to show: a rule at 07:00, a manual override at 11:00,
a campus network failure at 13:00 for 45 min and a BESCOM outage at 18:00 for 3 h. Frames every 5 simulated minutes."""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT)]
SCRIPT = {7 * 60: ("rule", "no DG unless outage"), 11 * 60: ("override", ("server_room", "battery", "hold", 30)),
          13 * 60: ("event", ("comms_loss", 45)), 18 * 60: ("event", ("outage", 180))}


def main(day: str):
    from qgrid import data as qd
    from qgrid.model import load_campus
    from qgrid.operate.session import Session
    Session._threaded = False
    s = Session(load_campus(), qd.load_processed(), mqtt=False)
    s.reset("historical", day)
    s.sim_s = 0.0; s.minute_done = -1; s.warm = {}
    s._build_day(); s.events = []
    s.plan_now(reason="session start")
    frames, plans, admms = [], [], []
    last_plan = None
    t0 = time.time()
    for minute in range(0, 24 * 60, 5):
        if minute in SCRIPT:
            k, a = SCRIPT[minute]
            if k == "rule":
                s.add_rule(a, who="estate office (scripted)")
            elif k == "override":
                s.override(*a, who="operator (scripted)")
            else:
                s.inject(*a, who="operator (scripted)")
            s.plan_now(reason={"rule": "rule added", "override": "manual override", "event": a[0].replace("_", " ") if k == "event" else ""}[k])
        s.advance(5 * 60 if minute < 24 * 60 - 5 else 5 * 60 - 1)
        snap = s.snapshot()
        if s.plan is not last_plan:
            last_plan = s.plan
            plans.append({"made_at_s": s.plan["made_at"], **snap["plan"]}); admms.append(snap["admm"])
        frames.append({"t": snap["clock"]["sim_s"], "time": snap["clock"]["time"], "zones": snap["zones"], "plan": len(plans) - 1,
                       "grid_up_now": snap["grid_up_now"], "comms_ok": snap["comms_ok"], "overrides": snap["overrides"], "rules": snap["rules"]})
        if minute % 120 == 0:
            print(f"{snap['clock']['time']} plans {len(plans)} events {len(s.events)} ({time.time() - t0:.0f} s)", flush=True)
    rep = s.reports[-1] if s.reports else s.report(final=True)
    final = s.snapshot()
    out = {"kind": "recorded", "date": day, "labels": s.data_label, "recorded_at": time.strftime("%Y-%m-%d %H:%M"),
           "generated_by": "scripts/record_operate.py", "frames": frames, "plans": plans, "admm": admms,
           "series": final["series"], "events": s.events, "audit": s.audit, "report": rep, "lines": final["lines"]}
    js = json.dumps(out, separators=(",", ":"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    p = ROOT / "results" / "replay" / "operate_session.json"
    p.write_text(js, encoding="utf-8")
    shutil.copy(p, ROOT / "frontend" / "public" / "replay" / "operate_session.json")
    print(f"wrote {p} ({len(js) / 1e6:.1f} MB), saving vs status quo Rs {rep['saving_rs']:,.0f}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(next((a.split("=")[1] for a in sys.argv if a.startswith("--day=")), "2023-06-15"))
