"""Status-quo 'problem' analysis on the RVCE model -> results/problem_day.json + results/plots/problem_day.png.
Uses B0 (no optimisation, no sharing) on the teaching scenario and the headline outage scenario, price A."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT)]
from baselines import b0_status_quo  # noqa: E402
from qgrid import data as qdata  # noqa: E402
from qgrid.model import DT, flex_load, load_campus, make_day, net_power, pump_load, settle  # noqa: E402


def main():
    campus = load_campus()
    proc = qdata.load_processed()
    scen = proc["meta"]["days"]["scenarios"]
    out = {"data_status": proc["meta"]["data_status"], "scenarios": {}}
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    for ax, key in zip(axs, ("teaching", campus.cfg["headline_scenario"])):
        sc = scen[key]
        day = make_day(campus, proc, sc["date"], "A", key, sc.get("outage"))
        s = b0_status_quo.solve(campus, day)
        m = settle(campus, day, s)
        net = net_power(campus, day, s)
        g = day.grid
        imp = np.where(g, np.maximum(net, 0).sum(0), 0)
        exp = np.where(g, np.maximum(-net, 0).sum(0), 0)
        h = np.arange(96) * DT
        noon, eve = (h >= 10) & (h < 15), (h >= 18) & (h < 22)
        out["scenarios"][key] = {"date": sc["date"], "note": sc["note"], "price_status": day.status["price_scenario"],
                                 **{k: round(v, 2) for k, v in m.items() if isinstance(v, float)},
                                 "noon_export_kwh": round(float(exp[noon].sum() * DT), 1),
                                 "evening_import_kwh": round(float(imp[eve].sum() * DT), 1)}
        ax.fill_between(h, 0, day.pv.sum(0), color="#f2b705", alpha=.35, label="Campus PV (kW)")
        ax.plot(h, (day.load + flex_load(campus, s) + pump_load(campus, s)).sum(0), color="#1f3b73", lw=2, label="Campus load (kW)")
        ax.fill_between(h, 0, -exp, color="#2a9d8f", alpha=.7, label="Export (kW, negative)")
        ax.fill_between(h, 0, imp, color="#e63946", alpha=.25, label="Grid import (kW)")
        if not g.all():
            a = np.where(~g)[0]
            ax.axvspan(a[0] * DT, (a[-1] + 1) * DT, color="#444", alpha=.4, label="BESCOM outage")
        ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 3)); ax.set_xlabel("hour"); ax.set_ylabel("kW")
        tag = "" if day.status["price_scenario"] == "REAL" else " [STAND-IN prices, PENDING]"
        ax.set_title(f"Status quo — {key} {sc['date']}{tag}\nRs {m['cost_rs']:,.0f} · DG {m['dg_hours']:.1f} h · "
                     f"critical downtime {m['critical_downtime_h']:.2f} h", fontsize=9, color="#b00020" if tag else "k")
        ax.legend(fontsize=7, loc="upper left")
    fig.text(0.01, 0.01, "RVCE layout © OpenStreetMap contributors; PV from NASA POWER; loads PENDING RVCE DATA (area x EPI, "
             "timetable / Ausgrid shapes). Capacities are ASSUMPTIONS (config.yaml).", fontsize=7, alpha=.75)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(ROOT / "results" / "plots" / "problem_day.png", dpi=130)
    (ROOT / "results" / "problem_day.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["scenarios"], indent=1))


if __name__ == "__main__":
    main()
