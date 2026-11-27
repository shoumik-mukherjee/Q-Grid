"""Statistics, documents and the replay file for benchmark v3, all from results/bench_v3/{runs,bounds}.csv.

  python scripts/write_bench_v3.py  -> results/bench_v3/{daily,summary,wilcoxon}.csv, docs/benchmark_v3.md,
                                       docs/scorecard.md, results/replay/bench_v3.json (+ frontend/public/replay)
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "bench_v3"
sys.path[:0] = [str(ROOT / "backend"), str(ROOT)]
from qgrid import provenance as _prov  # noqa: E402

NAMES = {   # display names (never version numbers)
    "status_quo": ("Status quo (no coordination)", "Today"),
    "rule_based": ("Rule-based battery controller", "Smart control"),
    "ga": ("Genetic algorithm", "Research methods"),
    "pso": ("Particle swarm optimisation", "Research methods"),
    "per_building_mpc": ("Per-building MPC (MILP)", "Smart control"),
    "central_mpc": ("Central MPC with forecasts", "Central control"),
    "central_mpc_comms_fail": ("Central MPC when communication fails", "Central control"),
    "qgrid": ("Q-Grid", "Ours"),
    "qgrid_comms_fail": ("Q-Grid when communication fails", "Ours"),
}
REF = ("optimum", "Best possible plan (central MILP, perfect information)")
DESIGN = {
    "design_prototype_1h_binary": ("1-hour horizon, discrete kW, binary coordination", "earlier design, realistic forecasts"),
    "design_earlier_prototype": ("Earlier prototype: 3-slot QUBO per building, binary coordination with QAOA/annealing tiers", "first working version, realistic forecasts; battery wear not priced"),
    "design_benders_exact": ("24 h hybrid Benders, exact master", "Benders between an hourly on/off master and the LP; realistic forecasts; battery wear not priced"),
    "design_benders_qaoa": ("24 h hybrid Benders, QAOA master (verified, simulator)", "same with a QAOA block master; realistic forecasts; battery wear not priced"),
    "design_benders_qiea": ("24 h hybrid Benders, QIEA master (quantum-inspired, classical)", "same with a QIEA master; realistic forecasts; battery wear not priced"),
}
METRICS = [  # key, label, group, better, unit, digits
    ("bill_rs", "Bill", "Cost", "lower", "Rs/day", 0), ("saving_pct", "Saving vs today", "Cost", "higher", "%", 1),
    ("npv10_rs_lakh", "10-yr NPV of saving", "Cost", "higher", "Rs lakh", 1),
    ("critical_downtime_h", "Critical downtime", "Reliability", "lower", "h/day", 2),
    ("critical_downtime_1pct_h", "…counting >1 % shortfalls", "Reliability", "lower", "h/day", 2),
    ("unserved_kwh", "Unserved energy", "Reliability", "lower", "kWh/day", 1),
    ("peak_kva", "Peak demand", "Grid", "lower", "kVA", 0), ("import_kwh", "Grid import", "Grid", "lower", "kWh/day", 0),
    ("co2_kg", "CO₂", "Environment", "lower", "kg/day", 0), ("dg_litres", "Diesel", "Environment", "lower", "L/day", 1),
    ("spill_kwh", "Solar curtailed", "Environment", "lower", "kWh/day", 1),
    ("battery_cycles", "Battery cycles", "Battery", "lower", "/day", 2),
    ("battery_degradation_rs", "Degradation cost", "Battery", "lower", "Rs/day", 0),
    ("gap_pct", "Gap to best plan", "Optimality", "lower", "%", 2),
    ("gap_novoll_pct", "…money only (no VoLL)", "Optimality", "lower", "%", 2),
    ("certified_gap_pct", "Certified gap", "Optimality", "lower", "%", 2),
    ("runtime_per_decision_s", "Time per decision", "Operation", "lower", "s", 2),
    ("messages_per_decision", "Messages per decision", "Operation", "lower", "", 0),
]


def daily() -> pd.DataFrame:
    r = pd.read_csv(OUT / "runs.csv")
    b = pd.read_csv(OUT / "bounds.csv")
    key = lambda d: d.dataset + "|" + d.day   # noqa: E731
    bm = b.assign(k=key(b)).set_index("k")
    # Reference for "gap to the best plan" = the cheapest feasible plan known for the day: the central MILP with perfect
    # information (its run in runs.csv, or the bounds solve if that found a cheaper plan), or any controller's realised
    # plan if it is cheaper within solver tolerance. So no gap is negative, and the reference row's own gap is >= 0.
    feas = r[r.feasible.astype(bool)]
    best_run = feas.assign(k=key(feas)).groupby("k").objective_rs.min()
    k_r = key(r)
    r["opt"] = np.fmin(k_r.map(best_run), k_r.map(bm.opt_central))
    # a valid lower bound can never exceed a feasible plan; where it does (solver tolerance) it is capped and counted
    lb_raw = k_r.map(bm.LB)
    r["LB_capped"] = lb_raw > r.opt + 1e-6 * r.opt.abs()
    r["LB"] = np.fmin(lb_raw, r.opt)
    r["bill_rs"] = r.cost_incl_demand_rs
    # every figure of a method-day (gap, downtime, unserved energy, bill...) comes from that one row of runs.csv
    r["gap_pct"] = 100 * (r.objective_rs - r.opt) / r.opt
    r["certified_gap_pct"] = 100 * (r.objective_rs - r.LB) / r.LB
    money = r.objective_rs - r.voll_rs
    opt_money = feas.assign(k=key(feas), mm=feas.objective_rs - feas.voll_rs).groupby("k").mm.min()
    r["gap_novoll_pct"] = 100 * (money - k_r.map(opt_money)) / k_r.map(opt_money)
    assert (r.gap_pct >= -1e-9).all() and (r.certified_gap_pct >= r.gap_pct - 1e-9).all(), "inconsistent gap / bound"
    b0 = r[r.method == "status_quo"].assign(k=lambda d: key(d)).set_index("k").bill_rs
    r["saving_rs"] = key(r).map(b0) - r.bill_rs
    r["saving_pct"] = 100 * r.saving_rs / key(r).map(b0)
    r["messages_per_decision"] = r.messages / np.where(r.method.isin(["status_quo", "rule_based", "optimum", "ga", "pso"]), 1, 24)
    r.to_csv(OUT / "daily.csv", index=False)
    return r


STALE_PREFIX = "[not regenerated] "
STALE_NOTE = "previous results, not regenerated with the new campus data"


def carry_previous(r: pd.DataFrame) -> pd.DataFrame:
    """Datasets absent from runs.csv keep their previous daily rows (results/bench_v3/daily_previous.csv), marked
    regenerated=False. They are labelled as previous results everywhere they appear, never as current."""
    r = r.assign(regenerated=True)
    prev = OUT / "daily_previous.csv"
    if prev.exists():
        old = pd.read_csv(prev)
        old = old[~old.dataset.isin(r.dataset.unique())].assign(regenerated=False)
        if len(old):
            r = pd.concat([r, old], ignore_index=True)
    r.to_csv(OUT / "daily.csv", index=False)
    return r


def summary(r: pd.DataFrame, rate: float) -> pd.DataFrame:
    rows = []
    for (ds, m), g in r.groupby(["dataset", "method"]):
        row = {"dataset": ds, "method": m, "n_days": len(g), "feasible_all": bool(g.feasible.all()),
               "regenerated": bool(g.regenerated.all()) if "regenerated" in g else True}
        for k, *_ in METRICS:
            if k == "npv10_rs_lakh":
                continue
            x = g[k].astype(float).dropna().to_numpy()
            if len(x) == 0:
                continue
            mu, sd = float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0
            h = float(stats.t.ppf(0.975, len(x) - 1) * sd / np.sqrt(len(x))) if len(x) > 1 else 0.0
            row.update({f"{k}__mean": mu, f"{k}__std": sd, f"{k}__ci_lo": mu - h, f"{k}__ci_hi": mu + h})
        ann = row.get("saving_rs__mean", float(g.saving_rs.mean())) * 365
        npv = sum(ann / (1 + rate) ** y for y in range(1, 11)) / 1e5
        row.update({"npv10_rs_lakh__mean": npv, "npv10_rs_lakh__std": np.nan, "npv10_rs_lakh__ci_lo": np.nan, "npv10_rs_lakh__ci_hi": np.nan})
        rows.append(row)
    for k in ("saving_rs",):
        pass
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "summary.csv", index=False)
    return s


def wilcoxon(r: pd.DataFrame) -> pd.DataFrame:
    out = []
    for ds, g in r.groupby("dataset"):
        q = g[g.method == "qgrid"].set_index("day")
        for m, gm in g.groupby("method"):
            if m == "qgrid":
                continue
            gm = gm.set_index("day")
            days = q.index.intersection(gm.index)
            for k, *_ in METRICS:
                if k not in q or k not in gm:
                    continue
                a, bb = q.loc[days, k].astype(float), gm.loc[days, k].astype(float)
                ok = a.notna() & bb.notna()
                d = (a - bb)[ok]
                if len(d) < 6 or np.allclose(d, 0):
                    p = np.nan
                else:
                    try:
                        p = float(stats.wilcoxon(a[ok], bb[ok], zero_method="wilcox").pvalue)
                    except ValueError:
                        p = np.nan
                out.append({"dataset": ds, "method": m, "metric": k, "n": int(ok.sum()), "median_diff_qgrid_minus_method": float(np.median(d)) if len(d) else np.nan, "p": p})
    w = pd.DataFrame(out)
    w.to_csv(OUT / "wilcoxon.csv", index=False)
    return w


def _stat(s, ds, m, k):
    row = s[(s.dataset == ds) & (s.method == m)]
    if not len(row) or f"{k}__mean" not in row:
        return None
    row = row.iloc[0]
    f = lambda c: None if pd.isna(row.get(c)) else float(row.get(c))   # noqa: E731
    return {"mean": f(f"{k}__mean"), "std": f(f"{k}__std"), "ci_lo": f(f"{k}__ci_lo"), "ci_hi": f(f"{k}__ci_hi"), "n": int(row.n_days)}


def comms_series(ds: str, day: str) -> dict:
    """Cumulative objective (energy + diesel + value of lost load) through the day for the four comms runs."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_bench_v3 as rb
    from qgrid.model import DT, flex_load, net_power, pump_load
    series = {}
    for m in ("central_mpc", "central_mpc_comms_fail", "qgrid", "qgrid_comms_fail"):
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            r = rb.run_one(ds, day, m, return_sched=True)
        campus, dd, sch = r["campus"], r["dd"], r["sched"]
        net = net_power(campus, dd, sch)
        g = dd.grid[None, :]
        imp = np.where(g, np.maximum(net, 0), 0).sum(0); exp = np.where(g, np.maximum(-net, 0), 0).sum(0)
        demand = dd.load + flex_load(campus, sch) + pump_load(campus, sch)
        deficit = np.where(g, 0.0, np.maximum(net, 0))
        nc = np.minimum(deficit, np.maximum(demand - dd.crit, 0))
        voll = ((dd.voll_nc * nc + dd.voll_c * (deficit - nc)).sum(0)) * DT
        fuel = np.zeros(96)
        for i, b in enumerate(campus.buildings):
            if b.dg:
                on = sch.dg_kw[i] > 1e-9
                fuel += b.dg.litres(on, sch.dg_kw[i]) * b.dg.price
        series[m] = np.round(np.cumsum((imp * dd.price_imp - exp * dd.price_exp) * DT + fuel + voll), 1).tolist()
    w = rb.comms_window(ds, day)
    return {"dataset": ds, "day": day, "window": [int(w[0]), int(w[1])],
            "t": [f"{k // 4:02d}:{k % 4 * 15:02d}" for k in range(96)], "series": series,
            "totals": {k: v[-1] for k, v in series.items()}}


def headline(s, w, labels) -> dict:
    nocoord = ["status_quo", "rule_based", "ga", "pso", "per_building_mpc", "qgrid"]
    best_on, details = [], []
    for ds in labels:
        g = {m: (_stat(s, ds, m, "gap_pct") or {}).get("mean") for m in nocoord}
        g = {m: v for m, v in g.items() if v is not None}
        if not g:
            continue
        others = {m: v for m, v in g.items() if m != "qgrid"}
        alt = min(others, key=others.get)
        q = g.get("qgrid")
        p = w[(w.dataset == ds) & (w.method == alt) & (w.metric == "gap_pct")].p
        p = float(p.iloc[0]) if len(p) and pd.notna(p.iloc[0]) else None
        if q is not None and q <= others[alt]:
            best_on.append(ds)
        details.append((ds, q, alt, others[alt], p))
    n = len(details)
    rv = next((d for d in details if d[0] == "rvce"), details[0] if details else None)
    if rv is None:
        return {"text": "Benchmark results are not available yet.", "detail": ""}
    _, q, alt, av, p = rv
    sig = "" if p is None else (f" (paired Wilcoxon p = {p:.3f})" if p >= 0.001 else " (paired Wilcoxon p < 0.001)")
    first = (f"Q-Grid lands {q:.1f}% from the best possible plan; the closest alternative without a central controller, "
             f"{NAMES[alt][0]}, lands {av:.1f}%{sig}.")
    if len(best_on) == n:
        second = f"Among methods that work without a central controller, Q-Grid is closest to the best possible plan on all {n} campuses."
    elif best_on:
        miss = [labels[d[0]] for d in details if d[0] not in best_on]
        second = (f"Among methods that work without a central controller, Q-Grid is closest to the best possible plan on "
                  f"{len(best_on)} of {n} campuses; elsewhere ({'; '.join(miss)}) another method is closer.")
    else:
        second = "Q-Grid is not the closest to the best possible plan among methods without a central controller on any campus."
    return {"text": first, "detail": second, "rvce": {"qgrid_gap": q, "alt": NAMES[alt][0], "alt_gap": av, "p": p},
            "best_on": best_on, "n": n}


def story_headlines(s, labels, cfg, ds: str = "rvce") -> dict:
    """Step 1 (the problem) and Step 7 (the result) sentences, generated from the summary on the case-study dataset.
    Wording follows the numbers (e.g. 'cuts' vs 'raises'); figures that depend on the stand-in tariff are flagged."""
    g = lambda m, k: (_stat(s, ds, m, k) or {}).get("mean")   # noqa: E731
    n = int(s[(s.dataset == ds) & (s.method == "qgrid")].n_days.iloc[0]) if len(s[(s.dataset == ds) & (s.method == "qgrid")]) else 0
    pend = cfg["tariff_v2"]["status"] != "REAL"
    sq_dt, sq_dt1, sq_bill = g("status_quo", "critical_downtime_h"), g("status_quo", "critical_downtime_1pct_h"), g("status_quo", "bill_rs")
    opt_sav = g("optimum", "saving_pct")
    q_dt, q_dt1, q_sav, q_gap = g("qgrid", "critical_downtime_h"), g("qgrid", "critical_downtime_1pct_h"), g("qgrid", "saving_pct"), g("qgrid", "gap_pct")
    c_gap, cf_gap = g("central_mpc", "gap_pct"), g("central_mpc_comms_fail", "gap_pct")
    qf_gap = g("qgrid_comms_fail", "gap_pct")
    if None in (sq_dt, sq_bill, q_dt, q_sav, q_gap):
        return {}
    hm = lambda h: f"{h * 60:.0f} min" if h < 1 else f"{h:.1f} h"   # noqa: E731
    step1 = (f"Today every building runs on its own: over {n} modelled days the campus loses power to critical loads for "
             f"{hm(sq_dt)} a day on average and pays Rs {sq_bill:,.0f} a day"
             + (f"; the best possible plan would pay {opt_sav:.1f}% less." if opt_sav is not None and opt_sav > 0 else "."))
    dt_verb = "cuts" if q_dt < sq_dt - 1e-9 else ("keeps" if abs(q_dt - sq_dt) <= 1e-9 else "raises")
    dt_txt = (f"{dt_verb} critical downtime from {hm(sq_dt)} to {hm(q_dt)} a day" if dt_verb != "keeps" else f"keeps critical downtime at {hm(q_dt)} a day")
    if q_dt1 is not None and q_dt1 < q_dt - 1e-9:
        dt_txt += f" ({hm(q_dt1)} counting only shortfalls above 1 %)"
    sav_txt = f"{'saves' if q_sav >= 0 else 'adds'} {abs(q_sav):.1f}% {'on' if q_sav >= 0 else 'to'} the bill versus today"
    step7 = f"Q-Grid {dt_txt}, {sav_txt}, and lands {q_gap:.1f}% from the best possible plan."
    comms = ""
    if None not in (c_gap, cf_gap, qf_gap):
        pc, pq = cf_gap - c_gap, qf_gap - q_gap
        comms = (f"When the campus network fails for 3 hours, central control loses {pc:.1f} points of gap and Q-Grid "
                 f"{pq:.1f}, because each building keeps optimising on its own.")
    return {"dataset": ds, "n_days": n, "tariff_pending": pend,
            "step1": {"text": step1, "tariff_dependent": True},
            "step7": {"text": step7, "comms": comms, "tariff_dependent": True},
            "numbers": {"status_quo_downtime_h": sq_dt, "status_quo_downtime_1pct_h": sq_dt1, "status_quo_bill_rs": sq_bill,
                        "optimum_saving_pct": opt_sav, "qgrid_downtime_h": q_dt, "qgrid_downtime_1pct_h": q_dt1,
                        "qgrid_saving_pct": q_sav, "qgrid_gap_pct": q_gap, "central_gap_pct": c_gap,
                        "central_comms_fail_gap_pct": cf_gap, "qgrid_comms_fail_gap_pct": qf_gap}}


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    from qgrid.datasets import SPECS
    r = carry_previous(daily())
    s = summary(r, float(cfg["economics"].get("discount_rate", 0.08)))
    w = wilcoxon(r)
    stale = sorted(set(r.dataset[~r.regenerated.astype(bool)]))
    labels = {k: (STALE_PREFIX + SPECS[k].label + f" — {STALE_NOTE}" if k in stale else SPECS[k].label) for k in r.dataset.unique()}
    order = [k for k in SPECS if k in labels]
    table = {ds: {m: {k: _stat(s, ds, m, k) for k, *_ in METRICS} for m in list(NAMES) + [REF[0]]} for ds in order}
    wil = {ds: {m: {k: (None if pd.isna(p) else float(p)) for k, p in zip(g.metric, g.p)} for m, g in gd.groupby("method")}
           for ds, gd in w.groupby("dataset")}
    # comms day: the RVCE day with the largest penalty for central MPC under the failure
    comms = None
    cf = r[(r.dataset == "rvce") & r.method.isin(["central_mpc", "central_mpc_comms_fail"])].pivot_table(index="day", columns="method", values="objective_rs")
    if len(cf):
        day = (cf["central_mpc_comms_fail"] - cf["central_mpc"]).idxmax()
        comms = comms_series("rvce", day)
    ab = ROOT / "results" / "ablation" / "ablation_summary.csv"
    design = []
    if ab.exists():
        a = pd.read_csv(ab, dtype={"combo": str})
        desc = lambda c: ", ".join([("24 h horizon" if c[0] == "1" else "1 h horizon"), ("continuous kW" if c[1] == "1" else "discrete kW"),   # noqa: E731
                                   ("LP coordination" if c[2] == "1" else "binary coordination"), ("realistic forecast" if c[3] == "1" else "perfect forecast")])
        for _, x in a.iterrows():
            c = str(x.combo).zfill(4)
            design.append({"key": f"ablation_{c}", "name": desc(c), "what_changed": "ablation (30 RVCE days)", "gap_pct": float(x.gap_pct),
                           "certified_gap_pct": float(x.certified_gap_pct), "critical_downtime_h": float(x.critical_downtime_h),
                           "runtime_per_decision_s": float(x.plan_time_s), "source": "results/ablation/ablation_summary.csv"})
    for m, (nm, what) in DESIGN.items():
        st = {k: _stat(s, "rvce", m, k) for k in ("gap_pct", "certified_gap_pct", "critical_downtime_h", "runtime_per_decision_s")}
        if st["gap_pct"]:
            design.append({"key": m, "name": nm, "what_changed": what, **{k: (v or {}).get("mean") for k, v in st.items()},
                           "source": "results/bench_v3/summary.csv"})
    hl = headline(s, w, {k: v for k, v in labels.items() if k not in stale})   # headline: regenerated datasets only
    if stale:
        hl["detail"] = (hl["detail"] + f" Not regenerated with the new campus data (previous results, not part of this headline): "
                        + ", ".join(SPECS[k].label for k in stale) + ".").strip()
    hl["stale_datasets"] = stale
    hl["story"] = story_headlines(s, labels, cfg)
    # shadow mode (Deploy -> Phase 0): per RVCE day, what staff did (status quo) vs what Q-Grid recommended, same data
    shadow = []
    rv = r[r.dataset == "rvce"]
    for day, g in rv.groupby("day"):
        a, q = g[g.method == "status_quo"], g[g.method == "qgrid"]
        if len(a) and len(q):
            a, q = a.iloc[0], q.iloc[0]
            pick = lambda x: {"bill_rs": float(x.bill_rs), "critical_downtime_h": float(x.critical_downtime_h), "dg_litres": float(x.dg_litres),   # noqa: E731
                              "co2_kg": float(x.co2_kg), "import_kwh": float(x.import_kwh), "unserved_kwh": float(x.unserved_kwh)}
            shadow.append({"day": day, "outage": bool(isinstance(a.outage, str) and a.outage), "status_quo": pick(a), "qgrid": pick(q)})
    out = {"tariff_status": cfg["tariff_v2"]["status"],
           "methods": [{"key": k, "name": v[0], "group": v[1], "without_coordinator": k not in ("central_mpc", "central_mpc_comms_fail")} for k, v in NAMES.items()],
           "reference": {"key": REF[0], "name": REF[1], "group": "Reference", "without_coordinator": False},
           "metrics": [{"key": k, "label": l, "group": g, "better": b, "unit": u, "digits": d} for k, l, g, b, u, d in METRICS],
           "datasets": [{"key": k, "label": labels[k], "n_days": int(r[r.dataset == k].day.nunique()), "regenerated": k not in stale}
                        for k in order],
           "table": table, "wilcoxon": wil, "comms_day": comms, "headline": hl, "design_choices": design, "shadow": shadow}
    rp = ROOT / "results" / "replay"
    (rp / "bench_v3.json").write_text(json.dumps(out, allow_nan=False, default=lambda o: None).replace("NaN", "null"), encoding="utf-8")
    shutil.copy(rp / "bench_v3.json", ROOT / "frontend" / "public" / "replay" / "bench_v3.json")
    write_docs(r, s, w, labels, order, hl, cfg)
    print("wrote results/bench_v3/*.csv, docs/benchmark_v3.md, docs/scorecard.md, replay bench_v3.json")


def _cell(s, ds, m, k, d):
    st = _stat(s, ds, m, k)
    if not st or st["mean"] is None:
        return "–"
    if st["ci_lo"] is None:
        return f"{st['mean']:,.{d}f}"
    return f"{st['mean']:,.{d}f} ± {st['std']:,.{d}f}"


def write_docs(r, s, w, labels, order, hl, cfg):
    key_metrics = [("gap_pct", 2), ("certified_gap_pct", 2), ("gap_novoll_pct", 2), ("bill_rs", 0), ("saving_pct", 1),
                   ("critical_downtime_h", 2), ("critical_downtime_1pct_h", 2), ("unserved_kwh", 1), ("peak_kva", 0),
                   ("co2_kg", 0), ("dg_litres", 1), ("battery_degradation_rs", 0), ("runtime_per_decision_s", 2)]
    lab = {k: l for k, l, *_ in METRICS}
    oc = cfg["outage_costs"]
    L = ["# Benchmark v3 — structured, fair, multi-dataset\n",
         "_Generated by `scripts/write_bench_v3.py` from `results/bench_v3/*.csv` (runs: `scripts/run_bench_v3.py`). "
         "Do not edit numbers by hand._\n",
         "> QAOA on a local simulator: verified against exact solutions, no quantum advantage claimed. " + _prov.data_sentence(cfg) + "\n",
         "## Headline (generated from the results)\n", f"**{hl['text']}** {hl['detail']}\n",
         *([f"Step 1 (problem): {hl['story']['step1']['text']}\n", f"Step 7 (result): {hl['story']['step7']['text']} {hl['story']['step7']['comms']}"
            + (" Bill figures use a stand-in tariff (PENDING).\n" if hl['story']['tariff_pending'] else "\n")] if hl.get("story") else []),
         "## Protocol\n",
         "- **Datasets** (same for every method): " + "; ".join(f"{labels[k]}" for k in order) +
         ". Licences and mappings: `data/SOURCES.md`; asset sizing rule for sites without asset data: `qgrid/datasets.py`.",
         f"- **Days**: {int(r.groupby('dataset').day.nunique().min())}–{int(r.groupby('dataset').day.nunique().max())} evaluation days per dataset "
         "(seeded sample across the year); a seeded 20 % of days carry one 1–3 h unplanned grid outage. 365 days for RVCE is pending "
         "RVCE meter data.",
         "- **Same information**: every non-oracle method uses the same gradient-boosted solar and load forecasts (day-ahead methods: "
         "the 00:00 forecast; MPC methods: rolling). Status quo and the rule-based controller react to the current measurement only. "
         "The best possible plan (central MILP with perfect information) is the only oracle and is shown as a reference line.",
         f"- **Same objective and VoLL**: energy + diesel fuel + battery wear (Rs per kWh charged or discharged, from the battery cost and cycle life — ASSUMPTION) + value of lost load (non-critical Rs {oc['voll_noncritical_rs_kwh']}/kWh, "
         f"critical Rs {oc['voll_critical_rs_kwh']}/kWh — ASSUMPTION, see docs/consistency.md) + terminal value. No method optimises the "
         "demand charge; it is reported in the bill. Every gap is also reported on money alone (no VoLL).",
         "- **Best plan**: the cheapest feasible plan known for the day — the central MILP with perfect information, or a "
         "controller's realised plan where it is cheaper within solver tolerance. Every figure of a method-day (gap, critical "
         "downtime, unserved energy, bill) comes from the same run.",
         "- **Communication failure**: one seeded 3-hour campus network outage per day (06:00–21:00). Central MPC cannot reach its "
         "coordinator and devices fall back to the rule-based controller; Q-Grid's buildings lose their neighbours and keep "
         "optimising alone (each runs its own solver).",
         "- **Statistics**: mean ± standard deviation over days; 95 % t-intervals in `summary.csv`; paired Wilcoxon signed-rank tests "
         "against Q-Grid on the same days in `wilcoxon.csv`; certified gap from proven lower bounds (docs/math.md).",
         "- Same machine, one process pool, fixed seeds. Reinforcement learning (PPO) was optional and is **not included**.\n"]
    for ds in order:
        L.append(f"## {labels[ds]}\n")
        L.append("| Method | " + " | ".join(lab[k] for k, _ in key_metrics) + " |")
        L.append("|---|" + "---:|" * len(key_metrics))
        for m in list(NAMES) + [REF[0]]:
            nm = NAMES[m][0] if m in NAMES else REF[1]
            nm = f"**{nm}**" if m == "qgrid" else nm
            L.append(f"| {nm} | " + " | ".join(_cell(s, ds, m, k, d) for k, d in key_metrics) + " |")
        L.append("")
    # interpretation: wins / ties / losses vs each method on gap (Wilcoxon 5 %)
    L.append("## Where Q-Grid wins, ties and loses (gap to the best plan, paired Wilcoxon, 5 % level)\n")
    L.append("| Dataset | Method | median(Q-Grid − method), points | p | verdict |")
    L.append("|---|---|---:|---:|---|")
    for _, x in w[w.metric == "gap_pct"].iterrows():
        v = "tie (not significant)" if pd.isna(x.p) or x.p >= 0.05 else ("Q-Grid better" if x.median_diff_qgrid_minus_method < 0 else "Q-Grid worse")
        L.append(f"| {labels.get(x.dataset, x.dataset)} | {NAMES.get(x.method, (REF[1],))[0]} | {x.median_diff_qgrid_minus_method:+.2f} | "
                 f"{'–' if pd.isna(x.p) else f'{x.p:.3g}'} | {v} |")
    L += ["", "## Honest interpretation\n",
          "- Read the verdict column: it is the test, not the means, that says whether a difference is real on these days.",
          "- The central MPC with working communication is a strong benchmark; the point of Q-Grid is to stay close to it without a "
          "single point of failure, which the communication-failure rows quantify.",
          "- Asset sizes on the BDG2 and CU-BEMS sites follow one rule, not real installations; they use RVCE's tariff and CO₂ "
          "factor for comparability (controlled variable); outages are synthetic."]
    stale = [k for k in order if labels[k].startswith(STALE_PREFIX)]
    if stale:
        L.append("- **Not regenerated with the new campus data:** " + ", ".join(labels[k][len(STALE_PREFIX):].split(" — ")[0] for k in stale)
                 + ". Their rows are the previous run (earlier stand-in tariff and CO₂ data), kept for reference only; they are "
                 "excluded from the headline and from docs/consistency.md.")
    (ROOT / "docs" / "benchmark_v3.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    # one-page scorecard (RVCE + per-dataset gap)
    S = ["# Q-Grid scorecard\n", f"**{hl['text']}** {hl['detail']}\n",
         f"_All figures: mean over evaluation days, from `results/bench_v3/summary.csv`. {_prov.tariff_label(cfg)}, synthetic outages._\n",
         "| Method (RVCE) | Gap to best plan % | Certified gap % | Bill Rs/day | Saving vs today % | Critical downtime h | Peak kVA | Works without a central controller |",
         "|---|---:|---:|---:|---:|---:|---:|:---:|"]
    for m in list(NAMES) + [REF[0]]:
        nm = NAMES[m][0] if m in NAMES else REF[1]
        S.append(f"| {('**' + nm + '**') if m == 'qgrid' else nm} | " + " | ".join(_cell(s, 'rvce', m, k, d).split(' ±')[0] for k, d in
                 (("gap_pct", 2), ("certified_gap_pct", 2), ("bill_rs", 0), ("saving_pct", 1), ("critical_downtime_h", 2), ("peak_kva", 0))) +
                 f" | {'—' if m == 'optimum' else ('✓' if m not in ('central_mpc', 'central_mpc_comms_fail') else '✗')} |")
    S += ["", "| Dataset | Q-Grid gap % | best alternative without a central controller | its gap % |", "|---|---:|---|---:|"]
    nocoord = ["status_quo", "rule_based", "ga", "pso", "per_building_mpc"]
    for ds in order:
        q = (_stat(s, ds, "qgrid", "gap_pct") or {}).get("mean")
        alts = {m: (_stat(s, ds, m, "gap_pct") or {}).get("mean") for m in nocoord}
        alts = {m: v for m, v in alts.items() if v is not None}
        if alts and q is not None:
            a = min(alts, key=alts.get)
            S.append(f"| {labels[ds]} | {q:.2f} | {NAMES[a][0]} | {alts[a]:.2f} |")
    (ROOT / "docs" / "scorecard.md").write_text("\n".join(S) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
