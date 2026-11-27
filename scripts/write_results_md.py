"""Generate docs/results.md from results/*.csv and render the doc templates (docs/templates/*.md -> docs/ and
README.md). Every number is read from the result files; placeholders look like {{key}} (see values())."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT)]
from qgrid import QUANTUM_LABEL  # noqa: E402

ORDER = ["B0", "B1", "B2-GA", "B2-PSO", "B3", "C-QIEA", "B5", "OURS-QIEA", "OURS", "B4"]
NAMES = {"B0": "B0 status quo", "B1": "B1 rule-based", "B2-GA": "B2 genetic algorithm", "B2-PSO": "B2 particle swarm",
         "B3": "B3 per-zone MPC-MILP", "B4": "B4 central MILP (optimum)", "B5": "B5 ADMM + exact local solver",
         "OURS": "**OURS ADMM + QAOA**", "OURS-QIEA": "OURS-QIEA ADMM + QIEA†", "C-QIEA": "C-QIEA central QUBO + QIEA†",
         "B5-nobattery": "B5 without batteries"}


def f(x, d=0):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{d}f}"


def table(df, price, day_set):
    d = df[(df.price_scenario == price) & (df.day_set == day_set)].set_index("method")
    cols = [("cost_rs", "Rs/day", 0), ("objective_rs", "Rs incl. VOLL", 0), ("gap_to_B4_pct", "gap to B4 %", 2),
            ("critical_downtime_h", "crit. downtime h", 2), ("dg_hours", "DG h", 1), ("dg_litres", "DG L", 0),
            ("co2_kg", "CO2 kg", 0), ("self_consumption_pct", "self-cons. %", 1), ("peak_import_kw", "peak kW", 0),
            ("replan_time_s", "re-plan s/slot", 3), ("feasibility_rate", "feasible %", 1), ("qaoa_p_opt", "QAOA P(opt)", 4),
            ("admm_convergence_rate", "ADMM conv. %", 1)]
    out = ["| Design | " + " | ".join(c[1] for c in cols) + " |", "|---|" + "---:|" * len(cols)]
    for m in ORDER:
        if m not in d.index:
            continue
        r = d.loc[m]
        vals = []
        for k, _, dd in cols:
            v = r.get(k, np.nan)
            v = 100 * v if k in ("feasibility_rate", "admm_convergence_rate") and not pd.isna(v) else v
            vals.append(f(v, dd))
        out.append(f"| {NAMES[m]} | " + " | ".join(vals) + " |")
    return "\n".join(out)


def values() -> dict:
    """Flat dict of every number the templates may use."""
    b = pd.read_csv(ROOT / "results" / "benchmark.csv")
    s = pd.read_csv(ROOT / "results" / "benchmark_summary.csv")
    v = {}
    for _, r in s.iterrows():
        for k in ("cost_rs", "objective_rs", "gap_to_B4_pct", "critical_downtime_h", "dg_hours", "dg_litres", "co2_kg",
                  "replan_time_s", "admm_convergence_rate", "qaoa_p_opt", "cost_rs_semester", "co2_t_year", "self_consumption_pct"):
            if k in r and not pd.isna(r[k]):
                v[f"{r.method}.{r.day_set}.{r.price_scenario}.{k}"] = float(r[k])
    h = json.loads((ROOT / "results" / "headline.json").read_text())
    for side in ("before", "after", "optimum_B4"):
        for k, x in h[side].items():
            v[f"headline.{side}.{k}"] = x
    eco = json.loads((ROOT / "results" / "economics.json").read_text())
    for sc_, e in eco.items():
        for k, x in e.items():
            if isinstance(x, (int, float)) or x is None:
                v[f"eco.{sc_}.{k}"] = x
    ss = json.loads((ROOT / "results" / "search_space.json").read_text())
    v["search.N"] = ss["centralised_full_day"]["N_binary"]
    v["search.log10"] = ss["centralised_full_day"]["log10_states"]
    v["search.max_qubits"] = max(ss["local_qubo_qubits"].values())
    o = b[(b.method == "OURS") & (b.scenario != "eval")]
    tq, ti, ta, tx = (o[f"tier_{k}"].sum() for k in ("qaoa", "qiea", "anneal", "exact"))
    tot = tq + ti + ta + tx
    v.update({"tiers.qaoa_pct": 100 * tq / tot, "tiers.qiea_pct": 100 * ti / tot, "tiers.anneal_pct": 100 * ta / tot,
              "tiers.exact_pct": 100 * tx / tot, "tiers.total": tot, "qaoa.solves": o.qaoa_solves.sum(),
              "qaoa.time_s": o.qaoa_time_s.sum(), "qaoa.p_opt_mean": o.qaoa_p_opt.mean(),
              "qaoa.found_opt_pct": 100 * o.qaoa_found_opt_rate.mean(), "admm.conv_pct": 100 * o.admm_convergence_rate.mean(),
              "admm.iters": o.admm_iters.mean(), "admm.osc": o.admm_oscillations.sum()})
    rob = pd.read_csv(ROOT / "results" / "robustness.csv")
    net = rob[rob["mode"] == "networked"]
    v["robust.recovery_s"] = float(net.recovery_time_s.max()) if len(net) else float("nan")
    v["robust.all_feasible"] = bool(rob.feasible.all())
    sc = pd.read_csv(ROOT / "results" / "qaoa_scaling.csv")
    t = sc[(sc.params == "trained")].groupby(["n_qubits", "p"]).p_opt.mean()
    nmax = int(sc.n_qubits.max())
    for p in (1, 2, 3):
        v[f"scaling.p{p}.nmax.p_opt"] = float(t.loc[(nmax, p)])
    v["scaling.nmax"] = nmax
    v["scaling.uniform_nmax"] = 2.0 ** -nmax
    meta = json.loads((ROOT / "data" / "processed" / "meta.json").read_text())
    v["price_A_status"] = meta["prices"]["A"]["status"]
    v["price_B_status"] = meta["prices"]["B"]["status"]
    v.update(values_v2())
    v.update(values_v3())
    return v


def values_v3() -> dict:
    """Current design: quantum placement test, ablation (Shapley), multi-dataset benchmark; keys q3.*"""
    v = {}
    qp = ROOT / "results" / "quantum_placement" / "summary.json"
    if qp.exists():
        for k, x in json.loads(qp.read_text()).items():
            if isinstance(x, (int, float)) and x is not None:
                v[f"q3.qp.{k}"] = float(x)
    sv = ROOT / "results" / "ablation" / "shapley.json"
    if sv.exists():
        for metric, d in json.loads(sv.read_text()).items():
            for f, x in d.items():
                v[f"q3.shap.{metric}.{f}"] = float(x)
    ab = ROOT / "results" / "ablation" / "ablation_summary.csv"
    if ab.exists():
        a = pd.read_csv(ab, dtype={"combo": str})
        for _, r in a.iterrows():
            c = str(r.combo).zfill(4)
            v[f"q3.abl.{c}.gap_pct"] = float(r.gap_pct)
            v[f"q3.abl.{c}.certified_gap_pct"] = float(r.certified_gap_pct)
    bs = ROOT / "results" / "bench_v3" / "summary.csv"
    if bs.exists():
        b = pd.read_csv(bs)
        for _, r in b.iterrows():
            for k in ("gap_pct", "certified_gap_pct", "saving_pct", "critical_downtime_h", "bill_rs", "critical_downtime_1pct_h",
                      "gap_novoll_pct", "unserved_kwh", "peak_kva", "co2_kg", "dg_litres", "battery_degradation_rs",
                      "runtime_per_decision_s", "messages_per_decision", "battery_cycles"):
                if f"{k}__mean" in r and not pd.isna(r[f"{k}__mean"]):
                    v[f"q3.b.{r.dataset}.{r.method}.{k}"] = float(r[f"{k}__mean"])
        v["q3.b.n_datasets"] = float(b.dataset.nunique())
    rb = ROOT / "results" / "replay" / "bench_v3.json"
    if rb.exists():
        h = json.loads(rb.read_text())["headline"]
        v["q3.headline"] = h["text"] + " " + h["detail"]
        st = h.get("story") or {}
        if st:
            v["q3.story.step1"] = st["step1"]["text"]; v["q3.story.step7"] = st["step7"]["text"]; v["q3.story.comms"] = st["step7"]["comms"]
            v["q3.story.n_days"] = float(st["n_days"])
            for k, x in st["numbers"].items():
                if x is not None:
                    v[f"q3.n.{k}"] = float(x)
            n = st["numbers"]
            if n.get("central_comms_fail_gap_pct") is not None:
                v["q3.n.central_comms_penalty"] = n["central_comms_fail_gap_pct"] - n["central_gap_pct"]
                v["q3.n.qgrid_comms_penalty"] = n["qgrid_comms_fail_gap_pct"] - n["qgrid_gap_pct"]
            if n.get("qgrid_downtime_h") is not None:
                v["q3.n.qgrid_downtime_min"] = 60 * n["qgrid_downtime_h"]; v["q3.n.status_quo_downtime_min"] = 60 * n["status_quo_downtime_h"]
        rv = h.get("rvce") or {}
        if rv:
            v["q3.alt_name"] = rv["alt"]; v["q3.alt_gap"] = float(rv["alt_gap"])
        v["q3.best_on_n"] = float(len(h.get("best_on", []))); v["q3.n_campuses"] = float(h.get("n", 0))
    im = ROOT / "results" / "impact" / "impact.json"
    if im.exists():
        m = json.loads(im.read_text())
        c = m["central"]
        for k, x in c.items():
            if isinstance(x, (int, float)):
                v[f"imp.{k}"] = float(x); v[f"imp.{k}_lakh"] = float(x) / 1e5
        for side in ("npv_gross_rs", "npv_net_rs"):
            for q, x in m["monte_carlo"][side].items():
                v[f"imp.mc.{side}.{q}"] = float(x) * (100 if q == "prob_positive" else 1e-5)
        v["imp.n_buildings"] = float(m["n_buildings"]); v["imp.hw_lo"] = float(m["per_building"][0]["hardware_rs"][0]); v["imp.hw_hi"] = float(m["per_building"][0]["hardware_rs"][1])
        v["imp.throughput_kwh_day"] = float(m["benefit"]["extra_throughput_kwh_per_day_mean"])
        if m.get("variant"):
            vv = m["variant"]
            v["imp.var.net_lakh"] = vv["net_rs_per_year"] / 1e5; v["imp.var.npv_lakh"] = vv["npv_net_rs"] / 1e5
            v["imp.var.saving_day"] = vv["bill_saving_rs_per_day_mean"]; v["imp.var.prob_pos"] = 100 * vv["mc_npv_net_rs"]["prob_positive"]
            v["imp.var.downtime_h"] = vv["critical_downtime_h"]; v["imp.var.wear"] = vv["wear_rs_per_kwh_in_objective"]
    lv = ROOT / "results" / "replay" / "live.json"
    if lv.exists():
        L = json.loads(lv.read_text())
        rm = L["remove"].get("heavy_labs")
        if rm:
            v["live.remove.rounds_after"] = float(rm["after"]["admm_rounds"]); v["live.remove.rounds_before"] = float(rm["before"]["admm_rounds"])
            v["live.remove.crit_after"] = float(rm["others"]["crit_shed_after_kwh"]); v["live.remove.compute_s"] = float(rm["compute_s"])
            v["live.remove.lines"] = float(len(rm["lines_dark"])); v["live.remove.time"] = rm["time"]
        v["live.remove.max_s"] = float(max(r["compute_s"] for r in L["remove"].values()))
        if L["qaoa"]:
            v["live.qaoa.max_s"] = float(max(q["qaoa"]["time_s"] + q["exact"]["time_s"] for q in L["qaoa"]))
            v["live.qaoa.n_qubits"] = float(L["qaoa"][0]["n_qubits"])
            v["live.qaoa.match_n"] = float(sum(q["qaoa"]["matches_exact"] for q in L["qaoa"])); v["live.qaoa.n"] = float(len(L["qaoa"]))
        for i, r in enumerate(L["rules"]):
            v[f"live.rule{i}.text"] = r["text"]; v[f"live.rule{i}.s"] = float(r["time_s"])
            v[f"live.rule{i}.ok"] = "accepted" if all(x["ok"] for x in r["results"]) else "rejected"
            if not all(x["ok"] for x in r["results"]):
                v[f"live.rule{i}.reason"] = "; ".join(x["reason"] for x in r["results"] if not x["ok"]).strip("; ")
            e = r.get("effect")
            if e and e["after"].get("status") == "Optimal":
                for k in ("cost_rs", "objective_rs", "dg_hours", "shed_noncritical_kwh", "critical_downtime_h"):
                    v[f"live.rule{i}.before.{k}"] = float(e["before"].get(k, 0.0)); v[f"live.rule{i}.after.{k}"] = float(e["after"].get(k, 0.0))
                v[f"live.rule{i}.delta_cost"] = float(e["after"]["cost_rs"] - e["before"]["cost_rs"])
                v[f"live.rule{i}.delta_obj"] = float(e["after"]["objective_rs"] - e["before"]["objective_rs"])
    dly = ROOT / "results" / "bench_v3" / "daily.csv"
    if dly.exists():
        d = pd.read_csv(dly)
        rv_ = d[d.dataset == "rvce"]
        pv = rv_.pivot_table(index="day", columns="method", values="objective_rs")
        if {"design_benders_exact", "design_benders_qaoa"} <= set(pv.columns):
            v["cons.qaoa_beats_exact_days"] = float((pv.design_benders_qaoa < pv.design_benders_exact * (1 - 1e-8)).sum())
            v["cons.days"] = float(len(pv))
            v["cons.qaoa_agree_pct"] = 100 * float(rv_[rv_.method == "design_benders_qaoa"].master_opt_share.mean())
    return v


def values_v2() -> dict:
    """Optimiser v2 numbers (results/v2/*), keyed v2.<method>.<day_set>.<metric>; empty if not run yet."""
    d = ROOT / "results" / "v2"
    v = {}
    if not (d / "benchmark_v2_summary.csv").exists():
        return v
    s = pd.read_csv(d / "benchmark_v2_summary.csv")
    for _, r in s.iterrows():
        for k, x in r.items():
            if k in ("day_set", "method") or isinstance(x, str) or pd.isna(x):
                continue
            v[f"v2.{r.method}.{r.day_set}.{k}"] = float(x)
    b = pd.read_csv(d / "benchmark_v2.csv")
    head = "outage_fest"
    for _, r in b[b.scenario == head].iterrows():
        for k in ("objective_incl_demand_rs", "cost_incl_demand_rs", "peak_kva", "demand_charge_rs", "co2_kg", "dg_hours",
                  "critical_downtime_h", "gap_to_opt_pct", "runtime_s", "replan_time_s", "benders_iters_mean",
                  "benders_gap_mean_pct", "master_opt_share", "qaoa_block_opt_share", "admm_converged_rate", "admm_iters_mean"):
            if k in r and not pd.isna(r[k]):
                v[f"v2.head.{r.method}.{k}"] = float(r[k])
    for m in b.method.unique():
        sub = b[(b.method == m) & (b.scenario != "eval")]
        if "master_opt_share" in sub and sub.master_opt_share.notna().any():
            v[f"v2.scen.{m}.master_opt_pct"] = 100 * float(sub.master_opt_share.mean())
        if "qaoa_block_opt_share" in sub and sub.qaoa_block_opt_share.notna().any():
            v[f"v2.scen.{m}.block_opt_pct"] = 100 * float(sub.qaoa_block_opt_share.mean())
            v[f"v2.scen.{m}.block_solves"] = float(sub.qaoa_block_solves.sum())
        if "admm_converged_rate" in sub and sub.admm_converged_rate.notna().any():
            v[f"v2.scen.{m}.admm_conv_pct"] = 100 * float(sub.admm_converged_rate.mean())
        if "benders_converged_rate" in sub and sub.benders_converged_rate.notna().any():
            v[f"v2.scen.{m}.benders_conv_pct"] = 100 * float(sub.benders_converged_rate.mean())
    fe = d / "forecast_error.csv"
    if fe.exists():
        f_ = pd.read_csv(fe)
        for k in ("nmae_pct", "nmae_1h_pct", "nmae_4-24h_pct", "bias_pct"):
            v[f"v2.fc.mean.{k}"] = float(f_[k].mean())
            v[f"v2.fc.max.{k}"] = float(f_[k].max())
    hw = d / "hardware.json"
    if hw.exists():
        hj = json.loads(hw.read_text())
        v["v2.hw.ibm_status"] = hj["ibm"]["status"]
        v["v2.hw.dwave_status"] = hj["dwave"].get("dwave", "")
        v["v2.hw.campus_n"] = hj["dwave"].get("n_binaries")
        v["v2.hw.sa_gap_pct"] = hj["dwave"].get("sa_gap_pct")
        rows = hj["ibm"].get("rows", [])
        if rows:
            v["v2.hw.aer_found_opt_pct"] = 100 * float(np.mean([r["aer_found_opt"] for r in rows]))
            v["v2.hw.n_blocks"] = len(rows)
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    v["v2.tariff_status"] = cfg["tariff_v2"]["status"]
    v["v2.demand_rate_month"] = float(cfg["tariff_v2"]["demand_rs_per_kva_month"])
    return v


V2_LABEL = {"B0": "Status quo", "B1": "Rule-based", "B2-GA": "Genetic algorithm", "B2-PSO": "Particle swarm",
            "B3": "Per-zone MILP (no sharing)", "B4": "Central MILP, energy only",
            "B4-DC": "Central MILP with demand charge (optimum)", "V1": "Earlier prototype (1-hour horizon, binary coordination)",
            "V2-QAOA": "Hybrid Benders, QAOA master", "V2-QIEA": "Hybrid Benders, QIEA† master", "V2-SA": "Hybrid Benders, annealing master",
            "V2-EXACT": "Hybrid Benders, exact master", "V2-EXACT-FC": "Hybrid Benders, exact master, solar forecast",
            "V2-QAOA-FC": "Hybrid Benders, QAOA master, solar forecast"}
V2_ORDER = list(V2_LABEL)


def table_v2(df) -> str:
    cols = [("objective_incl_demand_rs", "Rs/day incl. demand charge + VOLL", 0), ("cost_incl_demand_rs", "Rs/day (money)", 0),
            ("peak_kva", "peak kVA", 0), ("co2_kg", "CO₂ kg", 0), ("dg_hours", "DG h", 2),
            ("critical_downtime_h", "critical downtime h", 2), ("gap_to_opt_pct", "gap to optimum %", 2),
            ("runtime_s", "runtime s", 1), ("benders_iters_mean", "Benders iters", 1),
            ("master_opt_share", "masters optimal (QAOA/QIEA/SA)", 2)]
    lines = ["| Design | " + " | ".join(c[1] for c in cols) + " |", "|---|" + "---:|" * len(cols)]
    for m in V2_ORDER:
        r = df[df.method == m]
        if not len(r):
            continue
        r = r.iloc[0]
        cells = []
        for k, _, d in cols:
            x = r.get(k, np.nan)
            cells.append("–" if pd.isna(x) else (f"{100 * x:.0f} %" if k == "master_opt_share" else f"{x:,.{d}f}"))
        nm = V2_LABEL[m]
        lines.append(f"| {'**' + nm + '**' if m.startswith('V2') else nm} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_results_v2(v: dict):
    d = ROOT / "results" / "v2"
    if not (d / "benchmark_v2_summary.csv").exists():
        return
    s = pd.read_csv(d / "benchmark_v2_summary.csv")
    b = pd.read_csv(d / "benchmark_v2.csv")
    meta = json.loads((ROOT / "data" / "processed" / "meta.json").read_text())
    out = [f"# Results — hybrid Benders design (an earlier design step; computed on the previous template load profiles)\n\n> {QUANTUM_LABEL}. "
           "† QIEA = quantum-inspired evolutionary algorithm, classical CPU — not quantum computing.\n",
           "_Generated by `scripts/write_results_md.py` from `results/v2/*.csv` (produced by `scripts/run_benchmark_v2.py`); "
           "do not edit numbers by hand._\n",
           "## ⚠️ Data status — read first\n",
           (f"Energy prices (price scenario A) and the demand charge (Rs {v.get('v2.demand_rate_month', float('nan')):,.0f}/kVA/month) are "
            "**REAL**: KERC Tariff Order 2023, BESCOM HT-2(c)(ii) with ToD (page references in `data/SOURCES.md`); daily figures carry "
            "1/30 of the monthly demand charge on the day's peak kVA (ASSUMPTION). Campus loads are area-based estimates.\n"
            if v.get("v2.tariff_status") == "REAL" else
            f"Energy prices (price scenario A) and the demand charge (Rs {v.get('v2.demand_rate_month', float('nan')):,.0f}/kVA/month) are "
            f"**PENDING stand-ins** (tariff v2 status: {v.get('v2.tariff_status', '?')}); daily figures carry 1/30 of the monthly demand "
            "charge on the day's peak kVA (ASSUMPTION). Rs figures are therefore not headline results; comparisons between methods on "
            "identical inputs are meaningful. Campus loads are area-based estimates.\n"),
           "## Setup\n",
           "Every method runs on the same days, data and constraints and is settled by the same `model.settle()`. v2 re-plans 24 h ahead "
           "every 15-min slot (dispatch: LP + consensus ADMM every slot; commitment master re-optimised at the start of every clock "
           "hour, carried forward in between). Benders stops at a 0.5 % bound gap or 6 iterations. Rows marked *solar forecast* use the "
           "clear-sky × clearness forecast instead of actual solar; all other v2 rows see actual solar over the horizon, like the MILP "
           "baselines. The optimum is the central MILP with the demand charge (B4-DC, 0.1 % MIP gap, 15-min binaries, continuous "
           "flows); v2's hourly commitments are a restriction of it.\n",
           "## Six named days (mean per day)\n\n" + table_v2(s[s.day_set == "scenarios"]) + "\n"]
    if (s.day_set == "eval30_mean").any():
        out.append("## 30-day evaluation set (mean per day)\n\n" + table_v2(s[s.day_set == "eval30_mean"]) + "\n")
    for k, sc in meta["days"]["scenarios"].items():
        sub = b[b.scenario == k]
        if len(sub):
            out.append(f"### {k.replace('_', ' ')} — {sc['date']} ({sc['note']})\n\n" + table_v2(sub) + "\n")
    conv = d / "convergence.csv"
    if conv.exists():
        c = pd.read_csv(conv).groupby("method")[["admm_converged_rate", "admm_iters_mean"]].mean()
        out.append("## Decentralisation: old binary ADMM vs ADMM on the convex LP\n\n| Method | ADMM runs converged | mean iterations |\n|---|---:|---:|")
        for m, r in c.iterrows():
            out.append(f"| {V2_LABEL.get(m, m)} | {100 * r.admm_converged_rate:.0f} % | {r.admm_iters_mean:.0f} |")
        out.append("\nv1's ADMM agrees on discrete tie-line levels (non-convex): 'converged' = all neighbours agreed within the "
                   "iteration cap in a slot. v2's ADMM runs on the continuous LP subproblem (convex): 'converged' = every cable's "
                   "copies within 1 kW and the consensus moving < 1 kW per iteration. Each building then fixes the agreed flows and "
                   "re-solves its own LP so the applied plan balances exactly.\n")
    fe = d / "forecast_error.csv"
    if fe.exists():
        f_ = pd.read_csv(fe)
        out.append("## Solar forecast error (forecasts issued every slot, lead 15 min – 24 h)\n\n| Day | nMAE % | 1 h | 1–4 h | 4–24 h | bias % |\n|---|---:|---:|---:|---:|---:|")
        for _, r in f_.iterrows():
            out.append(f"| {r.scenario} | {r.nmae_pct:.1f} | {r['nmae_1h_pct']:.1f} | {r['nmae_1-4h_pct']:.1f} | {r['nmae_4-24h_pct']:.1f} | {r.bias_pct:+.1f} |")
        out.append("\nnMAE = mean absolute error / mean actual output over daylight slots. NASA POWER hourly irradiance is satellite-derived "
                   "and smooth, so real rooftop forecasting error would be larger.\n")
    hw = d / "hardware.json"
    if hw.exists():
        hj = json.loads(hw.read_text())
        out.append("## Real quantum hardware (optional, experimental)\n")
        out.append(f"- IBM Quantum: {hj['ibm']['status']}. Qiskit Aer on the same {len(hj['ibm']['rows'])} recorded master blocks: "
                   f"best measured assignment = exact optimum in {sum(r['aer_found_opt'] for r in hj['ibm']['rows'])} of {len(hj['ibm']['rows'])}.")
        dw = hj["dwave"]
        out.append(f"- D-Wave Leap: {dw.get('dwave')}. Whole-campus 24 h commitment QUBO: {dw.get('n_binaries')} binaries; simulated "
                   f"annealing {dw.get('sa_gap_pct', float('nan')):.2f} % from the exact MILP masters.")
        out.append("\nHardware results, when run, are noisy and are shown next to simulator and exact results.\n")
    (ROOT / "docs" / "results_v2.md").write_text("\n".join(out))


def render(text: str, v: dict) -> str:
    def rep(m):
        key, _, fmt_ = m.group(1).partition("|")
        x = v.get(key.strip())
        if x is None:
            return f"⟨missing {key.strip()}⟩"
        if isinstance(x, bool):
            return "yes" if x else "no"
        d = int(fmt_) if fmt_ else 0
        return f"{x:,.{d}f}" if isinstance(x, (int, float)) else str(x)
    return re.sub(r"\{\{([^}]+)\}\}", rep, text)


def main():
    s = pd.read_csv(ROOT / "results" / "benchmark_summary.csv")
    meta = json.loads((ROOT / "data" / "processed" / "meta.json").read_text())
    st = meta["data_status"]
    pend = [f"{k}: {x}" for k, x in st.items() if x != "REAL"]
    out = [f"# Results (RVCE campus model)\n\n> {QUANTUM_LABEL}. † QIEA = quantum-inspired evolutionary algorithm, classical CPU — not quantum computing.\n",
           "_Generated by `scripts/write_results_md.py` from `results/*.csv`; do not edit numbers by hand. "
           "Earlier generic 5-building results: `docs/results_v1.md`._\n",
           "## ⚠️ Data status — read first\n",
           "Real: NASA POWER (PV), Ausgrid (hostel load shape), OpenStreetMap (layout). **PENDING:** " + "; ".join(pend) + ". "
           "Both price scenarios use labelled STAND-IN prices and campus loads are area-based estimates, so **no Rs figure "
           "below is a headline result yet**; comparisons between methods on identical inputs are meaningful. "
           "Re-run `scripts/refresh_all.sh` when the files in `docs/data_request.md` arrive.\n",
           "## Setup\n",
           "9 OSM-derived zones, tie-lines from the real layout (MST + 2 mesh links), 96 × 15-min slots. QAOA in the loop on "
           "the 6 named scenarios; on the 30-day set the OURS row uses the classical local tiers (SA → exact). B4 = MILP to a "
           "0.1 % gap with continuous tie-line flows (a relaxation of the decentralised methods' discrete levels). Objective "
           "comparisons (gap to B4) include value of lost load (VOLL, an ASSUMPTION) so that shedding is never 'cheaper'.\n"]
    for p in ("A", "B"):
        out.append(f"## Price scenario {p} ({'headline: BESCOM HT ToD + KERC export' if p == 'A' else 'sensitivity: IEX DAM both ways'}) — {meta['prices'][p]['status']}\n")
        out.append("### 30-day evaluation set (mean per day)\n\n" + table(s, p, "eval30_mean") + "\n")
        for k in meta["days"]["scenarios"]:
            sc = meta["days"]["scenarios"][k]
            out.append(f"### {k.replace('_', ' ')} — {sc['date']} ({sc['note']})\n\n" + table(s, p, k) + "\n")
    v = values()
    tmpl = (ROOT / "docs" / "templates" / "results_tail.md")
    if tmpl.exists():
        out.append(render(tmpl.read_text(), v))
    (ROOT / "docs" / "results.md").write_text("\n".join(out))
    for t in (ROOT / "docs" / "templates").glob("*.md"):
        if t.name == "results_tail.md":
            continue
        dest = ROOT / ("README.md" if t.name == "README.md" else f"docs/{t.name}")
        dest.write_text(render(t.read_text(), v))
    write_results_v2(v)
    missing = sum(p.read_text().count("⟨missing") for p in [ROOT / "README.md", *(ROOT / "docs").glob("*.md")])
    print(f"wrote docs/results.md and rendered templates ({missing} missing placeholders)")


if __name__ == "__main__":
    main()
