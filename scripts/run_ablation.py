"""Ablation study (2^4 = 16 combinations of A horizon, B dispatch, C coordination, D forecast) on the 30-day
evaluation set, every combination on identical days, data and objective, plus per-day certified lower bounds.

  python scripts/run_ablation.py [--workers=8] [--days=N]      -> results/ablation/*.csv, docs/ablation.md
  python scripts/run_ablation.py --report                      -> re-write docs/ablation.md from the CSVs

Gap to optimum uses the central MILP (15-min binaries, perfect information, 0.01 % MIP gap) on the same day;
certified gap uses the best proven lower bound (central MILP dual bound, LP relaxation, Lagrangian bound).
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT)]
OUT = ROOT / "results" / "ablation"
GBM = OUT / "gbm.pkl"
COMBOS = ["".join(map(str, c)) for c in itertools.product((0, 1), repeat=4)]          # "0000" .. "1111"
FACTOR = {"A": "24 h horizon (vs 1 h)", "B": "LP continuous dispatch (vs discretised kW)",
          "C": "consensus ADMM on the convex LP (vs binary ADMM)", "D": "realistic forecasts (vs perfect)"}
_CTX = {}


def _ctx():
    if not _CTX:
        from qgrid import data as qdata
        from qgrid.model import load_campus
        from qgrid.v2.horizon import YearData
        campus = load_campus()
        proc = qdata.load_processed()
        yd = YearData(campus, proc, "A", forecast=False)
        _CTX.update(campus=campus, proc=proc, yd=yd)
    return _CTX


def _forecaster():
    if "fc" not in _CTX:
        from qgrid.ablation.forecast import GBMForecaster
        c = _ctx()
        meta = c["proc"]["meta"]
        hold = meta["days"]["eval_days"] + [v["date"] for v in meta["days"]["scenarios"].values()]
        _CTX["fc"] = GBMForecaster(c["campus"], c["proc"], c["yd"], hold, cache=GBM)
    return _CTX["fc"]


def run_combo(day: str, code: str, outage=None) -> dict:
    from qgrid.ablation.engine import ACfg, run_day
    from qgrid.model import check_feasibility, make_day, settle
    c = _ctx()
    campus, yd = c["campus"], c["yd"]
    cfg = ACfg(*[ch == "1" for ch in code])
    dd = make_day(campus, c["proc"], day, "A", "eval", outage)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        sched, st = run_day(campus, yd, day, outage, cfg, forecaster=_forecaster() if cfg.D else None)
    m = settle(campus, dd, sched)
    f = check_feasibility(campus, sched, dd)
    pf = float(campus.cfg["tariff_v2"]["power_factor"])
    return {"day": day, "set": "outage" if outage else "eval30", "combo": code, "name": cfg.name, **{k: m[k] for k in (
        "objective_rs", "cost_rs", "critical_downtime_h", "critical_unserved_kwh", "unserved_kwh", "import_kwh",
        "export_kwh", "spill_kwh", "dg_litres", "dg_hours", "self_consumption_pct", "co2_kg")},
        "peak_kva": m["peak_import_kw"] / pf, "feasible": f["feasible"], **st}


def run_bounds(day: str, outage=None) -> dict:
    from qgrid.ablation.bounds import day_bounds
    c = _ctx()
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        b = day_bounds(c["campus"], c["yd"], day, outage)
    return {"day": day, "set": "outage" if outage else "eval30", **b}


def _task(kind, *args):
    t0 = time.perf_counter()
    r = run_combo(*args) if kind == "combo" else run_bounds(*args)
    r["combo"] = str(r.get("combo", "")).zfill(4) if kind == "combo" else None
    r["wall_s"] = time.perf_counter() - t0
    return kind, r


# ------------------------------------------------------------------------------------------------ analysis
def shapley(means: dict, metric: str) -> dict:
    """Shapley value of each factor for the change of `metric` from 0000 to 1111 (exact, from the 16 cells)."""
    out = {}
    for k, f in enumerate("ABCD"):
        tot = 0.0
        others = [j for j in range(4) if j != k]
        for r in range(4):
            for S in itertools.combinations(others, r):
                base = ["0"] * 4
                for j in S:
                    base[j] = "1"
                with_f = base.copy(); with_f[k] = "1"
                wgt = factorial(len(S)) * factorial(3 - len(S)) / factorial(4)
                tot += wgt * (means["".join(with_f)][metric] - means["".join(base)][metric])
        out[f] = tot
    return out


def report():
    df = pd.read_csv(OUT / "ablation.csv", dtype={"combo": str})
    df["combo"] = df.combo.str.zfill(4)
    bd_all = pd.read_csv(OUT / "bounds.csv")
    key = lambda d: d["day"] + "|" + d["set"]   # noqa: E731
    bmap = bd_all.assign(k=key(bd_all)).set_index("k")
    df["opt"] = key(df).map(bmap.opt_central); df["LB"] = key(df).map(bmap.LB)
    stress = df[df.set == "outage"].copy()
    bd = bd_all[bd_all.set == "eval30"].set_index("day")
    df["gap_pct"] = 100 * (df.objective_rs - df.opt) / df.opt
    df["certified_gap_pct"] = 100 * (df.objective_rs - df.LB) / df.LB
    df.to_csv(OUT / "ablation.csv", index=False)
    stress = df[df.set == "outage"].copy()
    df = df[df.set == "eval30"]
    cols = ["gap_pct", "certified_gap_pct", "critical_downtime_h", "critical_unserved_kwh", "import_kwh", "peak_kva",
            "dg_litres", "self_consumption_pct", "spill_kwh", "objective_rs", "plan_time_s", "admm_converged_rate",
            "mip_timelimit"]
    g = df.groupby("combo")[cols].mean()
    g["n_days"] = df.groupby("combo").size()
    g["feasible_all"] = df.groupby("combo").feasible.all()
    g.to_csv(OUT / "ablation_summary.csv")
    means = g.to_dict(orient="index")
    sv = {m: shapley({k: v for k, v in means.items()}, m) for m in ("gap_pct", "critical_downtime_h", "peak_kva", "import_kwh")}
    json.dump(sv, open(OUT / "shapley.json", "w", encoding="utf-8"), indent=1)
    fe = json.load(open(OUT / "forecast_mae.json", encoding="utf-8")) if (OUT / "forecast_mae.json").exists() else {}
    # waterfall: start at 0000, add each factor's Shapley contribution, end at 1111
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (ROOT / "docs" / "media").mkdir(parents=True, exist_ok=True)
    start, end = means["0000"]["gap_pct"], means["1111"]["gap_pct"]
    steps = [("start: first-prototype\ndesign (none)", start)] + [(f"{f}: {FACTOR[f].split(' (')[0]}", sv["gap_pct"][f]) for f in "ABCD"]
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=150)
    run = 0.0
    for k, (lab, v) in enumerate(steps):
        if k == 0:
            ax.bar(k, v, color="#6B6B6B"); run = v
            ax.text(k, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
        else:
            bottom = run + min(v, 0)
            ax.bar(k, abs(v), bottom=bottom, color="#1E9E5A" if v < 0 else "#D64545")
            ax.text(k, run + max(v, 0), f"{v:+.1f}", ha="center", va="bottom", fontsize=9)
            run += v
    ax.bar(len(steps), end, color="#2F6BFF"); ax.text(len(steps), end, f"{end:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(steps) + 1), [s[0] for s in steps] + ["all four\n(ABCD)"], fontsize=8)
    ax.set_ylabel("gap to optimum, % (30-day mean)")
    ax.set_title("How many points each change removes (Shapley attribution over all 16 combinations)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(ROOT / "docs" / "media" / "ablation_waterfall.png"); plt.close(fig)
    _write_md(g, sv, fe, bd, stress.set_index("combo"))
    _write_math(df, g, bd)


def _write_md(g, sv, fe, bd, stress=None):
    from qgrid import QUANTUM_LABEL
    lab = lambda c: "+".join(f for f, ch in zip("ABCD", c) if ch == "1") or "none (first-prototype design)"   # noqa: E731
    lines = ["# Ablation study — which change removes how much of the gap\n",
             f"> {QUANTUM_LABEL}. No quantum solver is used in the ablation: every master problem is solved exactly "
             "(HiGHS branch and bound) so the factors are measured without solver noise.\n",
             "_Generated by `scripts/run_ablation.py` from `results/ablation/*.csv`; do not edit numbers by hand._\n",
             "## Setup\n",
             f"- {int(g.n_days.max())} evaluation days (the 30-day set), identical data, days, constraints and objective for every "
             "combination; loads from BDG2 meter shapes, solar from NASA POWER; prices are the PENDING stand-in tariff (A).",
             "- Every combination re-plans every hour (4 slots) and applies the first hour; the realised schedule is settled "
             "by `model.settle()`. Objective = energy + diesel fuel + value of lost load + terminal value of stored energy "
             "and water (the central MILP's objective). Peak kVA is reported, not optimised.",
             "- Branch and bound gets the same budget in every combination: 0.5 s and a 1 % gap per building solve; "
             "`B&B limit hits` counts solves that stopped at the time limit (incumbent used).",
             "- Factors: " + "; ".join(f"**{k}** = {v}" for k, v in FACTOR.items()) + ".",
             "- Under C (consensus ADMM on the LP) cable flows are continuous, because ADMM needs a convex problem; "
             "without B and without C, flows use the first prototype's discrete levels {−120, 0, +120} kW.",
             "- 1 h horizon (no A) uses the first prototype's per-zone cost-to-go (dynamic programming over the day on its own forecast) "
             "as the terminal value of the battery.\n",
             "## How many points each change removes\n",
             "![waterfall](media/ablation_waterfall.png)\n",
             "Shapley attribution: each factor's average marginal effect over all orders in which the four changes could be "
             "added (exact, from the 16 cells; the contributions sum to the total change from none to all four).\n",
             "| Factor | gap-to-optimum, points | critical downtime, h | peak, kVA | grid import, kWh |",
             "|---|---:|---:|---:|---:|"]
    for f in "ABCD":
        lines.append(f"| {f}: {FACTOR[f]} | {sv['gap_pct'][f]:+.2f} | {sv['critical_downtime_h'][f]:+.2f} | "
                     f"{sv['peak_kva'][f]:+.0f} | {sv['import_kwh'][f]:+.0f} |")
    lines += ["", "## All 16 combinations (30-day means)\n",
              "| Combination | gap % | certified gap % | critical downtime h | critical unserved kWh | grid import kWh | "
              "peak kVA | DG litres | solar self-consumption % | curtailment kWh | plan time s | ADMM converged | B&B limit hits |",
              "|---|" + "---:|" * 12]
    for c, r in g.sort_values("gap_pct", ascending=False).iterrows():
        c = str(c).zfill(4)
        lines.append(f"| {lab(c)} | {r.gap_pct:.2f} | {r.certified_gap_pct:.2f} | {r.critical_downtime_h:.2f} | "
                     f"{r.critical_unserved_kwh:.1f} | {r.import_kwh:,.0f} | {r.peak_kva:,.0f} | {r.dg_litres:.1f} | "
                     f"{r.self_consumption_pct:.1f} | {r.spill_kwh:.1f} | {r.plan_time_s:.2f} | "
                     f"{100 * r.admm_converged_rate:.0f} % | {r.mip_timelimit:.1f} |")
    if stress is not None and len(stress):
        lines += ["", "## Outage stress day (BESCOM outage 18:00–21:00 during the fest, 15 Jun 2023) — critical loads\n",
                  "| Combination | gap % | certified gap % | critical downtime h | critical unserved kWh | DG litres | peak kVA |",
                  "|---|" + "---:|" * 6]
        for c, r in stress.sort_values("gap_pct", ascending=False).iterrows():
            c = str(c).zfill(4)
            lines.append(f"| {lab(c)} | {r.gap_pct:.2f} | {r.certified_gap_pct:.2f} | {r.critical_downtime_h:.2f} | "
                         f"{r.critical_unserved_kwh:.1f} | {r.dg_litres:.1f} | {r.peak_kva:,.0f} |")
    lines += ["", "## Lower bounds (per day, proven)\n",
              f"Mean over days: LP relaxation is {100 * ((bd.opt_central - bd.LB_lp) / bd.opt_central).mean():.3f} % below the "
              f"central optimum, the Lagrangian bound {100 * ((bd.opt_central - bd.LB_lag) / bd.opt_central).mean():.3f} %, and the "
              f"MILP's own dual bound {100 * ((bd.opt_central - bd.LB_mip) / bd.opt_central).mean():.4f} %. Certified gap = "
              "(our cost − best bound) / best bound; see docs/math.md for what is proven and what is measured.\n"]
    if fe:
        lines += ["## Forecast error (factor D, held-out evaluation days, forecasts issued hourly, lead 15 min – 24 h)\n",
                  f"- Solar: MAE {fe['solar_mae_kw']:.0f} kW (campus), nMAE {fe['solar_nmae_pct']:.1f} % "
                  f"(1 h {fe['solar_nmae_1h_pct']:.1f} %, 1–4 h {fe['solar_nmae_1-4h_pct']:.1f} %, 4–24 h {fe['solar_nmae_4-24h_pct']:.1f} %).",
                  f"- Load: MAE {fe['load_mae_kw']:.0f} kW (campus), nMAE {fe['load_nmae_pct']:.1f} % "
                  f"(1 h {fe['load_nmae_1h_pct']:.1f} %, 4–24 h {fe['load_nmae_4-24h_pct']:.1f} %).",
                  "- Gradient-boosted trees trained on other 2023 days (held-out days excluded); BDG2 zone shapes are medians "
                  "over many buildings and NASA POWER irradiance is hourly and satellite-derived, so both are smoother than a "
                  "single real meter — real forecast errors would be larger.\n"]
    lines += ["## Limitations\n",
              "- 30 days, not 365 (the full year will be run when RVCE data is available); one price scenario (stand-in).",
              "- Loads are BDG2 shapes scaled by floor area × EPI (PENDING RVCE meters); see data/SOURCES.md.",
              "- Discrete-dispatch combinations hit the branch-and-bound time limit more often; their gap includes that."]
    (ROOT / "docs" / "ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote docs/ablation.md")


def _write_math(df, g, bd):
    """docs/math.md: what is proven (with verified citations) and what is measured (from results/ablation)."""
    from qgrid.model import load_campus
    campus = load_campus()
    nl, nz = len(campus.lines), campus.n
    m_coup = nl * 96
    lb_lp = 100 * ((bd.opt_central - bd.LB_lp) / bd.opt_central)
    lb_lag = 100 * ((bd.opt_central - bd.LB_lag) / bd.opt_central)
    lb_mip = 100 * ((bd.opt_central - bd.LB_mip) / bd.opt_central)
    best = g.sort_values("gap_pct").iloc[0]
    c1 = df[df.combo.str[2] == "1"]; c0 = df[df.combo.str[2] == "0"]
    txt = f"""# The mathematics behind Q-Grid: what is proven and what is measured

_Generated by `scripts/run_ablation.py` (numbers from `results/ablation/`); citations checked to exist._

## 1. The day problem

For one day with perfect information the campus problem is

$$\min_{{x_1..x_n}} \sum_{{i=1}}^{{n}} f_i(x_i) \quad\text{{s.t.}}\quad x_i \in X_i,\qquad f^{{(a)}}_{{l,t}} = f^{{(b)}}_{{l,t}}\ \ \forall\, l,t$$

with $n = {nz}$ buildings, $f_i$ linear (energy, diesel fuel, value of lost load, terminal value) and $X_i$ the building's
mixed-integer feasible set (battery, diesel commitment with minimum on-time, pump and tank, flexible-job starts, 15-min
slots). The only coupling is that the two ends of each of the {nl} cables agree on its flow in each of 96 slots:
$m = {m_coup}$ equality constraints. `baselines/milp.py` (B4) solves this centrally; `qgrid/ablation/bounds.py`
assembles the same problem from the per-building models (`qgrid/ablation/local.py`) and reproduces B4's optimum.

## 2. Proven

**Lower bounds.** Each of the following is a valid lower bound on the day optimum $p^\star$ (weak duality / relaxation):
- $LB_{{lp}}$: the LP relaxation (integrality dropped) — a relaxation has a feasible set containing the original one.
- $LB_{{mip}}$: the dual bound reported by branch and bound for the central MILP.
- $LB_{{lag}}(\lambda) = \sum_i \min_{{x_i\in X_i}} \big(f_i(x_i) + \lambda_i^\top f_i^{{\text{{copies}}}}\big)$ for any prices with
  $\lambda^{{(a)}}_{{l,t}} + \lambda^{{(b)}}_{{l,t}} = 0$: the Lagrangian dual function of the cable-agreement constraints. We use the
  scaled multipliers of consensus ADMM, which keep that sum at zero. Each building's term is taken as its own branch-and-bound
  dual bound, so the sum stays valid even if a building's MILP is not solved to optimality.

Therefore **certified gap** $= (J - LB)/LB$ with $LB = \max(LB_{{lp}}, LB_{{mip}}, LB_{{lag}})$ is a proven upper bound on how far a
realised schedule with cost $J$ is from the best possible schedule for that day.

**ADMM converges in the convex case.** For a convex problem (closed, proper, convex $f_i$ and a saddle point of the
Lagrangian), ADMM's residuals go to zero, the objective converges to the optimum and the dual variables converge to an optimal
dual point (Boyd, Parikh, Chu, Peleato and Eckstein, *Distributed Optimization and Statistical Learning via the Alternating
Direction Method of Multipliers*, Foundations and Trends in Machine Learning 3(1):1–122, 2011, §3.2). In design C the binaries
are fixed locally first, so the negotiated problem is a linear programme and the theorem applies. The guarantee is
asymptotic and assumes exact local solves; in practice we stop at a 1 kW residual and the local QPs are solved by OSQP to a
tolerance (measured below).

**Binary ADMM (the first prototype's coordination, no C) has no such guarantee.** With integer variables inside the negotiation the problem is non-convex;
ADMM is then a heuristic (Boyd et al. 2011, §9) and may cycle. That is why the first prototype needed bilateral resolution.

**Shapley–Folkman duality-gap bound — what it does and does not give here.** For separable problems with $m$ coupling
constraints the duality gap is bounded by a term that grows with the number of coupling constraints and the non-convexity of
the individual terms, and becomes small *relative to the objective* when the number of separable terms is much larger than $m$
(Aubin and Ekeland, *Estimates of the duality gap in nonconvex optimization*, Mathematics of Operations Research 1(3):225–245,
1976; the underlying lemma: Starr, *Quasi-equilibria in markets with non-convex preferences*, Econometrica 37(1):25–38, 1969;
a constructive bound: Udell and Boyd, *Bounding duality gap for separable problems with linear constraints*, Computational
Optimization and Applications 64(2):355–378, 2016). Here $m = {m_coup}$ coupling constraints but only $n = {nz}$ separable
terms, so the bound can involve the non-convexity of every building: **the theorem does not certify a small gap for this
campus.** The small Lagrangian gap below is a measurement, not a consequence of the theorem. (It would become meaningful in a
city-feeder setting with many more buildings than cable-slot constraints.)

## 3. Measured (30 evaluation days)

- Tightness of the bounds (mean below the central optimum): LP relaxation {lb_lp.mean():.3f} %, Lagrangian bound with ADMM
  prices {lb_lag.mean():.3f} % (max {lb_lag.max():.3f} %), branch-and-bound dual bound {lb_mip.mean():.4f} %. The certified gap is
  therefore almost equal to the true gap.
- Best combination ({'+'.join(f for f, ch in zip('ABCD', best.name) if ch == '1') or 'none'}): true gap {best.gap_pct:.2f} %, certified gap
  {best.certified_gap_pct:.2f} %.
- ADMM on the convex LP (combinations with C) reached the 1 kW tolerance in {100 * c1.admm_converged_rate.mean():.0f} % of hourly
  plans; binary ADMM (without C) agreed within its 6 rounds in {100 * c0.admm_converged_rate.mean():.0f} % (it then rounds and fixes
  the consensus).
"""
    (ROOT / "docs" / "math.md").write_text(txt, encoding="utf-8")
    print("wrote docs/math.md")


def main(workers: int, ndays: int | None):
    c = _ctx()
    meta = c["proc"]["meta"]
    days = meta["days"]["eval_days"][:ndays] if ndays else meta["days"]["eval_days"]
    OUT.mkdir(parents=True, exist_ok=True)
    fc = _forecaster()                                     # trains and caches once before the workers start
    from qgrid.ablation.forecast import forecast_errors
    json.dump(forecast_errors(fc, days), open(OUT / "forecast_mae.json", "w", encoding="utf-8"), indent=1)
    heavy = lambda code: (code[0] == "1") * (2 if code[2] == "0" else 1) * (2 if code[1] == "0" else 1)   # noqa: E731
    hs = meta["days"]["scenarios"][c["campus"].cfg["headline_scenario"]]
    stress = [(hs["date"], hs["outage"])]                 # outage stress day, reported separately
    tasks = [("combo", d, code, None) for d in days for code in COMBOS] + [("bounds", d, None) for d in days]
    tasks += [("combo", d, code, o) for d, o in stress for code in COMBOS] + [("bounds", d, o) for d, o in stress]
    tasks.sort(key=lambda t: -heavy(t[2]) if t[0] == "combo" else 0)
    rows, bounds = [], []
    if RETRY:                                  # keep finished rows, re-run only the missing combos
        old = pd.read_csv(OUT / "ablation.csv", dtype={"combo": str})
        old["combo"] = old.combo.str.zfill(4)
        done = {(r.day, r.combo, r.set) for r in old.itertuples()}
        tasks = [t for t in tasks if t[0] == "combo" and (t[1], t[2], "outage" if t[3] else "eval30") not in done]
        rows = old.drop(columns=[c for c in ("opt", "LB", "gap_pct", "certified_gap_pct") if c in old]).to_dict("records")
        bounds = pd.read_csv(OUT / "bounds.csv").to_dict("records")
        print(f"retrying {len(tasks)} missing runs", flush=True)
    t_all = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_task, *t): t for t in tasks}
        for k, fu in enumerate(as_completed(futs), 1):
            t = futs[fu]
            try:
                kind, r = fu.result()
            except Exception as e:
                print(f"FAILED {t}: {e!r}", flush=True)
                continue
            (rows if kind == "combo" else bounds).append(r)
            if kind == "combo":
                print(f"[{k}/{len(tasks)}] {r['day']} {r['combo']} obj {r['objective_rs']:,.0f} {r['wall_s']:.0f}s", flush=True)
            if k % 40 == 0 or k == len(tasks):
                pd.DataFrame(rows).to_csv(OUT / "ablation.csv", index=False)
                pd.DataFrame(bounds).to_csv(OUT / "bounds.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "ablation.csv", index=False)
    pd.DataFrame(bounds).to_csv(OUT / "bounds.csv", index=False)
    json.dump({"runtime_s": time.time() - t_all, "workers": workers, "days": len(days), "combos": len(COMBOS)},
              open(OUT / "run_info.json", "w", encoding="utf-8"))
    report()


RETRY = "--retry-missing" in sys.argv

if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    else:
        from qgrid.parallel import workers as _workers
        w = _workers(next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--workers=")), None))
        nd = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--days=")), None)
        main(w, nd)
