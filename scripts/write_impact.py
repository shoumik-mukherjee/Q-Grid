"""Is it worth it? Cost per building, payback, 10-year NPV with a Monte Carlo range, and the scale path.

  python scripts/write_impact.py   -> results/impact/impact.json, docs/impact.md, results/replay/impact.json (+ frontend)

Benefits come from the scorecard benchmark (results/bench_v3/daily.csv, RVCE days, Q-Grid vs status quo on the same
days): the bill saving and the extra battery throughput. Costs are the labelled assumptions in config.yaml -> impact.
Solar (500 kWp) and the diesel sets exist at RVCE; the batteries, tie-line cables and EV-charging load are PROPOSED (Q-Grid
design) and their purchase cost is NOT in this NPV, which covers the controller hardware only; the extra battery wear
from cycling them is a cost and is counted (net value). The tariff label comes from the data status (qgrid.provenance);
every cost is an ASSUMPTION until vendor quotes arrive.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT)]
from qgrid import provenance as _prov  # noqa: E402


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    im = cfg["impact"]
    from qgrid.model import load_campus
    campus = load_campus()
    names = campus.names
    kwh = np.array([b.battery.kwh if b.battery.present else 0.0 for b in campus.buildings])
    cap = float(kwh.sum())
    d = pd.read_csv(ROOT / "results" / "bench_v3" / "daily.csv")
    rv = d[d.dataset == "rvce"]
    q = rv[rv.method == "qgrid"].set_index("day").sort_index()
    sq = rv[rv.method == "status_quo"].set_index("day").loc[q.index]
    sav = (sq.bill_rs - q.bill_rs).to_numpy()                                   # Rs/day, paired days
    dthr = ((q.battery_cycles - sq.battery_cycles) * 2 * cap).to_numpy()       # extra kWh charged + discharged per day
    n_days = len(sav)
    pb = im["per_building"]
    lo = sum(v[0] for v in pb.values()); hi = sum(v[1] for v in pb.values())
    nb = campus.n
    years = int(im["years"])
    dod = float(im["battery_dod"])

    def annuity(r):
        return sum(1 / (1 + r) ** y for y in range(1, years + 1))

    # ---- central case (mid-points)
    mid = lambda v: 0.5 * (v[0] + v[1])   # noqa: E731
    hw_mid = sum(mid(v) for v in pb.values()) * nb
    om_mid = mid(im["om_share_per_year"]) * hw_mid
    r_mid = mid(im["discount_rate"])
    # central battery wear = the rate every optimiser prices (config economics + battery_wear), so the impact page and the
    # scorecard agree; the Monte Carlo varies it over the impact ranges
    eco = cfg["economics"]
    wear_kwh_mid = float(eco["battery_capex_rs_per_kwh"]) / (2 * float(eco["battery_cycle_life"]) * dod)
    S = sav.mean() * 365; W = dthr.mean() * 365 * wear_kwh_mid
    central = {"hardware_rs": hw_mid, "om_rs_per_year": om_mid, "bill_saving_rs_per_year": S, "battery_wear_rs_per_year": W,
               "net_rs_per_year": S - om_mid - W, "discount_rate": r_mid,
               "npv_gross_rs": -hw_mid + (S - om_mid) * annuity(r_mid), "npv_net_rs": -hw_mid + (S - om_mid - W) * annuity(r_mid),
               "payback_gross_years": hw_mid / (S - om_mid) if S - om_mid > 0 else None,
               "payback_net_years": hw_mid / (S - om_mid - W) if S - om_mid - W > 0 else None,
               "wear_rs_per_kwh_throughput": wear_kwh_mid}
    # break-even battery wear cost per kWh of throughput (net = 0 at the central costs)
    central["breakeven_wear_rs_per_kwh"] = (S - om_mid - hw_mid / annuity(r_mid)) / (dthr.mean() * 365) if dthr.mean() > 0 else None
    # ---- Monte Carlo
    mc = im["monte_carlo"]
    rng = np.random.default_rng(mc["seed"])
    N = int(mc["samples"])
    U = lambda v, n=N: rng.uniform(v[0], v[1], n)   # noqa: E731
    idx = rng.integers(0, n_days, (N, n_days))
    Sd = sav[idx].mean(1) * 365; Td = dthr[idx].mean(1) * 365                  # bootstrap over days (paired)
    hw = sum(U(v) for v in pb.values()) * nb
    om = U(im["om_share_per_year"]) * hw
    r = U(im["discount_rate"])
    wear = U(im["battery_capex_rs_per_kwh"]) / (2 * U(im["battery_cycle_life"]) * dod)
    A = np.array([annuity(x) for x in r])
    npv_g = -hw + (Sd - om) * A
    npv_n = -hw + (Sd - om - Td * wear) * A
    pct = lambda x: {"p5": float(np.percentile(x, 5)), "p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)),   # noqa: E731
                     "prob_positive": float((x > 0).mean())}
    payg = np.where(Sd - om > 0, hw / np.maximum(Sd - om, 1e-9), np.inf)
    # ---- per building
    per_b = [{"zone": z, "battery_kwh": float(k), "hardware_rs": [lo, hi],
              "battery_value_rs": [float(k) * im["battery_capex_rs_per_kwh"][0], float(k) * im["battery_capex_rs_per_kwh"][1]]}
             for z, k in zip(names, kwh)]
    # ---- scale path evidence from the benchmark (other campuses)
    s = pd.read_csv(ROOT / "results" / "bench_v3" / "summary.csv")
    from qgrid.datasets import SPECS
    ds_rows = []
    for ds in [k for k in SPECS if k in set(s.dataset)]:
        row = s[(s.dataset == ds) & (s.method == "qgrid")]
        if len(row):
            row = row.iloc[0]
            ds_rows.append({"dataset": ds, "label": SPECS[ds].label, "gap_pct": float(row.gap_pct__mean),
                            "saving_pct": float(row.saving_pct__mean), "runtime_per_decision_s": float(row.runtime_per_decision_s__mean),
                            "messages_per_decision": float(row.messages_per_decision__mean)})
    # ---- design change under test: wear priced in the objective (scripts/run_wear_variant.py), same days and draws
    variant = None
    wv = ROOT / "results" / "impact" / "wear_variant.csv"   # only for a design not yet in the scorecard
    if wv.exists():
        v = pd.read_csv(wv).set_index("day")
        days = q.index.intersection(v.index)
        if len(days) == n_days:
            vs = (sq.loc[days].bill_rs - v.loc[days].cost_incl_demand_rs).to_numpy()
            vt = ((v.loc[days].battery_cycles - sq.loc[days].battery_cycles) * 2 * cap).to_numpy()
            Sv, Tv = vs[idx].mean(1) * 365, vt[idx].mean(1) * 365
            nv = -hw + (Sv - om - Tv * wear) * A
            Svm, Wvm = vs.mean() * 365, vt.mean() * 365 * wear_kwh_mid
            variant = {"name": "Q-Grid with battery wear priced in its objective (design change under test, not in the scorecard)",
                       "wear_rs_per_kwh_in_objective": float(v.wear_rs_per_kwh.iloc[0]), "n_days": int(len(days)),
                       "bill_saving_rs_per_day_mean": float(vs.mean()), "extra_throughput_kwh_per_day_mean": float(vt.mean()),
                       "critical_downtime_h": float(v.loc[days].critical_downtime_h.mean()),
                       "net_rs_per_year": float(Svm - om_mid - Wvm),
                       "npv_net_rs": float(-hw_mid + (Svm - om_mid - Wvm) * annuity(r_mid)),
                       "payback_net_years": float(hw_mid / (Svm - om_mid - Wvm)) if Svm - om_mid - Wvm > 0 else None,
                       "mc_npv_net_rs": pct(nv), "source": "results/impact/wear_variant.csv"}
    out = {"status": f"ASSUMPTION costs; {_prov.tariff_label(cfg)}; PROPOSED batteries not costed", "n_buildings": nb, "n_days": n_days,
           "variant": variant,
           "battery_kwh_total": cap,
           "benefit": {"bill_saving_rs_per_day_mean": float(sav.mean()), "bill_saving_rs_per_day_ci95": [float(np.percentile(sav[idx].mean(1), 2.5)), float(np.percentile(sav[idx].mean(1), 97.5))],
                       "extra_throughput_kwh_per_day_mean": float(dthr.mean()), "source": "results/bench_v3/daily.csv (RVCE, Q-Grid vs status quo, same days)"},
           "central": central,
           "monte_carlo": {"samples": N, "seed": mc["seed"], "npv_gross_rs": pct(npv_g), "npv_net_rs": pct(npv_n),
                           "payback_gross_years": {"p5": float(np.percentile(payg, 5)), "p50": float(np.percentile(payg, 50)), "p95": float(np.percentile(payg, 95))}},
           "per_building": per_b,
           "assumptions": [
               {"name": "Edge controller per building", "value": pb["edge_controller_rs"], "unit": "Rs", "source": "ASSUMPTION — quote PENDING"},
               {"name": "Energy meter per building", "value": pb["energy_meter_rs"], "unit": "Rs", "source": "ASSUMPTION — quote PENDING"},
               {"name": "Network per building", "value": pb["network_rs"], "unit": "Rs", "source": "ASSUMPTION — quote PENDING"},
               {"name": "Integration per building", "value": pb["integration_rs"], "unit": "Rs", "source": "ASSUMPTION — quote PENDING"},
               {"name": "Maintenance per year", "value": [100 * x for x in im["om_share_per_year"]], "unit": "% of hardware", "source": "ASSUMPTION"},
               {"name": "Discount rate", "value": [100 * x for x in im["discount_rate"]], "unit": "%", "source": "ASSUMPTION"},
               {"name": "Battery cost", "value": im["battery_capex_rs_per_kwh"], "unit": "Rs/kWh", "source": "ASSUMPTION (config.yaml economics / impact)"},
               {"name": "Battery cycle life", "value": im["battery_cycle_life"], "unit": "full cycles", "source": "ASSUMPTION (LFP datasheet class)"},
               {"name": "Depth of discharge per cycle", "value": [100 * dod, 100 * dod], "unit": "%", "source": "ASSUMPTION"},
               {"name": "Tariff (energy + demand charge)", "value": None, "unit": "", "source": _prov.tariff_label(cfg)},
               {"name": "Batteries, solar, diesel sets", "value": None, "unit": "", "source": "already in the campus model (sizes ASSUMPTION until RVCE asset data)"},
           ],
           "scale": [
               {"stage": "RVCE pilot", "what": f"{nb} buildings, one node each", "evidence": "this page"},
               {"stage": "Bengaluru campuses", "what": "the same node per building; no campus-wide server", "evidence": "the benchmark's four other campuses", "datasets": ds_rows},
               {"stage": "Feeders", "what": "campuses and buildings on one distribution feeder agree the same way, with neighbours only",
                "evidence": "per-decision work and messages depend on a building's neighbours, not on the size of the network (consensus ADMM); not yet tested on a feeder model"},
           ]}
    rd = ROOT / "results" / "impact"; rd.mkdir(parents=True, exist_ok=True)
    (rd / "impact.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    rp = ROOT / "results" / "replay" / "impact.json"
    shutil.copy(rd / "impact.json", rp); shutil.copy(rp, ROOT / "frontend" / "public" / "replay" / "impact.json")
    c = central
    lakh = lambda x: f"Rs {x / 1e5:,.1f} lakh"   # noqa: E731
    mcn, mcg = out["monte_carlo"]["npv_net_rs"], out["monte_carlo"]["npv_gross_rs"]
    L = ["# Is it worth it? (impact)\n",
         "_Generated by `scripts/write_impact.py` from `results/bench_v3/daily.csv` and `config.yaml` → `impact`. "
         "Every cost is an ASSUMPTION (vendor quotes PENDING); " + _prov.data_sentence(cfg) + " The PROPOSED batteries' purchase cost is not included._\n",
         "## Central case (mid-point assumptions)\n",
         f"- Hardware for {nb} building nodes: {lakh(c['hardware_rs'])}; maintenance {lakh(c['om_rs_per_year'])} per year.",
         f"- Bill saving vs today: {lakh(c['bill_saving_rs_per_year'])} per year (mean of {n_days} RVCE days × 365).",
         f"- Extra battery wear from cycling the existing batteries: {lakh(c['battery_wear_rs_per_year'])} per year "
         f"({out['benefit']['extra_throughput_kwh_per_day_mean']:,.0f} kWh more throughput per day × Rs {c['wear_rs_per_kwh_throughput']:.2f}/kWh).",
         f"- Net value per year: {lakh(c['net_rs_per_year'])}.",
         f"- 10-year NPV at {100 * c['discount_rate']:.0f} %: {lakh(c['npv_gross_rs'])} before battery wear, {lakh(c['npv_net_rs'])} after.",
         f"- Payback: {c['payback_gross_years']:.1f} years before battery wear" if c["payback_gross_years"] else "- Payback: not reached before battery wear",
         (f"; {c['payback_net_years']:.1f} years after." if c["payback_net_years"] else "; not reached after battery wear."),
         (f"- Break-even battery wear: Rs {c['breakeven_wear_rs_per_kwh']:.2f} per kWh of throughput (the central assumption is "
          f"Rs {c['wear_rs_per_kwh_throughput']:.2f})." if c["breakeven_wear_rs_per_kwh"] is not None else ""),
         "",
         f"## Monte Carlo ({N:,} samples, seed {mc['seed']}): days bootstrapped, costs uniform in their ranges\n",
         "| 10-year NPV | P5 | median | P95 | chance it is positive |", "|---|---:|---:|---:|---:|",
         f"| before battery wear | {lakh(mcg['p5'])} | {lakh(mcg['p50'])} | {lakh(mcg['p95'])} | {100 * mcg['prob_positive']:.0f} % |",
         f"| after battery wear | {lakh(mcn['p5'])} | {lakh(mcn['p50'])} | {lakh(mcn['p95'])} | {100 * mcn['prob_positive']:.0f} % |",
         *([ "", "## Design change under test: battery wear in the objective\n",
             f"Same {variant['n_days']} days, with each building pricing wear at Rs {variant['wear_rs_per_kwh_in_objective']:.2f} per kWh "
             f"charged or discharged (`scripts/run_wear_variant.py`): bill saving Rs {variant['bill_saving_rs_per_day_mean']:,.0f}/day, "
             f"extra throughput {variant['extra_throughput_kwh_per_day_mean']:,.0f} kWh/day, critical downtime {variant['critical_downtime_h']:.2f} h/day; "
             f"net {lakh(variant['net_rs_per_year'])} per year, 10-year NPV after wear {lakh(variant['npv_net_rs'])} "
             f"(Monte Carlo P5–P95 {lakh(variant['mc_npv_net_rs']['p5'])} to {lakh(variant['mc_npv_net_rs']['p95'])}, "
             f"{100 * variant['mc_npv_net_rs']['prob_positive']:.0f} % positive). Not part of the scorecard."] if variant else []),
         "", "## What this means\n",
         "- Every optimiser in the scorecard prices battery wear in its objective (`config.yaml` → `battery_wear`), so the "
         "batteries cycle only when a price spread beats the wear. An earlier run without that term cycled the batteries "
         "for small spreads and lost money after wear (`results/impact/wear_pilot_2.96.csv` documents the pilot that "
         "led to the change).",
         "- Reliability (critical downtime) and resilience to a network failure are benefits not priced here.",
         "", "## Assumptions\n", "| Assumption | Value | Source |", "|---|---|---|"]
    for a in out["assumptions"]:
        v = "–" if a["value"] is None else (f"{a['value'][0]:,.0f}–{a['value'][1]:,.0f} {a['unit']}" if a["value"][0] != a["value"][1] else f"{a['value'][0]:,.0f} {a['unit']}")
        L.append(f"| {a['name']} | {v} | {a['source']} |")
    L += ["", "## Scale path\n"]
    for st in out["scale"]:
        L.append(f"- **{st['stage']}**: {st['what']} ({st['evidence']}).")
        for r_ in st.get("datasets", []):
            L.append(f"  - {r_['label']}: gap to the best plan {r_['gap_pct']:.1f} %, bill change {r_['saving_pct']:+.1f} % (RVCE KERC tariff for comparability), "
                     f"{r_['runtime_per_decision_s']:.1f} s per decision")
    (ROOT / "docs" / "impact.md").write_text("\n".join(x for x in L if x is not None) + "\n", encoding="utf-8")
    print(json.dumps(central, indent=1))
    print("MC net", mcn, "gross", mcg)


if __name__ == "__main__":
    main()
