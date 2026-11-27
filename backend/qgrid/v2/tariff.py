"""Tariff v2: energy charges (ToD, as in price scenario A) plus a demand charge on the campus peak kVA.

The campus is one HT connection: peak kVA = max over the day's grid-connected slots of the campus import
(sum over zones) / power factor. BESCOM bills the demand charge monthly on the recorded maximum demand; daily
figures here carry 1/days_per_month of that monthly charge on the day's peak (ASSUMPTION, labelled).
The rate comes from config.yaml tariff_v2: REAL = KERC Tariff Order 2023, BESCOM HT-2(c)(ii), Rs 350/kVA/month
(page references in config.yaml and data/SOURCES.md).
"""
from __future__ import annotations

from ..model import Campus, DayData, Schedule, settle


def demand_params(campus: Campus) -> dict:
    tv = campus.cfg["tariff_v2"]
    return {"rate_day": tv["demand_rs_per_kva_month"] / tv["days_per_month"], "pf": float(tv["power_factor"]),
            "rate_month": float(tv["demand_rs_per_kva_month"]), "status": tv["status"]}


def settle_v2(campus: Campus, day: DayData, sched: Schedule) -> dict:
    m = settle(campus, day, sched)
    dp = demand_params(campus)
    peak_kva = m["peak_import_kw"] / dp["pf"]
    dc = dp["rate_day"] * peak_kva
    m.update({"peak_kva": peak_kva, "demand_charge_rs": dc, "demand_charge_month_rs": dp["rate_month"] * peak_kva,
              "cost_incl_demand_rs": m["cost_rs"] + dc, "objective_incl_demand_rs": m["objective_rs"] + dc,
              "tariff_v2_status": dp["status"]})
    return m
