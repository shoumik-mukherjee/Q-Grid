"""Tariff v2: energy charges (ToD, as in price scenario A) plus a demand charge on the campus peak kVA.

The campus is one HT connection: peak kVA = max over the day's grid-connected slots of the campus import
(sum over zones) / power factor. BESCOM bills the demand charge monthly on the recorded maximum demand; daily
figures here carry 1/days_per_month of that monthly charge on the day's peak (ASSUMPTION, labelled).
The rate is date-dependent (config.yaml prices.A.periods, REAL): BESCOM HT-2(c)(ii) Rs 265/kVA/month before
2023-04-01 (KERC Tariff Order 2022), Rs 350/kVA/month from 2023-04-01 (KERC Tariff Order 2023); page references in
config.yaml and data/SOURCES.md.
"""
from __future__ import annotations

from .. import data as qdata
from ..model import Campus, DayData, Schedule, settle


def demand_params(campus: Campus, day: str) -> dict:
    """Demand charge in force on `day` (date-dependent: config.yaml prices.A.periods, see qgrid.data.tariff_period)."""
    tv = campus.cfg["tariff_v2"]
    rate = qdata.demand_rate_month(campus.cfg, day)
    return {"rate_day": rate / tv["days_per_month"], "pf": float(tv["power_factor"]),
            "rate_month": rate, "status": tv["status"]}


def settle_v2(campus: Campus, day: DayData, sched: Schedule) -> dict:
    m = settle(campus, day, sched)
    dp = demand_params(campus, day.day)
    peak_kva = m["peak_import_kw"] / dp["pf"]
    dc = dp["rate_day"] * peak_kva
    m.update({"peak_kva": peak_kva, "demand_charge_rs": dc, "demand_charge_month_rs": dp["rate_month"] * peak_kva,
              "cost_incl_demand_rs": m["cost_rs"] + dc, "objective_incl_demand_rs": m["objective_rs"] + dc,
              "tariff_v2_status": dp["status"]})
    return m
