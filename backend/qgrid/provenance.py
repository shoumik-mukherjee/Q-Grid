"""Provenance labels for generated documents, read from the data status (never hard-coded).

Every generated document (benchmark, scorecard, impact) says where the tariff and the CO2 factor come from. These
helpers build that wording from config.yaml (prices.A / tariff_v2 status) and data/raw/cea/emission_factor.yaml, so the
documents switch between "REAL" and "PENDING stand-in" when the data does.
"""
from __future__ import annotations

from . import data as qdata


def tariff_real(cfg: dict) -> bool:
    return cfg["prices"]["A"]["status"] == "REAL" and cfg.get("tariff_v2", {}).get("status") == "REAL"


def tariff_label(cfg: dict) -> str:
    """Short label for the energy tariff + demand charge (price scenario A)."""
    if not tariff_real(cfg):
        return "stand-in tariff (PENDING)"
    names = [p["name"] for p in cfg["prices"]["A"].get("periods", [])]
    return f"KERC tariff, BESCOM HT-2(c)(ii) ({' / '.join(names)}) (REAL)" if names else "KERC tariff (REAL)"


def co2_label(cfg: dict) -> str:
    c = qdata.load_co2_factor(cfg)
    if c["status"] != "REAL":
        return "CO₂ factor stand-in (PENDING)"
    return f"CEA CO₂ factor v{c.get('version', '?')} (FY{c.get('fiscal_year', '?')}, REAL)"


def demand_charge_note(cfg: dict) -> str:
    return (f"the demand charge is counted as 1/{cfg['tariff_v2']['days_per_month']} of the monthly charge on each day's peak "
            "(ASSUMPTION; monthly billing demand with the 85 % contract-demand floor is not yet modelled)")


def data_sentence(cfg: dict) -> str:
    """One sentence for a document header: tariff, CO2, outages, and whether Rs figures may be read as results."""
    t, c = tariff_label(cfg), co2_label(cfg)
    tail = ("Rs figures are not headline results." if not tariff_real(cfg)
            else "Price scenario B (IEX) is still a stand-in and is not used here.")
    return f"Tariff: {t}; {c}; {demand_charge_note(cfg)}; outages are SYNTHETIC (seeded). {tail}"
