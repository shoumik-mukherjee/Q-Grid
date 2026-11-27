"""Fetch / verify raw data for Q-Grid into data/raw/.

Automatic:  NASA POWER hourly irradiance + temperature (public API, no key).
Manual:     Ausgrid Solar Home, IEX DAM prices, CEA CO2 baseline: these sites block scripted
            downloads or render via JavaScript, so this script checks for the files and prints
            exact instructions when they are missing (exit code 2).

Usage:  python scripts/fetch_data.py [--year 2023]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

LAT, LON = 12.92, 77.50  # Bengaluru campus site

NASA_URL = (
    "https://power.larc.nasa.gov/api/temporal/hourly/point"
    "?parameters=ALLSKY_SFC_SW_DWN,T2M&community=RE&longitude={lon}&latitude={lat}"
    "&start={y}0101&end={y}1231&format=JSON&time-standard=LST"
)


def fetch_nasa(year: int) -> Path:
    out = RAW / "nasa_power" / f"bengaluru_{year}_hourly.json"
    if out.exists():
        print(f"[nasa]    cached {out.relative_to(ROOT)}")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    url = NASA_URL.format(lat=LAT, lon=LON, y=year)
    print(f"[nasa]    GET {url}")
    r = requests.get(url, timeout=120, headers={"User-Agent": "qgrid-data-fetch/0.1"})
    r.raise_for_status()
    data = r.json()
    params = data["properties"]["parameter"]
    assert {"ALLSKY_SFC_SW_DWN", "T2M"} <= set(params), params.keys()
    data["_qgrid_fetch"] = {"url": url, "fetched_utc": dt.datetime.now(dt.UTC).isoformat()}
    out.write_text(json.dumps(data), encoding="utf-8")
    print(f"[nasa]    saved {out.relative_to(ROOT)} ({len(params['T2M'])} hours)")
    return out


NASA_CS_URL = NASA_URL.replace("ALLSKY_SFC_SW_DWN,T2M", "CLRSKY_SFC_SW_DWN")


def fetch_nasa_clearsky(year: int) -> Path:
    """Clear-sky surface shortwave (NASA POWER CLRSKY_SFC_SW_DWN), used by the v2 solar forecast."""
    out = RAW / "nasa_power" / f"bengaluru_{year}_hourly_clearsky.json"
    if out.exists():
        print(f"[nasa-cs] cached {out.relative_to(ROOT)}")
        return out
    url = NASA_CS_URL.format(lat=LAT, lon=LON, y=year)
    print(f"[nasa-cs] GET {url}")
    r = requests.get(url, timeout=120, headers={"User-Agent": "qgrid-data-fetch/0.1"})
    r.raise_for_status()
    data = r.json()
    assert "CLRSKY_SFC_SW_DWN" in data["properties"]["parameter"]
    data["_qgrid_fetch"] = {"url": url, "fetched_utc": dt.datetime.now(dt.UTC).isoformat()}
    out.write_text(json.dumps(data), encoding="utf-8")
    print(f"[nasa-cs] saved {out.relative_to(ROOT)}")
    return out


NASA_SITE_URL = (
    "https://power.larc.nasa.gov/api/temporal/hourly/point"
    "?parameters=ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN,T2M&community=RE&longitude={lon}&latitude={lat}"
    "&start={y}0101&end={y}1231&format=JSON&time-standard=LST"
)


def fetch_nasa_site(name: str, lat: float, lon: float, year: int) -> Path:
    """NASA POWER hourly irradiance (all-sky, clear-sky) and temperature for another benchmark site."""
    out = RAW / "nasa_power" / f"{name}_{year}_hourly.json"
    if out.exists():
        print(f"[nasa]    cached {out.relative_to(ROOT)}")
        return out
    url = NASA_SITE_URL.format(lat=round(lat, 3), lon=round(lon, 3), y=year)
    print(f"[nasa]    GET {url}")
    r = requests.get(url, timeout=180, headers={"User-Agent": "qgrid-data-fetch/0.1"})
    r.raise_for_status()
    data = r.json()
    assert {"ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN", "T2M"} <= set(data["properties"]["parameter"])
    data["_qgrid_fetch"] = {"url": url, "fetched_utc": dt.datetime.now(dt.UTC).isoformat()}
    out.write_text(json.dumps(data), encoding="utf-8")
    return out


BENCH_SITES = {   # benchmark v3 datasets: BDG2 university sites (site coordinates from BDG2 metadata) and CU-BEMS (Bangkok)
    "bdg2_fox": (33.424425, -111.928140, 2017), "bdg2_bull": (30.2672, -97.7431, 2017),
    "bdg2_cockatoo": (42.459837, -76.485292, 2017), "cubems": (13.7367, 100.5332, 2019),
}


BDG2_FILES = {
    "LICENSE": "https://raw.githubusercontent.com/buds-lab/building-data-genome-project-2/master/LICENSE",
    "metadata.csv": "https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/master/data/metadata/metadata.csv",
    "electricity_cleaned.csv": "https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/master/data/meters/cleaned/electricity_cleaned.csv",
}


def fetch_bdg2() -> Path:
    """Building Data Genome Project 2 (CC BY-SA 4.0): education-campus hourly electricity meters, 2016-17."""
    d = RAW / "bdg2"
    d.mkdir(parents=True, exist_ok=True)
    for name, url in BDG2_FILES.items():
        out = d / name
        if out.exists() and out.stat().st_size > 1000:
            print(f"[bdg2]    cached {out.relative_to(ROOT)}")
            continue
        print(f"[bdg2]    GET {url}")
        r = requests.get(url, timeout=1200, headers={"User-Agent": "qgrid-data-fetch/0.1"})
        r.raise_for_status()
        out.write_bytes(r.content)
    return d


MANUAL = {
    "ausgrid": dict(
        dir=RAW / "ausgrid",
        patterns=["*.zip", "*.csv"],
        instructions="""\
Ausgrid Solar Home Electricity Data (2012-13, half-hourly, 300 homes, Sydney).
Ausgrid removed the original page; the dataset is mirrored by a researcher:
  1. Download https://pierreh.eu/downloads/Ausgrid_solar_home_data.zip
     (or any copy of '2012-2013 Solar home electricity data v2.zip' / the CSV
      'Solar-home-half-hour-data---1-July-2012-to-30-June-2013').
  2. Put the .zip (or extracted .csv) in data/raw/ausgrid/
Licence: as published by Ausgrid for research use; cite Ratnam et al. 2017,
doi:10.1080/14786451.2015.1100196.""",
    ),
    "iex": dict(
        dir=RAW / "iex",
        patterns=["*.xlsx", "*.xls", "*.csv"],
        instructions="""\
IEX Day-Ahead Market (DAM) 15-minute market clearing prices, calendar year {year}.
  1. Open https://www.iexindia.com/market-data/day-ahead-market/market-snapshot
  2. Set Interval = 15 Min Block, Delivery period = a date range (the site limits
     ranges; download month by month if needed), Area = MCP (all-India).
  3. Click Export -> Excel. Repeat to cover 01-Jan-{year} .. 31-Dec-{year}.
  4. Put all exported files in data/raw/iex/  (any filenames).""",
    ),
    "tariff": dict(
        dir=RAW / "tariff",
        patterns=["*.pdf"],
        instructions="""\
KERC tariff order (scenario A, headline): BESCOM HT educational-institution tariff with ToD,
and the KERC solar/rooftop export tariff order.
  1. Open https://kerc.karnataka.gov.in/  -> Tariff -> Tariff Orders -> the order
     applicable to FY2023-24 (BESCOM), and the KERC order on solar rooftop/export tariff.
     (BESCOM also mirrors tariff orders at https://bescom.karnataka.gov.in/ -> Tariff.)
  2. Save as data/raw/tariff/KERC_tariff_order_BESCOM_FY2023-24.pdf and
     data/raw/tariff/KERC_solar_export_tariff.pdf
  3. Copy the HT educational energy charge, ToD adders and export rate into config.yaml
     prices.A (with page numbers in data/SOURCES.md) and set prices.A.status: REAL.""",
    ),
    "cea": dict(
        dir=RAW / "cea",
        patterns=["*.pdf"],
        instructions="""\
CEA CO2 Baseline Database for the Indian Power Sector (latest User Guide PDF).
  1. Open https://cea.nic.in/  -> Reports -> 'CO2 Baseline Database'
     (cea.nic.in blocks scripted downloads).
  2. Download the latest 'User Guide Version NN.0' PDF.
  3. Put it in data/raw/cea/. The value is then read from the PDF into
     data/raw/cea/emission_factor.yaml (version, fiscal_year,
     weighted_average_tco2_per_mwh, page), checked by hand against the PDF.""",
    ),
}


def check_manual(name: str, year: int) -> bool:
    spec = MANUAL[name]
    d: Path = spec["dir"]
    d.mkdir(parents=True, exist_ok=True)
    found = [p for pat in spec["patterns"] for p in d.glob(pat)]
    if name == "cea":
        ok = bool(found) and (d / "emission_factor.yaml").exists()
    else:
        ok = bool(found)
    if ok:
        print(f"[{name:7s}] found {len(found)} file(s) in {d.relative_to(ROOT)}")
    else:
        print(f"\n[{name:7s}] MISSING — manual step required:\n")
        print("  " + spec["instructions"].format(year=year).replace("\n", "\n  ") + "\n")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2023)
    args = ap.parse_args()
    fetch_nasa(args.year)
    fetch_nasa_clearsky(args.year)
    fetch_bdg2()
    for nm, (la, lo, yr) in BENCH_SITES.items():
        fetch_nasa_site(nm, la, lo, yr)
    ok = all([check_manual(n, args.year) for n in MANUAL])
    if not ok:
        print("Some raw data is missing; complete the manual steps above, then re-run.")
        return 2
    print("All raw data present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
