# Data sources

Status legend: **REAL** = official/primary data in `data/raw/` and used for results ·
**PENDING** = file not yet obtained; a clearly labelled STAND-IN keeps the pipeline runnable and
**no stand-in value is ever presented as a headline result**. After adding the missing files run
`scripts/refresh_all.sh`.

| Dataset | Status | Used for | Raw file(s) |
|---|---|---|---|
| NASA POWER hourly (ALLSKY_SFC_SW_DWN, T2M), 12.92N 77.50E, 2023 | **REAL** | PV output, sun position | `data/raw/nasa_power/bengaluru_2023_hourly.json` |
| NASA POWER hourly clear-sky (CLRSKY_SFC_SW_DWN), same site, 2023 | **REAL** | v2 solar forecast (clear-sky envelope) | `data/raw/nasa_power/bengaluru_2023_hourly_clearsky.json` |
| Building Data Genome Project 2 (BDG2): hourly electricity meters, 1,578 non-residential buildings, 2016-17 | **REAL** | zone load shapes (7 of 9 zones) | `data/raw/bdg2/electricity_cleaned.csv`, `metadata.csv`, `LICENSE` |
| Ausgrid Solar Home Electricity Data 2012-13 (half-hourly, 300 homes) | REAL, **superseded by BDG2** (used only if `loads.source: templates`) | hostel load shape (v1) | `data/raw/ausgrid/Solar home 2012-2013.csv` (+ archive zip, notes PDF) |
| KERC Tariff Orders 2022 (FY2022-23) and 2023 (FY2023-24), BESCOM HT-2(c)(ii) energy + demand charge + HT ToD, by date; KERC FY24 solar tariff order (export) | **REAL** | price scenario A (headline), demand charge (`tariff_v2`) | `data/raw/tariff/KERC_tariff_order_BESCOM_FY2022-23.pdf`, `KERC_tariff_order_BESCOM_FY2023-24.pdf`, `KERC_solar_export_tariff.pdf` |
| IEX Day-Ahead Market 15-min MCP, 2023 | **PENDING** | price scenario B | `data/raw/iex/` |
| CEA CO2 Baseline Database User Guide v20.0 (weighted average, FY2023-24) | **REAL** | CO2 accounting | `data/raw/cea/CEA_CO2_Baseline_User_Guide_v20.0_Dec2024.pdf`, `emission_factor.yaml` |
| OpenStreetMap: RVCE campus outline + building footprints | **REAL** | zones, roof area → PV kWp, tie-line layout, 3D campus | `data/raw/osm_rvce.json`, `osm_rvce.geojson` |
| RVCE academic calendar | **PENDING** (fest dates sourced) | day types / scenario dates | `data/raw/rvce_calendar.csv` (optional) |
| RVCE BESCOM bills / meter data | **PENDING RVCE DATA** | campus load totals | `data/raw/rvce_bills.csv` (optional) |
| Faculty lab schedule | **TEMPLATE** | lab loads, critical labs | `data/raw/lab_schedule.csv` |

## NASA POWER — REAL
- URL: `https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=ALLSKY_SFC_SW_DWN,T2M&community=RE&longitude=77.5&latitude=12.92&start=20230101&end=20231231&format=JSON&time-standard=LST`
- Fetched: 2026-11-21 by `scripts/fetch_data.py` (timestamp stored inside the JSON under `_qgrid_fetch`).
- Licence: NASA POWER data are freely available without restriction (NASA open data policy); cite
  "NASA Langley Research Center POWER Project".
- Processing: hourly means in local solar time repeated to 4 x 15-min slots (energy preserving);
  PV = kWp · G/1000 · (1 + γ(T_cell − 25)) · derate, T_cell = T_air + (NOCT − 20)/800 · G
  (γ = −0.4 %/°C, NOCT 45 °C, derate 0.86 — config.yaml).

## NASA POWER clear-sky — REAL
- URL: `https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=CLRSKY_SFC_SW_DWN&community=RE&longitude=77.5&latitude=12.92&start=20230101&end=20231231&format=JSON&time-standard=LST`
- Fetched: 2026-11-22 by `scripts/fetch_data.py` (`fetch_nasa_clearsky`; timestamp inside the JSON under `_qgrid_fetch`).
- Licence: as NASA POWER above.
- Use: the v2 rolling-horizon optimiser forecasts PV as clear-sky PV x a clearness index (persistence of the
  last observed hour, blended towards yesterday's daily clearness for longer lead times). The forecast only
  uses data up to the decision time; its error against the actual (ALLSKY-based) PV is reported.

## Building Data Genome Project 2 (BDG2) — REAL
- Paper: Miller, C., Kathirgamanathan, A., Picchetti, B. et al. "The Building Data Genome Project 2, energy meter data
  from the ASHRAE Great Energy Predictor III competition." *Scientific Data* 7, 368 (2020). Repository
  `https://github.com/buds-lab/building-data-genome-project-2` (Zenodo record 3887306).
- Files (Git LFS media URLs, fetched 2026-11-22 by `scripts/fetch_data.py`, `fetch_bdg2`):
  `data/meters/cleaned/electricity_cleaned.csv` (174,977,911 bytes, SHA-256 b6ffc9b4dfcefe5c753594730a08ae822b0d50fec6815abb8f185591e6c630a3),
  `data/metadata/metadata.csv` (272,024 bytes, SHA-256 992d0b29f24f96ad4332bc4dbb534b7bdd7dd2689aad093f94e93068ecddca02), `LICENSE`.
- Licence: **Creative Commons Attribution-ShareAlike 4.0** (`data/raw/bdg2/LICENSE`). Derived zone load profiles in
  `data/processed/load.parquet` are shared under the same licence; attribution as above.
- Mapping (config.yaml `bdg2.zones`; buildings kept if >= 90 % of hours present, each normalised by its own mean,
  median across buildings per hour):

  | Zone | BDG2 primary use / sub use | buildings |
  |---|---|---:|
  | academic | Education / College Classroom, Academic, Classroom | 130 |
  | heavy_labs | Education / College Laboratory, Research | 124 |
  | hostels | Lodging/residential / Dormitory, Residence Hall | 86 |
  | library | Public services / Library | 48 |
  | canteen | Education / Student Union | 11 |
  | admin | Office / Office | 235 |
  | health_centre | Healthcare / Hospital, Patient Care | 11 |
  | server_room | none (no data centre in BDG2): flat profile, ASSUMPTION | – |

- Calendar mapping: each 2023 date takes one real 2016/17 source day — RVCE teaching/exam/fest weekdays -> source
  term-time weekdays (same weekday), RVCE vacation -> source break weekdays (June-August and 20 Dec - 5 Jan,
  ASSUMPTION), weekends -> source term-time weekends; seeded per zone and date (`qgrid/bdg2.py`).
- Limitations: the buildings are in North America and Europe (BDG2 sites), so HVAC base load and climate differ from
  Bengaluru, and the source "break" months are northern summer. Shapes are real; magnitudes are still floor area x EPI
  (PENDING RVCE meter data / bills).

## Ausgrid Solar Home Electricity Data — REAL (superseded by BDG2 for loads)
- Original publisher: Ausgrid (dataset page `https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data`,
  no longer online in 2026; metadata record: `https://data.gov.au/data/en/dataset/nsw-solar-home-electricty-data`).
- Copy used: archive by Pierre Haessig, `https://pierreh.eu/downloads/Ausgrid_solar_home_data.zip`
  (linked from `https://github.com/pierre-haessig/ausgrid-solar-data`), fetched 2026-11-21,
  SHA-256 `5a766f52b6c8b3b72730380f4422e478934bc94640a4b089dd0e0e3c055c5d82`. Contains Ausgrid's original
  CSVs and Ausgrid's notes PDF (Aug 2014).
- Licence: Creative Commons Attribution 3.0 Australia (per the archive README); Ausgrid disclaimer in the notes PDF.
- Reference: E. L. Ratnam, S. R. Weller, C. M. Kellett, A. T. Murray, "Residential load and rooftop PV
  generation: an Australian distribution network dataset", Int. J. Sustainable Energy, 2017,
  doi:10.1080/14786451.2015.1100196.
- Processing (`backend/qgrid/data.py`):
  - Load = GC (general consumption) + CL (controlled load) per customer; kWh per half hour → kW.
    Times are Sydney local (EST/EDT) as recorded.
  - k-means (k = 5, fixed seed) on each customer's normalised mean daily profile; clusters assigned
    to buildings by daytime (09–17 h) energy share: highest → lab_block, then admin, library, canteen,
    hostel (lowest daytime share, evening-heavy).
  - **Date alignment: Ausgrid 2012-13 load shapes are weekday-aligned to 2023.** Each 2023 date takes
    the same month/day from Jul-2012..Jun-2013 (Jul–Dec from 2012, Jan–Jun from 2013), shifted by at
    most ±3 days so the weekday matches.
  - Cluster-mean household kW scaled so the annual 99th percentile equals the building's `peak_kw`
    (a design assumption, config.yaml). Half-hourly values repeated to 15-min.
  - Limitation: these are residential shapes from Sydney used as a proxy for campus buildings
    (no open, licence-clear Indian campus metering dataset was available); the lab/admin proxies
    therefore peak in the evening more than a real lab block would.

## Price scenario A — KERC Tariff Orders 2022 and 2023 (BESCOM HT-2(c)(ii), by date) + KERC FY24 solar export rate — REAL
The tariff is date-dependent (`config.yaml prices.A.periods`, `qgrid.data.tariff_period`): days before 2023-04-01 use
Tariff Order 2022 (FY2022-23), later days Tariff Order 2023 (FY2023-24). Both orders take effect from the first
meter-reading date on or after 1 April; the switch is modelled on the calendar date (ASSUMPTION: RVCE's meter-reading
date is unknown).
Page refs "pN (printed M)": N = page of the merged PDF, M = page number printed on the order.
- **RVCE's category — HT-2(c)(ii), sourced:** RVCE NAAC Self Study Report, "Final submitted SSR_28 June 2024"
  (`https://rvce.edu.in/sites/default/files/Final%20submitted%20SSR_28%20June%202024_latest.pdf`; rvce.edu.in refuses
  scripted downloads, so the PDF was saved by hand on 2026-11-27) → `data/raw/rvce/Final-submitted-SSR_28-June-2024_latest.pdf`
  (124 pages, 538,066 bytes, SHA-256 `01a3125406d4360b0881c61f2fb4eb20ac539b9f2c88fae990cbee7240f3f787`; page footers
  read "Page N/124", so PDF page = printed page).
  - p2: "RV College of Engineering (RVCE) is a flagship institution under the Trust, established in 1963 is one of the
    earliest self-financing engineering colleges in the country. The institution is run by RSST, a not-for-profit
    Trust. RVCE is an Autonomous college." RSST = "Rashtreeya Shikshana Samithi Trust (RSST), founded in the year 1940" (p2).
  - p11 (2.1 Basic information): "Status of the Institution — Self Financing".
  - Self-financing means RVCE is not a Government, local-body or aided institution (and it is a college, not a
    university), so HT-2(c)(i) does not apply and the category is HT-2(c)(ii).
- **ASSUMPTION pending RVCE's BESCOM bill — single connection:** the whole campus, including hostels, is on one HT
  connection billed under HT-2(c)(ii). Hostels on a separate connection would fall under HT-2(c)(i) ("Hostels of all
  Educational Institutions"; TO2023 Rs 300/kVA, 750 paise, p215 and p382).
- **Import tariff FY2022-23 (days before 2023-04-01):** KERC Tariff Order 2022 for BESCOM, dated 4th April 2022,
  "Effective for the Electricity consumed from the first meter reading date falling on or after 01.04.2022"
  (p384, printed 375). KERC publishes it as 18 PDFs (`https://kerc.karnataka.gov.in/198/_gallery_/en`, listed under
  `https://kerc.karnataka.gov.in/69/tariff-order-2022/en`); fetched 2026-11-27, kept unchanged in
  `data/raw/tariff/KERC_TO2022_BESCOM_parts/` (not committed) and merged in order (pypdf) into
  `data/raw/tariff/KERC_tariff_order_BESCOM_FY2022-23.pdf` (424 pages, SHA-256 `6869d16a62ddb40a69dd2117241e6c9f038641c5100bcda2f5c69ad51905a55e`).
  - HT-2(c)(ii) "Applicable to Hospitals/Educational Institutions other than those covered under HT2(c) (i)":
    demand charges "Rs.265 / kVA of billing demand / month"; energy charges "For the first one lakh units 825 paise
    per unit", "For the balance units 865 paise per unit": p222 (printed 213), rate schedule p421 (printed 412).
  - **Two-slab implementation:** the slab is per billing month (ASSUMPTION: billing month = calendar month). Each slot's
    rate comes from the campus's cumulative grid import since the 1st of the month (`qgrid.data.slab_energy_rate`):
    8.25 Rs/kWh while it is within the first 1,00,000 kWh, 8.65 after; a slot straddling the boundary gets the
    energy-weighted mix, so the month's energy bill equals the slab bill exactly. The cumulative import is the
    modelled campus load minus PV without battery/DG dispatch (status-quo basis). The modelled campus imports about
    540–625 MWh a month, so the first lakh is used up on day 5–6 of each month; no evaluation or scenario day falls in
    that window, so all seven FY2022-23 evaluation days are priced at the balance rate.
  - ToD (HT-2(a), (b), (c)): identical to 2023 — December–June 18–22 h "(+) 100 paise per unit", 22–06 h "(-) 100
    paise per unit", 06–10 h and 10–18 h 0, July–November 0: p222 (printed 213), p421 (printed 412). Mandatory for
    contract demand ≥ 500 kVA: p213 (printed 204).
- **Import tariff FY2023-24 (from 2023-04-01):** KERC Tariff Order 2023 for BESCOM, dated 12 May 2023, effective 1 April 2023 (press note p2).
  KERC publishes it as 16 chapter PDFs (`https://kerc.karnataka.gov.in/309/_gallery_/en`, listed under
  `https://kerc.karnataka.gov.in/84/tariff-order-2023/en`); fetched 2026-11-27, kept unchanged in
  `data/raw/tariff/KERC_TO2023_BESCOM_parts/` and merged in order (pypdf) into
  `data/raw/tariff/KERC_tariff_order_BESCOM_FY2023-24.pdf` (385 pages, SHA-256 `018d3be54cca27e46803b584e0f676fc2c025a78c4ebc14eac422eeb40987b5b`).
  Press note: `https://kerc.karnataka.gov.in/uploads/36411683893437.pdf` → `KERC_TO2023_press_note_all_ESCOMs.pdf`.
  - Category HT-2(c)(ii) "Applicable to Hospitals and Educational Institutions other than those covered under
    HT-2(c)(i)": p213 (printed 204), tariff schedule p382 (printed 373). HT-2(c)(i) covers Government/charitable/ESI
    hospitals, universities, Government/local-body and aided educational institutions and **hostels of all
    educational institutions**.
  - Energy charge 850 paise/kWh "for entire consumption" (single slab); demand charge Rs 350 per kVA of billing
    demand per month: p215 (printed 206), p382 (printed 373) → `prices.A.periods[FY2023-24]`.
  - ToD (HT-2(a), (b), (c)(i), (c)(ii)): December–June 18–22 h +100 paise, 22–06 h −100 paise, 06–10 h and
    10–18 h 0; July–November (monsoon period) 0 in every slot: p215 (printed 206); Annexure V note 11 p378
    (printed 369). No morning peak: BESCOM's proposed 06–10 h peak was not approved ("continue with the existing
    approved ToD Tariff"): p170–172 (printed 161–163). Encoded with `months: [12, 1, …, 6]` per window
    (`qgrid.data.tod_adder`); a slot's own calendar month decides the season.
  - ToD applicability: mandatory for HT-2(c) with contract demand ≥ 500 kVA, optional below: p208 (printed 199).
    RVCE ≥ 500 kVA is an **ASSUMPTION** pending RVCE's bill.
- **Export rate:** KERC order dated 01.06.2023, "Determination of Tariff and norms in respect of Solar Power Projects
  (including Solar Rooftop Photovoltaic Projects) for FY24", `https://kerc.karnataka.gov.in/uploads/media_to_upload1685615785.pdf`
  → `data/raw/tariff/KERC_solar_export_tariff.pdf` (SHA-256 `5e5d1c2edbbc3456341b0c9e410ccc6cfee41e0ecab2d9d8532fb3bb005930ad`), fetched 2026-11-27. Rooftop PV
  1–2000 kW (excluding 1–10 kW domestic) Rs 3.74 per unit, for PPAs entered into on or after 01.04.2023, 25 years:
  p30 → `prices.A.export_rs_kwh` = 3.74. Under third-party models only domestic consumers get net metering; others
  gross metering (p19).
- **Export rate on every day = 3.74:** a solar PPA tariff is fixed for 25 years by the PPA date, not the billing date
  ("applicable to all the new solar power projects for which PPAs are entered into on or after 01.04.2023", p30), so
  the model uses one rate for the whole year: a new RVCE rooftop PPA under the FY24 order. (A PPA signed in FY2021-23
  would carry Rs 3.19/unit, quoted in the FY24 order p2.)
- **Benchmark sites (Fox, Bull, Cockatoo, Bangkok):** their years (2017/2019) are outside every KERC period, so they get
  the FY2023-24 period (`prices.A.out_of_range_period`) on local clock time, labelled "RVCE KERC tariff applied for
  comparability (controlled variable)"; they also use the CEA factor below with the same label. These are not the
  sites' own tariffs or grid factors.
- Licence: Government of Karnataka regulatory orders, public documents.
- Earlier stand-in (no longer used): base 8.00 Rs/kWh, +1.00 06–10 h and 18–22 h, −1.00 22–06 h, export 3.00.

## Price scenario B — IEX DAM — PENDING
- Needed: 15-min MCP for 2023 from `https://www.iexindia.com/market-data/day-ahead-market/market-snapshot`
  (Excel export). Parser: `qgrid.data.parse_iex_file` (tolerant to header rows / merged date cells).
- STAND-IN currently used: synthetic hour-window price shape (config.yaml `prices.B.standin_profile_rs_kwh`).

## CEA CO2 baseline — REAL
- CEA "CO2 Baseline Database for the Indian Power Sector — User Guide", Version 20.0, December 2024,
  `https://cea.nic.in/api/cdm-report-version/view-pdf/en/76` → `data/raw/cea/CEA_CO2_Baseline_User_Guide_v20.0_Dec2024.pdf`
  (SHA-256 `b881c258f58fd77c8faeb36479ccd1ed8bd111c45e708196df8a66ff7b29afc6`), fetched 2026-11-27. Publisher: Central Electricity Authority, Ministry of Power, GoI.
- Value used: **0.727 tCO2/MWh**, weighted average of the Indian Grid for FY 2023-24 (adjusted for cross-border
  transfers, including RES & captive injection), Table S, PDF page 4 (printed "Page | 1"). Recorded in
  `data/raw/cea/emission_factor.yaml`. It matches the earlier secondary-source stand-in.
- Also downloaded for reference: v19.0 (Dec 2023; FY2022-23: 0.716 incl. RES, PDF p9) and v22.0 (Aug 2026, latest;
  FY2025-26: 0.675, PDF p6). v22.0 says it replaced provisional 2022-23 and 2023-24 data; any revised FY2023-24
  value in v22.0 has not been checked.

## Synthetic data
Synthetic inputs appear only in unit tests (`backend/tests/`), labelled SYNTHETIC in the test code.

## OpenStreetMap — REAL
- Overpass API (`https://overpass-api.de/api/interpreter`), campus element `way(55685638)` "R. V. College of Engineering",
  everything inside its area (buildings, amenities). Fetched 2026-11-21 by `qgrid.campus.fetch_osm`; query and
  timestamp stored in the JSON (`_qgrid_fetch`); SHA-256 recorded in `data/processed/campus.json`.
- Licence: ODbL 1.0 — **© OpenStreetMap contributors**.
- Processing (`backend/qgrid/campus.py`, rules in `config.yaml campus.zone_rules`): footprints → 9 zones by name/tag
  regex; floor area = footprint × `building:levels` (or zone default, ASSUMPTION); PV kWp = footprint × 45 % usable
  roof × 0.18 kWp/m² (ASSUMPTION); tie-lines = minimum spanning tree on zone centroids + 2 shortest extra links,
  degree ≤ 3. The pump house is not mapped in OSM; its location/size are ASSUMPTIONS.

## RVCE campus assets and floor areas (config.yaml `campus.pv`, `campus.area_targets`, diesel) — SOURCED / PROPOSED
Source: RVCE NAAC SSR 2024, `data/raw/rvce/Final-submitted-SSR_28-June-2024_latest.pdf` (PDF page = printed page).
- **Campus extent:** SSR p12 "Main campus area ... Campus Area in Acres 16.85 ... Built up Area in sq.mts. 67721". The OSM
  outline (way 55685638) is 49.6 acres, the whole RV Vidyaniketan campus. RV University buildings are excluded by their OSM
  names ("RVU A Block", "RVU Block", "RVU C Block", "RVU Admin Block"); unnamed buildings cannot be attributed from OSM.
- **Floor areas:** hostels = 38,659.11 m² (p78, "three Boys Hostels(1185 rooms, 38659.11 sq. m, 1450 capacity)"; girls'
  hostels have no area in the SSR and are not added). The other OSM zones are scaled together (×0.584) to 67,721 m²
  (INTERPRETATION, ASSUMPTION: the p12 figure excludes the boys' hostels; including them would leave the other buildings
  29,062 m², below their mapped roof area). Zones scaled down keep their OSM roofs; the hostel roof grows by the same factor
  as its floor area (unmapped hostel blocks, ASSUMPTION). Total 106,440 m² (was 156,588 m² from OSM × assumed levels).
- **Existing rooftop PV: 500 kWp, SOURCED, VERIFICATION PENDING.** RVCE "21. Electrical Grid Power Supply Connection"
  (`https://rvce.edu.in/sites/default/files/21.%20Electrical%20Grid%20Power%20Supply%20Connection.pdf`): 1,564 modules ×
  320 Wp on 21 rooftops = 500.48 kWp; inverters 6 × 50 + 6 × 30 + 1 × 20 kW = 500 kW, as reported by the team. The PDF and the
  NAAC building list (`https://naac.rvce.edu.in/sites/default/files/2024-05/Cr_7.1.2_File_4_Solar_Power_Facility_in_various_departments_Geotagged_Photos.pdf`)
  refused scripted downloads on 2026-11-27 and are not yet in `data/raw/rvce/`; split across zones by roof area
  (ASSUMPTION) until the building list is available. The earlier roof-potential PV (≈ 4,000 kWp: 45 % of roof × 0.18 kWp/m²)
  is kept only as the PROPOSED scenario (`campus.pv.scenario: proposed`), never the baseline.
- **Diesel:** SSR p69 "UPS along with two generators of capacity 980 KVA are available. It is planned to add three more
  generators with capacity 750 KVA each". Read as 980 kVA in total for the two existing sets (INTERPRETATION), 490 kVA each ×
  0.8 pf (ASSUMPTION) = 392 kW, placed at hostels and admin (ASSUMPTION). The planned 3 × 750 kVA are excluded.
- **PROPOSED (Q-Grid design, not on campus today):** batteries (1,460 kWh; no battery documented — the server-room 120 kWh is
  the documented UPS, capacity ASSUMPTION), inter-building tie-line cables, the 44 kW EV-charging load (no chargers
  documented; SSR p108 says staff and students use EVs).

## Campus loads — PENDING RVCE DATA
- Annual kWh per zone = floor area × energy-use intensity (kWh/m²/yr, ASSUMPTION per zone in `config.yaml`).
- Shapes: timetable templates by building type and day type (`configs/profiles.yaml`, ASSUMPTION); hostels use the
  real Ausgrid residential shape (above). Lab schedule from `data/raw/lab_schedule.csv` (TEMPLATE until faculty input).
- If `data/raw/rvce_bills.csv` (month, kwh) exists, totals are scaled month by month to the bills.

## Calendar — PENDING (partly sourced)
- Fest (8th Mile) 13–15 June 2023: RVCE cultural activities report and technical club report 2023-24
  (`https://rvce.edu.in/sites/default/files/All%20Techclubreport_2023-2024.pdf`, `https://naac.rvce.edu.in/sites/default/files/2024-05/CAT-Report-2019-2023.pdf`).
- Teaching / exam / vacation windows in `config.yaml calendar` are PLACEHOLDERS: rvce.edu.in returned HTTP 403 to
  scripted downloads of the academic-calendar PDFs. Provide `data/raw/rvce_calendar.csv` (date, day_type) to override.

## Engineering assumptions (not data)
Diesel price 90 Rs/L, fuel curve 0.08 L/h per kW rated + 0.25 L/kWh, 2.68 kg CO2/L; VOLL 40 / 1000 Rs/kWh
(optimisation only); battery capex 25,000 Rs/kWh; semester = 120 days; battery/DG/tank sizes per zone. All in
`config.yaml`, all labelled ASSUMPTION; none is presented as data.


## Benchmark datasets (multi-dataset benchmark, `qgrid/datasets.py`)

| Dataset | Status | Loads | Weather | Licence |
|---|---|---|---|---|
| RVCE (case study) | as above | BDG2 shapes x floor area x EPI | NASA POWER 2023 | as above |
| BDG2 site "Fox" (Arizona), 2017 | **REAL** loads | the site's own metered buildings (>= 90 % complete), grouped into zones | NASA POWER at the BDG2 site coordinates | CC BY-SA 4.0 |
| BDG2 site "Bull" (Texas), 2017 | **REAL** loads | same | same | CC BY-SA 4.0 |
| BDG2 site "Cockatoo" (New York), 2017 | **REAL** loads | same | same | CC BY-SA 4.0 |
| CU-BEMS, Chamchuri 5 office building, Bangkok, 2019 | **REAL** loads | the 7 floors' AC, lighting and plug meters (1-min, summed per floor, hourly means) | NASA POWER at 13.7367 N, 100.5332 E | CC BY 4.0 |

- **BDG2 sites**: same files and licence as the BDG2 entry above; zone groups (academic, labs, dorms, library, offices, other)
  from `metadata.csv` primary/sub use; site coordinates from `metadata.csv`. Meter spikes clipped at each building's 99.9th
  percentile.
- **CU-BEMS**: Pipattanasomporn, M., Chitalia, G., Songsiri, J. et al. "CU-BEMS, smart building electricity consumption and
  indoor environmental sensor datasets." *Scientific Data* 7, 241 (2020). Files from
  `https://github.com/SGRUdata/SGRUdata.github.io/tree/master/data` (fetched 2026-11-22):
  - `2019Floor1.csv.zip` SHA-256 f88d0707a4571598bc1be307cf44b6e13d5bf356b9255ed7742f8a279e272e50
  - `2019Floor2.csv.zip` SHA-256 f6a40fa250d095fe9cdb17a302b4f81a7e3f4c2bdf709460b06b242c57ab2516
  - `2019Floor3.csv.zip` SHA-256 a3b10899e070d3a32f3a1e3e8bfc2874eee81d573ddb8031b59801b39064a0c1
  - `2019Floor4.csv.zip` SHA-256 f8ce104a9f1ed938d90cc2a0b02a72dee895c2c672b66ca896461feab02b0e61
  - `2019Floor5.csv.zip` SHA-256 310ca9ae005326975e5161b0672cace46fab73c15d828078552af50ffb7be51c
  - `2019Floor6.csv.zip` SHA-256 ded70a7938030b9bc19a7038ae09cb31d3d8dc18801a054a6a216c483828520d
  - `2019Floor7.csv.zip` SHA-256 f6fb56ee65bb07d0035f51148b86e0caf8030c96c46cf7c0aa2ec9e38196ce05
  Licence: Creative Commons Attribution 4.0 (CC BY 4.0) per the dataset record. It is one office building, so its
  "zones" are floors linked by assumed internal cables.
- **NASA POWER** (sites): `data/raw/nasa_power/<dataset>_<year>_hourly.json`, parameters ALLSKY_SFC_SW_DWN, CLRSKY_SFC_SW_DWN,
  T2M, fetched 2026-11-22 by `scripts/fetch_data.py` (`fetch_nasa_site`); licence as NASA POWER above.
- **ASSUMPTIONS for sites without asset data** (one rule for all, `qgrid/datasets.py`): PV = 30 % of each zone's annual energy,
  battery = 2 h of mean load (C-rate 0.5), one diesel set at the zone with the largest critical load (50 % of its 99th-percentile
  load), critical share 10 % (dorms) / 5 % (others), cables in a chain plus two seeded cross-links rated at 25 % of the median
  zone peak. Tariff and CO2: RVCE's KERC tariff (FY2023-24 period) on local clock time and RVCE's CEA factor, applied for
  comparability (controlled variable), not the sites' own.
- **SYNTHETIC (seeded) events**: unplanned grid outages (20 % of evaluation days, 1–3 h, start 08:00–20:00) and one 3-hour
  campus network outage per day (06:00–21:00) for the communication-failure methods. Official BESCOM planned-cut schedules are
  **PENDING**.
