"""Campus layout from OpenStreetMap: footprints -> zones -> tie-lines -> PV capacity.

Generic: the campus (name, lat/lon), zone keyword rules, assumptions (levels, PV factor, line degree)
all come from config.yaml; nothing campus-specific lives in code.
Data © OpenStreetMap contributors, ODbL 1.0 (https://www.openstreetmap.org/copyright).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OVERPASS = "https://overpass-api.de/api/interpreter"
OVERPASS_ALT = ["https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter",
                "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]


def _overpass(q: str, tries: int = 6) -> dict:
    import time
    last = None
    for k in range(tries):
        ep = OVERPASS_ALT[k % len(OVERPASS_ALT)]
        try:
            r = requests.post(ep, data={"data": q}, timeout=180, headers={"User-Agent": "qgrid/0.2"})
            if r.status_code == 200 and r.text.lstrip().startswith("{"):
                return r.json()
            last = f"{ep}: HTTP {r.status_code}"
        except requests.RequestException as e:
            last = f"{ep}: {e}"
        time.sleep(3 * (k + 1))
    raise RuntimeError(f"Overpass unavailable: {last}")
OSM_CREDIT = "© OpenStreetMap contributors (ODbL)"


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def fetch_osm(name: str, lat: float, lon: float, slug: str | None = None, force: bool = False) -> Path:
    """Find the campus outline by name near (lat, lon) and download everything inside it.
    Cached to data/raw/osm_<slug>.json (raw Overpass) and osm_<slug>.geojson."""
    slug = slug or _slug(name)
    raw_p = RAW / f"osm_{slug}.json"
    if raw_p.exists() and not force:
        return raw_p
    q_find = (f'[out:json][timeout:60];(way["amenity"~"university|college|school"](around:2000,{lat},{lon});'
              f'relation["amenity"~"university|college|school"](around:2000,{lat},{lon}););out tags center;')
    cands = [e for e in _overpass(q_find)["elements"] if name.lower() in e.get("tags", {}).get("name", "").lower()]
    if not cands:
        raise RuntimeError(f"No campus named like '{name}' near {lat},{lon} in OSM")
    c = cands[0]
    sel = f'{c["type"]}({c["id"]})'
    q = (f'[out:json][timeout:90];{sel}->.c;.c out geom tags;.c map_to_area->.a;'
         f'(way["building"](area.a);relation["building"](area.a);way["amenity"](area.a);way["leisure"](area.a););'
         f'out geom tags;')
    data = _overpass(q)
    data["_qgrid_fetch"] = {"query": q, "endpoint": OVERPASS, "campus_element": sel,
                            "fetched_utc": dt.datetime.now(dt.UTC).isoformat(), "licence": "ODbL 1.0",
                            "credit": OSM_CREDIT}
    RAW.mkdir(parents=True, exist_ok=True)
    raw_p.write_text(json.dumps(data))
    _to_geojson(data, RAW / f"osm_{slug}.geojson")
    return raw_p


def _rings(e: dict) -> list[list[tuple[float, float]]]:
    if e["type"] == "way" and e.get("geometry"):
        return [[(p["lon"], p["lat"]) for p in e["geometry"]]]
    if e["type"] == "relation":
        return [[(p["lon"], p["lat"]) for p in m["geometry"]] for m in e.get("members", [])
                if m.get("role") == "outer" and m.get("geometry")]
    return []


def _to_geojson(data: dict, path: Path):
    feats = []
    for e in data["elements"]:
        for ring in _rings(e):
            feats.append({"type": "Feature", "properties": {"osm_id": f'{e["type"]}/{e["id"]}', **e.get("tags", {})},
                          "geometry": {"type": "Polygon", "coordinates": [ring]}})
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats,
                                "properties": {"credit": OSM_CREDIT, "licence": "ODbL 1.0"}}))


def _xy(ring, lat0, lon0):
    k = 111_320.0
    return [((lo - lon0) * k * math.cos(math.radians(lat0)), (la - lat0) * k) for lo, la in ring]


def _area(xy) -> float:
    a = 0.0
    for (x1, y1), (x2, y2) in zip(xy, xy[1:] + xy[:1]):
        a += x1 * y2 - x2 * y1
    return abs(a) / 2


def classify(tags: dict, rules: list[dict]) -> str | None:
    """First matching rule wins. rule = {zone, name_regex?, tags?: {key: regex}, exclude?: bool}."""
    name = tags.get("name", "")
    for r in rules:
        ok = True
        if "name_regex" in r:
            ok &= bool(re.search(r["name_regex"], name, re.I))
        for k, pat in r.get("tags", {}).items():
            ok &= bool(re.search(pat, str(tags.get(k, "")), re.I))
        if ok:
            return None if r.get("exclude") else r["zone"]
    return None


def build_layout(cfg: dict) -> dict:
    """Zones with footprint area, floor area, centroid (m), PV kWp, polygons; tie-lines by geometry."""
    cc = cfg["campus"]
    raw_p = fetch_osm(cc["osm_name"], cfg["site"]["lat"], cfg["site"]["lon"], cc["slug"])
    data = json.loads(raw_p.read_text())
    sha = hashlib.sha256(raw_p.read_bytes()).hexdigest()
    lat0, lon0 = cfg["site"]["lat"], cfg["site"]["lon"]
    zones = {z: {"footprint_m2": 0.0, "floor_m2": 0.0, "cx": 0.0, "cy": 0.0, "buildings": []} for z in cc["zones"]}
    seen = set()
    campus_outline = None
    for e in data["elements"]:
        key = (e["type"], e["id"])
        tags = e.get("tags", {})
        if f'{e["type"]}({e["id"]})' == data["_qgrid_fetch"]["campus_element"] or (
                tags.get("amenity") in ("university", "college") and "building" not in tags):
            rings = _rings(e)
            if rings:
                campus_outline = [list(p) for p in _xy(rings[0], lat0, lon0)]
            continue
        if key in seen:
            continue
        seen.add(key)
        if "building" not in tags and tags.get("amenity") not in ("hospital", "clinic", "library", "fast_food", "restaurant", "cafe"):
            continue
        z = classify(tags, cc["zone_rules"])
        if z is None:
            continue
        for ring in _rings(e):
            xy = _xy(ring, lat0, lon0)
            a = _area(xy)
            if a < 30:
                continue
            levels = float(tags.get("building:levels", cc["zones"][z].get("levels", 2)))
            cx = sum(p[0] for p in xy) / len(xy)
            cy = sum(p[1] for p in xy) / len(xy)
            zz = zones[z]
            zz["cx"] += cx * a
            zz["cy"] += cy * a
            zz["footprint_m2"] += a
            zz["floor_m2"] += a * levels
            zz["buildings"].append({"osm_id": f'{e["type"]}/{e["id"]}', "name": tags.get("name"),
                                    "levels": levels, "area_m2": round(a, 1),
                                    "polygon": [[round(x, 2), round(y, 2)] for x, y in xy]})
    for z, zz in zones.items():
        spec = cc["zones"][z]
        if zz["footprint_m2"] > 0:
            zz["cx"] /= zz["footprint_m2"]
            zz["cy"] /= zz["footprint_m2"]
            zz["source"] = "osm"
        else:  # virtual zone (not mapped in OSM): assumed location + floor area from config
            zz["cx"], zz["cy"] = spec["virtual"]["x_m"], spec["virtual"]["y_m"]
            zz["footprint_m2"] = zz["floor_m2"] = spec["virtual"]["floor_m2"]
            zz["source"] = "ASSUMPTION (not mapped in OSM)"
    apply_area_targets(zones, cc)
    assign_pv(zones, cc)
    lines = tie_lines({z: (v["cx"], v["cy"]) for z, v in zones.items()}, cc["tie_lines"]["max_degree"],
                      cc["tie_lines"]["extra_links"])
    out = {"campus": cc["display_name"], "credit": OSM_CREDIT, "osm_sha256": sha, "osm_file": str(raw_p.relative_to(ROOT)),
           "origin": {"lat": lat0, "lon": lon0}, "outline": campus_outline, "zones": zones,
           "pv": {"scenario": cc["pv"].get("scenario", "existing"), "existing_kwp": cc["pv"].get("existing_kwp"),
                  "status": cc["pv"].get("status", "ASSUMPTION"),
                  "total_kwp": round(sum(z["pv_kwp"] for z in zones.values()), 1)},
           "area_targets": cc.get("area_targets"),
           "lines": [list(l) for l in lines]}
    PROC.mkdir(parents=True, exist_ok=True)
    (PROC / "campus.json").write_text(json.dumps(out, indent=1))
    return out


def apply_area_targets(zones: dict, cc: dict) -> None:
    """Scale OSM floor areas to documented totals (config campus.area_targets, RVCE NAAC SSR 2024).
    zones_floor_m2 fixes named zones; the other OSM-mapped zones are scaled together so their floor area sums to
    other_zones_floor_m2. Footprints (roofs) stay as mapped in OSM when a zone is scaled down (the OSM excess is in the
    assumed levels); a zone scaled up has unmapped buildings, so its footprint grows by the same factor (ASSUMPTION).
    The OSM values are kept as osm_floor_m2 / osm_footprint_m2."""
    at = cc.get("area_targets")
    for zz in zones.values():
        zz["osm_floor_m2"], zz["osm_footprint_m2"], zz["area_scale"] = zz["floor_m2"], zz["footprint_m2"], 1.0
    if not at:
        return
    fixed = {z: float(v) for z, v in (at.get("zones_floor_m2") or {}).items()}
    rest = [z for z, zz in zones.items() if z not in fixed and zz["source"] == "osm"]
    targets = dict(fixed)
    if at.get("other_zones_floor_m2") is not None and rest:
        tot = sum(zones[z]["floor_m2"] for z in rest)
        for z in rest:
            targets[z] = zones[z]["floor_m2"] * float(at["other_zones_floor_m2"]) / tot
    for z, target in targets.items():
        zz = zones[z]
        scale = target / zz["floor_m2"] if zz["floor_m2"] > 0 else 1.0
        zz["area_scale"] = scale
        zz["floor_m2"] = target
        if scale > 1.0:              # more floor area than OSM maps -> unmapped buildings at the zone's mean OSM height
            zz["footprint_m2"] *= scale
            zz["footprint_note"] = ("ASSUMPTION: documented floor area exceeds OSM; footprint scaled by the same factor "
                                    "(unmapped buildings at the zone's mean OSM levels)")


def assign_pv(zones: dict, cc: dict) -> None:
    """pv_kwp per zone. With campus.pv.existing_kwp: the existing (SOURCED) array, split by existing_zone_kwp where the
    building list names zones, the rest by roof area (footprint x pv_roof, ASSUMPTION). pv_kwp_potential = roof
    potential (footprint x usable_roof_frac x kwp_per_m2 x pv_roof). campus.pv.scenario: existing (baseline) uses the
    existing array; proposed (PROPOSED scenario, not the baseline) uses max(existing, potential) per zone.
    Without existing_kwp (other campuses): pv_kwp = roof potential, as before."""
    pv = cc["pv"]
    w = {z: zz["footprint_m2"] * cc["zones"][z].get("pv_roof", 1.0) for z, zz in zones.items()}
    for z, zz in zones.items():
        zz["pv_kwp_potential"] = round(w[z] * pv["usable_roof_frac"] * pv["kwp_per_m2"], 1)
    if pv.get("existing_kwp") is None:
        for zz in zones.values():
            zz["pv_kwp"] = zz["pv_kwp_potential"]
        return
    named = {z: float(v) for z, v in (pv.get("existing_zone_kwp") or {}).items()}
    rest_kwp = float(pv["existing_kwp"]) - sum(named.values())
    wr = sum(v for z, v in w.items() if z not in named)
    for z, zz in zones.items():
        zz["pv_kwp_existing"] = round(named[z] if z in named else rest_kwp * w[z] / wr if wr > 0 else 0.0, 1)
        zz["pv_kwp_basis"] = "SOURCED (building list)" if z in named else "SOURCED total, split by roof area (ASSUMPTION)"
        scen = pv.get("scenario", "existing")
        if scen not in ("existing", "proposed"):
            raise ValueError(f"campus.pv.scenario must be existing or proposed, not {scen!r}")
        zz["pv_kwp"] = zz["pv_kwp_existing"] if scen == "existing" else max(zz["pv_kwp_existing"], zz["pv_kwp_potential"])


def tie_lines(pos: dict[str, tuple[float, float]], max_degree: int = 3, extra_links: int = 2) -> list[tuple[str, str]]:
    """Minimum spanning tree on zone centroids (cables follow the shortest real distances), plus the
    `extra_links` shortest remaining links that keep every degree <= max_degree (meshing for resilience)."""
    names = list(pos)
    d = {(a, b): math.dist(pos[a], pos[b]) for i, a in enumerate(names) for b in names[i + 1:]}
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    deg = {n: 0 for n in names}
    chosen = []
    for (a, b), _ in sorted(d.items(), key=lambda kv: kv[1]):
        if find(a) != find(b) and deg[a] < max_degree and deg[b] < max_degree:
            parent[find(a)] = find(b)
            chosen.append((a, b)); deg[a] += 1; deg[b] += 1
    added = 0
    for (a, b), _ in sorted(d.items(), key=lambda kv: kv[1]):
        if added >= extra_links:
            break
        if (a, b) not in chosen and deg[a] < max_degree and deg[b] < max_degree:
            chosen.append((a, b)); deg[a] += 1; deg[b] += 1; added += 1
    return chosen


def search_space(var_counts: dict[str, int], slots: int = 96) -> dict:
    """Size of the full-day centralised binary decision space: N = slots * sum_i n_i, |space| = 2^N."""
    N = slots * sum(var_counts.values())
    return {"vars_per_slot": var_counts, "slots": slots, "N_binary": N, "log10_states": round(N * math.log10(2), 1),
            "formula": "N = slots x sum_i n_i ; |space| = 2^N"}


def fetch_roads(cfg: dict, force: bool = False) -> list[list[list[float]]]:
    """Footways/roads inside the campus outline from OSM (visual context only), cached to data/raw/osm_<slug>_roads.json.
    Returns polylines in local metres."""
    cc = cfg["campus"]
    p = RAW / f"osm_{cc['slug']}_roads.json"
    if not p.exists() or force:
        raw = json.loads((RAW / f"osm_{cc['slug']}.json").read_text())
        sel = raw["_qgrid_fetch"]["campus_element"]
        q = f'[out:json][timeout:90];{sel}->.c;.c map_to_area->.a;(way["highway"](area.a););out geom tags;'
        data = _overpass(q)
        data["_qgrid_fetch"] = {"query": q, "fetched_utc": dt.datetime.now(dt.UTC).isoformat(), "licence": "ODbL 1.0",
                                "credit": OSM_CREDIT}
        p.write_text(json.dumps(data))
    data = json.loads(p.read_text())
    lat0, lon0 = cfg["site"]["lat"], cfg["site"]["lon"]
    return [[[round(x, 1), round(y, 1)] for x, y in _xy([(g["lon"], g["lat"]) for g in e["geometry"]], lat0, lon0)]
            for e in data["elements"] if e.get("geometry")]
