#!/usr/bin/env python3
"""
Synthetic dengue surveillance benchmark, Visakhapatnam (GVMC), Andhra Pradesh. Version 3.

Outputs 8 CSVs to ./data/:
  Static:
    zones_demographics.csv          7 GVMC zones (zone attributes, M-SDT constants)
    city_infrastructure.csv         8 facilities (3 hospitals, 2 labs, 3 vector depots)
  Graph:
    spatial_network_edges.csv       16 zone-zone edges, DengueGNN gravity schema
    city_edges.csv                  zone-zone edges + zone-facility links, search-graph schema
  Dynamic:
    daily_environmental_surveillance.csv   210 zone-days, weather and vector indices
    clinical_reporting_logs.csv            210 zone-days, clinic and hotline records
    daily_surveillance.csv                 210 zone-days, simulation stream for BN/RL
                                           (joined from the two tables above, so values agree)
    interventions_and_resources.csv        37 operations with stress-test rows

Zone and facility names are real GVMC zones and Visakhapatnam hospitals. Coordinates are
approximate. Population, area, bed/ICU/kit/team counts (except KGH beds) are estimates.
M-SDT constants (b0, m, p, R0) are Ahmedabad values carried over, not re-fitted for AP.

Zero dependencies (csv, random, math, datetime, os, sys). Seeded (42). Runs a cross-table
consistency check after writing; exits non-zero if any check fails.
"""
import csv
import math
import os
import random
import sys
from datetime import date, timedelta

SEED = 42
random.seed(SEED)
OUT = "data"
os.makedirs(OUT, exist_ok=True)

START = date(2026, 8, 1)
DAYS = 30
MONSOON = range(12, 17)          # Days 12-16 cloudburst
SOUTH_OUTBREAK = range(13, 18)   # Days 13-17 under-reporting stress test (ZONE_SOUTH)
LAG_WINDOW = range(15, 27)       # post-monsoon transmission window
ROAD_FACTOR = 1.25               # straight-line -> estimated road distance

RAIN_MSG = "Precipitation exceeded 20mm; larvicide washout hazard"
INFEAS_MSG = ("Total diagnostic kits requested (350) exceeded available depot inventory (250). "
              "Deficit of 100 kits")
ACTIONS = ["ULV_Fogging", "Residual_Spraying", "Breeding_Site_Inspection", "Distribute_Kits",
           "Awareness_Campaign", "Hospital_Bed_Surge", "Monitor_Only"]
OUTDOOR = {"ULV_Fogging", "Residual_Spraying"}
STATUSES = {"Completed", "Suspended_Rain", "Infeasible_Shortage"}
REPORT_STATUSES = {"Active", "Delayed", "Missing"}

# zone_id, GVMC zone name, lat, lon, population (est.), area_sqkm (est.),
# base_biting_rate_b0, vector_to_host_ratio_m, symptomatic_fraction_p, baseline_r0
ZONES = [
    ("ZONE_CENTRAL", "Suryabagh", 17.7215, 83.3050, 720000, 24.5, 0.132, 3.385, 0.028, 2.21),
    ("ZONE_WEST", "Gnanapuram", 17.7240, 83.2930, 610000, 31.2, 0.103, 3.397, 0.054, 1.75),
    ("ZONE_NORTH", "Asilmetta", 17.7450, 83.3150, 540000, 38.0, 0.125, 2.021, 0.062, 1.64),
    ("ZONE_EAST", "Madhurawada", 17.8150, 83.3800, 480000, 19.6, 0.131, 2.211, 0.060, 1.80),
    ("ZONE_SOUTH", "Gajuwaka", 17.6870, 83.2000, 690000, 27.4, 0.101, 4.620, 0.060, 2.00),
    ("ZONE_NORTH_WEST", "Vepagunta", 17.8150, 83.2180, 410000, 35.8, 0.101, 4.815, 0.078, 2.06),
    ("ZONE_SOUTH_WEST", "Anakapalli", 17.6900, 83.0050, 455000, 22.1, 0.120, 3.000, 0.053, 1.90),
]
ZONE = {z[0]: z for z in ZONES}
ZONE_IDS = [z[0] for z in ZONES]

# facility_id, name, type, zone_id, lat, lon, total_beds, icu_beds, pcr_kits,
# spray_teams, surveillance_teams, vehicles
FACILITIES = [
    ("HOSP_01", "King George Hospital (KGH), Maharanipeta", "Hospital", "ZONE_CENTRAL",
     17.7178, 83.3019, 1237, 35, 90, 0, 3, 4),
    ("HOSP_02", "Area Hospital Seethammadhara", "Hospital", "ZONE_NORTH",
     17.7437, 83.3250, 150, 25, 75, 0, 3, 3),
    ("HOSP_03", "Area Hospital Gajuwaka", "Hospital", "ZONE_SOUTH",
     17.6891, 83.1989, 150, 25, 60, 0, 2, 2),
    ("LAB_01", "District Public Health Laboratory, Visakhapatnam", "Testing_Lab", "ZONE_CENTRAL",
     17.7180, 83.3030, 0, 0, 450, 0, 0, 0),
    ("LAB_02", "Gajuwaka Zonal Testing Laboratory", "Testing_Lab", "ZONE_SOUTH",
     17.6905, 83.2010, 0, 0, 320, 0, 0, 0),
    ("DEPOT_01", "GVMC Suryabagh Zonal Office (Fogging Depot)", "Vector_Depot", "ZONE_CENTRAL",
     17.7220, 83.3040, 0, 0, 0, 4, 3, 5),
    ("DEPOT_02", "GVMC Gajuwaka Zonal Office (Vector Depot)", "Vector_Depot", "ZONE_SOUTH",
     17.6880, 83.1995, 0, 0, 0, 3, 4, 6),
    ("DEPOT_03", "GVMC Gnanapuram Zonal Office (Field Office)", "Vector_Depot", "ZONE_WEST",
     17.7250, 83.2920, 0, 0, 0, 2, 2, 2),
]
FAC = {f[0]: f for f in FACILITIES}
HOSPS = [f for f in FACILITIES if f[2] == "Hospital"]
IDX_BEDS, IDX_PCR, IDX_SPRAY, IDX_SURV = 6, 8, 9, 10


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def clip(x, lo, hi):
    return max(lo, min(hi, x))


def poisson(lam):
    if lam <= 0:
        return 0
    if lam > 30:
        return max(0, round(random.gauss(lam, math.sqrt(lam))))
    L = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        p *= random.random()
        if p <= L:
            return k
        k += 1


def tier_of(pop, area):
    density = pop / area
    if density >= 25000:
        return "High"
    if density >= 15000:
        return "Medium"
    return "Low"


TIERS = {z[0]: tier_of(z[4], z[5]) for z in ZONES}
TIER_BONUS = {"High": 8, "Medium": 4, "Low": 0}

# Primary hospital per zone = nearest hospital to the zone centroid
PRIMARY = {z[0]: min(HOSPS, key=lambda f: haversine(z[2], z[3], f[4], f[5]))[0] for z in ZONES}


def zone_cap(zid, idx):
    return sum(f[idx] for f in FACILITIES if f[3] == zid)


def spray_cap(zid):
    return zone_cap(zid, IDX_SPRAY)


def surv_cap(zid):
    return zone_cap(zid, IDX_SURV)


def kit_cap(zid):
    labs = sum(f[IDX_PCR] for f in FACILITIES if f[3] == zid and f[2] == "Testing_Lab")
    return FAC[PRIMARY[zid]][IDX_PCR] + labs


def dstr(d):
    return (START + timedelta(days=d - 1)).isoformat()


def write_csv(name, header, rows):
    path = os.path.join(OUT, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


# ---------------- Static tables ----------------
def zones_rows():
    return [[z[0], z[1], z[4], z[5], z[2], z[3], z[6], z[7], z[8], z[9], TIERS[z[0]]] for z in ZONES]


def infra_rows():
    return [list(f) for f in FACILITIES]


# ---------------- Graph: zone-zone edges (shared by two files) ----------------
def zone_pairs():
    """16 shortest zone-zone pairs with gravity flow and transmission weight.
    Random draws happen here, so keep this call before the dynamic tables."""
    cands = []
    for i, a in enumerate(ZONES):
        for b in ZONES[i + 1:]:
            d = clip(round(haversine(a[2], a[3], b[2], b[3]) * ROAD_FACTOR, 2), 2.0, 18.0)
            cands.append((d, a, b))
    cands.sort(key=lambda x: x[0])
    cands = cands[:16]
    gs = [a[4] * b[4] / (d * d * 10000) for d, a, b in cands]
    gmax = max(gs)
    out = []
    for k, ((d, a, b), g) in enumerate(zip(cands, gs), 1):
        speed = random.uniform(20, 35)  # km/h average traffic
        t = int(clip(round(d / speed * 60), 8, 45))
        w = round(clip(0.10 + 0.85 * (g / gmax), 0.10, 0.95), 2)
        out.append({"edge_id": f"EDGE_{k:03d}", "src": a[0], "tgt": b[0], "distance_km": d,
                    "travel_min": t, "gravity": int(g), "weight": w})
    return out


def spatial_rows(pairs):
    return [[p["edge_id"], p["src"], p["tgt"], p["distance_km"], p["travel_min"], p["gravity"],
             p["weight"]] for p in pairs]


# ---------------- Graph: search graph (zone-zone + zone-facility) ----------------
def facility_links():
    """Each facility links to its own zone; each zone links to its primary hospital.
    Deduplicated, ordered. Deterministic (no random draws)."""
    seen, links = set(), []

    def add(z, f):
        if (z, f) not in seen:
            seen.add((z, f))
            links.append((z, f))

    for f in FACILITIES:
        add(f[3], f[0])
    for z in ZONES:
        add(z[0], PRIMARY[z[0]])
    return links


def city_edge_rows(pairs):
    rows = [[p["src"], p["tgt"], p["distance_km"], p["travel_min"], p["gravity"], "road_and_mobility"]
            for p in pairs]
    for zid, fid in facility_links():
        z, f = ZONE[zid], FAC[fid]
        d = round(max(0.1, haversine(z[2], z[3], f[4], f[5]) * ROAD_FACTOR), 2)
        t = max(2, round(d / 25 * 60))
        rows.append([zid, fid, d, t, 0, "road_only"])
    return rows


# ---------------- Dynamic tables ----------------
def env_rows():
    rows = []
    for d in range(1, DAYS + 1):
        wet = d in MONSOON
        lr = 1.0 if d in LAG_WINDOW else 0.0
        lag = 1.0 if 15 <= d <= 21 else 0.0
        for z in ZONES:
            zid = z[0]
            bonus = TIER_BONUS[TIERS[zid]]
            if wet:
                rain = round(random.uniform(45.0, 95.0), 1)
                temp = round(random.uniform(27.0, 30.0), 1)
                hum = round(random.uniform(84.0, 92.0), 1)
            else:
                rain = 0.0 if random.random() < 0.35 else round(random.uniform(0.0, 12.0), 1)
                temp = round(random.uniform(28.0, 36.0), 1)
                hum = round(random.uniform(55.0, 80.0), 1)
            ovi = round(clip(random.gauss(15 + 10 * lr + bonus, 3), 5.0, 45.0), 2)
            bre = round(clip(random.gauss(18 + 20 * lr + 1.5 * bonus, 4), 8.0, 65.0), 2)
            lar = round(clip(random.gauss(2.0 + 1.8 * lag, 0.4), 1.0, 5.0), 2)
            adu = round(clip(random.gauss(2.2 + 0.6 * lr, 0.5), 1.0, 5.0), 2)
            rows.append([dstr(d), d, zid, rain, temp, hum, ovi, bre, lar, adu])
    return rows


def clinical_rows():
    rows = []
    for d in range(1, DAYS + 1):
        lr = 1.6 if d in LAG_WINDOW else 1.0
        for z in ZONES:
            zid, p, r0 = z[0], z[8], z[9]
            fac = PRIMARY[zid]
            cap_fever = FAC[fac][IDX_BEDS] // 2
            if zid == "ZONE_SOUTH" and d in SOUTH_OUTBREAK:
                conf, susp = 0, random.randint(40, 70)
                hot = random.randint(35, 55)
                status = "Missing"
                kits = random.randint(10, 20)
            else:
                conf = poisson((3 + 60 * p) * (r0 / 2) * lr)
                susp = poisson((25 + 150 * p) * lr)
                hot = poisson((6 + 40 * p) * lr)
                status = "Delayed" if random.random() < 0.06 else "Active"
                kits = int(clip(round(susp * 1.2 + random.gauss(0, 4)), 10, min(120, kit_cap(zid))))
            beds = int(clip(round(susp * 0.5 + random.gauss(0, 3)), 0, cap_fever))
            rows.append([dstr(d), d, zid, fac, conf, susp, hot, status, kits, beds])
    return rows


def daily_rows(env, clin):
    """Simulation stream: one row per zone-day, joined from env and clinical tables."""
    clin_map = {(r[1], r[2]): r for r in clin}
    rows = []
    for e in env:
        c = clin_map[(e[1], e[2])]
        # env: [date, day, zone, rain, temp, hum, ovi, bre, lar, adu]
        # clin: [date, day, zone, fac, conf, susp, hot, status, kits, beds]
        rows.append([e[1], e[2], e[3], e[4], e[5], c[4], c[6], c[7]])
    return rows


def intervention_rows():
    recs = []
    for d, zid, act in [(13, "ZONE_CENTRAL", "Residual_Spraying"), (13, "ZONE_WEST", "ULV_Fogging"),
                        (14, "ZONE_CENTRAL", "Residual_Spraying"), (14, "ZONE_WEST", "ULV_Fogging"),
                        (14, "ZONE_SOUTH", "ULV_Fogging")]:
        spray = random.randint(1, min(4, spray_cap(zid)))
        surv = random.randint(0, min(5, surv_cap(zid)))
        recs.append((d, zid, act, spray, surv, 0, "Suspended_Rain", RAIN_MSG))
    recs.append((18, "ZONE_CENTRAL", "Distribute_Kits", 0,
                 min(2, surv_cap("ZONE_CENTRAL")), 0, "Infeasible_Shortage", INFEAS_MSG))
    spray_zones = [z for z in ZONE_IDS if spray_cap(z) > 0]
    for _ in range(31):
        d = random.randint(1, DAYS)
        act = random.choice(ACTIONS)
        if act in OUTDOOR:
            zid = random.choice(spray_zones)
            spray = random.randint(1, min(4, spray_cap(zid)))
        else:
            zid = random.choice(ZONE_IDS)
            spray = 0
        surv = random.randint(0, min(5, surv_cap(zid)))
        kits = random.randint(0, min(200, kit_cap(zid))) if act == "Distribute_Kits" else 0
        if act in OUTDOOR and d in MONSOON:
            status, msg = "Suspended_Rain", RAIN_MSG
        else:
            status, msg = "Completed", ""
        recs.append((d, zid, act, spray, surv, kits, status, msg))
    recs.sort(key=lambda r: r[0])
    rows = []
    for d, zid, act, spray, surv, kits, status, msg in recs:
        cost = round(random.uniform(3000, 45000), 2)
        rows.append([dstr(d), d, zid, act, spray, surv, kits, cost, status, msg])
    return rows


# ---------------- Consistency validation ----------------
def load(name):
    with open(os.path.join(OUT, name), newline="") as f:
        return list(csv.DictReader(f))


def header_of(name):
    with open(os.path.join(OUT, name), newline="") as f:
        return next(csv.reader(f))


def validate():
    errs = []

    def check(cond, msg):
        if not cond:
            errs.append(msg)

    zones = load("zones_demographics.csv")
    infra = load("city_infrastructure.csv")
    sp_edges = load("spatial_network_edges.csv")
    city_edges = load("city_edges.csv")
    env = load("daily_environmental_surveillance.csv")
    clin = load("clinical_reporting_logs.csv")
    daily = load("daily_surveillance.csv")
    ints = load("interventions_and_resources.csv")

    zids = {z["zone_id"] for z in zones}
    check(len(zones) == 7 and len(zids) == 7, "zones: expected 7 unique zone_ids")
    fac = {f["facility_id"]: f for f in infra}
    check(len(fac) == len(infra) == 8, "infrastructure: expected 8 unique facility_ids")
    for f in infra:
        check(f["zone_id"] in zids, f"facility {f['facility_id']}: unknown zone {f['zone_id']}")
        if f["facility_type"] != "Hospital":
            check(f["total_bed_capacity"] == "0", f"{f['facility_id']}: non-hospital has beds")
    pop = {z["zone_id"]: int(z["population"]) for z in zones}
    nodes = zids | set(fac)

    # spatial_network_edges (DengueGNN schema)
    check(14 <= len(sp_edges) <= 18, f"spatial edges: count {len(sp_edges)} outside 14-18")
    sp_by_pair = {}
    for e in sp_edges:
        s, t = e["source_node_id"], e["target_node_id"]
        check(s in zids and t in zids, f"{e['edge_id']}: non-zone endpoint {s}/{t}")
        d = float(e["distance_km"])
        check(2.0 <= d <= 18.0, f"{e['edge_id']}: distance {d} out of range")
        check(8 <= int(e["avg_travel_time_min"]) <= 45, f"{e['edge_id']}: travel time out of range")
        check(0.10 <= float(e["transmission_weight"]) <= 0.95, f"{e['edge_id']}: weight out of range")
        g = int(pop[s] * pop[t] / (d * d * 10000))
        check(int(e["mobility_flow_gravity"]) == g, f"{e['edge_id']}: gravity mismatch")
        sp_by_pair[frozenset((s, t))] = e

    # city_edges (search graph)
    check(header_of("city_edges.csv") ==
          ["source_id", "target_id", "distance_km", "travel_time_min", "daily_commuter_flow", "edge_type"],
          "city_edges: header mismatch")
    seen_pairs = set()
    fac_hit = set()
    zone_zone = 0
    for r in city_edges:
        s, t = r["source_id"], r["target_id"]
        check(s in nodes and t in nodes, f"city_edges {s}->{t}: unknown node")
        check(s != t, f"city_edges {s}: self-loop")
        key = frozenset((s, t))
        check(key not in seen_pairs, f"city_edges {s}-{t}: duplicate undirected edge")
        seen_pairs.add(key)
        check(float(r["distance_km"]) > 0 and int(r["travel_time_min"]) > 0,
              f"city_edges {s}-{t}: non-positive distance or time")
        if r["edge_type"] == "road_and_mobility":
            zone_zone += 1
            check(s in zids and t in zids, f"city_edges {s}-{t}: mobility edge touches a facility")
            check(int(r["daily_commuter_flow"]) > 0, f"city_edges {s}-{t}: zero flow on mobility edge")
            match = sp_by_pair.get(key)
            check(match is not None, f"city_edges {s}-{t}: missing from spatial_network_edges")
            if match:
                check(match["distance_km"] == r["distance_km"] and
                      match["avg_travel_time_min"] == r["travel_time_min"] and
                      match["mobility_flow_gravity"] == r["daily_commuter_flow"],
                      f"city_edges {s}-{t}: values differ from spatial_network_edges")
        elif r["edge_type"] == "road_only":
            check(r["daily_commuter_flow"] == "0", f"city_edges {s}-{t}: road_only with nonzero flow")
            fac_hit.add(t if t in fac else s)
        else:
            errs.append(f"city_edges {s}-{t}: bad edge_type {r['edge_type']}")
    check(zone_zone == len(sp_edges), f"city_edges: {zone_zone} mobility edges vs {len(sp_edges)} spatial edges")
    check(set(fac) <= fac_hit, f"city_edges: facilities unreachable: {sorted(set(fac) - fac_hit)}")
    for zid in zids:
        check(any(zid in k for k in seen_pairs), f"city_edges: zone {zid} has no edges")

    # Environmental and clinical: complete zone x day grid
    expected = {(dstr(d), d, zid) for d in range(1, DAYS + 1) for zid in ZONE_IDS}
    env_keys = {(r["date"], int(r["day_index"]), r["zone_id"]) for r in env}
    clin_keys = {(r["date"], int(r["day_index"]), r["zone_id"]) for r in clin}
    check(len(env) == 210 and env_keys == expected, "env: not a complete 30-day x 7-zone grid")
    check(len(clin) == 210 and clin_keys == expected, "clinical: not a complete 30-day x 7-zone grid")
    for r in env:
        day, rain = int(r["day_index"]), float(r["rainfall_mm"])
        if day in MONSOON:
            check(45.0 <= rain <= 95.0, f"env day {day} {r['zone_id']}: cloudburst rain {rain} out of range")
        else:
            check(rain <= 12.0, f"env day {day} {r['zone_id']}: normal rain {rain} above 12")

    zone_fac = {}
    for r in clin:
        fid, zid, day = r["facility_id"], r["zone_id"], int(r["day_index"])
        check(fid in fac and fac[fid]["facility_type"] == "Hospital",
              f"clinical day {day} {zid}: facility {fid} is not a hospital")
        zone_fac.setdefault(zid, set()).add(fid)
        if fid in fac:
            check(int(r["hospital_beds_occupied"]) <= int(fac[fid]["total_bed_capacity"]),
                  f"clinical day {day} {zid}: beds exceed capacity")
        check(10 <= int(r["diagnostic_kits_used"]) <= 120,
              f"clinical day {day} {zid}: kits out of 10-120")
        check(r["reporting_status"] in REPORT_STATUSES, f"clinical day {day}: bad status")
        if r["reporting_status"] == "Missing":
            check(zid == "ZONE_SOUTH" and day in SOUTH_OUTBREAK,
                  f"clinical day {day} {zid}: Missing status outside stress window")
        if zid == "ZONE_SOUTH" and day in SOUTH_OUTBREAK:
            check(r["reporting_status"] == "Missing" and int(r["confirmed_cases"]) == 0,
                  f"stress day {day}: ZONE_SOUTH must be Missing with 0 confirmed")
            check(35 <= int(r["citizen_hotline_fever_calls"]) <= 55,
                  f"stress day {day}: hotline calls not in 35-55")
            check(40 <= int(r["suspected_fever_cases"]) <= 70,
                  f"stress day {day}: suspected cases not in 40-70")
    for zid, fids in zone_fac.items():
        check(len(fids) == 1, f"{zid}: multiple primary facilities {sorted(fids)}")
        # The primary hospital must be linked to its zone in city_edges
        if zid in zids and fids:
            fid = next(iter(fids))
            check(any({r["source_id"], r["target_id"]} == {zid, fid} for r in city_edges),
                  f"city_edges: {zid} not linked to its primary facility {fid}")

    # daily_surveillance: exact header, full grid, values agree with env + clinical
    check(header_of("daily_surveillance.csv") ==
          ["day", "zone_id", "rainfall_mm", "temp_c", "humidity_pct", "reported_cases",
           "citizen_fever_calls", "reporting_status"], "daily_surveillance: header mismatch")
    env_map = {(r["day_index"], r["zone_id"]): r for r in env}
    clin_map = {(r["day_index"], r["zone_id"]): r for r in clin}
    daily_keys = {(int(r["day"]), r["zone_id"]) for r in daily}
    check(len(daily) == 210 and daily_keys == {(d, z) for _, d, z in expected},
          "daily_surveillance: not a complete 30-day x 7-zone grid")
    for r in daily:
        k = (r["day"], r["zone_id"])
        e, c = env_map.get(k), clin_map.get(k)
        if e is None or c is None:
            errs.append(f"daily_surveillance {k}: no matching env/clinical row")
            continue
        check(r["rainfall_mm"] == e["rainfall_mm"] and r["temp_c"] == e["temp_c"]
              and r["humidity_pct"] == e["humidity_pct"],
              f"daily_surveillance {k}: weather differs from env table")
        check(r["reported_cases"] == c["confirmed_cases"] and
              r["citizen_fever_calls"] == c["citizen_hotline_fever_calls"] and
              r["reporting_status"] == c["reporting_status"],
              f"daily_surveillance {k}: clinical fields differ from clinical table")

    # Interventions
    for r in ints:
        d, zid, status = int(r["day_index"]), r["zone_id"], r["execution_status"]
        check(zid in zids, f"intervention day {d}: unknown zone {zid}")
        check(r["date"] == dstr(d), f"intervention day {d}: date {r['date']} mismatch")
        check(status in STATUSES, f"intervention day {d}: bad status {status}")
        spray, surv, kits = (int(r["spray_teams_deployed"]), int(r["surveillance_teams_deployed"]),
                             int(r["diagnostic_kits_allocated"]))
        check(spray <= spray_cap(zid), f"intervention day {d} {zid}: spray {spray} exceeds zone teams")
        check(surv <= surv_cap(zid), f"intervention day {d} {zid}: surv {surv} exceeds zone teams")
        check(kits <= min(200, kit_cap(zid)), f"intervention day {d} {zid}: kits {kits} exceeds zone capacity")
        check(3000 <= float(r["operational_cost_inr"]) <= 45000, f"intervention day {d}: cost out of range")
        if status == "Completed":
            check(r["bottleneck_reason"] == "", f"intervention day {d}: Completed row has bottleneck")
        else:
            check(r["bottleneck_reason"] != "", f"intervention day {d}: {status} row lacks bottleneck")
    check(len([r for r in ints if int(r["day_index"]) in (13, 14)
               and r["execution_status"] == "Suspended_Rain" and r["bottleneck_reason"] == RAIN_MSG]) >= 2,
          "stress: Days 13-14 Suspended_Rain rows missing")
    check(len([r for r in ints if int(r["day_index"]) == 18 and r["execution_status"] == "Infeasible_Shortage"
               and r["zone_id"] in ("ZONE_CENTRAL", "ZONE_WEST") and r["bottleneck_reason"] == INFEAS_MSG]) == 1,
          "stress: Day 18 Infeasible_Shortage row missing or wrong zone")
    return errs


def main():
    pairs = zone_pairs()  # first: consumes the same random draws as v2 for the shared tables
    write_csv("zones_demographics.csv",
              ["zone_id", "zone_name", "population", "area_sqkm", "centroid_lat", "centroid_lon",
               "base_biting_rate_b0", "vector_to_host_ratio_m", "symptomatic_fraction_p",
               "baseline_r0", "vulnerability_tier"], zones_rows())
    write_csv("city_infrastructure.csv",
              ["facility_id", "facility_name", "facility_type", "zone_id", "lat", "lon",
               "total_bed_capacity", "icu_bed_capacity", "daily_pcr_kit_capacity",
               "spray_teams_stationed", "surveillance_teams_stationed", "vehicles_available"],
              infra_rows())
    write_csv("spatial_network_edges.csv",
              ["edge_id", "source_node_id", "target_node_id", "distance_km",
               "avg_travel_time_min", "mobility_flow_gravity", "transmission_weight"],
              spatial_rows(pairs))
    write_csv("city_edges.csv",
              ["source_id", "target_id", "distance_km", "travel_time_min", "daily_commuter_flow",
               "edge_type"], city_edge_rows(pairs))
    env = env_rows()
    clin = clinical_rows()
    write_csv("daily_environmental_surveillance.csv",
              ["date", "day_index", "zone_id", "rainfall_mm", "temp_c", "humidity_pct",
               "ovitrap_index_pct", "breteau_index", "larval_density_index",
               "adult_vector_index"], env)
    write_csv("clinical_reporting_logs.csv",
              ["date", "day_index", "zone_id", "facility_id", "confirmed_cases",
               "suspected_fever_cases", "citizen_hotline_fever_calls", "reporting_status",
               "diagnostic_kits_used", "hospital_beds_occupied"], clin)
    write_csv("daily_surveillance.csv",
              ["day", "zone_id", "rainfall_mm", "temp_c", "humidity_pct", "reported_cases",
               "citizen_fever_calls", "reporting_status"], daily_rows(env, clin))
    write_csv("interventions_and_resources.csv",
              ["date", "day_index", "zone_id", "action_name", "spray_teams_deployed",
               "surveillance_teams_deployed", "diagnostic_kits_allocated",
               "operational_cost_inr", "execution_status", "bottleneck_reason"],
              intervention_rows())

    errs = validate()
    if errs:
        print(f"\nCONSISTENCY CHECK FAILED: {len(errs)} issue(s)")
        for e in errs[:50]:
            print(" -", e)
        sys.exit(1)
    print("\nCONSISTENCY CHECK PASSED: all 8 tables agree on keys, values, capacities, and stress tests.")


if __name__ == "__main__":
    main()
