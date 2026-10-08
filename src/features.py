"""
Tract-level feature engineering for the Bias Bounty Mapping Equity Challenge.

Everything here is computed from the challenge-provided data only:
  reference/<region>/<region>-overture-{roads,buildings,pois,infrastructure,rail}.parquet
  reference/<region>/<region>-census-acs-housing.parquet
  strata/<region>/<region>-census-tracts.parquet          (TIGER 2025 tract polygons)
  strata/<region>/<region>-strata-tract-table.parquet     (SVI, RUCA, USFS, ... 227 cols)

Domain logic behind the features (see README.md for the full write-up):

* Road gap numerator  = Overture motorway/trunk/primary/secondary length clipped to the
  tract. We compute it EXACTLY (same data the organisers used).
* Road gap denominator = TIGER S1100+S1200 length (removed from the challenge). We proxy
  it with Overture segments that carry an Interstate / US / State route reference
  (TIGER S1100/S1200 is, by definition, the Interstate/US/State highway system), plus
  the length of the *tract boundary* that runs along such a highway. Census tract
  boundaries are drawn on TIGER road edges, so a TIGER highway that forms a tract
  boundary is counted in BOTH neighbouring tracts, while Overture's copy of the same
  road is a few metres off and lands in only one of them. That asymmetry is the
  single biggest source of urban road "gaps".
* Building gap denominator = Microsoft footprint count (removed). Proxies: USFS
  Wildfire-Risk BuildingCount, CarbonPlan building count (both in the strata), and the
  Overture building mix by upstream source (Microsoft ML / OSM / Google / Esri).
* POI gap = HIFLD facilities (fire, EMS, schools) + CBP establishments (removed).
  Overture's exact numerator counts (categories.primary) are computed exactly; the
  reference side is proxied by *any* Overture evidence of the facility (alternate
  categories, names like "Volunteer Fire Department", "... Elementary"), and by the
  tract's population / area / county establishment density.
"""
from __future__ import annotations

import os
import time

import duckdb
import pandas as pd

DATA = os.environ.get("BB_DATA", "data")
FEAT = os.path.join(DATA, "features")
REGIONS = {"eastern-ok": "OK", "maricopa-az": "AZ", "northern-ca": "CA", "south-central-tx": "TX",
           "eastern-wa": "WA"}

FIRE_NAME_RE = (r"(?i)(fire (station|dept|department|district|rescue|house|company|hall|co\b)|"
                r"\bvfd\b|volunteer fire|fire protection district|\bfpd\b|fire & rescue|fire and rescue|"
                r"cal ?fire|fire authority|fire control)")
EMS_NAME_RE = r"(?i)(\bems\b|ambulance|emergency medical|paramedic|rescue squad|medic \d)"
SCHOOL_NAME_RE = (r"(?i)(elementary|middle school|high school|junior high|intermediate school|"
                  r"primary school|\bisd\b|academy|charter school|\bschool\b)")
SCHOOL_CATS = ["elementary_school", "middle_school", "high_school", "school", "private_school", "public_school"]
SCHOOL_LIKE_CATS = SCHOOL_CATS + ["charter_school", "religious_school", "montessori_school",
                                   "preschool", "education", "college_university", "specialty_school"]


def _con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.sql("INSTALL spatial; LOAD spatial; SET threads=10; SET memory_limit='10GB';")
    con.sql("SET preserve_insertion_order=false; SET enable_progress_bar=false;")
    return con


def _t(g: str) -> str:
    # CRS84 -> CONUS Albers (metres). always_xy is REQUIRED (see challenge README).
    return f"ST_Transform({g}, 'EPSG:4326', 'EPSG:5070', always_xy := true)"


def _sql_list(xs) -> str:
    return "[" + ",".join(f"'{x}'" for x in xs) + "]"


def tract_table(con, region):
    con.sql(f"""
        CREATE OR REPLACE TABLE tr AS
        SELECT GEOID, COUNTYFP, STATEFP, geometry g4326, {_t('geometry')} g
        FROM '{DATA}/strata/{region}/{region}-census-tracts.parquet'""")
    con.sql("CREATE OR REPLACE TABLE trb AS SELECT GEOID, ST_Boundary(g) bd FROM tr")


ROUTE_NAME_RE = r"(?i)^((us|u\.s\.|state|sh|hwy|highway|interstate|i|ok|az|ca|tx|fm|rm|sr|loop|spur)[ -]*(hwy|highway|route|rte)?[ -]*\d+[a-z]?)$"


def _route_sql(st: str) -> str:
    """Overture route networks that correspond to TIGER S1100/S1200 (Interstate, US and State
    highway systems, incl. their loops/spurs/business routes and state toll roads). County roads
    (US:TX:Burnet, US:OK:Washita, ...) and Texas FM/RM roads are kept separate."""
    core = (f"^US:(I|US)(:.*)?$|^US:{st}(:(Loop|Spur|Business|Toll|Beltway|Alternate|Truck|Bypass|NTTA|Express|Scenic))?$"
            f"|^US:{st}:(NTTA|Harris:HCTRA|Toll)")
    nets = "list_transform(coalesce(routes, []), x -> coalesce(x.network, ''))"
    return f"""
      len(list_filter({nets}, n -> regexp_matches(n, '{core}'))) > 0
        OR (class IN ('motorway','trunk') AND len(list_filter({nets}, n -> n = '')) > 0) AS is_route,
      len(list_filter({nets}, n -> regexp_matches(n, '^US:I(:.*)?$'))) > 0 AS is_interstate,
      len(list_filter({nets}, n -> regexp_matches(n, '^US:TX:(FM|RM|PR|RE|UR)$'))) > 0 AS is_fm,
      len(list_filter({nets}, n -> regexp_matches(n, '^US:{st}:[A-Z][a-z_]+$')
                                AND NOT regexp_matches(n, '^US:{st}:(Loop|Spur|Business|Toll|Beltway|Alternate|Truck|Bypass|Scenic|Park)$'))) > 0 AS is_county,
      coalesce(len(routes) > 0, false) AS any_route,
      len(list_distinct(list_filter(list_transform(coalesce(routes, []), x -> x.network || '|' || x.ref), x -> x IS NOT NULL))) AS n_ref,
      coalesce(names.primary IS NOT NULL AND NOT regexp_matches(names.primary, '{ROUTE_NAME_RE}'), false) AS local_name"""


def road_features(con, region, st) -> pd.DataFrame:
    con.sql(f"""
        CREATE OR REPLACE TABLE rd AS
        SELECT class, subclass, {_t('geometry')} g, {_route_sql(st)}
        FROM '{DATA}/reference/{region}/{region}-overture-roads.parquet'""")
    con.sql("""
        CREATE OR REPLACE TABLE rd AS SELECT *,
          class IN ('motorway','trunk','primary','secondary') AS is_named FROM rd""")
    # --- clipped lengths per class / flags (intersection with tract polygon)
    con.sql("""
        CREATE OR REPLACE TABLE rl AS
        SELECT tr.GEOID, rd.* EXCLUDE (g), ST_Intersection(rd.g, tr.g) gi
        FROM tr JOIN rd ON ST_Intersects(tr.g, rd.g)""")
    con.sql("CREATE OR REPLACE TABLE rl AS SELECT * EXCLUDE gi, ST_Length(gi) L, gi FROM rl")
    agg = con.sql("""
        SELECT GEOID,
          sum(L) FILTER (is_named)                       AS L_named,
          sum(L) FILTER (class='motorway')               AS L_motorway,
          sum(L) FILTER (class='trunk')                  AS L_trunk,
          sum(L) FILTER (class='primary')                AS L_primary,
          sum(L) FILTER (class='secondary')              AS L_secondary,
          sum(L) FILTER (class='tertiary')               AS L_tertiary,
          sum(L) FILTER (class='residential')            AS L_residential,
          sum(L) FILTER (class='unclassified')           AS L_unclassified,
          sum(L) FILTER (subclass='link' AND is_named)   AS L_named_link,
          sum(L) FILTER (is_route)                       AS L_route,
          sum(L * (greatest(n_ref, 1) + local_name::INT)) FILTER (is_route) AS L_route_mult,
          sum(L) FILTER (is_route AND local_name)        AS L_route_localname,
          sum(L) FILTER (is_route AND n_ref >= 2)        AS L_route_multiref,
          sum(L) FILTER (is_interstate)                  AS L_interstate,
          sum(L) FILTER (is_fm)                          AS L_fm,
          sum(L) FILTER (is_fm AND local_name)           AS L_fm_localname,
          sum(L) FILTER (is_county)                      AS L_county,
          sum(L) FILTER (any_route)                      AS L_anyroute,
          sum(L) FILTER (is_route AND is_named)          AS L_route_named,
          sum(L) FILTER (is_route AND NOT is_named)      AS L_route_lowclass,
          sum(L) FILTER (is_named AND NOT is_route AND NOT is_fm) AS L_named_noroute,
          sum(L) FILTER (is_fm AND is_named)             AS L_fm_named,
          sum(L)                                         AS L_all,
          count(*) FILTER (is_named)                     AS n_named_seg
        FROM rl GROUP BY GEOID""").df()

    # --- Overture length lying within 30 m of the tract boundary (the "boundary highway" copy)
    nb = con.sql("""
        WITH bb AS (SELECT GEOID, ST_Buffer(bd, 30) bz FROM trb)
        SELECT rl.GEOID,
          sum(ST_Length(ST_Intersection(rl.gi, bb.bz))) FILTER (rl.is_named) AS L_named_nearB,
          sum(ST_Length(ST_Intersection(rl.gi, bb.bz))) FILTER (rl.is_route) AS L_route_nearB,
          sum(ST_Length(ST_Intersection(rl.gi, bb.bz))) FILTER (rl.is_fm)    AS L_fm_nearB
        FROM rl JOIN bb USING (GEOID)
        WHERE (rl.is_named OR rl.is_route OR rl.is_fm) AND ST_Intersects(rl.gi, bb.bz)
        GROUP BY rl.GEOID""").df()

    # --- TIGER-side proxy: tract boundary length that runs ALONG a highway (parallel pieces only;
    # crossings produce pieces of ~2*buffer and are dropped by the length filter)
    outs = []
    for name, cond in [("route", "is_route"), ("named", "is_named"), ("fm", "is_fm"),
                       ("interstate", "is_interstate"), ("tertiary", "class='tertiary'")]:
        buf = 25
        q = f"""
          WITH j AS (
            SELECT b.GEOID, any_value(b.bd) bd, ST_Union_Agg(ST_Buffer(h.g, {buf})) u
            FROM trb b JOIN (SELECT g FROM rd WHERE {cond}) h ON ST_DWithin(b.bd, h.g, {buf})
            GROUP BY b.GEOID),
          x AS (SELECT GEOID, unnest(ST_Dump(ST_Intersection(bd, u))) d FROM j)
          SELECT GEOID, sum(CASE WHEN ST_Length(d.geom) > {4 * buf} THEN ST_Length(d.geom) ELSE 0 END) AS Lb_{name}
          FROM x GROUP BY GEOID"""
        outs.append(con.sql(q).df().set_index("GEOID"))
    bnd = pd.concat(outs, axis=1).reset_index()

    geo = con.sql("SELECT GEOID, ST_Area(g) area_m2, ST_Perimeter(g) perim_m FROM tr").df()
    df = geo.merge(agg, on="GEOID", how="left").merge(nb, on="GEOID", how="left").merge(bnd, on="GEOID", how="left")
    return df


def rail_infra_features(con, region) -> pd.DataFrame:
    rail = con.sql(f"""
        SELECT tr.GEOID, sum(ST_Length(ST_Intersection({_t('r.geometry')}, tr.g))) L_rail
        FROM tr JOIN '{DATA}/reference/{region}/{region}-overture-rail.parquet' r
          ON ST_Intersects(tr.g4326, r.geometry)
        GROUP BY 1""").df()
    inf = con.sql(f"""
        SELECT tr.GEOID, count(*) n_infra,
               count(*) FILTER (i.class ILIKE '%bus%') n_infra_bus,
               count(*) FILTER (i.subtype='airport' OR i.class ILIKE '%airport%' OR i.class ILIKE '%aerodrome%') n_infra_air
        FROM tr JOIN '{DATA}/reference/{region}/{region}-overture-infrastructure.parquet' i
          ON ST_Intersects(tr.g4326, ST_Centroid(i.geometry))
        GROUP BY 1""").df()
    return rail.merge(inf, on="GEOID", how="outer")


def building_features(con, region) -> pd.DataFrame:
    con.sql(f"""
        CREATE OR REPLACE TABLE bld AS
        SELECT ST_Centroid(geometry) c, sources[1].dataset ds, len(sources) n_src,
               sources[1].confidence conf, height, num_floors, class, subtype,
               ST_Area({_t('geometry')}) a
        FROM '{DATA}/reference/{region}/{region}-overture-buildings.parquet'""")
    df = con.sql("""
        SELECT tr.GEOID,
          count(*) n_bld,
          count(*) FILTER (ds ILIKE '%microsoft%') n_bld_ms,
          count(*) FILTER (ds ILIKE '%openstreetmap%') n_bld_osm,
          count(*) FILTER (ds ILIKE '%google%') n_bld_google,
          count(*) FILTER (ds ILIKE '%esri%') n_bld_esri,
          count(*) FILTER (n_src > 1) n_bld_multisrc,
          count(*) FILTER (a < 40) n_bld_small,
          count(*) FILTER (a > 1000) n_bld_large,
          count(*) FILTER (height IS NOT NULL) n_bld_height,
          count(*) FILTER (class IS NOT NULL OR subtype IS NOT NULL) n_bld_tagged,
          sum(a) bld_area_sum, median(a) bld_area_med,
          avg(conf) bld_conf_mean
        FROM tr JOIN bld ON ST_Contains(tr.g4326, bld.c)
        GROUP BY 1""").df()
    return df


def poi_features(con, region) -> pd.DataFrame:
    con.sql(f"""
        CREATE OR REPLACE TABLE poi AS
        SELECT geometry p, categories.primary cat, coalesce(categories.alternate, []) alts,
               names.primary nm, confidence conf, operating_status os, basic_category bc,
               sources[1].dataset ds, brand.names.primary brand_nm,
               len(coalesce(websites, [])) > 0 has_web, len(coalesce(phones, [])) > 0 has_phone,
               len(coalesce(addresses, [])) > 0 has_addr
        FROM '{DATA}/reference/{region}/{region}-overture-pois.parquet'""")
    sc, sl = _sql_list(SCHOOL_CATS), _sql_list(SCHOOL_LIKE_CATS)
    df = con.sql(f"""
        SELECT tr.GEOID,
          count(*) n_poi,
          count(*) FILTER (os = 'open') n_poi_open,
          count(*) FILTER (os = 'permanently_closed') n_poi_closed,
          count(*) FILTER (conf >= 0.5) n_poi_c50,
          count(*) FILTER (conf >= 0.8) n_poi_c80,
          avg(conf) poi_conf_mean,
          count(*) FILTER (brand_nm IS NOT NULL) n_poi_brand,
          count(*) FILTER (has_web) n_poi_web, count(*) FILTER (has_phone) n_poi_phone,
          count(*) FILTER (has_addr) n_poi_addr,
          count(DISTINCT cat) n_poi_cats,
          -- fire
          count(*) FILTER (cat = 'fire_department') n_fd,
          count(*) FILTER (cat <> 'fire_department' AND list_contains(alts, 'fire_department')) n_fd_alt,
          count(*) FILTER (coalesce(cat,'') <> 'fire_department' AND regexp_matches(coalesce(nm,''), '{FIRE_NAME_RE}')) n_fire_name,
          count(*) FILTER (cat = 'fire_protection_service') n_fire_prot,
          count(*) FILTER (cat = 'fire_department' AND coalesce(os,'') = 'permanently_closed') n_fd_closed,
          -- ems
          count(*) FILTER (cat = 'ambulance_and_ems_services') n_ems,
          count(*) FILTER (cat <> 'ambulance_and_ems_services' AND list_contains(alts, 'ambulance_and_ems_services')) n_ems_alt,
          count(*) FILTER (coalesce(cat,'') <> 'ambulance_and_ems_services' AND regexp_matches(coalesce(nm,''), '{EMS_NAME_RE}')) n_ems_name,
          -- schools
          count(*) FILTER (cat IN {sc.replace('[', '(').replace(']', ')')}) n_sch,
          count(*) FILTER (cat IN {sl.replace('[', '(').replace(']', ')')} AND cat NOT IN {sc.replace('[', '(').replace(']', ')')}) n_sch_like,
          count(*) FILTER (coalesce(cat,'') NOT IN {sc.replace('[', '(').replace(']', ')')} AND list_has_any(alts, {sc})) n_sch_alt,
          count(*) FILTER (coalesce(cat,'') NOT IN {sc.replace('[', '(').replace(']', ')')} AND regexp_matches(coalesce(nm,''), '{SCHOOL_NAME_RE}')
                           AND NOT regexp_matches(coalesce(cat,''), '(driving|dance|music|art|martial|cosmetology|swim|beauty|flight)')) n_sch_name,
          count(*) FILTER (cat = 'elementary_school') n_sch_elem,
          count(*) FILTER (cat = 'high_school') n_sch_high,
          -- health / civic richness
          count(*) FILTER (cat ILIKE '%hospital%') n_hosp,
          count(*) FILTER (cat IN ('police_department')) n_police,
          count(*) FILTER (bc IN ('restaurant','fast_food_restaurant','cafe','bar')) n_food,
          count(*) FILTER (bc ILIKE '%store%' OR bc ILIKE '%shop%') n_shop,
          count(*) FILTER (bc ILIKE '%religious%' OR cat ILIKE '%church%') n_relig,
          count(*) FILTER (ds ILIKE '%meta%') n_poi_meta,
          count(*) FILTER (ds ILIKE '%microsoft%') n_poi_msft,
          count(*) FILTER (ds ILIKE '%foursquare%') n_poi_fsq
        FROM tr JOIN poi ON ST_Contains(tr.g4326, poi.p)
        GROUP BY 1""").df()
    return df


def facility_points(region: str, radius: float = 500.0, force: bool = False) -> pd.DataFrame:
    """Neighbour-aware facility evidence. One row per (Overture facility point, candidate tract)
    for every tract within `radius` m: d > 0 is the distance to the boundary of the tract that
    contains the point, d < 0 is minus the distance to a neighbouring tract. The model spreads
    each facility's reference (HIFLD) copy over its candidate tracts with Phi(d / sigma_loc):
    the two maps place the same station a few tens of metres apart, so stations near a tract
    boundary leak into the neighbour."""
    out = os.path.join(FEAT, f"{region}-facpts.parquet")
    if os.path.exists(out) and not force:
        return pd.read_parquet(out)
    con = _con()
    tract_table(con, region)
    sc = "(" + ",".join(f"'{c}'" for c in SCHOOL_CATS) + ")"
    con.sql(f"""
        CREATE OR REPLACE TABLE pts AS
        WITH p AS (
          SELECT id, categories.primary cat, coalesce(categories.alternate, []) alts,
                 coalesce(names.primary, '') nm, {_t('geometry')} g
          FROM '{DATA}/reference/{region}/{region}-overture-pois.parquet')
        SELECT id, g, 'fire' typ, cat = 'fire_department' AS strong FROM p
          WHERE cat = 'fire_department' OR list_contains(alts, 'fire_department')
             OR regexp_matches(nm, '{FIRE_NAME_RE}')
        UNION ALL
        SELECT id, g, 'ems', cat = 'ambulance_and_ems_services' FROM p
          WHERE cat = 'ambulance_and_ems_services' OR list_contains(alts, 'ambulance_and_ems_services')
             OR regexp_matches(nm, '{EMS_NAME_RE}')
        UNION ALL
        SELECT id, g, 'sch', coalesce(cat IN {sc}, false) FROM p
          WHERE coalesce(cat IN {sc}, false) OR list_has_any(alts, {_sql_list(SCHOOL_CATS)})
             OR (regexp_matches(nm, '{SCHOOL_NAME_RE}')
                 AND NOT regexp_matches(coalesce(cat, ''), '(driving|dance|music|art|martial|cosmetology|swim|beauty|flight)'))""")
    df = con.sql(f"""
        SELECT pts.id, pts.typ, pts.strong, tr.GEOID,
               CASE WHEN ST_Contains(tr.g, pts.g) THEN ST_Distance(pts.g, trb.bd)
                    ELSE -ST_Distance(pts.g, tr.g) END AS d
        FROM pts JOIN tr ON ST_DWithin(tr.g, pts.g, {radius}) JOIN trb USING (GEOID)""").df()
    df.to_parquet(out, index=False)
    print(f"  [{region}] facility points: {df.id.nunique()} points, {len(df)} point-tract pairs")
    return df


def neighbor_table(con) -> pd.DataFrame:
    """Queen-adjacency with shared-boundary length (metres)."""
    return con.sql("""
        SELECT a.GEOID g1, b.GEOID g2, ST_Length(ST_Intersection(a.bd, b.bd)) shared_m
        FROM trb a JOIN trb b ON ST_Intersects(a.bd, b.bd) AND a.GEOID < b.GEOID""").df()


def build_region(region: str, force: bool = False) -> pd.DataFrame:
    os.makedirs(FEAT, exist_ok=True)
    out = os.path.join(FEAT, f"{region}.parquet")
    st = REGIONS[region]
    if os.path.exists(out) and not force:
        return pd.read_parquet(out)
    t0 = time.time()
    con = _con()
    tract_table(con, region)
    parts = []
    for name, fn in [("roads", lambda: road_features(con, region, st)),
                     ("rail/infra", lambda: rail_infra_features(con, region)),
                     ("buildings", lambda: building_features(con, region)),
                     ("pois", lambda: poi_features(con, region))]:
        parts.append(fn())
        print(f"  [{region}] {name:10s} done  {time.time() - t0:6.1f}s", flush=True)
    nbr = neighbor_table(con)
    nbr.to_parquet(os.path.join(FEAT, f"{region}-neighbors.parquet"), index=False)

    df = parts[0]
    for p in parts[1:]:
        df = df.merge(p, on="GEOID", how="left")
    acs = pd.read_parquet(f"{DATA}/reference/{region}/{region}-census-acs-housing.parquet",
                          columns=["GEOID", "housing_units"])
    strata = pd.read_parquet(f"{DATA}/strata/{region}/{region}-strata-tract-table.parquet")
    df = df.merge(acs, on="GEOID", how="left").merge(strata, on="GEOID", how="left")
    df.insert(1, "region", region)
    df.to_parquet(out, index=False)
    print(f"  [{region}] saved {df.shape} -> {out}  ({time.time() - t0:.0f}s)", flush=True)
    return df


def build_all(force: bool = False) -> pd.DataFrame:
    return pd.concat([build_region(r, force) for r in REGIONS], ignore_index=True)


if __name__ == "__main__":
    import sys
    build_all(force="--force" in sys.argv)
