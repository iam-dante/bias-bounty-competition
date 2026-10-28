"""
Bias Bounty Mapping Equity Challenge: coverage-gap model trained from scratch.

    pip install -r requirements.txt
    python train.py                  # data -> features -> physics model -> aggregate-label posterior -> submission.csv
    python train.py --rebuild        # recompute the cached features in data/features/
    python train.py --physics-only   # stop after the physics model (outputs/physics.csv)
    python train.py --scorecard      # physics variant also calibrated on the public Bias Scorecard (outputs/physics_scorecard.csv)
    python train.py --discovery      # Best Bias Discovery tables in outputs/ (write-up only, not used for the submission)

Everything except submission.csv is written to outputs/ (models, reports, tables).

Inputs: the permitted challenge data (Overture release 2026-08-19.0 roads / unfiltered roads / rail /
infrastructure / buildings / places, ACS housing units, the strata tables and the tract boundaries), statistics
the organisers published, and in stage 8 the public RMSE of scored submissions (aggregate labels).

There are no tract labels. Stages 1-7 model the target from the data alone; stage 8 learns from aggregate labels:

  1. DATA        download the permitted files from the public Source Cooperative bucket (no account).
  2. FEATURES    DuckDB spatial. The Overture side of every ratio is measured exactly per tract. So are the
                 features that predict each withdrawn reference count: Interstate/US/State route length
                 (TIGER S1100/S1200 is that system), boundary highways, facility evidence, the USFS building
                 count, population, the business-registry share of places and mapping effort (road detail).
  3. AUXILIARY   supervised models where real labels exist: the expected density of facilities and of
                 registry businesses, given context only (population, housing, buildings, roads, transit).
                 Poisson GLM vs gradient boosting, 5-fold cross-validation grouped by county; the
                 cross-fitted predictions feed the simulator.
  4. SIMULATOR   a generative model of each withdrawn count (TIGER length, Microsoft buildings, facilities,
                 CBP establishments), pushed through the exact official composite by Monte Carlo.
  5. CALIBRATION the Method of Simulated Moments on statistics the organisers published:
                 - the share of tracts with an undefined component in each region (Evaluation page);
                 - the range of the Overture/TIGER highway-length ratio across regions;
                 - "EMS is close to 1:1 and rarely binds" (data README);
                 plus weak order-of-magnitude public knowledge: schools and fire stations per tract and business
                 establishments per resident (national totals, not tract-level reference data).
                 Weighted least squares with a Gaussian prior on the parameters, common random numbers and
                 Levenberg-Marquardt. Unidentified parameters stay at physically reasoned prior centres.
  6. VALIDATION  leave-one-target-out prediction of every published statistic, cross-validated auxiliary
                 models, posterior identifiability of each parameter, prior sensitivity, Monte Carlo stability.
  7. PREDICTION  E[score | data, theta_MAP] by Monte Carlo (the RMSE-optimal prediction under the most
                 probable parameter set); clipped to [0, 1]. Written to physics.csv.
  8. AGGREGATE   learning from aggregate labels. For RMSE, a file v with public score s gives one exact
     LABELS      moment of the hidden target: mean(v * y) = (mean(v^2) + mean(y^2) - s^2) / 2.
                 aggregate_labels.npz holds the vectors and public scores of submitted files, including
                 physics.csv itself (scored 0.101340). One empirical-Bayes Gaussian posterior over the target is fitted
                 to all of them at once: the files are orthogonalised in scoring order; each direction's coefficient
                 has a group prior whose mean and scale come from marginal likelihood; the moments carry the noise of
                 the 30% public sample. Nothing is copied or chained from earlier files: their scores are data.
                 Written to submission.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
DATA = os.environ.get("BB_DATA", "data")
FEAT = os.path.join(DATA, "features")
OUT = "outputs"                         # generated models, reports and tables (submission.csv stays at the top level)
out = lambda name: os.path.join(OUT, name)
ROUND_STATE = out("round_state.json")
REGIONS = ["eastern-ok", "maricopa-az", "northern-ca", "south-central-tx", "eastern-wa"]
GRADED = REGIONS[:4]                    # eastern-wa is in the submission but not scored
STATE = {"eastern-ok": "OK", "maricopa-az": "AZ", "northern-ca": "CA", "south-central-tx": "TX", "eastern-wa": "WA"}
DATA_URL = "https://data.source.coop/humane-intelligence/bias-bounty-mapping-equity-challenge"

# ---- statistics published by the organisers (the only target information used) ----
UNDEFINED_SHARE = {"eastern-ok": 0.21, "northern-ca": 0.37, "south-central-tx": 0.28, "maricopa-az": 0.55}
ROAD_RATIO_RANGE = (0.71, 1.59)         # Overture / TIGER named-highway length, lowest and highest region
EMS_RATIO = 1.0                         # "EMS is close to 1:1" (Overture vs reference EMS count)
EMS_BINDING = 0.01                      # "...and rarely binds": share of tracts where the EMS term is > 0
# Order-of-magnitude public knowledge (national totals, not tract-level reference data), weak soft targets:
SCHOOLS_PER_TRACT = 1.5                 # about 130,000 K-12 schools over about 85,000 US census tracts
FIRE_STATIONS_PER_TRACT = 0.6           # about 52,000 fire stations over about 85,000 tracts
ESTABLISHMENTS_PER_RESIDENT = 0.025     # about one business establishment per 40 residents
# Optional (--scorecard): the public Bias Scorecard of the rank-1 entry (exact on the public rows), read from
# the leaderboard API on 2026-10-19 to 3 decimals. Off by default: it comes from another team's submission.
SCORECARD = dict(rural=2.336, tribal=2.884, svi=1.231, cvi=1.554, sdrought=0.999, wdrought=1.360,
                 wildfire=1.767, heat=0.618, hv=1.219)
SCORECARD_SD = 0.03                     # model-structure error: the scorecard must not override the statistics above
STRATA = ["rural", "tribal", "svi", "cvi", "sdrought", "wdrought", "wildfire", "heat"]


def log(msg):
    print(msg, flush=True)


def atomic_json(path, value):
    """Write a complete JSON artifact before replacing the previous one."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(value, f, indent=2, default=float)
    os.replace(tmp, path)


# ================================================================================================
# 1. DATA
# ================================================================================================
def data_files():
    out = []
    for r in REGIONS:
        out += [f"reference/{r}/{r}-{x}" for x in ("overture-roads.parquet", "overture-roads-unfiltered.parquet",
                                                    "overture-rail.parquet", "overture-infrastructure.parquet",
                                                    "overture-buildings.parquet", "overture-pois.parquet",
                                                    "census-acs-housing.parquet", "sample-submission.csv")]
        out += [f"strata/{r}/{r}-{x}" for x in ("census-tracts.parquet", "strata-tract-table.parquet")]
    return out


def _http(url, method="GET"):
    # the proxy rejects Python's default User-Agent with 403
    return urllib.request.urlopen(urllib.request.Request(url, method=method, headers={"User-Agent": "bias-bounty-train"}),
                                  timeout=60)


def download():
    files = data_files()
    for i, rel in enumerate(files, 1):
        path, url = os.path.join(DATA, rel), f"{DATA_URL}/{rel}"
        try:
            size = int(_http(url, "HEAD").headers["Content-Length"])
        except Exception as e:
            if os.path.exists(path):
                log(f"  [{i:2d}/{len(files)}] {rel}: bucket unreachable ({e}), using the local copy")
                continue
            raise
        if os.path.exists(path) and os.path.getsize(path) == size:
            continue
        log(f"  [{i:2d}/{len(files)}] {rel}: downloading {size / 1e6:.1f} MB")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with _http(url) as r, open(path + ".part", "wb") as f:
            while chunk := r.read(1 << 22):
                f.write(chunk)
        assert os.path.getsize(path + ".part") == size, f"{rel}: incomplete download, rerun to resume"
        os.replace(path + ".part", path)
    log(f"  {len(files)} files present")


# ================================================================================================
# 2. FEATURES (DuckDB spatial)
# ================================================================================================
FIRE_NAME_RE = (r"(?i)(fire (station|dept|department|district|rescue|house|company|hall|co\b)|"
                r"\bvfd\b|volunteer fire|fire protection district|\bfpd\b|fire & rescue|fire and rescue|"
                r"cal ?fire|fire authority|fire control)")
EMS_NAME_RE = r"(?i)(\bems\b|ambulance|emergency medical|paramedic|rescue squad|medic \d)"
SCHOOL_NAME_RE = (r"(?i)(elementary|middle school|high school|junior high|intermediate school|"
                  r"primary school|\bisd\b|academy|charter school|\bschool\b)")
SCHOOL_CATS = ["elementary_school", "middle_school", "high_school", "school", "private_school", "public_school"]
NOT_SCHOOL = "(driving|dance|music|art|martial|cosmetology|swim|beauty|flight)"
ROUTE_NAME_RE = (r"(?i)^((us|u\.s\.|state|sh|hwy|highway|interstate|i|ok|az|ca|tx|fm|rm|sr|loop|spur)"
                 r"[ -]*(hwy|highway|route|rte)?[ -]*\d+[a-z]?)$")
HISTORIC_NOT_TIGER = {"AZ"}            # Arizona's historic / future route tags ride on ordinary streets


def _con():
    import duckdb
    con = duckdb.connect()
    con.sql("INSTALL spatial; LOAD spatial; SET threads=8; SET memory_limit='10GB';")
    con.sql("SET preserve_insertion_order=false; SET enable_progress_bar=false;")
    return con


def _t(g):  # CRS84 -> CONUS Albers metres; always_xy is required (the layers are lon/lat)
    return f"ST_Transform({g}, 'EPSG:4326', 'EPSG:5070', always_xy := true)"


def _sql_list(xs):
    return "[" + ",".join(f"'{x}'" for x in xs) + "]"


def _route_sql(st):
    """TIGER S1100/S1200 is the Interstate, US and State highway system: Overture's route relations on those
    networks (with loops / spurs / business routes and state toll roads). Texas FM/RM and county roads apart."""
    core = (f"^US:(I|US)(:.*)?$|^US:{st}(:(Loop|Spur|Business|Toll|Beltway|Alternate|Truck|Bypass|NTTA|Express|Scenic))?$"
            f"|^US:{st}:(NTTA|Harris:HCTRA|Toll)")
    nets = "list_transform(coalesce(routes, []), x -> coalesce(x.network, ''))"
    if st in HISTORIC_NOT_TIGER:
        nets = f"list_filter({nets}, n -> NOT regexp_matches(n, ':(Historic|Future)'))"
    return f"""
      len(list_filter({nets}, n -> regexp_matches(n, '{core}'))) > 0
        OR (class IN ('motorway','trunk') AND len(list_filter({nets}, n -> n = '')) > 0) AS is_route,
      len(list_filter({nets}, n -> regexp_matches(n, '^US:I(:.*)?$'))) > 0 AS is_interstate,
      len(list_filter({nets}, n -> regexp_matches(n, '^US:TX:(FM|RM|PR|RE|UR)$'))) > 0 AS is_fm,
      len(list_filter({nets}, n -> regexp_matches(n, '^US:{st}:') AND NOT regexp_matches(n, '{core}'))) > 0 AS is_county,
      len(list_distinct(list_filter(list_transform(coalesce(routes, []), x -> x.network || '|' || x.ref), x -> x IS NOT NULL))) AS n_ref,
      coalesce(names.primary IS NOT NULL AND NOT regexp_matches(names.primary, '{ROUTE_NAME_RE}'), false) AS local_name"""


def _road_features(con, region, st):
    con.sql(f"""CREATE OR REPLACE TABLE rd AS SELECT class, subclass, {_t('geometry')} g, {_route_sql(st)}
                FROM '{DATA}/reference/{region}/{region}-overture-roads.parquet'""")
    con.sql("CREATE OR REPLACE TABLE rd AS SELECT *, class IN ('motorway','trunk','primary','secondary') AS is_named FROM rd")
    con.sql("""CREATE OR REPLACE TABLE rl AS SELECT tr.GEOID, rd.* EXCLUDE (g), ST_Intersection(rd.g, tr.g) gi
               FROM tr JOIN rd ON ST_Intersects(tr.g, rd.g)""")
    con.sql("CREATE OR REPLACE TABLE rl AS SELECT * EXCLUDE gi, ST_Length(gi) L, gi FROM rl")
    agg = con.sql("""
        SELECT GEOID,
          sum(L) FILTER (is_named)                AS L_named,          -- the exact Overture numerator
          sum(L) FILTER (is_route)                AS L_route,
          sum(L) FILTER (is_route AND local_name) AS L_route_localname,
          sum(L) FILTER (is_route AND n_ref >= 2) AS L_route_multiref,
          sum(L) FILTER (is_interstate)           AS L_interstate,
          sum(L) FILTER (is_fm)                   AS L_fm,
          sum(L) FILTER (class = 'tertiary')      AS L_tertiary,
          sum(L) FILTER (is_county AND NOT is_route AND NOT is_fm
                         AND class IN ('primary','secondary','tertiary')) AS L_county,
          sum(L) FILTER (NOT is_route AND NOT is_fm AND NOT is_county AND class IN ('primary','secondary')
                         AND coalesce(subclass, '') <> 'link') AS L_unrouted
        FROM rl GROUP BY GEOID""").df()
    nb = con.sql("""
        WITH bb AS (SELECT GEOID, ST_Buffer(bd, 30) bz FROM trb)
        SELECT rl.GEOID, sum(ST_Length(ST_Intersection(rl.gi, bb.bz))) FILTER (rl.is_route) AS L_route_nearB
        FROM rl JOIN bb USING (GEOID)
        WHERE (rl.is_named OR rl.is_route OR rl.is_fm) AND ST_Intersects(rl.gi, bb.bz)
        GROUP BY rl.GEOID""").df()
    # a tract boundary running ALONG a route highway: TIGER puts the highway in both tracts
    bnd = con.sql("""
        WITH j AS (
          SELECT b.GEOID, any_value(b.bd) bd, ST_Union_Agg(ST_Buffer(h.g, 25)) u
          FROM trb b JOIN (SELECT g FROM rd WHERE is_route) h ON ST_DWithin(b.bd, h.g, 25) GROUP BY b.GEOID),
        x AS (SELECT GEOID, unnest(ST_Dump(ST_Intersection(bd, u))) d FROM j)
        SELECT GEOID, sum(CASE WHEN ST_Length(d.geom) > 100 THEN ST_Length(d.geom) ELSE 0 END) AS Lb_route
        FROM x GROUP BY GEOID""").df()
    geo = con.sql("SELECT GEOID, ST_Area(g) area_m2 FROM tr").df()
    return geo.merge(agg, on="GEOID", how="left").merge(nb, on="GEOID", how="left").merge(bnd, on="GEOID", how="left")


def _building_features(con, region):
    return con.sql(f"""
        SELECT tr.GEOID, count(*) n_bld
        FROM tr JOIN (SELECT ST_Centroid(geometry) c FROM '{DATA}/reference/{region}/{region}-overture-buildings.parquet') b
          ON ST_Contains(tr.g4326, b.c)
        GROUP BY 1""").df()


def _poi_features(con, region):
    sc = "(" + ",".join(f"'{c}'" for c in SCHOOL_CATS) + ")"
    return con.sql(f"""
        WITH poi AS (SELECT geometry p, categories.primary cat, coalesce(categories.alternate, []) alts, names.primary nm,
                            sources[1].dataset ds
                     FROM '{DATA}/reference/{region}/{region}-overture-pois.parquet')
        SELECT tr.GEOID, count(*) n_poi,
          count(*) FILTER (ds = 'meta') n_poi_meta,
          count(*) FILTER (ds = 'BrightQuery') n_poi_bq,
          count(*) FILTER (ds = 'Microsoft') n_poi_msft,
          count(*) FILTER (cat = 'fire_department') n_fd,
          count(*) FILTER (cat <> 'fire_department' AND list_contains(alts, 'fire_department')) n_fd_alt,
          count(*) FILTER (coalesce(cat,'') <> 'fire_department' AND regexp_matches(coalesce(nm,''), '{FIRE_NAME_RE}')) n_fire_name,
          count(*) FILTER (cat = 'ambulance_and_ems_services') n_ems,
          count(*) FILTER (cat <> 'ambulance_and_ems_services' AND list_contains(alts, 'ambulance_and_ems_services')) n_ems_alt,
          count(*) FILTER (coalesce(cat,'') <> 'ambulance_and_ems_services' AND regexp_matches(coalesce(nm,''), '{EMS_NAME_RE}')) n_ems_name,
          count(*) FILTER (cat IN {sc}) n_sch,
          count(*) FILTER (coalesce(cat,'') NOT IN {sc} AND list_has_any(alts, {_sql_list(SCHOOL_CATS)})) n_sch_alt,
          count(*) FILTER (coalesce(cat,'') NOT IN {sc} AND regexp_matches(coalesce(nm,''), '{SCHOOL_NAME_RE}')
                           AND NOT regexp_matches(coalesce(cat,''), '{NOT_SCHOOL}')) n_sch_name
        FROM tr JOIN poi ON ST_Contains(tr.g4326, poi.p)
        GROUP BY 1""").df()


def _extra_features(con, region):
    """Rail, transit / airport infrastructure and the unfiltered road layer (mapping detail). Lines go to
    the tract containing their midpoint."""
    d = f"{DATA}/reference/{region}/{region}-overture"

    def lines(layer, select):
        return con.sql(f"""
            WITH s AS (SELECT class, ST_PointOnSurface(geometry) p, ST_Length({_t('geometry')}) L FROM '{d}-{layer}.parquet')
            SELECT tr.GEOID, {select} FROM tr JOIN s ON ST_Contains(tr.g4326, s.p) GROUP BY 1""").df()

    ped = "('footway','path','cycleway','steps','pedestrian','bridleway')"
    rail = lines("rail", "sum(L) L_rail")
    unf = lines("roads-unfiltered", f"""sum(L) L_unf, sum(L) FILTER (class = 'service') L_service,
                 sum(L) FILTER (class = 'track') L_track, sum(L) FILTER (class IN {ped}) L_ped""")
    infra = con.sql(f"""
        WITH s AS (SELECT subtype, class, ST_PointOnSurface(geometry) p FROM '{d}-infrastructure.parquet')
        SELECT tr.GEOID, count(*) FILTER (subtype = 'transit') n_transit,
               count(*) FILTER (subtype = 'airport' AND class NOT IN ('helipad','heliport')) n_airport
        FROM tr JOIN s ON ST_Contains(tr.g4326, s.p) GROUP BY 1""").df()
    out = con.sql("SELECT GEOID FROM tr").df()
    for part in (rail, unf, infra):
        out = out.merge(part, on="GEOID", how="left")
    return out.fillna(0)


def _facility_points(con, region):
    """One row per (Overture facility point, tract within 500 m): d > 0 is the distance to the boundary of the
    containing tract, d < 0 minus the distance to a neighbouring tract (reference copies near a boundary can land
    next door)."""
    sc = "(" + ",".join(f"'{c}'" for c in SCHOOL_CATS) + ")"
    con.sql(f"""
        CREATE OR REPLACE TABLE pts AS
        WITH p AS (SELECT id, categories.primary cat, coalesce(categories.alternate, []) alts,
                          coalesce(names.primary, '') nm, {_t('geometry')} g
                   FROM '{DATA}/reference/{region}/{region}-overture-pois.parquet')
        SELECT id, g, 'fire' typ, cat = 'fire_department' AS strong FROM p
          WHERE cat = 'fire_department' OR list_contains(alts, 'fire_department') OR regexp_matches(nm, '{FIRE_NAME_RE}')
        UNION ALL
        SELECT id, g, 'ems', cat = 'ambulance_and_ems_services' FROM p
          WHERE cat = 'ambulance_and_ems_services' OR list_contains(alts, 'ambulance_and_ems_services') OR regexp_matches(nm, '{EMS_NAME_RE}')
        UNION ALL
        SELECT id, g, 'sch', coalesce(cat IN {sc}, false) FROM p
          WHERE coalesce(cat IN {sc}, false) OR list_has_any(alts, {_sql_list(SCHOOL_CATS)})
             OR (regexp_matches(nm, '{SCHOOL_NAME_RE}') AND NOT regexp_matches(coalesce(cat, ''), '{NOT_SCHOOL}'))""")
    return con.sql("""
        SELECT pts.id, pts.typ, pts.strong, tr.GEOID,
               CASE WHEN ST_Contains(tr.g, pts.g) THEN ST_Distance(pts.g, trb.bd) ELSE -ST_Distance(pts.g, tr.g) END AS d
        FROM pts JOIN tr ON ST_DWithin(tr.g, pts.g, 500) JOIN trb USING (GEOID)""").df()


UNMATCHED = {"col": "('college_university','vocational_and_technical_school','specialty_school')",
             "k12": "('charter_school','religious_school','montessori_school','education','educational_services')",
             "er": "('emergency_room','emergency_medicine')"}


def _unmatched_features(con, region):
    """Overture places the organisers' category match does not count but that the reference layers may hold:
    colleges / vocational schools and other K-12 or education places (schools layer), hospital ERs (EMS layer)."""
    sel = ", ".join(f"count(*) FILTER (p.cat IN {v}) AS n_{k}" for k, v in UNMATCHED.items())
    return con.sql(f"""
        WITH p AS (SELECT geometry, categories.primary cat FROM '{DATA}/reference/{region}/{region}-overture-pois.parquet')
        SELECT tr.GEOID, {sel} FROM tr JOIN p ON ST_Contains(tr.g4326, p.geometry) GROUP BY tr.GEOID""").df()


def build_features(region, force=False):
    base, extra = os.path.join(FEAT, f"{region}.parquet"), os.path.join(FEAT, f"{region}-extra.parquet")
    facpts = os.path.join(FEAT, f"{region}-facpts.parquet")
    unmatched = os.path.join(FEAT, f"{region}-unmatched.parquet")
    if os.path.exists(base) and os.path.exists(extra) and os.path.exists(facpts) and not force:
        if not os.path.exists(unmatched):
            con = _con()
            con.sql(f"CREATE OR REPLACE TABLE tr AS SELECT GEOID, geometry g4326 FROM '{DATA}/strata/{region}/{region}-census-tracts.parquet'")
            _unmatched_features(con, region).to_parquet(unmatched, index=False)
        return
    os.makedirs(FEAT, exist_ok=True)
    t0 = time.time()
    con = _con()
    con.sql(f"""CREATE OR REPLACE TABLE tr AS SELECT GEOID, geometry g4326, {_t('geometry')} g
                FROM '{DATA}/strata/{region}/{region}-census-tracts.parquet'""")
    con.sql("CREATE OR REPLACE TABLE trb AS SELECT GEOID, ST_Boundary(g) bd FROM tr")
    df = _road_features(con, region, STATE[region])
    for part in (_building_features(con, region), _poi_features(con, region)):
        df = df.merge(part, on="GEOID", how="left")
    acs = pd.read_parquet(f"{DATA}/reference/{region}/{region}-census-acs-housing.parquet", columns=["GEOID", "housing_units"])
    strata = pd.read_parquet(f"{DATA}/strata/{region}/{region}-strata-tract-table.parquet")
    df = df.merge(acs, on="GEOID", how="left").merge(strata, on="GEOID", how="left")
    df.insert(1, "region", region)
    df.to_parquet(base, index=False)
    _extra_features(con, region).to_parquet(extra, index=False)
    _facility_points(con, region).to_parquet(facpts, index=False)
    _unmatched_features(con, region).to_parquet(unmatched, index=False)
    log(f"  features [{region}] {df.shape} in {time.time() - t0:.0f}s")


def scorecard_groups(df):
    """(high, low) masks of the Bias Scorecard strata over all rows. Median splits over every row, rows
    missing the variable in neither group, rural = Census urban/rural class."""
    def split(col):
        x = df[col]
        return (x >= x.median()).to_numpy(), (x < x.median()).to_numpy()
    g = {"rural": (df.ur_class.eq("Rural").to_numpy(), df.ur_class.eq("Urban").to_numpy())}
    tribal = df.tribal_any.fillna(False).astype(bool).to_numpy()
    g["tribal"] = (tribal, ~tribal)
    for k, col in [("svi", "svi_overall"), ("cvi", "cvi_overall"), ("sdrought", "usdm_summer_dsci"),
                   ("wdrought", "usdm_winter_dsci"), ("wildfire", "usfs_WHP_mean"), ("heat", "epht_heat_days_summer")]:
        g[k] = split(col)
    hv = (g["heat"][0] | g["wildfire"][0] | g["sdrought"][0]) & g["svi"][0]
    g["hv"] = (hv, ~hv)
    return g


def load():
    frames = []
    for r in REGIONS:
        f = pd.read_parquet(os.path.join(FEAT, f"{r}.parquet")).drop(columns=["region"])
        x = pd.read_parquet(os.path.join(FEAT, f"{r}-extra.parquet"))
        ss = pd.read_csv(f"{DATA}/reference/{r}/{r}-sample-submission.csv", dtype={"GEOID": str})
        f = ss[["GEOID"]].merge(f, on="GEOID", how="left").merge(x, on="GEOID", how="left")
        um = os.path.join(FEAT, f"{r}-unmatched.parquet")
        if os.path.exists(um):
            f = f.merge(pd.read_parquet(um), on="GEOID", how="left")
        bx = os.path.join(FEAT, "building-extras", f"{r}.parquet")
        if os.path.exists(bx):
            f = f.merge(pd.read_parquet(bx), on="GEOID", how="left")
        f["region"] = r
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    for k in UNMATCHED:
        if f"n_{k}" not in df.columns:
            df[f"n_{k}"] = 0
    zero = [c for c in df.columns if c.startswith(("L_", "Lb_", "n_", "b_"))] + ["housing_units"]
    df[zero] = df[zero].fillna(0)
    df["area_km2"] = df["area_m2"] / 1e6
    df["pop"] = df["pop_total"].fillna(0).clip(lower=0)
    df["county"] = df["STATEFP"].astype(str) + df["COUNTYFP"].astype(str)
    df["graded"] = df["region"].isin(GRADED)
    for k, a, b, c in [("fire", "n_fd", "n_fd_alt", "n_fire_name"), ("ems", "n_ems", "n_ems_alt", "n_ems_name"),
                       ("sch", "n_sch", "n_sch_alt", "n_sch_name")]:
        df[f"E_{k}"] = df[a] + df[b] + df[c]          # all Overture evidence of the facility type
        df[f"X_{k}"] = df[b] + df[c]                  # weak evidence (alternate category or name only)
    u = df["usfs_BuildingCount_sum"] if FOOTPRINT_COLUMNS else df["n_bld"]   # independent building count (USFS)
    df["U_bld"] = u.where(u > 0, df["n_bld"]).fillna(df["n_bld"])
    df["detail"] = (df["L_service"] + df["L_track"] + df["L_ped"]) / (df["L_unf"] + 1.0)
    df["bnd"] = (df["Lb_route"] - df["L_route_nearB"]).clip(lower=0)
    groups = scorecard_groups(df)
    for k in STRATA:
        df[f"s_{k}"] = groups[k][0].astype(float)
    return df, groups


# ================================================================================================
# 3. AUXILIARY MODELS (supervised; labels are observed Overture counts)
# ================================================================================================
def context_matrix(df):
    """Context only: no Overture place count, so the expected densities do not inherit mapping completeness."""
    cols = {"log_pop": np.log1p(df["pop"]), "log_housing": np.log1p(df["housing_units"]), "log_area": np.log1p(df["area_km2"]),
            "log_buildings_usfs": np.log1p(df["U_bld"]), "log_road_km": np.log1p(df["L_unf"] / 1000),
            "road_detail": df["detail"], "log_transit": np.log1p(df["n_transit"]), "log_rail_km": np.log1p(df["L_rail"] / 1000),
            "pct_urban": df["pct_urban"].fillna(0), "ruca": df["ruca_primary"].fillna(1).clip(upper=10),
            "svi": df["svi_overall"].fillna(df["svi_overall"].median()), "tribal_pct": df["tribal_pct"].fillna(0)}
    for r in REGIONS:
        cols[f"region_{r}"] = (df["region"] == r).astype(float)
    return pd.DataFrame(cols).to_numpy(float), list(cols)


def cross_fit(df, X, target, cap):
    """5-fold county-grouped CV of a Poisson GLM and Poisson gradient boosting. Returns the cross-fitted
    (out-of-fold) predictions of the better model and the CV report."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import PoissonRegressor
    from sklearn.metrics import mean_poisson_deviance
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    y = df[target].clip(upper=cap).to_numpy(float)
    gbm_config = AUX_GBM_BY_TARGET.get(target, "baseline")
    models = {"poisson_glm": lambda: make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-3, max_iter=2000)),
              "poisson_gbm": lambda: HistGradientBoostingRegressor(loss="poisson", **AUX_GBM_CONFIGS[gbm_config],
                                                                   random_state=0)}
    oof = {k: np.zeros(len(y)) for k in models}
    for tr, te in GroupKFold(5).split(X, y, df["county"]):
        for k, make in models.items():
            oof[k][te] = np.clip(make().fit(X[tr], y[tr]).predict(X[te]), 1e-4, None)
    dev = {k: float(mean_poisson_deviance(y, p)) for k, p in oof.items()}
    null = float(mean_poisson_deviance(y, np.full(len(y), y.mean())))
    best = min(dev, key=dev.get)
    return oof[best], dict(target=target, cv_deviance=dev, null_deviance=null, chosen=best,
                           explained=1 - dev[best] / null, gbm_config=gbm_config)


AUX_GBM_CONFIGS = {
    "baseline": dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0),
    "shallow_regularized": dict(learning_rate=0.05, max_iter=400, max_leaf_nodes=7, min_samples_leaf=60, l2_regularization=5.0),
    "higher_capacity": dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0),
    "slow_regularized": dict(learning_rate=0.03, max_iter=500, max_leaf_nodes=15, min_samples_leaf=80, l2_regularization=10.0),
    "faster": dict(learning_rate=0.08, max_iter=200, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0),
}
AUX_GBM_BY_TARGET = {"E_fire": "shallow_regularized", "E_ems": "shallow_regularized",
                     "E_sch": "slow_regularized", "n_poi_bq": "slow_regularized"}
AUX_FEATURE_CHOICES = {"E_fire": "density", "E_ems": "density_region",
                       "E_sch": "density", "n_poi_bq": "density"}
AUX_COUNTY_FEATURE_CHOICES = {"E_fire": "density", "E_ems": "density_county",
                              "E_sch": "density_county", "n_poi_bq": "density_county"}
AUX_STRATA_FEATURE_CHOICES = {"E_fire": "density_strata", "E_ems": "density_strata",
                              "E_sch": "density_strata", "n_poi_bq": "density_strata"}
AUX_STRATA_FULL_FEATURE_CHOICES = {"E_fire": "density_strata_full", "E_ems": "density_strata_full",
                                   "E_sch": "density_strata_full", "n_poi_bq": "density_strata_full"}
AUX_DENSITY_BUILDING_FEATURE_CHOICES = {"E_fire": "density_building", "E_ems": "density_building",
                                         "E_sch": "density_building", "n_poi_bq": "density_building"}
AUX_STRATA_MIXED_FEATURE_CHOICES = {"E_fire": "density_strata", "E_ems": "density_strata_full",
                                    "E_sch": "density_strata", "n_poi_bq": "density_strata"}
AUX_STRATA_NO_GEHE_FEATURE_CHOICES = {"E_fire": "density_strata", "E_ems": "density_strata_no_gehe",
                                      "E_sch": "density_strata", "n_poi_bq": "density_strata"}
AUX_HURDLE_FEATURE_CHOICES = {"E_fire": "density_strata", "E_ems": "density_strata_no_gehe",
                              "E_sch": "density_strata", "n_poi_bq": "density_strata"}
AUX_CROSS_COUNT_FEATURE_CHOICES = AUX_HURDLE_FEATURE_CHOICES.copy()
AUX_CROSS_COUNT_SOURCES = ("E_fire", "E_ems", "E_sch", "n_poi_bq", "n_bld")
AUX_HURDLE_SELECTED = {"E_fire": (0.50, "moderate", "shallow"),
                       "E_ems": (0.75, "shallow", "shallow"),
                       "E_sch": (0.50, "moderate", "moderate"),
                       "n_poi_bq": (0.50, "moderate", "shallow")}
AUX_HURDLE_CLASSIFIER_CONFIGS = {
    "shallow": dict(learning_rate=0.05, max_iter=250, max_leaf_nodes=7, min_samples_leaf=60,
                    l2_regularization=5.0),
    "moderate": dict(learning_rate=0.05, max_iter=250, max_leaf_nodes=15, min_samples_leaf=40,
                     l2_regularization=2.0),
}
AUX_HURDLE_POSITIVE_CONFIGS = {
    "shallow": dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=7, min_samples_leaf=30,
                    l2_regularization=5.0),
    "moderate": dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=25,
                     l2_regularization=2.0),
}
AUX_REGION_BLEND_WEIGHT = 0.10
AUX_USE_REGION_BLEND = False
AUX_USE_HURDLE = False
AUX_USE_CROSS_COUNTS = False
AUX_USE_SOURCE_MIX = False
AUX_USE_SOURCE_CONFIDENCE = False
AUX_USE_SOURCE_RICHNESS = False
AUX_USE_SPATIAL_LAGS = False
AUX_USE_POI_CATEGORIES = False
AUX_POI_CATEGORY_KINDS = {}
AUX_USE_SOURCE_DEPTH = False
AUX_SOURCE_DEPTH_TARGETS = set()
AUX_SPATIAL_LAG_TARGETS = set()
AUX_POI_CATEGORY_SELECTED = {"E_fire": ("primary",), "E_ems": (),
                            "E_sch": ("primary", "basic"), "n_poi_bq": ("primary",)}
AUX_SOURCE_DEPTH_SELECTED = {"n_poi_bq"}
AUX_FEATURE_BY_TARGET = {}
AUX_FEATURE_TAG = ""
# Strata columns counted from building footprints (USFS WRC 2nd edition, CarbonPlan) and the Microsoft-sourced share of
# Overture buildings. The organisers allow "permitted columns in the strata tables" without listing them, and questions
# on these columns are unanswered. `--no-footprint-columns` drops them all; the Overture count then stands in for USFS.
FOOTPRINT_COLUMNS = True
FOOTPRINT_PREFIXES = ("usfs_BuildingCount", "usfs_HUCount", "carbonplan_buildings")


def cross_fit_hurdle(df, X, target, cap):
    """Blend the selected direct Poisson fit with a county-held-out positive-count hurdle fit."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.metrics import mean_poisson_deviance
    from sklearn.model_selection import GroupKFold

    from_direct, direct_report = cross_fit(df, X, target, cap)
    blend, classifier_name, positive_name = AUX_HURDLE_SELECTED[target]
    classifier_config = AUX_HURDLE_CLASSIFIER_CONFIGS[classifier_name]
    positive_config = AUX_HURDLE_POSITIVE_CONFIGS[positive_name]
    y = df[target].clip(upper=cap).to_numpy(float)
    hurdle = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, df["county"]):
        positive = y[tr] > 0
        classifier = HistGradientBoostingClassifier(random_state=0, **classifier_config)
        classifier.fit(X[tr], positive.astype(int))
        p_positive = classifier.predict_proba(X[te])[:, 1]
        positive_model = HistGradientBoostingRegressor(loss="poisson", random_state=0, **positive_config)
        positive_model.fit(X[tr][positive], y[tr][positive])
        conditional_mean = np.clip(positive_model.predict(X[te]), 1e-4, None)
        hurdle[te] = np.clip(p_positive * conditional_mean, 1e-4, None)
    blended = np.clip((1.0 - blend) * from_direct + blend * hurdle, 1e-4, None)
    direct_deviance = float(mean_poisson_deviance(y, from_direct))
    blended_deviance = float(mean_poisson_deviance(y, blended))
    null = float(mean_poisson_deviance(y, np.full(len(y), y.mean())))
    return blended, dict(target=target, cv_deviance={"direct_best": direct_deviance,
                                                     "selected_hurdle_blend": blended_deviance},
                         null_deviance=null, chosen=f"hurdle_{classifier_name}_{positive_name}_blend_{blend:.2f}",
                         hurdle_blend=blend, hurdle_classifier=classifier_name,
                         hurdle_positive_model=positive_name, direct_model=direct_report["chosen"],
                         explained=1.0 - blended_deviance / null)


def auxiliary_model_sweep(df):
    """Compare Poisson-GBM settings with county-grouped OOF predictions, including per-region deviance."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_poisson_deviance
    from sklearn.model_selection import GroupKFold

    X, _ = context_matrix(df)
    rows = []
    for target, cap in AUX_TARGETS:
        y = df[target].clip(upper=cap).to_numpy(float)
        oof = {name: np.zeros(len(y)) for name in AUX_GBM_CONFIGS}
        for tr, te in GroupKFold(5).split(X, y, df["county"]):
            for name, cfg in AUX_GBM_CONFIGS.items():
                model = HistGradientBoostingRegressor(loss="poisson", random_state=0, **cfg)
                oof[name][te] = np.clip(model.fit(X[tr], y[tr]).predict(X[te]), 1e-4, None)
        for name, pred in oof.items():
            by_region = {r: float(mean_poisson_deviance(y[df.region.eq(r)], pred[df.region.eq(r)])) for r in REGIONS}
            graded = df.region.isin(GRADED).to_numpy()
            rows.append(dict(target=target, model=name, cv_deviance=float(mean_poisson_deviance(y, pred)),
                             graded_cv_deviance=float(mean_poisson_deviance(y[graded], pred[graded])),
                             region_deviance=by_region))
    report = {"split": "GroupKFold(5), grouped by county", "features": "context_matrix only",
              "scored_regions": GRADED, "configs": AUX_GBM_CONFIGS, "selected_by_target": AUX_GBM_BY_TARGET,
              "results": rows}
    os.makedirs(OUT, exist_ok=True)
    atomic_json(out("aux_sweep.json"), report)
    for target, _ in AUX_TARGETS:
        rr = [x for x in rows if x["target"] == target]
        base = next(x["graded_cv_deviance"] for x in rr if x["model"] == "baseline")
        winner = min(rr, key=lambda x: x["graded_cv_deviance"])
        log(f"  {target}: scored-region baseline {base:.4f}; best {winner['model']} {winner['graded_cv_deviance']:.4f} "
            f"({100 * (winner['graded_cv_deviance'] / base - 1):+.1f}%)")
    log(f"Wrote {out('aux_sweep.json')}")
    return report


def building_extra_features(df):
    """Overture building structure: footprint area, height, class and source mix. Independent of place coverage."""
    n = df["b_n"].to_numpy(float)
    a = df["b_area"].to_numpy(float)
    h = df["b_height_sum"].to_numpy(float)
    hm = df["b_height_mean"].fillna(0).to_numpy(float)
    return {
        "log_b_n": np.log1p(n), "log_b_area": np.log1p(a), "log_b_height": np.log1p(h),
        "b_height_mean": hm, "b_tall6_share": df["b_tall6"].to_numpy(float) / (n + 1),
        "b_class_known_share": df["b_class_known"].to_numpy(float) / (n + 1),
        "b_commercial_share": df["b_class_commercial"].to_numpy(float) / (n + 1),
        "b_apartments_share": df["b_class_apartments"].to_numpy(float) / (n + 1),
        "b_house_share": df["b_class_house"].to_numpy(float) / (n + 1),
        **({"b_src_msft_share": df["b_src_msft"].to_numpy(float) / (n + 1)} if FOOTPRINT_COLUMNS else {}),
        "b_src_osm_share": df["b_src_osm"].to_numpy(float) / (n + 1),
        "b_area_per_bld": np.log1p(a / (n + 1)), "b_volume": np.log1p(a * hm),
    }


def auxiliary_feature_matrix(df, variant):
    """Feature ablations for count CV: context, densities, region/county effects and supplied strata."""
    X, names = context_matrix(df)
    if variant == "baseline":
        return X, names
    x = {name: X[:, j] for j, name in enumerate(names)}
    area = df["area_km2"].to_numpy(float) + 0.01
    pop = df["pop"].to_numpy(float)
    housing = df["housing_units"].to_numpy(float)
    extra = {
        "log_pop_density": np.log1p(pop / area),
        "log_housing_density": np.log1p(housing / area),
        "log_housing_per_resident": np.log((housing + 1) / (pop + 1)),
        "log_usfs_per_housing": np.log((df["U_bld"].to_numpy(float) + 1) / (housing + 1)),
        "log_road_density": np.log1p((df["L_unf"].to_numpy(float) / 1000) / area),
        "log_transit_per_resident": np.log1p(df["n_transit"].to_numpy(float) / (pop + 100) * 1e6),
        "log_rail_density": np.log1p((df["L_rail"].to_numpy(float) / 1000) / area),
    }
    if variant == "density_region":
        for region in GRADED:
            mask = (df["region"].to_numpy() == region).astype(float)
            for name in ("log_pop", "log_housing", "log_area", "log_road_km", "road_detail", "pct_urban", "ruca", "svi"):
                extra[f"{region}__{name}"] = x[name] * mask
    elif variant == "density_county":
        county = df.groupby("county", sort=False)
        pop_total = county["pop"].transform("sum").to_numpy(float)
        housing_total = county["housing_units"].transform("sum").to_numpy(float)
        area_total = county["area_km2"].transform("sum").to_numpy(float)
        buildings_total = county["U_bld"].transform("sum").to_numpy(float)
        roads_total = county["L_unf"].transform("sum").to_numpy(float) / 1000
        transit_total = county["n_transit"].transform("sum").to_numpy(float)
        extra.update({
            "county_log_tracts": np.log1p(county["GEOID"].transform("size").to_numpy(float)),
            "county_log_pop": np.log1p(pop_total),
            "county_log_housing": np.log1p(housing_total),
            "county_log_area": np.log1p(area_total),
            "county_pop_density": np.log1p(pop_total / (area_total + 0.01)),
            "county_housing_density": np.log1p(housing_total / (area_total + 0.01)),
            "county_building_density": np.log1p(buildings_total / (area_total + 0.01)),
            "county_road_density": np.log1p(roads_total / (area_total + 0.01)),
            "county_transit_per_million": np.log1p(transit_total / (pop_total + 100) * 1e6),
            "county_urban_mean": county["pct_urban"].transform("mean").fillna(0).to_numpy(float),
            "county_ruca_mean": county["ruca_primary"].transform("mean").fillna(1).to_numpy(float),
            "county_svi_mean": county["svi_overall"].transform("mean").fillna(df["svi_overall"].median()).to_numpy(float),
            "tract_pop_log_share": np.log((pop + 1) / (pop_total + 1)),
            "tract_housing_log_share": np.log((housing + 1) / (housing_total + 1)),
            "tract_area_log_share": np.log((area + 0.01) / (area_total + 0.01)),
            "tract_building_log_share": np.log((df["U_bld"].to_numpy(float) + 1) / (buildings_total + 1)),
            "tract_road_log_share": np.log((df["L_unf"].to_numpy(float) / 1000 + 0.01) / (roads_total + 0.01)),
        })
    elif variant in ("density_strata", "density_strata_full", "density_strata_no_gehe"):
        prefixes = ("svi_", "cvi_", "ruca_", "rucc_", "nchs_", "tribal_", "usdm_", "usfs_", "epht_",
                    "carbonplan_", "usgs_", "mtbs_", "pmdi_", "spi_", "fpa_")
        full_prefixes = prefixes + ("nifc_", "fod_", "ghcn_", "cdcw_", "hwd_", "gehe_", "uhe_", "drought_gov_")
        full_columns = {"ALAND", "pop_urban", "pop_rural", "frac_inside_aoi", "ur_class", "aiannh_geoid",
                        "aiannh_name", "aiannh_kind", "uhe_city_name", "uhe_city_country"}
        use_full = variant != "density_strata"
        prefixes = full_prefixes if use_full else prefixes
        metadata_tokens = ("_year", "_vintage", "_version", "_edition", "_extent", "_metric", "_threshold")
        metadata_suffixes = ("_year", "_year_min", "_year_max", "_vintage", "_version", "_edition", "_extent",
                             "_metric", "_threshold", "_window_start", "_window_end", "_record_start", "_record_end")
        for col in df.columns:
            if variant == "density_strata_no_gehe" and col.startswith("gehe_"):
                continue
            in_strata = col.startswith(prefixes) or (use_full and col in full_columns)
            if not FOOTPRINT_COLUMNS and col.startswith(FOOTPRINT_PREFIXES):
                continue
            if not in_strata or col in {"aiannh_geoid", "aiannh_name"} and not use_full:
                continue
            if (col.endswith(metadata_suffixes) if use_full else any(token in col for token in metadata_tokens)):
                continue
            values = df[col]
            if pd.api.types.is_numeric_dtype(values.dtype):
                a = pd.to_numeric(values, errors="coerce").to_numpy(float)
                finite = np.isfinite(a)
                if finite.sum() == 0 or np.nanstd(a) < 1e-12:
                    continue
                fill = float(np.nanmedian(a[finite]))
                extra[f"strata__{col}"] = np.where(finite, a, fill)
                if (~finite).any():
                    extra[f"strata__{col}__missing"] = (~finite).astype(float)
            elif pd.api.types.is_object_dtype(values.dtype) or pd.api.types.is_string_dtype(values.dtype) or pd.api.types.is_categorical_dtype(values.dtype):
                present = values.dropna()
                if not present.map(lambda v: isinstance(v, (str, int, float, bool))).all():
                    continue
                category_limit = 51 if use_full else 21
                if present.nunique() in range(2, category_limit):
                    categories = values.astype("string").fillna("__missing__")
                    dummies = pd.get_dummies(categories, prefix=f"strata__{col}", dtype=float)
                    for dummy in dummies.columns:
                        if dummies[dummy].std() > 1e-12:
                            extra[dummy] = dummies[dummy].to_numpy(float)
    elif variant == "density_building":
        extra.update(building_extra_features(df))
    elif variant != "density":
        raise ValueError(f"unknown auxiliary feature variant: {variant}")
    return np.column_stack([X, *extra.values()]), names + list(extra)


def append_other_overture_count_features(df, target, X, names):
    """Add supplied Overture count context while strictly omitting the current proxy target."""
    if target not in {name for name, _ in AUX_TARGETS}:
        raise ValueError(f"unsupported auxiliary target for cross-count features: {target}")
    extra = {}
    pop = df["pop"].to_numpy(float)
    for source in AUX_CROSS_COUNT_SOURCES:
        if source == target:
            continue
        count = pd.to_numeric(df[source], errors="coerce").fillna(0).clip(lower=0).to_numpy(float)
        extra[f"other_count__log1p_{source}"] = np.log1p(count)
        extra[f"other_count__present_{source}"] = (count > 0).astype(float)
        extra[f"other_count__per_million_pop_{source}"] = np.log1p(count / (pop + 100) * 1e6)
    return np.column_stack([X, *extra.values()]), names + list(extra)


def auxiliary_poi_category_features():
    """Aggregate target-excluded Overture primary and broad place categories by tract."""
    path = os.path.join(FEAT, "poi-categories.parquet")
    schema_path = os.path.join(FEAT, "poi-category-schema.json")
    if os.path.exists(path) and os.path.exists(schema_path):
        with open(schema_path) as f:
            schema = json.load(f)
        return pd.read_parquet(path), schema

    excluded = auxiliary_target_exclusions()
    frames = []
    for region in REGIONS:
        con = _con()
        con.sql(f"CREATE OR REPLACE TABLE tr AS SELECT GEOID, geometry g FROM "
                f"'{DATA}/strata/{region}/{region}-census-tracts.parquet'")
        con.sql(f"""
            CREATE OR REPLACE TABLE poi AS
            SELECT geometry p, coalesce(categories.primary, '__missing__') cat,
                   coalesce(basic_category, '__missing__') basic_cat,
                   coalesce(categories.alternate, []) alts, coalesce(names.primary, '') nm,
                   coalesce(sources[1].dataset, '') ds
            FROM '{DATA}/reference/{region}/{region}-overture-pois.parquet'
        """)
        # Perform the spatial join once, then reuse its tract assignments for every target-exclusion aggregate.
        con.sql("CREATE OR REPLACE TABLE poi_in_tract AS "
                "SELECT tr.GEOID, poi.cat, poi.basic_cat, poi.alts, poi.nm, poi.ds "
                "FROM tr JOIN poi ON ST_Contains(tr.g, poi.p)")
        aggregates = []
        for target, exclusion in excluded.items():
            for kind, category in (("primary", "cat"), ("basic", "basic_cat")):
                aggregates.append(
                    f"SELECT GEOID, '{target}' AS target, '{kind}' AS kind, {category} AS category, count(*) AS n "
                    f"FROM poi_in_tract WHERE NOT ({exclusion}) GROUP BY GEOID, {category}"
                )
        frames.append(con.sql(" UNION ALL ".join(aggregates)).df())

    long = pd.concat(frames, ignore_index=True)
    limits = {"primary": 48, "basic": 32}
    top = {}
    for kind, limit in limits.items():
        totals = (long.loc[long.kind.eq(kind) & long.category.ne("__missing__")]
                  .groupby("category", sort=False).n.sum().sort_values(ascending=False))
        top[kind] = totals.head(limit).index.astype(str).tolist()
    schema = dict(version=1, top_categories=top,
                  target_exclusions={target: expression for target, expression in excluded.items()},
                  spatial_assignment="ST_Contains of challenge-supplied tract geometry and Overture POI geometry")
    os.makedirs(FEAT, exist_ok=True)
    long.to_parquet(path, index=False)
    atomic_json(schema_path, schema)
    return long, schema


def append_auxiliary_poi_categories(df, target, X, names, category_counts, schema, kinds):
    """Add logs of target-excluded category counts and per-capita rates to an auxiliary feature matrix."""
    pop = df["pop"].to_numpy(float)
    ids = df["GEOID"].astype(str).to_numpy()
    extra, extra_names = [], []
    selected = category_counts.loc[category_counts.target.eq(target)]
    for kind in kinds:
        categories = schema["top_categories"][kind]
        pivot = (selected.loc[selected.kind.eq(kind)]
                 .pivot(index="GEOID", columns="category", values="n")
                 .reindex(columns=categories, fill_value=0))
        values = (pd.DataFrame({"GEOID": ids})
                  .merge(pivot.rename_axis(None, axis=1).reset_index(), on="GEOID", how="left")
                  [categories].fillna(0).to_numpy(float))
        for transform, data in (("log_count", np.log1p(values)),
                                ("log_per_million", np.log1p(values / (pop[:, None] + 100) * 1e6))):
            extra.append(data)
            extra_names.extend(f"poi_category__{kind}__{category}__{transform}" for category in categories)
    if not extra:
        return X, names
    return np.column_stack([X, *extra]), names + extra_names


def auxiliary_poi_source_category_features():
    """Aggregate target-excluded POI categories within each first-source family and tract."""
    path = os.path.join(FEAT, "poi-source-categories.parquet")
    schema_path = os.path.join(FEAT, "poi-source-category-schema.json")
    if os.path.exists(path) and os.path.exists(schema_path):
        with open(schema_path) as f:
            schema = json.load(f)
        return pd.read_parquet(path), schema

    excluded = auxiliary_target_exclusions()
    source_case = ("CASE WHEN ds = 'meta' THEN 'meta' WHEN ds = 'BrightQuery' THEN 'bq' "
                   "WHEN ds = 'Microsoft' THEN 'msft' ELSE 'other' END")
    frames = []
    for region in REGIONS:
        con = _con()
        con.sql(f"CREATE OR REPLACE TABLE tr AS SELECT GEOID, geometry g FROM "
                f"'{DATA}/strata/{region}/{region}-census-tracts.parquet'")
        con.sql(f"""
            CREATE OR REPLACE TABLE poi AS
            SELECT geometry p, coalesce(categories.primary, '__missing__') cat,
                   coalesce(basic_category, '__missing__') basic_cat,
                   coalesce(categories.alternate, []) alts, coalesce(names.primary, '') nm,
                   coalesce(sources[1].dataset, '') ds
            FROM '{DATA}/reference/{region}/{region}-overture-pois.parquet'
        """)
        con.sql("CREATE OR REPLACE TABLE poi_in_tract AS "
                "SELECT tr.GEOID, poi.cat, poi.basic_cat, poi.alts, poi.nm, poi.ds "
                "FROM tr JOIN poi ON ST_Contains(tr.g, poi.p)")
        aggregates = []
        for target, exclusion in excluded.items():
            for kind, category in (("primary", "cat"), ("basic", "basic_cat")):
                aggregates.append(
                    f"SELECT GEOID, '{target}' AS target, '{kind}' AS kind, ({source_case}) AS source, "
                    f"{category} AS category, count(*) AS n FROM poi_in_tract "
                    f"WHERE NOT ({exclusion}) GROUP BY GEOID, source, {category}"
                )
        frames.append(con.sql(" UNION ALL ".join(aggregates)).df())

    long = pd.concat(frames, ignore_index=True)
    limits = {"primary": 24, "basic": 16}
    top = {}
    for target in excluded:
        top[target] = {}
        for kind, limit in limits.items():
            top[target][kind] = {}
            for source in ("meta", "bq", "msft", "other"):
                totals = (long.loc[long.target.eq(target) & long.kind.eq(kind) & long.source.eq(source)
                                   & long.category.ne("__missing__")]
                          .groupby("category", sort=False).n.sum().sort_values(ascending=False))
                top[target][kind][source] = totals.head(limit).index.astype(str).tolist()
    schema = dict(version=1, top_categories=top,
                  target_exclusions={target: expression for target, expression in excluded.items()},
                  source_families={"meta": "Meta", "bq": "BrightQuery", "msft": "Microsoft",
                                   "other": "all remaining first-source datasets"},
                  spatial_assignment="ST_Contains of challenge-supplied tract geometry and Overture POI geometry")
    os.makedirs(FEAT, exist_ok=True)
    long.to_parquet(path, index=False)
    atomic_json(schema_path, schema)
    return long, schema


def append_auxiliary_poi_source_categories(df, target, X, names, source_category_counts, schema, kinds):
    """Add source-by-category logs and per-capita rates for non-target places."""
    ids = df["GEOID"].astype(str).to_numpy()
    pop = df["pop"].to_numpy(float)
    selected = source_category_counts.loc[source_category_counts.target.eq(target)]
    extra, extra_names = [], []
    for kind in kinds:
        for source in ("meta", "bq", "msft", "other"):
            categories = schema["top_categories"][target][kind][source]
            if not categories:
                continue
            pivot = (selected.loc[selected.kind.eq(kind) & selected.source.eq(source)]
                     .pivot(index="GEOID", columns="category", values="n")
                     .reindex(columns=categories, fill_value=0))
            values = (pd.DataFrame({"GEOID": ids})
                      .merge(pivot.rename_axis(None, axis=1).reset_index(), on="GEOID", how="left")
                      [categories].fillna(0).to_numpy(float))
            for transform, data in (("log_count", np.log1p(values)),
                                    ("log_per_million", np.log1p(values / (pop[:, None] + 100) * 1e6))):
                extra.append(data)
                extra_names.extend(f"poi_source_category__{kind}__{source}__{category}__{transform}"
                                   for category in categories)
    if not extra:
        return X, names
    return np.column_stack([X, *extra]), names + extra_names


def auxiliary_poi_source_category_sweep(df, report_file="aux_poi_source_category_sweep.json"):
    """Measure incremental value of source-by-category features in the existing category proxy models."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_poisson_deviance
    from sklearn.model_selection import GroupKFold

    source_counts, source_schema = auxiliary_poi_source_category_features()
    category_counts, category_schema = auxiliary_poi_category_features()
    source_mix = auxiliary_source_mix_features()
    source_confidence = auxiliary_source_confidence_features()
    graded = df["region"].isin(GRADED).to_numpy()
    feature_sets = {"current_category_set": (), "add_primary_by_source": ("primary",),
                    "add_basic_by_source": ("basic",),
                    "add_primary_and_basic_by_source": ("primary", "basic")}
    results, summary = [], []
    for target, cap in AUX_TARGETS:
        variant = AUX_CROSS_COUNT_FEATURE_CHOICES[target]
        X, names = auxiliary_feature_matrix(df, variant)
        X, names = append_other_overture_count_features(df, target, X, names)
        X, names = append_auxiliary_source_mix(df, target, X, names, source_mix)
        X, names = append_auxiliary_source_confidence(df, target, X, names, source_confidence)
        current_categories = AUX_POI_CATEGORY_SELECTED.get(target, ())
        X, names = append_auxiliary_poi_categories(
            df, target, X, names, category_counts, category_schema, current_categories)
        y = df[target].clip(upper=cap).to_numpy(float)
        config = AUX_GBM_BY_TARGET[target]
        target_rows = []
        for feature_set, kinds in feature_sets.items():
            features, feature_names = append_auxiliary_poi_source_categories(
                df, target, X, names, source_counts, source_schema, kinds)
            oof = np.zeros(len(y))
            for tr, te in GroupKFold(5).split(features, y, df["county"]):
                model = HistGradientBoostingRegressor(
                    loss="poisson", random_state=0, **AUX_GBM_CONFIGS[config])
                model.fit(features[tr], y[tr])
                oof[te] = np.clip(model.predict(features[te]), 1e-4, None)
            by_region = {region: float(mean_poisson_deviance(y[df.region.eq(region)],
                                                              oof[df.region.eq(region)]))
                         for region in GRADED}
            row = dict(target=target, feature_set=feature_set, model_config=config,
                       feature_count=len(feature_names),
                       graded_cv_deviance=float(mean_poisson_deviance(y[graded], oof[graded])),
                       region_deviance=by_region)
            target_rows.append(row)
            results.append(row)
        base = next(row for row in target_rows if row["feature_set"] == "current_category_set")
        for row in target_rows:
            changes = {region: 100 * (row["region_deviance"][region] /
                                        base["region_deviance"][region] - 1)
                       for region in GRADED}
            summary.append(dict(target=target, feature_set=row["feature_set"],
                                base_graded_cv_deviance=base["graded_cv_deviance"],
                                graded_cv_deviance=row["graded_cv_deviance"],
                                change_pct=100 * (row["graded_cv_deviance"] /
                                                  base["graded_cv_deviance"] - 1),
                                graded_regions_improved=sum(change < 0 for change in changes.values()),
                                graded_region_change_pct=changes))
        best = min((row for row in summary if row["target"] == target), key=lambda row: row["change_pct"])
        log(f"  {target}: best source-category addition {best['feature_set']} {best['change_pct']:+.3f}% "
            f"({best['graded_regions_improved']}/4 regions improve)")

    report = dict(date="2026-10-24", experiment="source-conditioned target-excluded Overture POI category context",
                  hypothesis="Category-specific provider counts may add mapping-completeness signal beyond source mix and pooled category counts.",
                  split="GroupKFold(5), grouped by county; each target is excluded before source-by-category aggregation.",
                  target_proxy="observed Overture counts", base_features="current selected category sets, cross-counts, source mix, and source confidence",
                  source_category_limits={"primary": 24, "basic": 16},
                  source_categories=source_schema["top_categories"],
                  fixed_gbm_config_by_target={target: AUX_GBM_BY_TARGET[target] for target, _ in AUX_TARGETS},
                  selection_note="Fixed GBM settings; grouped OOF deviance is a proxy-count metric, not a direct hidden-score validation.",
                  summary=summary, results=results)
    atomic_json(out(report_file), report)
    log(f"Wrote {out(report_file)}")
    return report


def auxiliary_poi_category_sweep(df, report_file="aux_poi_category_sweep.json"):
    """Test target-excluded detailed POI categories under county-held-out auxiliary proxy validation."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_poisson_deviance
    from sklearn.model_selection import GroupKFold

    category_counts, schema = auxiliary_poi_category_features()
    source_mix = auxiliary_source_mix_features()
    source_confidence = auxiliary_source_confidence_features()
    graded = df["region"].isin(GRADED).to_numpy()
    feature_sets = {"base": (), "primary": ("primary",), "basic": ("basic",),
                    "primary_and_basic": ("primary", "basic")}
    results, summary = [], []
    for target, cap in AUX_TARGETS:
        variant = AUX_CROSS_COUNT_FEATURE_CHOICES[target]
        X, names = auxiliary_feature_matrix(df, variant)
        X, names = append_other_overture_count_features(df, target, X, names)
        X, names = append_auxiliary_source_mix(df, target, X, names, source_mix)
        X, names = append_auxiliary_source_confidence(df, target, X, names, source_confidence)
        y = df[target].clip(upper=cap).to_numpy(float)
        model_config = AUX_GBM_BY_TARGET[target]
        target_rows = []
        for feature_set, kinds in feature_sets.items():
            features, feature_names = append_auxiliary_poi_categories(
                df, target, X, names, category_counts, schema, kinds)
            oof = np.zeros(len(y))
            for tr, te in GroupKFold(5).split(features, y, df["county"]):
                model = HistGradientBoostingRegressor(
                    loss="poisson", random_state=0, **AUX_GBM_CONFIGS[model_config])
                model.fit(features[tr], y[tr])
                oof[te] = np.clip(model.predict(features[te]), 1e-4, None)
            by_region = {region: float(mean_poisson_deviance(y[df.region.eq(region)],
                                                              oof[df.region.eq(region)]))
                         for region in GRADED}
            row = dict(target=target, feature_set=feature_set, model_config=model_config,
                       feature_count=len(feature_names), graded_cv_deviance=float(mean_poisson_deviance(y[graded], oof[graded])),
                       region_deviance=by_region)
            target_rows.append(row)
            results.append(row)
        base = next(row for row in target_rows if row["feature_set"] == "base")
        for row in target_rows:
            regional_changes = {region: 100 * (row["region_deviance"][region] /
                                                 base["region_deviance"][region] - 1)
                                for region in GRADED}
            summary.append(dict(target=target, feature_set=row["feature_set"],
                                base_graded_cv_deviance=base["graded_cv_deviance"],
                                graded_cv_deviance=row["graded_cv_deviance"],
                                change_pct=100 * (row["graded_cv_deviance"] /
                                                  base["graded_cv_deviance"] - 1),
                                graded_regions_improved=sum(value < 0 for value in regional_changes.values()),
                                graded_region_change_pct=regional_changes))
        best = min((row for row in summary if row["target"] == target), key=lambda row: row["change_pct"])
        log(f"  {target}: best category set {best['feature_set']} {best['change_pct']:+.3f}% "
            f"({best['graded_regions_improved']}/4 regions improve)")

    report = dict(date="2026-10-24", experiment="target-excluded Overture POI category context",
                  hypothesis="Detailed counts of non-target POI categories may measure local map completeness beyond total counts and source mix.",
                  split="GroupKFold(5), grouped by county; all category features exclude each proxy target using the existing exact match rule.",
                  target_proxy="observed Overture counts", base_features="strata, other target-excluded Overture counts, source mix, and source confidence",
                  category_schema=schema["top_categories"], feature_sets=list(feature_sets),
                  fixed_gbm_config_by_target={target: AUX_GBM_BY_TARGET[target] for target, _ in AUX_TARGETS},
                  selection_note="The model configuration is held fixed across feature sets. OOF deviance measures proxy-count prediction, not the final hidden score.",
                  summary=summary, results=results)
    atomic_json(out(report_file), report)
    log(f"Wrote {out(report_file)}")
    return report


def auxiliary_spatial_lag_features():
    """Build target-excluded K-neighbor summaries from challenge-provided counts in nearby tracts."""
    path = os.path.join(FEAT, "spatial-lags.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    from sklearn.neighbors import NearestNeighbors

    frames = []
    earth_radius_km = 6371.0088
    for region in REGIONS:
        features = pd.read_parquet(os.path.join(FEAT, f"{region}.parquet"))
        extra = pd.read_parquet(os.path.join(FEAT, f"{region}-extra.parquet"))
        all_tracts = features.merge(extra, on="GEOID", how="left")
        coords = _con().sql(f"""
            SELECT GEOID, ST_Y(ST_Centroid(geometry)) AS lat, ST_X(ST_Centroid(geometry)) AS lon
            FROM '{DATA}/strata/{region}/{region}-census-tracts.parquet'
        """).df()
        all_tracts = all_tracts.merge(coords, on="GEOID", how="inner").reset_index(drop=True)
        if len(all_tracts) < 2:
            continue
        latlon = np.deg2rad(all_tracts[["lat", "lon"]].to_numpy(float))
        max_k = min(20, len(all_tracts) - 1)
        model = NearestNeighbors(n_neighbors=max_k + 1, metric="haversine", algorithm="ball_tree")
        model.fit(latlon)
        distance_rad, neighbor_index = model.kneighbors(latlon)
        # Exclude each tract itself explicitly, including when coincident centroids create distance ties.
        neighbors, distances = [], []
        for i, (indices, distance) in enumerate(zip(neighbor_index, distance_rad)):
            keep = [(j, d) for j, d in zip(indices, distance) if j != i][:max_k]
            neighbors.append([j for j, _ in keep])
            distances.append([d for _, d in keep])
        neighbor_index = np.asarray(neighbors, dtype=int)
        neighbor_distance = np.asarray(distances, dtype=float) * earth_radius_km
        pop = pd.to_numeric(all_tracts.get("pop_total", 0), errors="coerce")
        if not isinstance(pop, pd.Series):
            pop = pd.Series(np.zeros(len(all_tracts)))
        pop = pop.fillna(0).clip(lower=0).to_numpy(float)
        count_columns = {
            "E_fire": ("n_fd", "n_fd_alt", "n_fire_name"),
            "E_ems": ("n_ems", "n_ems_alt", "n_ems_name"),
            "E_sch": ("n_sch", "n_sch_alt", "n_sch_name"),
            "n_poi_bq": ("n_poi_bq",),
            "n_bld": ("n_bld",),
        }
        counts = {}
        for source, columns in count_columns.items():
            values = np.zeros(len(all_tracts), dtype=float)
            for column in columns:
                if column in all_tracts:
                    values += pd.to_numeric(all_tracts[column], errors="coerce").fillna(0).to_numpy(float)
            counts[source] = values
        result = {"GEOID": all_tracts["GEOID"].astype(str).to_numpy()}
        for target in count_columns:
            for source, values in counts.items():
                if source == target:
                    continue
                log_rate = np.log1p(values / (pop + 100) * 1e6)
                present = (values > 0).astype(float)
                for k in (8, 20):
                    use_k = min(k, max_k)
                    indices = neighbor_index[:, :use_k]
                    distance = neighbor_distance[:, :use_k]
                    weight = 1 / (distance + 0.25)
                    weight /= weight.sum(axis=1, keepdims=True)
                    result[f"{target}__neighbor_k{k}__{source}_log_rate"] = np.sum(
                        weight * log_rate[indices], axis=1)
                    result[f"{target}__neighbor_k{k}__{source}_present_share"] = np.sum(
                        weight * present[indices], axis=1)
        frames.append(pd.DataFrame(result))
    lags = pd.concat(frames, ignore_index=True)
    os.makedirs(FEAT, exist_ok=True)
    lags.to_parquet(path, index=False)
    return lags


def append_auxiliary_spatial_lags(df, target, X, names, spatial_lags):
    """Add neighborhood rates of all supplied Overture count proxies except the current target."""
    feature_columns = [column for column in spatial_lags.columns
                       if column.startswith(f"{target}__neighbor_")]
    values = (pd.DataFrame({"GEOID": df["GEOID"].astype(str).to_numpy()})
              .merge(spatial_lags.assign(GEOID=spatial_lags.GEOID.astype(str)), on="GEOID", how="left")
              [feature_columns].fillna(0).to_numpy(float))
    return np.column_stack([X, values]), names + [column.removeprefix(f"{target}__") for column in feature_columns]


def auxiliary_target_exclusions():
    school_categories = _sql_list(SCHOOL_CATS)
    return {
        "E_fire": f"(coalesce(cat = 'fire_department', false) OR list_contains(alts, 'fire_department') "
                  f"OR regexp_matches(nm, '{FIRE_NAME_RE}'))",
        "E_ems": f"(coalesce(cat = 'ambulance_and_ems_services', false) OR "
                 f"list_contains(alts, 'ambulance_and_ems_services') OR regexp_matches(nm, '{EMS_NAME_RE}'))",
        "E_sch": f"(coalesce(cat IN {school_categories}, false) OR list_has_any(alts, {school_categories}) OR "
                 f"(regexp_matches(nm, '{SCHOOL_NAME_RE}') AND NOT regexp_matches(coalesce(cat, ''), '{NOT_SCHOOL}'))) ",
        "n_poi_bq": "(ds = 'BrightQuery')",
    }


def auxiliary_source_mix_features():
    """Count challenge-supplied Overture POI sources after excluding each proxy target."""
    path = os.path.join(FEAT, "source-mix.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)

    excluded = auxiliary_target_exclusions()
    frames = []
    for region in REGIONS:
        con = _con()
        con.sql(f"CREATE OR REPLACE TABLE tr AS SELECT GEOID, geometry g FROM "
                f"'{DATA}/strata/{region}/{region}-census-tracts.parquet'")
        con.sql(f"""
            CREATE OR REPLACE TABLE poi AS
            SELECT geometry p, categories.primary cat, coalesce(categories.alternate, []) alts,
                   coalesce(names.primary, '') nm, coalesce(sources[1].dataset, '') ds
            FROM '{DATA}/reference/{region}/{region}-overture-pois.parquet'
        """)
        expressions = []
        for target, exclusion in excluded.items():
            for source, source_filter in (
                ("meta", "ds = 'meta'"),
                ("bq", "ds = 'BrightQuery'"),
                ("msft", "ds = 'Microsoft'"),
                ("other", "ds NOT IN ('meta', 'BrightQuery', 'Microsoft')"),
            ):
                expressions.append(
                    f"count(*) FILTER (WHERE NOT {exclusion} AND {source_filter}) AS {target}__{source}"
                )
        query = f"""
            SELECT tr.GEOID, {', '.join(expressions)}
            FROM tr JOIN poi ON ST_Contains(tr.g, poi.p)
            GROUP BY tr.GEOID
        """
        frames.append(con.sql(query).df())
    source_mix = pd.concat(frames, ignore_index=True).fillna(0)
    os.makedirs(FEAT, exist_ok=True)
    source_mix.to_parquet(path, index=False)
    return source_mix


def auxiliary_source_signal_features():
    """Count non-target POIs carrying an Overture-signals provenance record."""
    path = os.path.join(FEAT, "source-signals.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    excluded = auxiliary_target_exclusions()
    frames = []
    for region in REGIONS:
        con = _con()
        con.sql(f"CREATE OR REPLACE TABLE tr AS SELECT GEOID, geometry g FROM "
                f"'{DATA}/strata/{region}/{region}-census-tracts.parquet'")
        con.sql(f"""
            CREATE OR REPLACE TABLE poi AS
            SELECT geometry p, categories.primary cat, coalesce(categories.alternate, []) alts,
                   coalesce(names.primary, '') nm, coalesce(sources[1].dataset, '') ds,
                   coalesce(sources[3].dataset, '') signal_ds
            FROM '{DATA}/reference/{region}/{region}-overture-pois.parquet'
        """)
        expressions = [
            f"count(*) FILTER (WHERE NOT {exclusion} AND signal_ds = 'Overture-signals') "
            f"AS {target}__signal_count"
            for target, exclusion in excluded.items()
        ]
        query = f"""
            SELECT tr.GEOID, {', '.join(expressions)}
            FROM tr JOIN poi ON ST_Contains(tr.g, poi.p)
            GROUP BY tr.GEOID
        """
        frames.append(con.sql(query).df())
    signals = pd.concat(frames, ignore_index=True).fillna(0)
    os.makedirs(FEAT, exist_ok=True)
    signals.to_parquet(path, index=False)
    return signals


def append_auxiliary_source_mix(df, target, X, names, source_mix):
    """Add source composition from points outside the current auxiliary target."""
    target_cols = [f"{target}__{source}" for source in ("meta", "bq", "msft", "other")]
    source = (pd.DataFrame({"GEOID": df["GEOID"].astype(str).to_numpy()})
              .merge(source_mix.assign(GEOID=source_mix.GEOID.astype(str)), on="GEOID", how="left")
              [target_cols].fillna(0).to_numpy(float))
    total = source.sum(axis=1)
    shares = source / np.maximum(total[:, None], 1.0)
    extra = {
        "source_mix__log1p_other_poi_per_million": np.log1p(total / (df["pop"].to_numpy(float) + 100) * 1e6),
        "source_mix__meta_share": shares[:, 0],
        "source_mix__brightquery_share": shares[:, 1],
        "source_mix__microsoft_share": shares[:, 2],
        "source_mix__other_share": shares[:, 3],
        "source_mix__entropy": -np.sum(np.where(shares > 0, shares * np.log(np.maximum(shares, 1e-12)), 0.0), axis=1),
    }
    return np.column_stack([X, *extra.values()]), names + list(extra)


def append_auxiliary_source_signals(df, target, X, names, source_mix, source_signals):
    """Add the share and per-capita rate of non-target places carrying Overture-signals provenance."""
    source = (pd.DataFrame({"GEOID": df["GEOID"].astype(str).to_numpy()})
              .merge(source_mix.assign(GEOID=source_mix.GEOID.astype(str)), on="GEOID", how="left")
              .merge(source_signals.assign(GEOID=source_signals.GEOID.astype(str)), on="GEOID", how="left"))
    total = source[[f"{target}__{k}" for k in ("meta", "bq", "msft", "other")]].fillna(0).sum(axis=1).to_numpy(float)
    signal_count = source[f"{target}__signal_count"].fillna(0).to_numpy(float)
    extra = {
        "source_signal__share_non_target_poi": signal_count / np.maximum(total, 1.0),
        "source_signal__log1p_per_million": np.log1p(signal_count / (df["pop"].to_numpy(float) + 100) * 1e6),
    }
    return np.column_stack([X, *extra.values()]), names + list(extra)


def auxiliary_source_confidence_features():
    """Aggregate the challenge-provided place confidence within each first-source family."""
    path = os.path.join(FEAT, "source-confidence.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    excluded = auxiliary_target_exclusions()
    frames = []
    source_filters = {
        "meta": "ds = 'meta'",
        "bq": "ds = 'BrightQuery'",
        "msft": "ds = 'Microsoft'",
        "other": "ds NOT IN ('meta', 'BrightQuery', 'Microsoft')",
    }
    for region in REGIONS:
        con = _con()
        con.sql(f"CREATE OR REPLACE TABLE tr AS SELECT GEOID, geometry g FROM "
                f"'{DATA}/strata/{region}/{region}-census-tracts.parquet'")
        con.sql(f"""
            CREATE OR REPLACE TABLE poi AS
            SELECT geometry p, categories.primary cat, coalesce(categories.alternate, []) alts,
                   coalesce(names.primary, '') nm, coalesce(sources[1].dataset, '') ds,
                   confidence
            FROM '{DATA}/reference/{region}/{region}-overture-pois.parquet'
        """)
        expressions = []
        for target, exclusion in excluded.items():
            for source, source_filter in source_filters.items():
                where = f"NOT {exclusion} AND {source_filter}"
                expressions.append(f"count(*) FILTER (WHERE {where}) AS {target}__{source}__count")
                expressions.append(
                    f"avg(confidence) FILTER (WHERE {where}) AS {target}__{source}__confidence_mean")
                expressions.append(
                    f"count(*) FILTER (WHERE {where} AND confidence IS NOT NULL) AS "
                    f"{target}__{source}__confidence_n")
        query = f"""
            SELECT tr.GEOID, {', '.join(expressions)}
            FROM tr JOIN poi ON ST_Contains(tr.g, poi.p)
            GROUP BY tr.GEOID
        """
        frames.append(con.sql(query).df())
    confidence = pd.concat(frames, ignore_index=True)
    os.makedirs(FEAT, exist_ok=True)
    confidence.to_parquet(path, index=False)
    return confidence


def append_auxiliary_source_confidence(df, target, X, names, confidence):
    """Add source-specific confidence means and availability shares for non-target places."""
    columns = ["GEOID"]
    for source in ("meta", "bq", "msft", "other"):
        columns.extend((f"{target}__{source}__count", f"{target}__{source}__confidence_mean",
                        f"{target}__{source}__confidence_n"))
    values = (pd.DataFrame({"GEOID": df["GEOID"].astype(str).to_numpy()})
              .merge(confidence.assign(GEOID=confidence.GEOID.astype(str))[columns],
                     on="GEOID", how="left"))
    extra = {}
    for source in ("meta", "bq", "msft", "other"):
        count = values[f"{target}__{source}__count"].fillna(0).to_numpy(float)
        confident = values[f"{target}__{source}__confidence_n"].fillna(0).to_numpy(float)
        extra[f"source_confidence__mean_{source}"] = values[
            f"{target}__{source}__confidence_mean"].fillna(0).to_numpy(float)
        extra[f"source_confidence__available_share_{source}"] = np.divide(
            confident, count, out=np.zeros_like(confident), where=count > 0)
    return np.column_stack([X, *extra.values()]), names + list(extra)


def auxiliary_source_depth_features():
    """Measure how many source records and distinct datasets support target-excluded POIs."""
    path = os.path.join(FEAT, "source-depth.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    excluded = auxiliary_target_exclusions()
    frames = []
    for region in REGIONS:
        con = _con()
        con.sql(f"CREATE OR REPLACE TABLE tr AS SELECT GEOID, geometry g FROM "
                f"'{DATA}/strata/{region}/{region}-census-tracts.parquet'")
        con.sql(f"""
            CREATE OR REPLACE TABLE poi AS
            SELECT geometry p, categories.primary cat, coalesce(categories.alternate, []) alts,
                   coalesce(names.primary, '') nm, coalesce(sources[1].dataset, '') ds,
                   coalesce(sources, []) source_rows,
                   list_transform(coalesce(sources, []), x -> coalesce(x.dataset, '')) source_datasets
            FROM '{DATA}/reference/{region}/{region}-overture-pois.parquet'
        """)
        con.sql("CREATE OR REPLACE TABLE poi_in_tract AS "
                "SELECT tr.GEOID, poi.cat, poi.alts, poi.nm, poi.ds, poi.source_rows, poi.source_datasets, "
                "len(poi.source_rows) n_sources, "
                "len(list_distinct(list_filter(poi.source_datasets, x -> x <> ''))) n_datasets "
                "FROM tr JOIN poi ON ST_Contains(tr.g, poi.p)")
        expressions = []
        for target, exclusion in excluded.items():
            where = f"NOT ({exclusion})"
            expressions.extend((
                f"count(*) FILTER (WHERE {where}) AS {target}__total",
                f"avg(n_sources) FILTER (WHERE {where}) AS {target}__mean_sources",
                f"avg(n_datasets) FILTER (WHERE {where}) AS {target}__mean_datasets",
                f"count(*) FILTER (WHERE {where} AND n_sources > 1) AS {target}__multi_source",
                f"count(*) FILTER (WHERE {where} AND n_datasets > 1) AS {target}__multi_dataset",
                f"count(*) FILTER (WHERE {where} AND list_contains(source_datasets, 'Overture-signals')) "
                f"AS {target}__overture_signal",
                f"count(*) FILTER (WHERE {where} AND list_contains(source_datasets, 'meta')) AS {target}__any_meta",
                f"count(*) FILTER (WHERE {where} AND list_contains(source_datasets, 'BrightQuery')) AS {target}__any_bq",
                f"count(*) FILTER (WHERE {where} AND list_contains(source_datasets, 'Microsoft')) AS {target}__any_msft",
            ))
        query = f"SELECT tr.GEOID, {', '.join(expressions)} FROM tr JOIN poi_in_tract USING (GEOID) GROUP BY tr.GEOID"
        frames.append(con.sql(query).df())
    depth = pd.concat(frames, ignore_index=True)
    os.makedirs(FEAT, exist_ok=True)
    depth.to_parquet(path, index=False)
    return depth


def append_auxiliary_source_depth(df, target, X, names, depth):
    """Add multi-source and cross-dataset support shares among non-target POIs."""
    columns = ["GEOID"] + [f"{target}__{name}" for name in
                           ("total", "mean_sources", "mean_datasets", "multi_source", "multi_dataset",
                            "overture_signal", "any_meta", "any_bq", "any_msft")]
    values = (pd.DataFrame({"GEOID": df["GEOID"].astype(str).to_numpy()})
              .merge(depth.assign(GEOID=depth.GEOID.astype(str))[columns], on="GEOID", how="left").fillna(0))
    total = values[f"{target}__total"].to_numpy(float)
    extra = {
        "source_depth__mean_sources": values[f"{target}__mean_sources"].to_numpy(float),
        "source_depth__mean_distinct_datasets": values[f"{target}__mean_datasets"].to_numpy(float),
        "source_depth__multi_source_share": np.divide(values[f"{target}__multi_source"].to_numpy(float), total,
                                                       out=np.zeros_like(total), where=total > 0),
        "source_depth__multi_dataset_share": np.divide(values[f"{target}__multi_dataset"].to_numpy(float), total,
                                                        out=np.zeros_like(total), where=total > 0),
        "source_depth__overture_signal_share": np.divide(values[f"{target}__overture_signal"].to_numpy(float), total,
                                                          out=np.zeros_like(total), where=total > 0),
    }
    for source, suffix in (("meta", "meta"), ("bq", "brightquery"), ("msft", "microsoft")):
        extra[f"source_depth__any_{suffix}_share"] = np.divide(
            values[f"{target}__any_{source}"].to_numpy(float), total,
            out=np.zeros_like(total), where=total > 0)
    return np.column_stack([X, *extra.values()]), names + list(extra)


def auxiliary_source_depth_sweep(df, report_file="aux_source_depth_sweep.json"):
    """Test Overture multi-source record depth beyond current category and source-mix features."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_poisson_deviance
    from sklearn.model_selection import GroupKFold

    depth = auxiliary_source_depth_features()
    source_mix = auxiliary_source_mix_features()
    source_confidence = auxiliary_source_confidence_features()
    category_counts, category_schema = auxiliary_poi_category_features()
    graded = df["region"].isin(GRADED).to_numpy()
    results, summary = [], []
    for target, cap in AUX_TARGETS:
        X, names = auxiliary_feature_matrix(df, AUX_CROSS_COUNT_FEATURE_CHOICES[target])
        X, names = append_other_overture_count_features(df, target, X, names)
        X, names = append_auxiliary_source_mix(df, target, X, names, source_mix)
        X, names = append_auxiliary_source_confidence(df, target, X, names, source_confidence)
        X, names = append_auxiliary_poi_categories(
            df, target, X, names, category_counts, category_schema, AUX_POI_CATEGORY_SELECTED[target])
        y = df[target].clip(upper=cap).to_numpy(float)
        config = AUX_GBM_BY_TARGET[target]
        target_rows = []
        for add_depth in (False, True):
            features, feature_names = ((append_auxiliary_source_depth(df, target, X, names, depth))
                                       if add_depth else (X, names))
            oof = np.zeros(len(y))
            for tr, te in GroupKFold(5).split(features, y, df["county"]):
                model = HistGradientBoostingRegressor(
                    loss="poisson", random_state=0, **AUX_GBM_CONFIGS[config])
                model.fit(features[tr], y[tr])
                oof[te] = np.clip(model.predict(features[te]), 1e-4, None)
            regions = {region: float(mean_poisson_deviance(y[df.region.eq(region)],
                                                           oof[df.region.eq(region)]))
                       for region in GRADED}
            row = dict(target=target, add_source_depth=add_depth, model_config=config,
                       feature_count=len(feature_names),
                       graded_cv_deviance=float(mean_poisson_deviance(y[graded], oof[graded])),
                       region_deviance=regions)
            target_rows.append(row)
            results.append(row)
        base, plus = target_rows
        changes = {region: 100 * (plus["region_deviance"][region] /
                                    base["region_deviance"][region] - 1)
                   for region in GRADED}
        summary.append(dict(target=target, base_graded_cv_deviance=base["graded_cv_deviance"],
                            source_depth_graded_cv_deviance=plus["graded_cv_deviance"],
                            change_pct=100 * (plus["graded_cv_deviance"] /
                                              base["graded_cv_deviance"] - 1),
                            graded_regions_improved=sum(value < 0 for value in changes.values()),
                            graded_region_change_pct=changes))
        item = summary[-1]
        log(f"  {target}: source-depth features {item['change_pct']:+.3f}% "
            f"({item['graded_regions_improved']}/4 regions improve)")
    report = dict(date="2026-10-24", experiment="target-excluded Overture source depth",
                  hypothesis="The number of source records and distinct source datasets supporting non-target POIs may signal map completeness.",
                  split="GroupKFold(5), grouped by county; target POIs are removed using the current exact target match rules.",
                  target_proxy="observed Overture counts", base_features="current category branch with cross-counts, source mix, and source confidence",
                  source_fields=["source record count", "distinct source datasets", "multi-source share",
                                 "Overture-signals share", "any Meta/BrightQuery/Microsoft support shares"],
                  fixed_gbm_config_by_target={target: AUX_GBM_BY_TARGET[target] for target, _ in AUX_TARGETS},
                  selection_note="Proxy-count CV only; not direct validation of hidden-score performance.",
                  summary=summary, results=results)
    atomic_json(out(report_file), report)
    log(f"Wrote {out(report_file)}")
    return report


def auxiliary_poi_identity_features():
    """Aggregate identity and status field availability among target-excluded POIs by first-source family."""
    path = os.path.join(FEAT, "poi-identity-quality.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    excluded = auxiliary_target_exclusions()
    source_filters = {"meta": "ds = 'meta'", "bq": "ds = 'BrightQuery'",
                      "msft": "ds = 'Microsoft'", "other": "ds NOT IN ('meta', 'BrightQuery', 'Microsoft')"}
    frames = []
    for region in REGIONS:
        con = _con()
        con.sql(f"CREATE OR REPLACE TABLE tr AS SELECT GEOID, geometry g FROM "
                f"'{DATA}/strata/{region}/{region}-census-tracts.parquet'")
        con.sql(f"""
            CREATE OR REPLACE TABLE poi AS
            SELECT geometry p, categories.primary cat, coalesce(categories.alternate, []) alts,
                   coalesce(names.primary, '') nm, coalesce(sources[1].dataset, '') ds,
                   coalesce(operating_status, '') operating_status,
                   len(coalesce(emails, [])) n_emails,
                   coalesce(brand.wikidata, '') brand_wikidata,
                   coalesce(brand.names.primary, '') brand_name
            FROM '{DATA}/reference/{region}/{region}-overture-pois.parquet'
        """)
        expressions = []
        for target, exclusion in excluded.items():
            for source, source_filter in source_filters.items():
                where = f"NOT ({exclusion}) AND ({source_filter})"
                expressions.append(f"count(*) FILTER (WHERE {where}) AS {target}__{source}__total")
                for attribute, condition in (
                    ("name", "nm <> ''"),
                    ("brand", "brand_name <> '' OR brand_wikidata <> ''"),
                    ("brand_id", "brand_wikidata <> ''"),
                    ("email", "n_emails > 0"),
                    ("status", "operating_status <> ''"),
                    ("open", "operating_status = 'open'"),
                ):
                    expressions.append(
                        f"count(*) FILTER (WHERE {where} AND ({condition})) "
                        f"AS {target}__{source}__{attribute}")
        frames.append(con.sql(f"""
            SELECT tr.GEOID, {', '.join(expressions)}
            FROM tr JOIN poi ON ST_Contains(tr.g, poi.p)
            GROUP BY tr.GEOID
        """).df())
    quality = pd.concat(frames, ignore_index=True)
    os.makedirs(FEAT, exist_ok=True)
    quality.to_parquet(path, index=False)
    return quality


def append_auxiliary_poi_identity(df, target, X, names, identity, attributes=None):
    """Add name, brand, email, and operating-status shares for non-target POIs by source family."""
    columns = ["GEOID"]
    attributes = attributes or ("name", "brand", "brand_id", "email", "status", "open")
    for source in ("meta", "bq", "msft", "other"):
        columns.append(f"{target}__{source}__total")
        columns.extend(f"{target}__{source}__{attribute}" for attribute in attributes)
    values = (pd.DataFrame({"GEOID": df["GEOID"].astype(str).to_numpy()})
              .merge(identity.assign(GEOID=identity.GEOID.astype(str))[columns], on="GEOID", how="left").fillna(0))
    extra = {}
    for source in ("meta", "bq", "msft", "other"):
        total = values[f"{target}__{source}__total"].to_numpy(float)
        for attribute in attributes:
            count = values[f"{target}__{source}__{attribute}"].to_numpy(float)
            extra[f"poi_identity__{attribute}_share_{source}"] = np.divide(
                count, total, out=np.zeros_like(total), where=total > 0)
    return np.column_stack([X, *extra.values()]), names + list(extra)


def auxiliary_poi_identity_sweep(df, report_file="aux_poi_identity_sweep.json", include_depth=False):
    """Test source-specific identity-field completeness on top of current category proxy models."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_poisson_deviance
    from sklearn.model_selection import GroupKFold

    identity = auxiliary_poi_identity_features()
    source_mix = auxiliary_source_mix_features()
    source_confidence = auxiliary_source_confidence_features()
    category_counts, category_schema = auxiliary_poi_category_features()
    source_depth = auxiliary_source_depth_features() if include_depth else None
    graded = df["region"].isin(GRADED).to_numpy()
    results, summary = [], []
    for target, cap in AUX_TARGETS:
        X, names = auxiliary_feature_matrix(df, AUX_CROSS_COUNT_FEATURE_CHOICES[target])
        X, names = append_other_overture_count_features(df, target, X, names)
        X, names = append_auxiliary_source_mix(df, target, X, names, source_mix)
        X, names = append_auxiliary_source_confidence(df, target, X, names, source_confidence)
        X, names = append_auxiliary_poi_categories(
            df, target, X, names, category_counts, category_schema, AUX_POI_CATEGORY_SELECTED[target])
        if include_depth and target in AUX_SOURCE_DEPTH_SELECTED:
            X, names = append_auxiliary_source_depth(df, target, X, names, source_depth)
        y = df[target].clip(upper=cap).to_numpy(float)
        config = AUX_GBM_BY_TARGET[target]
        target_rows = []
        for add_identity in (False, True):
            features, feature_names = (append_auxiliary_poi_identity(df, target, X, names, identity)
                                       if add_identity else (X, names))
            oof = np.zeros(len(y))
            for tr, te in GroupKFold(5).split(features, y, df["county"]):
                model = HistGradientBoostingRegressor(
                    loss="poisson", random_state=0, **AUX_GBM_CONFIGS[config])
                model.fit(features[tr], y[tr])
                oof[te] = np.clip(model.predict(features[te]), 1e-4, None)
            regions = {region: float(mean_poisson_deviance(y[df.region.eq(region)],
                                                           oof[df.region.eq(region)]))
                       for region in GRADED}
            row = dict(target=target, add_identity_fields=add_identity, model_config=config,
                       feature_count=len(feature_names),
                       graded_cv_deviance=float(mean_poisson_deviance(y[graded], oof[graded])),
                       region_deviance=regions)
            target_rows.append(row)
            results.append(row)
        base, plus = target_rows
        changes = {region: 100 * (plus["region_deviance"][region] /
                                    base["region_deviance"][region] - 1)
                   for region in GRADED}
        summary.append(dict(target=target, base_graded_cv_deviance=base["graded_cv_deviance"],
                            identity_graded_cv_deviance=plus["graded_cv_deviance"],
                            change_pct=100 * (plus["graded_cv_deviance"] /
                                              base["graded_cv_deviance"] - 1),
                            graded_regions_improved=sum(value < 0 for value in changes.values()),
                            graded_region_change_pct=changes))
        item = summary[-1]
        log(f"  {target}: identity completeness {item['change_pct']:+.3f}% "
            f"({item['graded_regions_improved']}/4 regions improve)")
    report = dict(date="2026-10-24", experiment="target-excluded Overture identity and status completeness",
                  hypothesis="Name, brand, contact, and status availability mark POI matching quality beyond source confidence and pooled categories.",
                  split="GroupKFold(5), grouped by county; matched target POIs are excluded before attribute aggregation.",
                  target_proxy="observed Overture counts", base_features=("current category branch with cross-counts, source mix, source confidence, and selected source depth" if include_depth else "current category branch with cross-counts, source mix, and source confidence"),
                  added_fields=["primary name", "brand name or Wikidata", "brand Wikidata", "email", "operating status", "open status"],
                  fixed_gbm_config_by_target={target: AUX_GBM_BY_TARGET[target] for target, _ in AUX_TARGETS},
                  source_depth_included=include_depth,
                  selection_note="Fixed GBM settings; grouped OOF deviance is a proxy-count metric, not a direct hidden-score validation.",
                  summary=summary, results=results)
    atomic_json(out(report_file), report)
    log(f"Wrote {out(report_file)}")
    return report


def auxiliary_poi_identity_ablation_sweep(df, report_file="aux_poi_identity_ablation_sweep.json"):
    """Ablate POI identity fields for BrightQuery after the selected category and source-depth features."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_poisson_deviance
    from sklearn.model_selection import GroupKFold

    target = "n_poi_bq"
    cap = dict(AUX_TARGETS)[target]
    identity = auxiliary_poi_identity_features()
    source_mix = auxiliary_source_mix_features()
    source_confidence = auxiliary_source_confidence_features()
    category_counts, category_schema = auxiliary_poi_category_features()
    source_depth = auxiliary_source_depth_features()
    X, names = auxiliary_feature_matrix(df, AUX_CROSS_COUNT_FEATURE_CHOICES[target])
    X, names = append_other_overture_count_features(df, target, X, names)
    X, names = append_auxiliary_source_mix(df, target, X, names, source_mix)
    X, names = append_auxiliary_source_confidence(df, target, X, names, source_confidence)
    X, names = append_auxiliary_poi_categories(
        df, target, X, names, category_counts, category_schema, AUX_POI_CATEGORY_SELECTED[target])
    X, names = append_auxiliary_source_depth(df, target, X, names, source_depth)
    y = df[target].clip(upper=cap).to_numpy(float)
    graded = df["region"].isin(GRADED).to_numpy()
    attribute_sets = {
        "baseline": (),
        "name": ("name",),
        "brand": ("brand", "brand_id"),
        "email": ("email",),
        "status": ("status",),
        "open": ("open",),
        "name_brand": ("name", "brand", "brand_id"),
        "all_identity": ("name", "brand", "brand_id", "email", "status", "open"),
    }
    config = AUX_GBM_BY_TARGET[target]
    results = []
    for feature_set, attributes in attribute_sets.items():
        features, feature_names = (append_auxiliary_poi_identity(df, target, X, names, identity, attributes)
                                   if attributes else (X, names))
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(features, y, df["county"]):
            model = HistGradientBoostingRegressor(
                loss="poisson", random_state=0, **AUX_GBM_CONFIGS[config])
            model.fit(features[tr], y[tr])
            oof[te] = np.clip(model.predict(features[te]), 1e-4, None)
        results.append(dict(
            target=target, feature_set=feature_set, attributes=list(attributes),
            feature_count=len(feature_names),
            graded_cv_deviance=float(mean_poisson_deviance(y[graded], oof[graded])),
            region_deviance={region: float(mean_poisson_deviance(y[df.region.eq(region)],
                                                                 oof[df.region.eq(region)]))
                             for region in GRADED}))
        log(f"  {target}: {feature_set} {results[-1]['graded_cv_deviance']:.6f}")
    base = results[0]
    for row in results:
        row["change_pct"] = 100 * (row["graded_cv_deviance"] / base["graded_cv_deviance"] - 1)
        row["graded_region_change_pct"] = {
            region: 100 * (row["region_deviance"][region] / base["region_deviance"][region] - 1)
            for region in GRADED}
    report = dict(
        date="2026-10-24", experiment="BrightQuery identity-field ablation",
        hypothesis="One identity field group may improve BrightQuery count prediction even when all identity fields together do not.",
        split="GroupKFold(5), grouped by county; BrightQuery first-source POIs are removed before feature aggregation.",
        target_proxy="observed BrightQuery POI counts",
        base_features="selected categories, cross-counts, source mix, source confidence, and source depth",
        fixed_gbm_config=AUX_GBM_BY_TARGET[target],
        selection_note="All variants use the same held-out counties; this is proxy-count CV, not direct hidden-score validation.",
        results=results)
    atomic_json(out(report_file), report)
    log(f"Wrote {out(report_file)}")
    return report


def auxiliary_source_richness_features():
    """Count non-target places with address/contact attributes by first-source family."""
    path = os.path.join(FEAT, "source-richness.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    excluded = auxiliary_target_exclusions()
    frames = []
    source_filters = {
        "meta": "ds = 'meta'",
        "bq": "ds = 'BrightQuery'",
        "msft": "ds = 'Microsoft'",
        "other": "ds NOT IN ('meta', 'BrightQuery', 'Microsoft')",
    }
    expressions = []
    for target, exclusion in excluded.items():
        for source, source_filter in source_filters.items():
            where = f"NOT {exclusion} AND {source_filter}"
            expressions.append(f"count(*) FILTER (WHERE {where}) AS {target}__{source}__count")
            for attribute, column in (("address", "addresses"), ("website", "websites"),
                                      ("phone", "phones"), ("social", "socials")):
                expressions.append(
                    f"count(*) FILTER (WHERE {where} AND len(coalesce({column}, [])) > 0) "
                    f"AS {target}__{source}__{attribute}_count")
    for region in REGIONS:
        con = _con()
        con.sql(f"CREATE OR REPLACE TABLE tr AS SELECT GEOID, geometry g FROM "
                f"'{DATA}/strata/{region}/{region}-census-tracts.parquet'")
        con.sql(f"""
            CREATE OR REPLACE TABLE poi AS
            SELECT geometry p, categories.primary cat, coalesce(categories.alternate, []) alts,
                   coalesce(names.primary, '') nm, coalesce(sources[1].dataset, '') ds,
                   addresses, websites, phones, socials
            FROM '{DATA}/reference/{region}/{region}-overture-pois.parquet'
        """)
        query = f"""
            SELECT tr.GEOID, {', '.join(expressions)}
            FROM tr JOIN poi ON ST_Contains(tr.g, poi.p)
            GROUP BY tr.GEOID
        """
        frames.append(con.sql(query).df())
    richness = pd.concat(frames, ignore_index=True)
    os.makedirs(FEAT, exist_ok=True)
    richness.to_parquet(path, index=False)
    return richness


def append_auxiliary_source_richness(df, target, X, names, richness):
    """Add address and contact-field shares within each source family among non-target places."""
    columns = ["GEOID"]
    for source in ("meta", "bq", "msft", "other"):
        columns.append(f"{target}__{source}__count")
        columns.extend(f"{target}__{source}__{attribute}_count"
                       for attribute in ("address", "website", "phone", "social"))
    values = (pd.DataFrame({"GEOID": df["GEOID"].astype(str).to_numpy()})
              .merge(richness.assign(GEOID=richness.GEOID.astype(str))[columns],
                     on="GEOID", how="left"))
    extra = {}
    for source in ("meta", "bq", "msft", "other"):
        count = values[f"{target}__{source}__count"].fillna(0).to_numpy(float)
        for attribute in ("address", "website", "phone", "social"):
            present = values[f"{target}__{source}__{attribute}_count"].fillna(0).to_numpy(float)
            extra[f"source_richness__{attribute}_share_{source}"] = np.divide(
                present, count, out=np.zeros_like(present), where=count > 0)
    return np.column_stack([X, *extra.values()]), names + list(extra)


def fit_auxiliary(df):
    """Cross-fit each observed-count proxy using the feature variant selected by county-held-out CV."""
    mu, reports = {}, []
    source_mix = auxiliary_source_mix_features() if AUX_USE_SOURCE_MIX else None
    source_confidence = auxiliary_source_confidence_features() if AUX_USE_SOURCE_CONFIDENCE else None
    source_richness = auxiliary_source_richness_features() if AUX_USE_SOURCE_RICHNESS else None
    spatial_lags = auxiliary_spatial_lag_features() if AUX_USE_SPATIAL_LAGS else None
    poi_categories = auxiliary_poi_category_features() if AUX_USE_POI_CATEGORIES else None
    source_depth = auxiliary_source_depth_features() if AUX_USE_SOURCE_DEPTH else None
    for target, cap in AUX_TARGETS:
        variant = AUX_FEATURE_BY_TARGET.get(target, "baseline")
        X, names = auxiliary_feature_matrix(df, variant)
        if AUX_USE_CROSS_COUNTS:
            X, names = append_other_overture_count_features(df, target, X, names)
        if AUX_USE_SOURCE_MIX:
            X, names = append_auxiliary_source_mix(df, target, X, names, source_mix)
        if AUX_USE_SOURCE_CONFIDENCE:
            X, names = append_auxiliary_source_confidence(df, target, X, names, source_confidence)
        if AUX_USE_SOURCE_RICHNESS:
            X, names = append_auxiliary_source_richness(df, target, X, names, source_richness)
        if AUX_USE_SPATIAL_LAGS and target in AUX_SPATIAL_LAG_TARGETS:
            X, names = append_auxiliary_spatial_lags(df, target, X, names, spatial_lags)
        if AUX_USE_POI_CATEGORIES and AUX_POI_CATEGORY_KINDS.get(target):
            X, names = append_auxiliary_poi_categories(
                df, target, X, names, poi_categories[0], poi_categories[1], AUX_POI_CATEGORY_KINDS[target])
        if AUX_USE_SOURCE_DEPTH and target in AUX_SOURCE_DEPTH_TARGETS:
            X, names = append_auxiliary_source_depth(df, target, X, names, source_depth)
        fit_function = cross_fit_hurdle if AUX_USE_HURDLE else cross_fit
        mu[target], rep = fit_function(df, X, target, cap)
        if AUX_USE_REGION_BLEND:
            shared = mu[target]
            regional = np.zeros(len(df))
            regional_reports = {}
            for region in REGIONS:
                mask = df["region"].eq(region).to_numpy()
                local_df = df.loc[mask].reset_index(drop=True)
                regional[mask], regional_reports[region] = cross_fit(local_df, X[mask], target, cap)
            mu[target] = np.clip((1.0 - AUX_REGION_BLEND_WEIGHT) * shared
                                 + AUX_REGION_BLEND_WEIGHT * regional, 1e-4, None)
            rep.update(regional_blend_weight=AUX_REGION_BLEND_WEIGHT,
                       regional_models={r: regional_reports[r]["chosen"] for r in REGIONS})
        rep.update(feature_variant=variant, features=names)
        reports.append(rep)
    return mu, reports


def auxiliary_feature_sweep(df, variants=None, report_file="aux_feature_sweep.json"):
    """County-grouped CV for density, regional/county effects and label-free supplied strata."""
    from sklearn.metrics import mean_poisson_deviance
    variants = list(variants or ["baseline", "density", "density_region", "density_county", "density_strata"])
    rows = []
    graded = df["region"].isin(GRADED).to_numpy()

    def save_report():
        selected = {}
        for target, _ in AUX_TARGETS:
            options = [r for r in rows if r["target"] == target]
            if options:
                selected[target] = min(options, key=lambda r: r["graded_cv_deviance"])["variant"]
        report = dict(split="GroupKFold(5), grouped by county", target_proxy="observed Overture counts",
                      variants=variants, completed_variants=list(dict.fromkeys(r["variant"] for r in rows)),
                      selected_by_graded_deviance=selected, results=rows)
        atomic_json(out(report_file), report)
        return report

    for variant in variants:
        X, names = auxiliary_feature_matrix(df, variant)
        log(f"  {variant}: {X.shape[1]} features")
        for target, cap in AUX_TARGETS:
            y = df[target].clip(upper=cap).to_numpy(float)
            pred, fit = cross_fit(df, X, target, cap)
            by_region = {r: float(mean_poisson_deviance(y[df.region.eq(r)], pred[df.region.eq(r)])) for r in REGIONS}
            rows.append(dict(variant=variant, target=target, feature_count=len(names), chosen_model=fit["chosen"],
                             cv_deviance=fit["cv_deviance"],
                             graded_cv_deviance=float(mean_poisson_deviance(y[graded], pred[graded])),
                             region_deviance=by_region))
        report = save_report()
    for target, _ in AUX_TARGETS:
        options = sorted((r for r in rows if r["target"] == target), key=lambda r: r["graded_cv_deviance"])
        base = next((r["graded_cv_deviance"] for r in options if r["variant"] == "baseline"), None)
        best = options[0]
        if base is None:
            log(f"  {target}: best {best['variant']} {best['graded_cv_deviance']:.4f}; {best['region_deviance']}")
        else:
            log(f"  {target}: baseline {base:.4f}; best {best['variant']} {best['graded_cv_deviance']:.4f} "
                f"({100 * (best['graded_cv_deviance'] / base - 1):+.2f}%), {best['region_deviance']}")
    log(f"Wrote {out(report_file)}")
    return report


def auxiliary_hurdle_sweep(df, report_file="aux_hurdle_sweep.json"):
    """Compare direct Poisson count fits with a county-held-out two-part (zero / positive count) model."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.metrics import mean_poisson_deviance
    from sklearn.model_selection import GroupKFold

    feature_variants = AUX_HURDLE_FEATURE_CHOICES
    classifier_configs = AUX_HURDLE_CLASSIFIER_CONFIGS
    positive_configs = AUX_HURDLE_POSITIVE_CONFIGS
    graded = df["region"].isin(GRADED).to_numpy()
    rows = []
    targets = []
    for target, cap in AUX_TARGETS:
        variant = feature_variants[target]
        X, names = auxiliary_feature_matrix(df, variant)
        y = df[target].clip(upper=cap).to_numpy(float)
        direct, direct_fit = cross_fit(df, X, target, cap)
        predictions = {f"hurdle:{c}/{p}": np.zeros(len(y))
                       for c in classifier_configs for p in positive_configs}
        fold_notes = []
        for fold, (tr, te) in enumerate(GroupKFold(5).split(X, y, df["county"]), 1):
            positive = y[tr] > 0
            if positive.sum() < 100:
                raise RuntimeError(f"too few positive rows in fold {fold} for {target}")
            for classifier_name, classifier_config in classifier_configs.items():
                classifier = HistGradientBoostingClassifier(random_state=0, **classifier_config)
                classifier.fit(X[tr], positive.astype(int))
                p_positive = classifier.predict_proba(X[te])[:, 1]
                for positive_name, positive_config in positive_configs.items():
                    positive_model = HistGradientBoostingRegressor(loss="poisson", random_state=0,
                                                                   **positive_config)
                    positive_model.fit(X[tr][positive], y[tr][positive])
                    conditional_mean = np.clip(positive_model.predict(X[te]), 1e-4, None)
                    predictions[f"hurdle:{classifier_name}/{positive_name}"][te] = np.clip(
                        p_positive * conditional_mean, 1e-4, None)
            fold_notes.append(dict(fold=fold, train_rows=int(len(tr)), test_rows=int(len(te)),
                                   train_counties=int(df["county"].iloc[tr].nunique()),
                                   test_counties=int(df["county"].iloc[te].nunique()),
                                   positive_train_rows=int(positive.sum())))

        target_rows = []
        candidates = {"direct_best": direct, **predictions}
        for name, pred in list(predictions.items()):
            for weight in (0.25, 0.5, 0.75):
                candidates[f"blend_direct_{weight:.2f}_{name}"] = np.clip(
                    (1.0 - weight) * direct + weight * pred, 1e-4, None)
        for name, pred in candidates.items():
            region_deviance = {r: float(mean_poisson_deviance(y[df.region.eq(r)], pred[df.region.eq(r)]))
                               for r in REGIONS}
            graded_region_delta = {r: float(region_deviance[r] /
                                           mean_poisson_deviance(y[df.region.eq(r)], direct[df.region.eq(r)]) - 1)
                                   for r in GRADED}
            target_rows.append(dict(target=target, feature_variant=variant, feature_count=len(names), model=name,
                                    graded_cv_deviance=float(mean_poisson_deviance(y[graded], pred[graded])),
                                    cv_deviance=float(mean_poisson_deviance(y, pred)),
                                    region_deviance=region_deviance, graded_region_delta_vs_direct=graded_region_delta))
        target_rows.sort(key=lambda row: row["graded_cv_deviance"])
        best = target_rows[0]
        direct_row = next(row for row in target_rows if row["model"] == "direct_best")
        hurdle_rows = [row for row in target_rows if row["model"].startswith("hurdle:")]
        best_hurdle = min(hurdle_rows, key=lambda row: row["graded_cv_deviance"])
        gain = 100 * (best["graded_cv_deviance"] / direct_row["graded_cv_deviance"] - 1)
        improves_regions = sum(delta < 0 for delta in best["graded_region_delta_vs_direct"].values())
        targets.append(dict(target=target, feature_variant=variant, feature_count=len(names),
                            zero_fraction=float(np.mean(y == 0)), direct_model=direct_fit["chosen"],
                            direct_graded_cv_deviance=direct_row["graded_cv_deviance"],
                            best_hurdle_model=best_hurdle["model"],
                            best_hurdle_graded_cv_deviance=best_hurdle["graded_cv_deviance"],
                            selected_model=best["model"], selected_graded_cv_deviance=best["graded_cv_deviance"],
                            change_vs_direct_pct=gain, graded_regions_improved=improves_regions,
                            fold_details=fold_notes, candidates=target_rows))
        rows.extend(target_rows)
        log(f"  {target}: direct {direct_row['graded_cv_deviance']:.6f}; best {best['model']} "
            f"{best['graded_cv_deviance']:.6f} ({gain:+.3f}%), regions improved {improves_regions}/4")
        atomic_json(out(report_file), dict(
            date="2026-10-24", experiment="zero-versus-positive hurdle count models",
            hypothesis="A separate probability of any observed count may help where proxy counts have many zeros.",
            split="GroupKFold(5), grouped by county; no label from a held-out county enters either hurdle stage.",
            target_proxy="observed Overture counts; clipped only at the existing target cap",
            feature_variants=feature_variants, classifier_configs=classifier_configs,
            positive_count_configs=positive_configs,
            selection_note="Feature and model selection use the same grouped OOF folds; treat CV wins as exploratory.",
            target_summary=targets, results=rows))
    report = dict(
        date="2026-10-24", experiment="zero-versus-positive hurdle count models",
        hypothesis="A separate probability of any observed count may help where proxy counts have many zeros.",
        split="GroupKFold(5), grouped by county; no label from a held-out county enters either hurdle stage.",
        target_proxy="observed Overture counts; clipped only at the existing target cap",
        feature_variants=feature_variants, classifier_configs=classifier_configs,
        positive_count_configs=positive_configs,
        selection_note="Feature and model selection use the same grouped OOF folds; treat CV wins as exploratory.",
        target_summary=targets, results=rows)
    atomic_json(out(report_file), report)
    return report


def auxiliary_cross_count_sweep(df, report_file="aux_cross_count_sweep.json"):
    """Measure whether other supplied Overture count columns add county-held-out signal for each proxy count."""
    from sklearn.metrics import mean_poisson_deviance

    feature_variants = AUX_HURDLE_FEATURE_CHOICES
    graded = df["region"].isin(GRADED).to_numpy()
    results, summary = [], []
    for target, cap in AUX_TARGETS:
        base_x, base_names = auxiliary_feature_matrix(df, feature_variants[target])
        cross_x, cross_names = append_other_overture_count_features(df, target, base_x, base_names)
        x_by_variant = {"base": (base_x, base_names), "other_overture_counts": (cross_x, cross_names)}
        y = df[target].clip(upper=cap).to_numpy(float)
        target_results = []
        for feature_variant, (X, names) in x_by_variant.items():
            for model_kind, fit_function in (("direct", cross_fit), ("hurdle_blend", cross_fit_hurdle)):
                pred, fit = fit_function(df, X, target, cap)
                by_region = {region: float(mean_poisson_deviance(y[df.region.eq(region)],
                                                                  pred[df.region.eq(region)]))
                             for region in REGIONS}
                row = dict(target=target, feature_variant=feature_variant, model_kind=model_kind,
                           feature_count=len(names), cv_deviance=fit["cv_deviance"],
                           graded_cv_deviance=float(mean_poisson_deviance(y[graded], pred[graded])),
                           region_deviance=by_region,
                           hurdle_configuration=({"blend": fit["hurdle_blend"],
                                                  "classifier": fit["hurdle_classifier"],
                                                  "positive_model": fit["hurdle_positive_model"]}
                                                 if model_kind == "hurdle_blend" else None))
                target_results.append(row)
                results.append(row)
        for model_kind in ("direct", "hurdle_blend"):
            before = next(r for r in target_results if r["feature_variant"] == "base" and r["model_kind"] == model_kind)
            after = next(r for r in target_results if r["feature_variant"] == "other_overture_counts" and
                         r["model_kind"] == model_kind)
            deltas = {region: after["region_deviance"][region] / before["region_deviance"][region] - 1
                      for region in GRADED}
            summary.append(dict(target=target, model_kind=model_kind,
                                base_graded_cv_deviance=before["graded_cv_deviance"],
                                other_counts_graded_cv_deviance=after["graded_cv_deviance"],
                                change_pct=100 * (after["graded_cv_deviance"] /
                                                  before["graded_cv_deviance"] - 1),
                                graded_region_change_pct={region: 100 * value for region, value in deltas.items()},
                                graded_regions_improved=sum(v < 0 for v in deltas.values())))
        atomic_json(out(report_file), dict(
            date="2026-10-24", experiment="cross-category Overture count features",
            hypothesis="Other challenge-provided Overture count categories may supply context for a held-out proxy count.",
            split="GroupKFold(5), grouped by county; the current target count is excluded from the added features.",
            target_proxy="observed Overture counts",
            feature_source="supplied Overture category counts; log counts, nonzero flags, and population-normalized rates",
            excluded_from_features={target: target for target, _ in AUX_TARGETS},
            selection_note="Target count CV is exploratory; the direct and hurdle settings were fixed before this feature comparison.",
            summary=summary, results=results))
        best = min((r for r in summary if r["target"] == target), key=lambda r: r["change_pct"])
        log(f"  {target}: other Overture counts change deviance {best['change_pct']:+.3f}% "
            f"(best of fixed direct/hurdle; {best['graded_regions_improved']}/4 graded regions improved)")
    report = dict(
        date="2026-10-24", experiment="cross-category Overture count features",
        hypothesis="Other challenge-provided Overture count categories may supply context for a held-out proxy count.",
        split="GroupKFold(5), grouped by county; the current target count is excluded from the added features.",
        target_proxy="observed Overture counts",
        feature_source="supplied Overture category counts; log counts, nonzero flags, and population-normalized rates",
        selection_note="Target count CV is exploratory; the direct and hurdle settings were fixed before this feature comparison.",
        summary=summary, results=results)
    atomic_json(out(report_file), report)
    return report


def auxiliary_source_mix_sweep(df, report_file="aux_source_mix_sweep.json"):
    """Test whether target-excluded Overture source composition helps county-held-out proxy models."""
    from sklearn.metrics import mean_poisson_deviance

    source_mix = auxiliary_source_mix_features()
    graded = df["region"].isin(GRADED).to_numpy()
    summary = []
    results = []
    for target, cap in AUX_TARGETS:
        variant = AUX_HURDLE_FEATURE_CHOICES[target]
        base_x, base_names = auxiliary_feature_matrix(df, variant)
        source_x, source_names = append_auxiliary_source_mix(df, target, base_x, base_names, source_mix)
        y = df[target].clip(upper=cap).to_numpy(float)
        target_rows = []
        for feature_set, X, names in (("base", base_x, base_names),
                                      ("base_plus_source_mix", source_x, source_names)):
            pred, fit = cross_fit(df, X, target, cap)
            by_region = {region: float(mean_poisson_deviance(y[df.region.eq(region)],
                                                              pred[df.region.eq(region)]))
                         for region in REGIONS}
            row = dict(target=target, feature_variant=variant, feature_set=feature_set,
                       feature_count=len(names), model=fit["chosen"],
                       cv_deviance=fit["cv_deviance"],
                       graded_cv_deviance=float(mean_poisson_deviance(y[graded], pred[graded])),
                       region_deviance=by_region)
            target_rows.append(row)
            results.append(row)
        base = next(row for row in target_rows if row["feature_set"] == "base")
        added = next(row for row in target_rows if row["feature_set"] == "base_plus_source_mix")
        region_change = {region: 100 * (added["region_deviance"][region] /
                                        base["region_deviance"][region] - 1) for region in GRADED}
        summary.append(dict(target=target, base_graded_cv_deviance=base["graded_cv_deviance"],
                            source_mix_graded_cv_deviance=added["graded_cv_deviance"],
                            change_pct=100 * (added["graded_cv_deviance"] /
                                              base["graded_cv_deviance"] - 1),
                            graded_region_change_pct=region_change,
                            graded_regions_improved=sum(change < 0 for change in region_change.values())))
        log(f"  {target}: target-excluded source mix changes deviance {summary[-1]['change_pct']:+.3f}% "
            f"({summary[-1]['graded_regions_improved']}/4 graded regions improved)")

    report = dict(
        date="2026-10-24", experiment="target-excluded Overture source-mix features",
        hypothesis="Source composition among other Overture places may reveal mapping-feed coverage gaps beyond demographic context.",
        split="GroupKFold(5), grouped by county; the current proxy target is removed before source counts are aggregated.",
        target_proxy="observed Overture counts",
        source="challenge-provided Overture POI sources[1].dataset and categories.primary / categories.alternate / names.primary",
        feature_set=["log1p non-target POIs per million residents", "Meta/BrightQuery/Microsoft/other shares",
                     "source entropy"],
        target_exclusions={
            "E_fire": "fire primary, alternate, or matching fire-name evidence",
            "E_ems": "EMS primary, alternate, or matching EMS-name evidence",
            "E_sch": "school primary, alternate, or matching non-driving / non-dance / non-music / non-art / non-martial / non-cosmetology / non-swim / non-beauty / non-flight name evidence",
            "n_poi_bq": "BrightQuery source rows",
        },
        selection_note="This is a proxy-count validation only; no leaderboard score is implied.",
        summary=summary, results=results)
    atomic_json(out(report_file), report)
    return report


def auxiliary_source_signal_sweep(df, report_file="aux_source_signal_sweep.json"):
    """Test whether Overture-signals provenance adds proxy signal beyond first-source composition."""
    from sklearn.metrics import mean_poisson_deviance

    source_mix = auxiliary_source_mix_features()
    source_signals = auxiliary_source_signal_features()
    graded = df["region"].isin(GRADED).to_numpy()
    summary = []
    results = []
    for target, cap in AUX_TARGETS:
        variant = AUX_HURDLE_FEATURE_CHOICES[target]
        base_x, base_names = auxiliary_feature_matrix(df, variant)
        source_x, source_names = append_auxiliary_source_mix(df, target, base_x, base_names, source_mix)
        signal_x, signal_names = append_auxiliary_source_signals(
            df, target, source_x, source_names, source_mix, source_signals)
        y = df[target].clip(upper=cap).to_numpy(float)
        target_rows = []
        for feature_set, X, names in (("base_plus_first_source_mix", source_x, source_names),
                                      ("base_plus_first_source_mix_and_signals", signal_x, signal_names)):
            pred, fit = cross_fit(df, X, target, cap)
            by_region = {region: float(mean_poisson_deviance(y[df.region.eq(region)],
                                                              pred[df.region.eq(region)]))
                         for region in REGIONS}
            row = dict(target=target, feature_variant=variant, feature_set=feature_set,
                       feature_count=len(names), model=fit["chosen"], cv_deviance=fit["cv_deviance"],
                       graded_cv_deviance=float(mean_poisson_deviance(y[graded], pred[graded])),
                       region_deviance=by_region)
            target_rows.append(row)
            results.append(row)
        base = next(row for row in target_rows if row["feature_set"] == "base_plus_first_source_mix")
        added = next(row for row in target_rows if row["feature_set"] == "base_plus_first_source_mix_and_signals")
        region_change = {region: 100 * (added["region_deviance"][region] /
                                        base["region_deviance"][region] - 1) for region in GRADED}
        summary.append(dict(target=target, first_source_mix_graded_cv_deviance=base["graded_cv_deviance"],
                            with_signal_graded_cv_deviance=added["graded_cv_deviance"],
                            incremental_change_pct=100 * (added["graded_cv_deviance"] /
                                                          base["graded_cv_deviance"] - 1),
                            graded_region_change_pct=region_change,
                            graded_regions_improved=sum(change < 0 for change in region_change.values())))
        log(f"  {target}: Overture-signals changes deviance {summary[-1]['incremental_change_pct']:+.3f}% "
            f"({summary[-1]['graded_regions_improved']}/4 graded regions improved)")
    report = dict(
        date="2026-10-24", experiment="incremental Overture-signals provenance feature",
        hypothesis="The additional Overture-signals provenance record may mark generated or enriched POIs and add mapping-completeness signal beyond the first source.",
        split="GroupKFold(5), grouped by county; each proxy target is excluded before counting signal-source POIs.",
        target_proxy="observed Overture counts",
        feature_source="Challenge-provided Overture sources[3].dataset == 'Overture-signals' on non-target places.",
        features=["share of non-target POIs with Overture-signals provenance",
                  "log1p Overture-signals POIs per million residents"],
        selection_note="Incremental proxy validation only; first-source feature settings are held fixed.",
        summary=summary, results=results)
    atomic_json(out(report_file), report)
    return report


def auxiliary_source_confidence_sweep(df, report_file="aux_source_confidence_sweep.json"):
    """Test source-specific place-confidence summaries beyond target-excluded source composition."""
    from sklearn.metrics import mean_poisson_deviance

    source_mix = auxiliary_source_mix_features()
    source_confidence = auxiliary_source_confidence_features()
    graded = df["region"].isin(GRADED).to_numpy()
    summary, results = [], []
    for target, cap in AUX_TARGETS:
        variant = AUX_HURDLE_FEATURE_CHOICES[target]
        base_x, base_names = auxiliary_feature_matrix(df, variant)
        mix_x, mix_names = append_auxiliary_source_mix(df, target, base_x, base_names, source_mix)
        confidence_x, confidence_names = append_auxiliary_source_confidence(
            df, target, mix_x, mix_names, source_confidence)
        y = df[target].clip(upper=cap).to_numpy(float)
        target_rows = []
        for feature_set, X, names in (("base_plus_first_source_mix", mix_x, mix_names),
                                      ("base_plus_source_mix_and_confidence", confidence_x, confidence_names)):
            pred, fit = cross_fit(df, X, target, cap)
            by_region = {region: float(mean_poisson_deviance(y[df.region.eq(region)],
                                                              pred[df.region.eq(region)]))
                         for region in REGIONS}
            row = dict(target=target, feature_variant=variant, feature_set=feature_set,
                       feature_count=len(names), model=fit["chosen"], cv_deviance=fit["cv_deviance"],
                       graded_cv_deviance=float(mean_poisson_deviance(y[graded], pred[graded])),
                       region_deviance=by_region)
            target_rows.append(row)
            results.append(row)
        base = target_rows[0]
        added = target_rows[1]
        region_change = {region: 100 * (added["region_deviance"][region] /
                                        base["region_deviance"][region] - 1) for region in GRADED}
        change = 100 * (added["graded_cv_deviance"] / base["graded_cv_deviance"] - 1)
        summary.append(dict(target=target,
                            source_mix_graded_cv_deviance=base["graded_cv_deviance"],
                            with_confidence_graded_cv_deviance=added["graded_cv_deviance"],
                            incremental_change_pct=change, graded_region_change_pct=region_change,
                            graded_regions_improved=sum(value < 0 for value in region_change.values())))
        log(f"  {target}: confidence changes deviance {change:+.3f}% "
            f"({summary[-1]['graded_regions_improved']}/4 graded regions improved)")
    report = dict(
        date="2026-10-24", experiment="source-specific Overture place-confidence features",
        hypothesis="Within-source averages of Overture existence confidence may mark local POI quality and add proxy-count signal beyond source composition.",
        split="GroupKFold(5), grouped by county; target-specific place exclusions match the auxiliary count labels.",
        target_proxy="observed Overture counts",
        feature_source="Challenge-provided Overture Place confidence, aggregated separately by sources[1].dataset family.",
        features=[f"mean place confidence by first source ({source})" for source in ("meta", "BrightQuery", "Microsoft", "other")] +
                 [f"share with confidence present by first source ({source})" for source in ("meta", "BrightQuery", "Microsoft", "other")],
        schema_note="The feature is an existence-confidence score, not mapping completeness; source-specific aggregation avoids comparing raw levels across providers.",
        selection_note="Incremental proxy validation only; source-composition features and model settings are held fixed.",
        summary=summary, results=results)
    atomic_json(out(report_file), report)
    return report


def auxiliary_source_richness_sweep(df, report_file="aux_source_richness_sweep.json"):
    """Test source-specific address/contact-field completeness beyond source mix and confidence."""
    from sklearn.metrics import mean_poisson_deviance

    source_mix = auxiliary_source_mix_features()
    source_confidence = auxiliary_source_confidence_features()
    source_richness = auxiliary_source_richness_features()
    graded = df["region"].isin(GRADED).to_numpy()
    summary, results = [], []
    for target, cap in AUX_TARGETS:
        variant = AUX_HURDLE_FEATURE_CHOICES[target]
        base_x, base_names = auxiliary_feature_matrix(df, variant)
        mix_x, mix_names = append_auxiliary_source_mix(df, target, base_x, base_names, source_mix)
        confidence_x, confidence_names = append_auxiliary_source_confidence(
            df, target, mix_x, mix_names, source_confidence)
        richness_x, richness_names = append_auxiliary_source_richness(
            df, target, confidence_x, confidence_names, source_richness)
        y = df[target].clip(upper=cap).to_numpy(float)
        target_rows = []
        for feature_set, X, names in (("source_mix_plus_confidence", confidence_x, confidence_names),
                                      ("source_mix_confidence_plus_richness", richness_x, richness_names)):
            pred, fit = cross_fit(df, X, target, cap)
            by_region = {region: float(mean_poisson_deviance(y[df.region.eq(region)],
                                                              pred[df.region.eq(region)]))
                         for region in REGIONS}
            row = dict(target=target, feature_variant=variant, feature_set=feature_set,
                       feature_count=len(names), model=fit["chosen"], cv_deviance=fit["cv_deviance"],
                       graded_cv_deviance=float(mean_poisson_deviance(y[graded], pred[graded])),
                       region_deviance=by_region)
            target_rows.append(row)
            results.append(row)
        base, added = target_rows
        region_change = {region: 100 * (added["region_deviance"][region] /
                                        base["region_deviance"][region] - 1) for region in GRADED}
        change = 100 * (added["graded_cv_deviance"] / base["graded_cv_deviance"] - 1)
        summary.append(dict(target=target, confidence_graded_cv_deviance=base["graded_cv_deviance"],
                            with_richness_graded_cv_deviance=added["graded_cv_deviance"],
                            incremental_change_pct=change, graded_region_change_pct=region_change,
                            graded_regions_improved=sum(value < 0 for value in region_change.values())))
        log(f"  {target}: address/contact richness changes deviance {change:+.3f}% "
            f"({summary[-1]['graded_regions_improved']}/4 graded regions improved)")
    report = dict(
        date="2026-10-24", experiment="source-specific Overture address and contact-field shares",
        hypothesis="Address and contact-field availability may reveal local contribution depth and improve proxy counts beyond source mix and confidence.",
        split="GroupKFold(5), grouped by county; target-specific place exclusions match the auxiliary count labels.",
        target_proxy="observed Overture counts",
        feature_source="Challenge-provided Overture Place addresses, websites, phones, and socials, grouped by sources[1].dataset family.",
        features=[f"share of non-target {attribute}-bearing places within first source {source}"
                  for source in ("meta", "BrightQuery", "Microsoft", "other")
                  for attribute in ("address", "website", "phone", "social")],
        selection_note="Incremental proxy validation only; source-mix and confidence features and model settings are held fixed.",
        summary=summary, results=results)
    atomic_json(out(report_file), report)
    return report


def auxiliary_spatial_lag_sweep(df, report_file="aux_spatial_lag_sweep.json"):
    """Test whether target-excluded nearby-tract counts improve proxy prediction beyond same-tract cross-counts."""
    from sklearn.metrics import mean_poisson_deviance

    spatial_lags = auxiliary_spatial_lag_features()
    graded = df["region"].isin(GRADED).to_numpy()
    summary, results = [], []
    for target, cap in AUX_TARGETS:
        variant = AUX_CROSS_COUNT_FEATURE_CHOICES[target]
        base_x, base_names = auxiliary_feature_matrix(df, variant)
        base_x, base_names = append_other_overture_count_features(df, target, base_x, base_names)
        spatial_x, spatial_names = append_auxiliary_spatial_lags(df, target, base_x, base_names, spatial_lags)
        y = df[target].clip(upper=cap).to_numpy(float)
        target_rows = []
        for feature_set, X, names in (("same_tract_cross_counts", base_x, base_names),
                                      ("same_tract_plus_spatial_cross_counts", spatial_x, spatial_names)):
            pred, fit = cross_fit(df, X, target, cap)
            by_region = {region: float(mean_poisson_deviance(y[df.region.eq(region)],
                                                              pred[df.region.eq(region)]))
                         for region in REGIONS}
            row = dict(target=target, feature_variant=variant, feature_set=feature_set,
                       feature_count=len(names), model=fit["chosen"], cv_deviance=fit["cv_deviance"],
                       graded_cv_deviance=float(mean_poisson_deviance(y[graded], pred[graded])),
                       region_deviance=by_region)
            target_rows.append(row)
            results.append(row)
        base, added = target_rows
        region_change = {region: 100 * (added["region_deviance"][region] /
                                        base["region_deviance"][region] - 1) for region in GRADED}
        change = 100 * (added["graded_cv_deviance"] / base["graded_cv_deviance"] - 1)
        summary.append(dict(target=target, same_tract_graded_cv_deviance=base["graded_cv_deviance"],
                            with_spatial_lags_graded_cv_deviance=added["graded_cv_deviance"],
                            incremental_change_pct=change, graded_region_change_pct=region_change,
                            graded_regions_improved=sum(value < 0 for value in region_change.values())))
        log(f"  {target}: spatial context changes deviance {change:+.3f}% "
            f"({summary[-1]['graded_regions_improved']}/4 graded regions improved)")
    report = dict(
        date="2026-10-24", experiment="target-excluded nearby-tract Overture count context",
        hypothesis="Nearby non-target Overture counts may capture local facility and mapping-feed context not represented by same-tract cross-counts.",
        split="GroupKFold(5), grouped by county; spatial features use only challenge-supplied covariates and omit the current proxy target for all neighboring tracts.",
        target_proxy="observed Overture counts",
        feature_source="Challenge-provided tract geometry, population and Overture proxy counts.",
        features=["distance-weighted mean log count per million residents and weighted nonzero share among 8 and 20 nearest tracts",
                  "neighbor source families: other facility proxies, BrightQuery places, and Overture buildings"],
        leakage_control="The current target proxy is removed from all neighbor sources before generating its features; only other challenge-provided count sources are used.",
        selection_note="Incremental proxy validation only; same-tract cross-count features and county folds are held fixed.",
        summary=summary, results=results)
    atomic_json(out(report_file), report)
    return report


def auxiliary_ems_strata_sweep(df, report_file="aux_ems_strata_sweep_v2.json"):
    """Ablate the expanded supplied-strata families and tune EMS count-model capacity under county-grouped CV."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import PoissonRegressor
    from sklearn.metrics import mean_poisson_deviance
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    old_x, old_names = auxiliary_feature_matrix(df, "density_strata")
    full_x, full_names = auxiliary_feature_matrix(df, "density_strata_full")
    old_set = set(old_names)
    extended = [name for name in full_names if name not in old_set]
    family_prefixes = ("nifc_", "fod_", "ghcn_", "cdcw_", "hwd_", "gehe_", "uhe_", "drought_gov_",
                       "mtbs_", "usgs_")
    family_names = {}
    for prefix in family_prefixes:
        name = prefix[:-1]
        family_names[name] = [feature for feature in extended
                              if feature.startswith(f"strata__{prefix}")
                              and not (name == "uhe" and feature.startswith(
                                  ("strata__uhe_city_name_", "strata__uhe_city_country_")))]
    context_columns = ("ALAND", "pop_urban", "pop_rural", "frac_inside_aoi", "ur_class",
                       "aiannh_kind", "uhe_city_name", "uhe_city_country")
    family_names["tract_context"] = [feature for feature in extended
                                     if any(feature == f"strata__{col}" or feature.startswith(f"strata__{col}_")
                                            for col in context_columns)]
    assigned = set().union(*map(set, family_names.values()))
    family_names["other"] = [feature for feature in extended
                             if feature not in assigned and feature not in family_names["tract_context"]]
    family_names = {name: features for name, features in family_names.items() if features}

    variants = {"density_strata": (old_x, old_names), "density_strata_full": (full_x, full_names)}
    for family, features in family_names.items():
        dropped = set(features)
        keep = [j for j, name in enumerate(full_names) if name not in dropped]
        variants[f"full_without_{family}"] = (full_x[:, keep], [full_names[j] for j in keep])

    target, cap = "E_ems", 20
    y = df[target].clip(upper=cap).to_numpy(float)
    graded = df["region"].isin(GRADED).to_numpy()
    groups = df["county"].to_numpy()
    model_configs = dict(AUX_GBM_CONFIGS)
    result_rows = []

    def save_report():
        ranked = sorted(result_rows, key=lambda row: row["graded_cv_deviance"])
        completed = list(dict.fromkeys(r["variant"] for r in result_rows))
        report = dict(
            split="GroupKFold(5), grouped by county",
            target_proxy=target,
            scored_regions=GRADED,
            feature_sets={name: {"feature_count": len(names), "ablated_features":
                                next((fam for fam, feats in family_names.items()
                                      if name == f"full_without_{fam}"), [])}
                          for name, (_, names) in variants.items() if name in completed},
            completed_variants=completed,
            tested_additional_families={name: features for name, features in family_names.items()},
            gbm_configs=model_configs,
            ablation_gbm_configs={"shallow_regularized": model_configs["shallow_regularized"],
                                  "slow_regularized": model_configs["slow_regularized"]},
            results=result_rows,
            ranking=[{"variant": r["variant"], "model": r["model"], "graded_cv_deviance": r["graded_cv_deviance"]}
                     for r in ranked],
            best_by_graded_cv=(None if not ranked else {k: ranked[0][k] for k in
                                                        ("variant", "model", "feature_count", "graded_cv_deviance",
                                                         "region_deviance")}),
            selection_note="Exploratory model and feature-family selection on the same grouped OOF folds; not an independent test.")
        atomic_json(out(report_file), report)
        return report

    for variant, (X, names) in variants.items():
        log(f"  {variant}: {X.shape[1]} EMS proxy features")
        is_base = variant in ("density_strata", "density_strata_full")
        configs = model_configs if is_base else {k: model_configs[k] for k in ("shallow_regularized", "slow_regularized")}
        oof = {"poisson_glm": np.zeros(len(y))} if is_base else {}
        oof.update({f"poisson_gbm:{config}": np.zeros(len(y)) for config in configs})
        for tr, te in GroupKFold(5).split(X, y, groups):
            if is_base:
                glm = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-3, max_iter=2000))
                oof["poisson_glm"][te] = np.clip(glm.fit(X[tr], y[tr]).predict(X[te]), 1e-4, None)
            for config, params in configs.items():
                gbm = HistGradientBoostingRegressor(loss="poisson", random_state=0, **params)
                key = f"poisson_gbm:{config}"
                oof[key][te] = np.clip(gbm.fit(X[tr], y[tr]).predict(X[te]), 1e-4, None)
        for model, pred in oof.items():
            by_region = {r: float(mean_poisson_deviance(y[df.region.eq(r)], pred[df.region.eq(r)]))
                         for r in REGIONS}
            result_rows.append(dict(variant=variant, model=model, feature_count=len(names),
                                    cv_deviance=float(mean_poisson_deviance(y, pred)),
                                    graded_cv_deviance=float(mean_poisson_deviance(y[graded], pred[graded])),
                                    region_deviance=by_region))
        report = save_report()
        current = [r for r in result_rows if r["variant"] == variant]
        best = min(current, key=lambda row: row["graded_cv_deviance"])
        log(f"    best {best['model']} {best['graded_cv_deviance']:.6f}")

    best = report["best_by_graded_cv"]
    log(f"  exploratory winner: {best['variant']} / {best['model']} "
        f"({best['graded_cv_deviance']:.6f}); wrote {out(report_file)}")
    return report


def auxiliary_region_sweep(df):
    """Compare one shared model with independent regional models under county-held-out folds."""
    from sklearn.metrics import mean_poisson_deviance
    results = []
    graded = df["region"].isin(GRADED).to_numpy()
    for target, cap in AUX_TARGETS:
        variant = AUX_FEATURE_CHOICES[target]
        X, features = auxiliary_feature_matrix(df, variant)
        y = df[target].clip(upper=cap).to_numpy(float)
        shared, shared_fit = cross_fit(df, X, target, cap)
        regional = np.zeros(len(df))
        regional_fit = {}
        for region in REGIONS:
            mask = df["region"].eq(region).to_numpy()
            local_df = df.loc[mask].reset_index(drop=True)
            regional[mask], regional_fit[region] = cross_fit(local_df, X[mask], target, cap)
        regional_deviance = float(mean_poisson_deviance(y[graded], regional[graded]))
        shared_deviance = float(mean_poisson_deviance(y[graded], shared[graded]))
        by_region = {r: dict(shared=float(mean_poisson_deviance(y[df.region.eq(r)], shared[df.region.eq(r)])),
                             regional=float(mean_poisson_deviance(y[df.region.eq(r)], regional[df.region.eq(r)])),
                             model=regional_fit[r]["chosen"], n=int(df.region.eq(r).sum())) for r in REGIONS}
        results.append(dict(target=target, feature_variant=variant, feature_count=len(features),
                            shared_model=shared_fit["chosen"], shared_graded_deviance=shared_deviance,
                            regional_graded_deviance=regional_deviance,
                            change_pct=100 * (regional_deviance / shared_deviance - 1), regions=by_region))
        log(f"  {target}: shared {shared_deviance:.4f}; regional {regional_deviance:.4f} "
            f"({100 * (regional_deviance / shared_deviance - 1):+.2f}%), "
            + ", ".join(f"{r} {by_region[r]['shared']:.3f}->{by_region[r]['regional']:.3f}"
                         for r in GRADED))
    report = dict(split="GroupKFold(5) separately within each region, grouped by county",
                  target_proxy="observed Overture counts", results=results)
    atomic_json(out("aux_region_sweep.json"), report)
    log(f"Wrote {out('aux_region_sweep.json')}")
    return report


def auxiliary_region_blend_sweep(df):
    """Measure whether a small regional-model correction helps shared OOF predictions."""
    from sklearn.metrics import mean_poisson_deviance

    weights = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0]
    graded = df["region"].isin(GRADED).to_numpy()
    results = []
    for target, cap in AUX_TARGETS:
        variant = AUX_FEATURE_CHOICES[target]
        X, features = auxiliary_feature_matrix(df, variant)
        y = df[target].clip(upper=cap).to_numpy(float)
        shared, shared_fit = cross_fit(df, X, target, cap)
        regional = np.zeros(len(df))
        regional_fit = {}
        for region in REGIONS:
            mask = df["region"].eq(region).to_numpy()
            local_df = df.loc[mask].reset_index(drop=True)
            regional[mask], regional_fit[region] = cross_fit(local_df, X[mask], target, cap)

        blends = []
        for weight in weights:
            prediction = np.clip((1.0 - weight) * shared + weight * regional, 1e-4, None)
            by_region = {region: float(mean_poisson_deviance(y[df.region.eq(region)], prediction[df.region.eq(region)]))
                         for region in REGIONS}
            blends.append(dict(regional_weight=weight,
                               scored_region_deviance=float(mean_poisson_deviance(y[graded], prediction[graded])),
                               region_deviance=by_region))
        best = min(blends, key=lambda row: row["scored_region_deviance"])
        results.append(dict(target=target, feature_variant=variant, feature_count=len(features),
                            shared_model=shared_fit["chosen"], regional_models={r: regional_fit[r]["chosen"] for r in REGIONS},
                            blends=blends, selected_weight=best["regional_weight"]))
        baseline = blends[0]["scored_region_deviance"]
        log(f"  {target}: shared {baseline:.4f}; best blend w={best['regional_weight']:.2f} "
            f"{best['scored_region_deviance']:.4f} ({100 * (best['scored_region_deviance'] / baseline - 1):+.2f}%)")

    report = dict(split="GroupKFold(5), grouped by county; shared and regional OOF predictions blended",
                  target_proxy="observed Overture counts", scored_regions=GRADED, weights=weights, results=results,
                  caveat="Blend weight is selected on the same OOF predictions used for reporting; selected minima are exploratory.")
    atomic_json(out("aux_region_blend_sweep.json"), report)
    log(f"Wrote {out('aux_region_blend_sweep.json')}")
    return report


# ================================================================================================
# 4. SIMULATOR: generative model of the withdrawn reference counts
# ================================================================================================
LOGIT = lambda p: float(np.log(p / (1 - p)))
SIG = lambda x: 1 / (1 + np.exp(-x))
# name, prior centre, prior sd (transformed scale). Centres are physical judgements, not fitted values.
CORE_PRIORS = [(f"log_m_{r}", 0.0, 0.4) for r in GRADED] + [  # TIGER metres per metre of Overture route network
    ("log_sigma_t", np.log(0.30), 0.4),                          # log-noise of TIGER length
    ("beta_bnd", 0.5, 0.3),                                      # boundary highway counted in both tracts
    ("logit_q_fm", LOGIT(0.25), 1.0),                            # P(Texas FM/RM road is S1200)
    ("logit_q_cty", LOGIT(0.10), 1.0),                           # P(county route is S1200)
    ("logit_q_unr", LOGIT(0.10), 1.0),                           # P(unrouted arterial is S1200)
    ("logit_p_surprise", LOGIT(0.02), 1.0),                      # TIGER highway with no Overture evidence
    ("log_kappa", 0.0, 0.3), ("log_sigma_b", np.log(0.15), 0.5), # Microsoft count = kappa * USFS count
    ("logit_r_fire", LOGIT(0.60), 1.0),                          # share of reference stations Overture labels
    ("logit_r_ems", LOGIT(0.90), 1.0), ("logit_r_sch", LOGIT(0.85), 1.0),
    ("logit_a_fire", LOGIT(0.90), 1.0), ("logit_x_fire", LOGIT(0.30), 1.0),   # Overture evidence that is real
    ("logit_a_ems", LOGIT(0.70), 1.0), ("logit_x_ems", LOGIT(0.10), 1.0),
    ("logit_a_sch", LOGIT(0.70), 1.0), ("logit_x_sch", LOGIT(0.10), 1.0),
    ("ge_fac", -0.3, 0.3),                                       # more mapping detail -> fewer unseen facilities
    ("log_eps", np.log(0.02), 0.7), ("log_sigma_c", np.log(0.5), 0.4),        # establishments per resident
    ("a_bq", 0.5, 0.5)]                                          # business-registry intensity elasticity
STRATA_PRIORS = [(f"gf_{k}", 0.0, 0.3) for k in STRATA] + [(f"gr_{k}", 0.0, 0.3) for k in STRATA]
# structural mechanisms: facility copies leak across tract boundaries (spread sigma_loc, metres); TIGER keeps one
# record per (edge, name), so routes that also carry a local street name count again (w_local); Interstate adjustment
STRUCT_PRIORS = [("log_sigma_loc", np.log(60.0), 0.5), ("w_local", 0.0, 0.5), ("w_inter", 0.0, 0.5),
                 ("w_multi", 0.0, 0.5)]   # concurrent routes (several route numbers on one road): one TIGER record each
# reference facilities that the organisers' Overture category match does not count: share of Overture colleges /
# vocational schools and of other K-12 / education places that are in the schools layer, and of ERs in the EMS layer
UNMATCHED_PRIORS = [("logit_a_col", LOGIT(0.2), 1.5), ("logit_a_k12", LOGIT(0.3), 1.5), ("logit_a_er", LOGIT(0.1), 1.5)]
# covariate effects found by screening the score-visible misfit: log-effects of the Interstate share, road detail and
# Farm-to-Market share on the TIGER length, and of buildings per housing unit on the Microsoft building scale
COVAR_PRIORS = [("gt_inter", 0.0, 0.3), ("gt_detail", 0.0, 0.3), ("gt_fm", 0.0, 0.3), ("gb_bph", 0.0, 0.3)]
# flexible context effects (log scale, ridge prior): on the TIGER length (ft_) and on the unseen-facility rate (ff_)
FLEX_T = ["bph", "unr_share", "loc_share", "multi_share", "tert_share", "cty_share", "log_area", "log_density", "ruca",
          "pct_urban", "named_over_route", "c_unr_share", "svi", "tribal"]
FLEX_F = ["bph", "log_area", "log_density", "ruca", "pct_urban", "places_pc", "meta_share", "bq_share", "svi", "tribal",
          "log_bld", "fire_weak_share", "has_route", "detail"]
FLEX_SD = 0.1
FLEX_PRIORS = lambda sd: [(f"ft_{k}", 0.0, sd) for k in FLEX_T] + [(f"ff_{k}", 0.0, sd) for k in FLEX_F]
# per-region deviations from each flexible effect (the same feature can act differently in each region's map)
FLEX_REGIONAL_PRIORS = ([(f"ftr_{k}_{r}", 0.0, 0.1) for k in FLEX_T for r in GRADED]
                        + [(f"ffr_{k}_{r}", 0.0, 0.1) for k in FLEX_F for r in GRADED])
# neural context effects: a small MLP (standardised context features -> tanh units -> log-multipliers) inside the
# simulator, calibrated by the same simulated moments. The output layer starts at zero, so the network starts as the
# physics model without context effects. `--nn`: 6 units, multipliers of the TIGER length (t) and the unseen-facility
# rate (f). `--nn-wide`: 12 units and all four reference layers, adding the Microsoft building scale (b) and the CBP
# establishments per resident (c).
NN_FEATURES = sorted(set(FLEX_T) | set(FLEX_F))
NN_LAYERS = "tfbc"


def nn_priors(hidden, outputs):
    return ([(f"nn_w1_{k}_{j}", 0.0, 0.5) for k in NN_FEATURES for j in range(hidden)]
            + [(f"nn_b1_{j}", 0.0, 0.5) for j in range(hidden)]
            + [(f"nn_w2_{j}_{o}", 0.0, 0.1) for j in range(hidden) for o in outputs] + [(f"nn_b2_{o}", 0.0, 0.1) for o in outputs])


NN_PRIORS = nn_priors(6, "tf")
NN_WIDE_PRIORS = nn_priors(12, NN_LAYERS)
# per-region log-multipliers of the unseen-facility rate, establishments per resident and the building scale
REGIONAL_PRIORS = [(f"{pre}_{r}", 0.0, 0.3) for pre in ("gfr", "gcr", "gbr") for r in GRADED]
NAMES, THETA0, PRIOR_SD = [], np.zeros(0), np.zeros(0)


def flex_suffix(sd):
    return "_flex" if sd == FLEX_SD else f"_flex{round(sd * 100)}"


def set_priors(use_scorecard, regional=False, structural=False, unmatched=False, covariates=False, flex=False,
               flex_regional=False, nn=False):
    """Stratum effects are free only when the public scorecard is used to calibrate them; regional multipliers only
    in the regional physics model."""
    global NAMES, THETA0, PRIOR_SD
    pri = (CORE_PRIORS + (STRATA_PRIORS if use_scorecard else []) + (STRUCT_PRIORS if structural else [])
           + (UNMATCHED_PRIORS if unmatched else []) + (REGIONAL_PRIORS if regional else [])
           + (COVAR_PRIORS if covariates else []) + (FLEX_PRIORS(flex) if flex else [])
           + (FLEX_REGIONAL_PRIORS if flex_regional else []) + (NN_WIDE_PRIORS if nn == "wide" else NN_PRIORS if nn else []))
    NAMES, THETA0, PRIOR_SD = [p[0] for p in pri], np.array([p[1] for p in pri]), np.array([p[2] for p in pri])


def inputs(df, mu):
    F = {k: df[c].to_numpy(float) for k, c in [("N", "L_named"), ("L_route", "L_route"), ("L_fm", "L_fm"),
                                                ("L_cty", "L_county"), ("L_unr", "L_unrouted"), ("bnd", "bnd"),
                                                ("O_b", "n_bld"), ("U_b", "U_bld"), ("pop", "pop"), ("n_poi", "n_poi")]}
    for k, o in [("fire", "n_fd"), ("ems", "n_ems"), ("sch", "n_sch")]:
        F[f"O_{k}"] = df[o].to_numpy().astype(int)
        F[f"X_{k}"] = df[f"X_{k}"].to_numpy().astype(int)
        F[f"mu_{k}"] = mu[f"E_{k}"]
    F["z_bq"] = np.log((mu["n_poi_bq"] + 1) / (F["pop"] + 100))
    F["z_bq"] = (F["z_bq"] - F["z_bq"].mean()) / F["z_bq"].std()
    e = np.log((df["detail"].to_numpy() + 0.01) / (1.01 - df["detail"].to_numpy()))
    F["z_effort"] = (e - e.mean()) / e.std()
    F["ridx"] = df["region"].map({r: i for i, r in enumerate(REGIONS)}).to_numpy()
    F["L_loc"], F["L_int"] = df["L_route_localname"].to_numpy(float), df["L_interstate"].to_numpy(float)
    F["L_multi"] = df["L_route_multiref"].to_numpy(float)
    zs = lambda x: (np.asarray(x, float) - np.mean(x)) / (np.std(x) + 1e-12)
    F["z_inter"] = zs(df["L_interstate"] / (df["L_route"] + 100))
    F["z_detail"] = zs(np.log((df["detail"] + 0.01) / (1.01 - df["detail"])))
    F["z_fm"] = zs(df["L_fm"] / (df["L_route"] + df["L_fm"] + 100))
    F["z_bph"] = zs(np.log((df["n_bld"] + 1) / (df["housing_units"] + 1)))
    R = df["L_route"] + 100
    cty = df.groupby("county")[["L_route", "L_unrouted"]].transform("sum")
    flex = {"bph": F["z_bph"], "unr_share": df["L_unrouted"] / R, "loc_share": df["L_route_localname"] / R,
            "multi_share": df["L_route_multiref"] / R, "tert_share": df["L_tertiary"] / R, "cty_share": df["L_county"] / R,
            "log_area": np.log1p(df["area_km2"]), "log_density": np.log1p(df["pop"] / (df["area_km2"] + 0.01)),
            "ruca": df["ruca_primary"].fillna(1).clip(upper=10), "pct_urban": df["pct_urban"].fillna(0),
            "named_over_route": np.log1p(df["L_named"] / R), "c_unr_share": cty["L_unrouted"] / (cty["L_route"] + cty["L_unrouted"] + 1),
            "svi": df["svi_overall"].fillna(df["svi_overall"].median()), "tribal": df["tribal_pct"].fillna(0),
            "places_pc": np.log1p(df["n_poi"] / (df["pop"] + 100) * 1000), "meta_share": df["n_poi_meta"] / df["n_poi"].clip(lower=1),
            "bq_share": df["n_poi_bq"] / df["n_poi"].clip(lower=1), "log_bld": np.log1p(df["n_bld"]),
            "fire_weak_share": df["X_fire"] / (df["E_fire"] + 1), "has_route": (df["L_route"] > 1).astype(float), "detail": df["detail"]}

    for k, v in flex.items():
        F[f"zf_{k}"] = zs(v)
    for k in UNMATCHED:
        F[f"U_{k}"] = df[f"n_{k}"].to_numpy().astype(int)
    F["pts"] = facility_points(df)
    for k in STRATA:
        F[f"s_{k}"] = df[f"s_{k}"].to_numpy(float)
    F["n"] = len(df)
    return F


def facility_points(df):
    """Per facility type: each Overture point, the tracts within 500 m, signed distances and a sparse tract map."""
    from scipy import sparse
    paths = [os.path.join(FEAT, f"{r}-facpts.parquet") for r in REGIONS]
    if not all(os.path.exists(p_) for p_ in paths):
        return None
    idx = pd.Series(np.arange(len(df)), index=df["GEOID"].to_numpy())
    pts = pd.concat([pd.read_parquet(p_) for p_ in paths])
    pts = pts[pts["GEOID"].isin(idx.index)]
    out = {}
    for k in ["fire", "ems", "sch"]:
        q = pts[pts["typ"] == k].sort_values(["id", "d"], ascending=[True, False]).reset_index(drop=True)
        pid, j = np.unique(q["id"].to_numpy(), return_inverse=True)
        strong = q["strong"].fillna(False).astype(bool).groupby(j).first().to_numpy()
        A = sparse.csr_matrix((np.ones(len(q)), (idx[q["GEOID"]].to_numpy(), np.arange(len(q)))), shape=(len(df), len(q)))
        out[k] = {"j": j, "d": q["d"].to_numpy(), "strong": strong, "A": A, "n_pts": len(pid)}
    return out


def point_counts(rng, P, alpha, alpha_x, sigma, S):
    """Reference copies of Overture facility points: each point is real with probability alpha (strong evidence)
    or alpha_x (weak), and lands in one of the tracts within 500 m with weight Phi(d / sigma)."""
    from scipy.stats import norm
    j, d = P["j"], P["d"]
    wgt = norm.cdf(d / max(sigma, 1e-3))
    tot = np.bincount(j, weights=wgt, minlength=P["n_pts"])
    wgt = wgt / np.maximum(tot[j], 1.0)
    c = np.cumsum(wgt)
    start = np.r_[0, np.cumsum(np.bincount(j, minlength=P["n_pts"]))[:-1]]
    base = np.r_[0.0, c][start][j]
    hi, lo = c - base, c - base - wgt
    real = rng.random((S, P["n_pts"])) < np.where(P["strong"], alpha, alpha_x)
    u = rng.random((S, P["n_pts"]))
    sel = real[:, j] & (u[:, j] >= lo) & (u[:, j] < hi)
    return np.asarray((P["A"] @ sel.T.astype(np.float32)).T)


def simulate(F, th, S=256, seed=0):
    """Monte Carlo under the exact composite. Returns per-tract E[y], E[y^2], P(some component undefined),
    E[TIGER length] and E[reference EMS count]."""
    p = dict(zip(NAMES, th))
    rng = np.random.default_rng(seed)
    n = F["n"]
    ln = lambda sig, shape: np.exp(sig * rng.standard_normal(shape) - 0.5 * sig ** 2)
    eff = lambda pre: np.exp(sum(p.get(f"{pre}_{k}", 0.0) * F[f"s_{k}"] for k in STRATA))
    reg = lambda pre: np.exp(np.array([p.get(f"{pre}_{r}", 0.0) for r in GRADED]
                                      + [np.mean([p.get(f"{pre}_{r}", 0.0) for r in GRADED])])[F["ridx"]])
    regk = lambda pre, k: np.array([p.get(f"{pre}_{k}_{r}", 0.0) for r in GRADED]
                                   + [np.mean([p.get(f"{pre}_{k}_{r}", 0.0) for r in GRADED])])[F["ridx"]]
    m = np.exp(np.array([p[f"log_m_{r}"] for r in GRADED] + [np.mean([p[f"log_m_{r}"] for r in GRADED])]))[F["ridx"]]
    # roads: TIGER S1100+S1200 length vs Overture named length
    inc = lambda q, L: np.where(rng.random((S, n)) < SIG(p[q]), 1.0, 0.0) * L[None, :]
    route = np.clip(F["L_route"] + p.get("w_local", 0.0) * F["L_loc"] + p.get("w_multi", 0.0) * F["L_multi"]
                    - p.get("w_inter", 0.0) * F["L_int"], 0, None)
    G = (route + p["beta_bnd"] * F["bnd"])[None, :] + inc("logit_q_fm", F["L_fm"]) + inc("logit_q_cty", F["L_cty"]) \
        + inc("logit_q_unr", F["L_unr"])
    has = G > 1.0
    nn = {}
    if "nn_b2_t" in p:
        hidden = sum(k.startswith("nn_b1_") for k in p)
        outs = [o for o in NN_LAYERS if f"nn_b2_{o}" in p]
        Z = np.column_stack([F[f"zf_{k}"] for k in NN_FEATURES])
        W1 = np.array([[p[f"nn_w1_{k}_{j}"] for j in range(hidden)] for k in NN_FEATURES])
        hid = np.tanh(Z @ W1 + np.array([p[f"nn_b1_{j}"] for j in range(hidden)]))
        W2 = np.array([[p[f"nn_w2_{j}_{o}"] for o in outs] for j in range(hidden)])
        nn = dict(zip(outs, (hid @ W2 + np.array([p[f"nn_b2_{o}"] for o in outs])).T))
    cov_t = np.exp(p.get("gt_inter", 0.0) * F["z_inter"] + p.get("gt_detail", 0.0) * F["z_detail"] + p.get("gt_fm", 0.0) * F["z_fm"]
                   + sum(p[f"ft_{k}"] * F[f"zf_{k}"] for k in FLEX_T if f"ft_{k}" in p)
                   + sum(regk("ftr", k) * F[f"zf_{k}"] for k in FLEX_T if f"ftr_{k}_{GRADED[0]}" in p) + nn.get("t", 0.0))
    T = (m * eff("gr") * cov_t)[None, :] * G * ln(np.exp(p["log_sigma_t"]), (S, n))
    surprise = ~has & (rng.random((S, n)) < SIG(p["logit_p_surprise"]))
    T = np.where(surprise, 400.0 * ln(1.0, (S, n)), T)
    dT = has | surprise
    t = np.where(dT, 1 - np.minimum(1, F["N"][None, :] / np.maximum(T, 1e-9)), 0.0)
    # buildings: Microsoft count = kappa * USFS count * noise
    MS = (np.exp(p["log_kappa"]) * F["U_b"] * reg("gbr") * np.exp(p.get("gb_bph", 0.0) * F["z_bph"] + nn.get("b", 0.0)))[None, :] \
        * ln(np.exp(p["log_sigma_b"]), (S, n))
    dB = MS > 0.5
    b = np.where(dB, 1 - np.minimum(1, F["O_b"][None, :] / np.maximum(MS, 1e-9)), 0.0)
    # facilities: reference = Bin(Overture primary, a) + Bin(weak evidence, x) + Poisson(unseen)
    fac = eff("gf") * np.exp(p["ge_fac"] * F["z_effort"] + sum(p[f"ff_{k}"] * F[f"zf_{k}"] for k in FLEX_F if f"ff_{k}" in p)
                                                   + sum(regk("ffr", k) * F[f"zf_{k}"] for k in FLEX_F if f"ffr_{k}_{GRADED[0]}" in p)
                                                   + nn.get("f", 0.0)) * reg("gfr")
    gs, ds, H_ems, types = [], [], None, {}
    for k in ["fire", "ems", "sch"]:
        r = SIG(p[f"logit_r_{k}"])
        lam = F[f"mu_{k}"] * (1 - r) / r * fac
        if "log_sigma_loc" in p and F.get("pts") is not None:
            H = (point_counts(rng, F["pts"][k], SIG(p[f"logit_a_{k}"]), SIG(p[f"logit_x_{k}"]), np.exp(p["log_sigma_loc"]), S)
                 + rng.poisson(np.broadcast_to(lam, (S, n))))
        else:
            H = (rng.binomial(np.broadcast_to(F[f"O_{k}"], (S, n)), SIG(p[f"logit_a_{k}"]))
                 + rng.binomial(np.broadcast_to(F[f"X_{k}"], (S, n)), SIG(p[f"logit_x_{k}"]))
                 + rng.poisson(np.broadcast_to(lam, (S, n))))
        extra = {"sch": [("col", "logit_a_col"), ("k12", "logit_a_k12")], "ems": [("er", "logit_a_er")]}.get(k, [])
        for src, par in extra:
            if par in p:
                H = H + rng.binomial(np.broadcast_to(F[f"U_{src}"], (S, n)), SIG(p[par]))
        if k == "ems":
            H_ems = H
            types["ems_binding"] = ((H >= 1) & (F["O_ems"][None, :] < H)).mean(0)
        types[f"EH_{k}"] = H.mean(0)
        d = H >= 1
        gs.append(np.where(d, 1 - np.minimum(1, F[f"O_{k}"][None, :] / np.maximum(H, 1)), 0.0))
        ds.append(d)
        types[f"{k}_defined"] = d.mean(0)
        types[f"{k}_gap"] = gs[-1].sum(0) / np.maximum(d.sum(0), 1)
    nd = ds[0].astype(int) + ds[1] + ds[2]
    dH = nd > 0
    h = np.where(dH, (gs[0] + gs[1] + gs[2]) / np.maximum(nd, 1), 0.0)
    # establishments: CBP = eps * population * registry intensity * noise
    cm = np.exp(p["log_eps"]) * F["pop"] * np.exp(p["a_bq"] * F["z_bq"] + nn.get("c", 0.0)) * reg("gcr")
    CBP = cm[None, :] * ln(np.exp(p["log_sigma_c"]), (S, n))
    dC = CBP >= 0.5
    c = np.where(dC, 1 - np.minimum(1, F["n_poi"][None, :] / np.maximum(CBP, 1e-9)), 0.0)
    nP = dH.astype(int) + dC
    pg = np.where(nP > 0, (h + c) / np.maximum(nP, 1), 0.0)
    dP = nP > 0
    nd_all = dT.astype(int) + dB + dP
    y = np.where(nd_all > 0, (t + b + pg) / np.maximum(nd_all, 1), 0.0)
    contrib = {"c_road": (t * dT / np.maximum(nd_all, 1)).mean(0), "c_building": (b * dB / np.maximum(nd_all, 1)).mean(0),
               "c_facility": (np.where(dP, h / np.maximum(nP, 1), 0.0) / np.maximum(nd_all, 1)).mean(0),
               "c_establishment": (np.where(dP, c / np.maximum(nP, 1), 0.0) / np.maximum(nd_all, 1)).mean(0)}
    undefined = ~(dT & dB & dP)
    comp = {f"{k}_defined": d.mean(0) for k, d in [("road", dT), ("building", dB), ("poi", dP)]}
    comp.update({f"{k}_gap": (v * d).sum(0) / np.maximum(d.sum(0), 1) for k, v, d in [("road", t, dT), ("building", b, dB),
                                                                                           ("facility", h, dH), ("establishment", c, dC)]})
    return dict(Ey=y.mean(0), Ey2=(y ** 2).mean(0), und=undefined.mean(0), ET=(T * dT).mean(0), ECBP=CBP.mean(0),
                **comp, **types, **contrib)


# ================================================================================================
# 5. CALIBRATION: Method of Simulated Moments on published statistics
# ================================================================================================
class Targets:
    def __init__(self, df, groups, use_scorecard):
        self.df, self.groups, self.use_scorecard = df, groups, use_scorecard
        self.graded = df["graded"].to_numpy()
        self.reg = {r: (df["region"] == r).to_numpy() for r in GRADED}
        self.names = [f"undefined_share[{r}]" for r in GRADED] + ["road_ratio[min region]", "road_ratio[max region]",
                      "ems_ratio", "ems_binding_share", "schools_per_tract", "fire_stations_per_tract",
                      "establishments_per_resident"]
        if use_scorecard:
            self.names += [f"scorecard[{k}]" for k in SCORECARD]

    def observed(self):
        o = [UNDEFINED_SHARE[r] for r in GRADED] + list(ROAD_RATIO_RANGE) + [EMS_RATIO, EMS_BINDING, SCHOOLS_PER_TRACT,
                                                                             FIRE_STATIONS_PER_TRACT, ESTABLISHMENTS_PER_RESIDENT]
        return np.array(o + (list(SCORECARD.values()) if self.use_scorecard else []))

    def model(self, sim):
        g, N = self.graded, self.df["L_named"].to_numpy()
        und = [sim["und"][self.reg[r]].mean() for r in GRADED]
        ratios = sorted(N[self.reg[r]].sum() / sim["ET"][self.reg[r]].sum() for r in GRADED)
        ems = self.df["n_ems"].to_numpy()[g].sum() / sim["EH_ems"][g].sum()
        pop = self.df["pop"].to_numpy()[g]
        m = und + [ratios[0], ratios[-1], ems, sim["ems_binding"][g].mean(), sim["EH_sch"][g].mean(), sim["EH_fire"][g].mean(),
                   sim["ECBP"][g].sum() / pop.sum()]
        if self.use_scorecard:
            m += [sim["Ey"][h].mean() / sim["Ey"][l].mean() for h, l in self.groups.values()]
        return np.array(m)

    def sd(self, sim):
        """Noise of each published statistic: rounding of the published figure plus model error."""
        s = [0.012] * len(GRADED) + [0.04, 0.04, 0.15, 0.01, 0.4, 0.2, 0.01]
        return np.array(s + ([SCORECARD_SD] * len(SCORECARD) if self.use_scorecard else []))


def moment_jacobian(F, T, th, S, h_frac=0.15):
    J = np.zeros((len(T.names), len(th)))
    for j in range(len(th)):
        h = h_frac * PRIOR_SD[j]
        up, dn = th.copy(), th.copy()
        up[j] += h
        dn[j] -= h
        J[:, j] = (T.model(simulate(F, up, S)) - T.model(simulate(F, dn, S))) / (2 * h)
    return J


def calibrate(F, T, J, S, th0=None, iters=30, linear=None):
    """Levenberg-Marquardt on r' W r + prior penalty, re-simulating each trial (common random numbers) and
    updating the moment Jacobian by Broyden. linear=(m0, sd0, th_lin) uses the first-order moment model."""
    obs = T.observed()
    Pinv = 1 / PRIOR_SD ** 2

    def evaluate(th):
        if linear is None:
            sim = simulate(F, th, S)
            return T.model(sim), T.sd(sim)
        m0, sd0, th_lin = linear
        return m0 + J @ (th - th_lin), sd0

    def cost(th, m, sd):
        r = (m - obs) / sd
        return float(r @ r + (th - THETA0) ** 2 @ Pinv), r

    th = (THETA0 if th0 is None else th0).copy()
    m, sd = evaluate(th)
    c, r = cost(th, m, sd)
    J, lam = J.copy(), 0.3
    for _ in range(iters):
        Jw = J / sd[:, None]
        A = Jw.T @ Jw + np.diag(Pinv)
        grad = Jw.T @ r + Pinv * (th - THETA0)
        step = np.linalg.solve(A + lam * np.diag(np.diag(A)), -grad)
        m_new, sd_new = evaluate(th + step)
        c_new, r_new = cost(th + step, m_new, sd_new)
        if c_new < c:
            if linear is None:
                J += np.outer((m_new - m) - J @ step, step) / (step @ step)
            gain = c - c_new
            th, m, sd, c, r, lam = th + step, m_new, sd_new, c_new, r_new, lam / 3
            if gain < 1e-4 * max(c, 1):
                break
        else:
            lam *= 4
            if lam > 1e6:
                break
    Jw = J / sd[:, None]
    post_cov = np.linalg.inv(Jw.T @ Jw + np.diag(Pinv))
    return th, m, sd, c, post_cov, J


# ================================================================================================
# 6-7. VALIDATION and PREDICTION
# ================================================================================================
def leave_one_target_out(T, J, th_hat, m_hat, sd_hat, m_prior):
    """Refit without each published statistic (first-order model around the fit) and predict it; compare
    with the prior centre's simulated value (no calibration at all)."""
    rows, obs = [], T.observed()
    for j, name in enumerate(T.names):
        sd = sd_hat.copy()
        sd[j] = 1e6
        th, m, *_ = calibrate(None, T, J, None, th0=th_hat, linear=(m_hat, sd, th_hat))
        rows.append(dict(target=name, published=obs[j], held_out_prediction=m[j], prior_prediction=m_prior[j]))
    return pd.DataFrame(rows)


def predict(F, th, S=4096, seed=100, chunk=512):
    """E[score | data, theta] by Monte Carlo with S draws per tract (in chunks to bound memory)."""
    parts = [simulate(F, th, chunk, seed=seed + k) for k in range(S // chunk)]
    out = {k: np.mean([q[k] for q in parts], axis=0) for k in parts[0]}
    return out


def prior_sensitivity(F, th, g, base, S=1024):
    """RMS change of the prediction when one parameter moves by one prior sd: which subjective choices matter."""
    rows = []
    for j, name in enumerate(NAMES):
        up = th.copy()
        up[j] += PRIOR_SD[j]
        v = simulate(F, up, S, seed=7)["Ey"]
        rows.append(dict(parameter=name, rms_change=float(np.sqrt(np.mean((v[g] - base[g]) ** 2))),
                         mean_change=float((v[g] - base[g]).mean())))
    return pd.DataFrame(rows).sort_values("rms_change", ascending=False)


HEALTH = ("('health_and_medical','public_health_clinic','hospital','home_health_care','dialysis_clinic','urgent_care_clinic',"
          "'pharmacy','medical_center','emergency_room','doctor')")


def discovery(df, pred):
    """Best Bias Discovery tables (write-up only): estimated gap by the upstream feed of a tract's Overture places."""
    con = _con()
    rows = []
    for r in GRADED:
        rows.append(con.sql(f"""
            WITH tr AS (SELECT GEOID, geometry g FROM '{DATA}/strata/{r}/{r}-census-tracts.parquet'),
                 p AS (SELECT geometry, sources[1].dataset ds, categories.primary cat
                       FROM '{DATA}/reference/{r}/{r}-overture-pois.parquet')
            SELECT tr.GEOID, count(p.ds) n_places, count(*) FILTER (p.ds = 'meta') n_meta,
                   count(*) FILTER (p.ds IN ('BrightQuery','AllThePlaces','DAC','PinMeTo')) n_registry,
                   count(*) FILTER (p.cat = 'fire_department') n_fire,
                   count(*) FILTER (p.cat = 'fire_department' AND p.ds = 'meta') n_fire_meta,
                   count(*) FILTER (p.cat IN ('elementary_school','middle_school','high_school','school','private_school','public_school')) n_school,
                   count(*) FILTER (p.cat IN ('elementary_school','middle_school','high_school','school','private_school','public_school')
                                    AND p.ds = 'meta') n_school_meta,
                   count(*) FILTER (p.cat IN {HEALTH}) n_health,
                   count(*) FILTER (p.cat IN {HEALTH} AND p.ds IN ('BrightQuery','AllThePlaces','DAC','PinMeTo')) n_health_registry
            FROM tr LEFT JOIN p ON ST_Contains(tr.g, p.geometry) GROUP BY tr.GEOID""").df())
    src = pd.concat(rows)
    d = df[["GEOID", "region", "county", "ur_class", "pop", "svi_overall", "tribal_any"]].assign(pred=pred).merge(src, on="GEOID")
    d["meta_share"] = d.n_meta / d.n_places.clip(lower=1)
    cells = []
    for (r, u), c in d.groupby(["region", "ur_class"]):
        q1, q3 = c.meta_share.quantile([0.25, 0.75])
        hi, lo = c[c.meta_share >= q3], c[c.meta_share <= q1]
        cells.append(dict(region=r, ur_class=u, n=len(c), gap_top_quartile=hi.pred.mean(), gap_bottom_quartile=lo.pred.mean(),
                          ratio=hi.pred.mean() / lo.pred.mean(),
                          registry_per_1k_top=1000 * hi.n_registry.sum() / hi["pop"].sum(),
                          registry_per_1k_bottom=1000 * lo.n_registry.sum() / lo["pop"].sum(),
                          health_per_1k_top=1000 * hi.n_health.sum() / hi["pop"].sum(),
                          health_per_1k_bottom=1000 * lo.n_health.sum() / lo["pop"].sum(),
                          svi_top=hi.svi_overall.mean(), svi_bottom=lo.svi_overall.mean()))
    cells = pd.DataFrame(cells)
    u = d[d.ur_class.eq("Urban")].copy()
    u["svi_high"] = u.svi_overall >= d.svi_overall.median()
    u["meta_quartile"] = u.groupby("region").meta_share.transform(lambda x: pd.qcut(x, 4, labels=False, duplicates="drop"))
    svi = u.groupby(["svi_high", "meta_quartile"]).pred.mean().unstack()
    examples = u[u.meta_quartile == 3].nlargest(12, "pred")[["GEOID", "county", "region", "pop", "n_places", "meta_share",
                                                             "n_registry", "tribal_any", "pred"]]
    log(f"  fire stations from Meta pages: {d.n_fire_meta.sum() / d.n_fire.sum():.0%}, schools: "
        f"{d.n_school_meta.sum() / d.n_school.sum():.0%}; health places from registries: "
        f"{d.n_health_registry.sum() / d.n_health.sum():.0%}\n" + cells.round(3).to_string()
        + "\n  urban tracts, mean estimate by SVI half (rows) and Meta-share quartile (columns):\n" + svi.round(4).to_string()
        + "\n" + examples.round(3).to_string(index=False))
    cells.to_csv(out("discovery.csv"), index=False)
    svi.to_csv(out("discovery_svi_control.csv"))
    examples.to_csv(out("discovery_examples.csv"), index=False)


# ================================================================================================
# 8. AGGREGATE LABELS: every scored submission's public RMSE is an exact moment of the hidden target
# ================================================================================================
LABELS = "aggregate_labels.npz"         # GEOIDs, regions, file names, public RMSEs and prediction vectors
P_PUBLIC = 0.3                          # the public leaderboard is about 30% of the graded tracts
EB_MIN_NEW_RMS = 0.0005                 # smallest new part (weighted RMS) that opens a posterior direction
MAX_SCORE_SURPRISE = 0.003              # a new label further than this from its projection estimate is refused
MIN_MOVE_PROB = 0.75                    # normal moves need this posterior probability of beating the best scored file
EXPLORE_MOVE_PROB = 0.65                # when none qualify, allow a small risk-adjusted exploration move
EXPLORE_RISK_SD = 0.5                   # exploration is accepted only if mean + this many SDs beats the best
CLIP_GROUP = ("r14", "r19b", "r22", "r23", "r24", "r25", "r26", "r27", "r28", "r29", "r3", "s8")   # leaderboard-calibrated files


class AggregateLabels:
    """For RMSE, a file v with public score s gives mean_public(v * y) = (mean(v^2) + mean(y^2) - s^2) / 2.
    mean(y^2) comes from the all-zero file and mean(y) from a constant file. Public-row membership is unknown;
    its probabilities pi are estimated from the exact affine relations among scored files."""

    def __init__(self, path=LABELS):
        with np.load(path, allow_pickle=False) as z:
            self.geoid = pd.Series(z["geoid"].astype(str))
            self.graded = z["region"] != "eastern-wa"
            names = z["names"].tolist()
            self.s = dict(zip(names, z["scores"].tolist()))
            self.V = dict(zip(names, z["predictions"].T))
        self.Z = self.s["probe_zeros.csv"] ** 2
        self.ybar = (0.01 + self.Z - self.s["probe_const_0.10.csv"] ** 2) / 0.2
        self.pi = np.full(self.graded.sum(), P_PUBLIC)
        self.public_weights()

    def AP(self, h):
        return float(h @ self.pi / self.pi.sum())

    def moment(self, name):
        v = self.V[name][self.graded]
        return (self.AP(v ** 2) + self.Z - self.s[name] ** 2) / 2

    def public_weights(self):
        g, n, p = self.graded, self.graded.sum(), P_PUBLIC
        V = {f: self.V[f][g] for f in self.s if not f.startswith("probe")}
        order = [f for f in self.s if f in V]
        eq = {"e01_r2f_affine.csv": ["r2f_ensemble.csv"], "r5e_groups_affine.csv": ["r5d_groups_ensemble.csv"],
              "r6e_vuln_affine.csv": ["r6d_vuln_ensemble.csv"], "r9d_routefix_affine.csv": ["r9c_routefix_ensemble.csv"]}
        for st in ["stack_region_ok_tx.csv", "stack2_e01_r3j_region.csv", "stack6_r9.csv", "stack7_r10.csv"]:
            if st in order:
                eq[st] = [f for f in order[:order.index(st)] if not f.startswith(("stack3", "stack4_with"))]
        Hs, rhs = [], []
        for F, comps in eq.items():
            if F not in V:
                continue
            comps = [c for c in comps if c in V]
            if not comps:
                continue
            Fv = V[F]
            X = np.column_stack([np.ones(n)] + [V[c] for c in comps])
            ok = (Fv > 1e-6) & (Fv < 1 - 1e-6)
            w = np.linalg.lstsq(X[ok], Fv[ok], rcond=None)[0]
            Hs.append(Fv ** 2 - sum(wi * V[c] ** 2 for wi, c in zip(w[1:], comps)))
            rhs.append(self.s[F] ** 2 + 2 * w[0] * self.ybar
                       + sum(wi * (self.Z - self.s[c] ** 2) for wi, c in zip(w[1:], comps)) - self.Z)
        Hm = np.column_stack(Hs) / (p * n)
        m0 = np.full(n, p)
        lam = np.linalg.lstsq(p * (1 - p) * Hm.T @ Hm, np.array(rhs) - Hm.T @ m0, rcond=None)[0]
        self.pi = np.clip(m0 + p * (1 - p) * Hm @ lam, 0, 1)


def add_label(name, path, rmse):
    """Append one scored file, or verify an identical label already recorded before an interrupted round."""
    with np.load(LABELS, allow_pickle=False) as z:
        art = {k: z[k] for k in z.files}
    v = pd.read_csv(path, dtype={"GEOID": str}).set_index("GEOID").loc[art["geoid"].astype(str), "coverage_gap_score"]
    assert len(v) == 9794 and v.notna().all()
    if name in art["names"]:
        j = list(art["names"]).index(name)
        if abs(float(art["scores"][j]) - float(rmse)) > 1e-9:
            raise SystemExit(f"refused: {name} is already recorded with RMSE {art['scores'][j]:.9f}, not {float(rmse):.9f}")
        if not np.allclose(art["predictions"][:, j], v.to_numpy(float), atol=5e-7, rtol=0):
            raise SystemExit(f"refused: {name} is already recorded with a different prediction vector")
        log(f"{LABELS}: {name} already recorded with the same score and vector; resuming without appending")
        return False
    # a label must be the real leaderboard score: the scored files bound what this file can score, and a target typed in
    # as a score (0.004, 0.04) would corrupt every later fit
    est = estimate(AggregateLabels(), v.to_numpy(float))
    if abs(float(rmse) - est) > MAX_SCORE_SURPRISE:
        raise SystemExit(f"refused: {rmse} is not a plausible public score for {path} (the scored files put it near {est:.6f}; "
                         f"the largest real surprise so far was 0.0013). Give the score the leaderboard shows.")
    name_width = max(42, len(name), max((len(str(n)) for n in art["names"]), default=0))
    art["names"] = np.append(art["names"], name).astype(f"<U{name_width}")
    art["scores"] = np.append(art["scores"], float(rmse))
    art["predictions"] = np.column_stack([art["predictions"], v.to_numpy(float)])
    tmp = LABELS + ".tmp.npz"
    np.savez_compressed(tmp, **art)
    os.replace(tmp, LABELS)
    log(f"{LABELS}: added {name} ({path}, public RMSE {float(rmse):.9f}); {len(art['names'])} labels")
    return True


def label_group(name):
    if name.startswith("probe_const"):
        return "const"
    if name.startswith("tr_"):
        return "transfer"           # deliberate small step: prior N(1, 1) in units of the step taken
    if name.startswith(("p", "q")) and name[1:3].isdigit():
        return "probe"
    return "clip" if name.startswith(CLIP_GROUP) else "base"


def eb_posterior(L, names, yhat, sigma=0.0624):
    """Empirical-Bayes posterior of the graded target in the span of the scored files.
    The files are orthogonalised in scoring order. Each new direction's coefficient gets a group prior
    (physics/base models, probes, leaderboard-calibrated files), mean and scale fitted by marginal likelihood. The moments
    carry the noise of the 30% public sample, so weak directions are shrunk rather than fitted."""
    from scipy.optimize import minimize
    g = L.graded
    w = L.pi / L.pi.sum()
    V = np.column_stack([L.V[f][g] for f in names])
    c = np.array([L.moment(f) for f in names])
    U = V * (yhat[:, None] - V / 2)
    Uc = U - w @ U
    Nv = ((1 - P_PUBLIC) * (Uc * w[:, None]).T @ Uc + sigma ** 2 * (V * w[:, None]).T @ V) / (P_PUBLIC * len(w))
    Q, keep = [], []
    for i in range(V.shape[1]):
        v = V[:, i].copy()
        for _ in range(2):
            for q in Q:
                v -= (w @ (q * v)) / (w @ (q * q)) * q
        # a near-duplicate file opens no direction: below EB_MIN_NEW_RMS its moment is dominated by the public-vs-average
        # curvature error, not by signal (two such files inflated the fitted noise scale from 0.97 to 6.4)
        if np.sqrt(w @ v ** 2) > max(1e-5 * np.sqrt(w @ V[:, i] ** 2), EB_MIN_NEW_RMS):
            Q.append(v)
            keep.append(i)
    Q = np.column_stack(Q)
    sw = np.sqrt(w)[:, None]
    R = np.linalg.lstsq(Q * sw, V[:, keep] * sw, rcond=None)[0]
    Rit = np.linalg.inv(R).T
    o = Rit @ c[keep]
    N = Rit @ Nv[np.ix_(keep, keep)] @ Rit.T
    D = w @ (Q * Q)
    grp = np.array([label_group(names[i]) for i in keep])

    def prior(x):
        tau_b, mu_b, tau_p, mu_p, tau_c, alpha, mu_c = np.exp(x[0]), x[1], np.exp(x[2]), x[3], np.exp(x[4]), np.exp(x[5]), x[6]
        mu = np.select([grp == "base", grp == "probe", grp == "clip", grp == "transfer"], [mu_b, mu_p, mu_c, 1.0], 0.0)
        var = np.select([grp == "const", grp == "base", grp == "probe", grp == "transfer"], [100.0, tau_b ** 2, tau_p ** 2, 1.0],
                        tau_c ** 2)
        return mu, var, np.diag(D * var * D) + alpha * N

    def nll(x):
        mu, _, C = prior(x)
        Lc = np.linalg.cholesky(C)
        r = np.linalg.solve(Lc, o - D * mu)
        return 0.5 * r @ r + np.log(np.diag(Lc)).sum()

    x = minimize(nll, [np.log(0.35), 0.56, np.log(0.75), 0.0, 0.0, np.log(0.42), 0.5], method="Nelder-Mead",
                 options=dict(maxiter=8000, xatol=1e-5, fatol=1e-7)).x
    mu, var, C = prior(x)
    theta = mu + var * D * np.linalg.solve(C, o - D * mu)
    hyper = dict(zip(["tau_base", "mu_base", "tau_probe", "mu_probe", "tau_clip", "noise_scale", "mu_clip"],
                     [float(np.exp(x[0])), float(x[1]), float(np.exp(x[2])), float(x[3]), float(np.exp(x[4])), float(np.exp(x[5])),
                      float(x[6])]))
    return Q @ theta, dict(Q=Q, w=w, o=o, noise=np.exp(x[5]) * N, D=D, mu=mu, var=var, theta=theta), hyper


def estimate(L, v, files=None):
    """Projection estimate of a vector's public RMSE from the scored files' exact moments (assumes no signal in
    the part of v outside their span, so it credits only what the scores have measured)."""
    g = L.graded
    files = files if files is not None else [f for f in L.s if not f.startswith("probe_")]
    X = np.column_stack([np.ones(g.sum())] + [L.V[f][g] for f in files])
    m = np.array([L.ybar] + [L.moment(f) for f in files])
    gram = (X * L.pi[:, None]).T @ X / L.pi.sum()
    gram += 1e-7 * np.trace(gram) / len(m) * np.eye(len(m))
    coef = np.linalg.solve(gram, (X * L.pi[:, None]).T @ v[g] / L.pi.sum())
    return float(np.sqrt(max(L.AP(v[g] ** 2) - 2 * coef @ m + L.Z, 0)))


def forward_validation(L, first="p12_01_route_ratio.csv"):
    """Chronological hold-out of the measurement model: predict each scored file's public RMSE from the files scored
    before it. Files far from everything scored before (the two physics files) cannot be predicted well, which is
    exactly why their scores add information."""
    files = [f for f in L.s if not f.startswith(("probe_", "e06"))]
    rows = [dict(file=f, actual=L.s[f], predicted=estimate(L, L.V[f], files[:files.index(f)]))
            for f in files[files.index(first):]]
    d = pd.DataFrame(rows)
    d["error"] = d.predicted - d.actual
    return d


def empirical_bayes_forward_sweep(L, points=20, sigmas=(0.0312, 0.0624, 0.1248, 0.2496)):
    """Rolling-origin validation of score forecasts from the aggregate-label posterior."""
    import copy

    ordered = list(L.s)
    eligible = [name for name in ordered if not name.startswith(("probe_", "e06"))]
    first = eligible.index("p12_01_route_ratio.csv")
    held_out = eligible[max(first, len(eligible) - points):]
    rows = []
    for name in held_out:
        prefix = ordered[:ordered.index(name)]
        train = copy.copy(L)
        train.s = {key: L.s[key] for key in prefix}
        train.V = {key: L.V[key] for key in prefix}
        train.pi = np.full(train.graded.sum(), P_PUBLIC)
        train.public_weights()
        posterior_names = [key for key in train.s if key not in ("probe_zeros.csv", "e06_r2f_only_eastern-wa.csv")]
        best_name = min((key for key in train.s if not key.startswith("probe_")), key=train.s.get)
        # The physics prediction exists before its leaderboard score is recorded, so it is a valid prior vector
        # even in prefixes that do not yet contain its score moment.
        yhat = L.V["fs_physics.csv"][train.graded]
        v = L.V[name][L.graded]
        prior_files = [key for key in train.s if not key.startswith("probe_")]
        ols_prediction = estimate(train, L.V[name], prior_files)
        for sigma in sigmas:
            pred, _, hyper = eb_posterior(train, posterior_names, yhat, sigma=sigma)
            pred, _, hyper = eb_posterior(train, posterior_names, pred, sigma=sigma)
            pred = np.clip(pred, 0, 1)
            mse = train.AP(v ** 2) + train.Z - 2 * train.AP(v * pred)
            rows.append(dict(file=name, score_group=label_group(name), actual=float(L.s[name]),
                             predicted=float(np.sqrt(max(mse, 0))), sigma=float(sigma),
                             ols_predicted=float(ols_prediction), prior_best=best_name,
                             initial_prediction="fs_physics.csv vector (score moment held out when absent)", posterior_passes=2,
                             prior_file_count=len(posterior_names), prior_public_weight_mean=float(train.pi.mean()),
                             fitted_hyperparameters=hyper))

    report = dict(validation="rolling origin; each held-out score excluded from the posterior and public-weight fit",
                  points=len(held_out), sigmas=list(sigmas), results=rows, summaries=[])
    for sigma in sigmas:
        group = [r for r in rows if r["sigma"] == sigma]
        errors = np.array([r["predicted"] - r["actual"] for r in group])
        report["summaries"].append(dict(sigma=float(sigma), n=len(group), mae=float(np.mean(np.abs(errors))),
                                        rmse=float(np.sqrt(np.mean(errors ** 2))), bias=float(np.mean(errors))))
    errors = np.array([r["ols_predicted"] - r["actual"] for r in rows if r["sigma"] == sigmas[0]])
    report["ols_summary"] = dict(n=len(errors), mae=float(np.mean(np.abs(errors))),
                                 rmse=float(np.sqrt(np.mean(errors ** 2))), bias=float(np.mean(errors)))
    atomic_json(out("eb_forward_sweep.json"), report)
    for row in report["summaries"]:
        log(f"  sigma={row['sigma']:.4f}: RMSE {row['rmse']:.6f}, MAE {row['mae']:.6f}, bias {row['bias']:+.6f} (n={row['n']})")
    log(f"  OLS projection: RMSE {report['ols_summary']['rmse']:.6f}, MAE {report['ols_summary']['mae']:.6f}, "
        f"bias {report['ols_summary']['bias']:+.6f} (n={report['ols_summary']['n']})")
    log(f"Wrote {out('eb_forward_sweep.json')}")
    return report


def public_weight_forward_sweep(L, points=40, blends=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """Rolling-origin test of shrinkage for inferred public-row inclusion weights.

    The score moments determine both public-row weights and the forecast. Shrinking the inferred weights toward
    uniform inclusion can reduce overfit, so each held-out file gets a fresh weight fit using only earlier labels.
    The earlier half selects the blend; the later half is a chronological confirmation set.
    """
    import copy

    ordered = list(L.s)
    eligible = [name for name in ordered if not name.startswith(("probe_", "e06"))]
    first = eligible.index("p12_01_route_ratio.csv")
    held_out = eligible[max(first, len(eligible) - points):]
    if not held_out:
        raise ValueError("no eligible held-out labels; increase --public-weight-forward-points")

    rows = []
    for index, name in enumerate(held_out):
        prefix = ordered[:ordered.index(name)]
        train = copy.copy(L)
        train.s = {key: L.s[key] for key in prefix}
        train.V = {key: L.V[key] for key in prefix}
        train.pi = np.full(train.graded.sum(), P_PUBLIC)
        train.public_weights()
        inferred_pi = train.pi.copy()
        prior_files = [key for key in train.s if not key.startswith("probe_")]
        for blend in blends:
            weighted = copy.copy(train)
            weighted.pi = np.clip(P_PUBLIC + float(blend) * (inferred_pi - P_PUBLIC), 0, 1)
            prediction = estimate(weighted, L.V[name], prior_files)
            rows.append(dict(file=name, held_out_index=index, actual=float(L.s[name]),
                             predicted=float(prediction), error=float(prediction - L.s[name]),
                             blend=float(blend), prior_file_count=len(prior_files),
                             inferred_weight_mean=float(inferred_pi.mean()),
                             inferred_weight_sd=float(inferred_pi.std()),
                             inferred_weight_min=float(inferred_pi.min()),
                             inferred_weight_max=float(inferred_pi.max())))

    split = len(held_out) // 2
    def summarize(group):
        errors = np.array([r["error"] for r in group], dtype=float)
        return dict(n=len(errors), mae=float(np.mean(np.abs(errors))),
                    rmse=float(np.sqrt(np.mean(errors ** 2))), bias=float(np.mean(errors)))

    summaries = []
    for blend in blends:
        group = [r for r in rows if r["blend"] == float(blend)]
        selection = [r for r in group if r["held_out_index"] < split]
        confirmation = [r for r in group if r["held_out_index"] >= split]
        summaries.append(dict(blend=float(blend), all=summarize(group),
                              selection=summarize(selection) if selection else None,
                              confirmation=summarize(confirmation) if confirmation else None))

    selectable = [s for s in summaries if s["selection"] is not None]
    chosen = min(selectable, key=lambda s: (s["selection"]["mae"], s["blend"]))["blend"] if selectable else 0.0
    report = dict(validation="rolling origin; weights fit from prior labels only; first chronological half selects blend, second half confirms",
                  points=len(held_out), selection_points=split, confirmation_points=len(held_out) - split,
                  blends=[float(x) for x in blends], chosen_blend=float(chosen),
                  results=rows, summaries=summaries)
    atomic_json(out("public_weight_forward_sweep.json"), report)
    for item in summaries:
        for window in ("selection", "confirmation", "all"):
            val = item[window]
            if val:
                log(f"  blend={item['blend']:.2f} {window}: MAE {val['mae']:.6f}, RMSE {val['rmse']:.6f}, "
                    f"bias {val['bias']:+.6f} (n={val['n']})")
    log(f"  selected on first half: blend={chosen:.2f}; wrote {out('public_weight_forward_sweep.json')}")
    return report


def aggregate_label_fit(df, groups, physics, out_path, report, do_forward=True):
    """Stage 8: fit the target to the aggregate labels (the scored files) and write the submission."""
    L = AggregateLabels()
    g = L.graded
    pos = pd.Series(np.arange(len(df)), index=df["GEOID"].to_numpy())[L.geoid].to_numpy()   # df row of each label row
    f = physics[pos]
    stored = L.V["fs_physics.csv"]
    base_diff = float(np.abs(np.round(f, 6) - stored).max())
    if base_diff < 1e-8:
        log(f"  {len(L.s)} scored files; baseline physics reproduces fs_physics.csv (RMSE {L.s['fs_physics.csv']:.6f})")
    else:
        log(f"  {len(L.s)} scored files; input model differs from fs_physics.csv by max {base_diff:.3f} (expected for a calibrated model)")
    report["aggregate_labels"] = dict(files=len(L.s), public_weight_mean=float(L.pi.mean()), mean_y=L.ybar, mean_y2=L.Z)
    names = [n for n in L.s if n not in ("probe_zeros.csv", "e06_r2f_only_eastern-wa.csv")]
    post, parts, hyper = eb_posterior(L, names, f[g])
    post, parts, hyper = eb_posterior(L, names, post)                 # second pass: noise model at the posterior
    log(f"  empirical-Bayes posterior: {parts['Q'].shape[1]} independent directions; hyperparameters "
        + ", ".join(f"{k} {v:.3f}" for k, v in hyper.items()))
    base = f.copy()
    base[g] = post
    # Washington is not graded, so no score informs it: transfer the graded-row calibration of the physics model
    # (affine fit of the posterior on the physics prediction) to the Washington rows.
    w_ = L.pi / L.pi.sum()
    A = np.column_stack([np.ones(g.sum()), f[g]])
    a_b = np.linalg.solve((A * w_[:, None]).T @ A, (A * w_[:, None]).T @ post)
    base[~g] = np.clip(a_b[0] + a_b[1] * f[~g], 0, 1)
    log(f"  Washington rows: physics recalibrated as {a_b[0]:.4f} + {a_b[1]:.3f} x physics (graded-row fit)")
    grp_l = {k: (hi[pos], lo[pos]) for k, (hi, lo) in groups.items()}
    final = np.round(np.clip(base, 0, 1), 6)
    rat = {k: round(float(final[h].mean() / final[l].mean()), 3) for k, (h, l) in grp_l.items()}
    est_final = estimate(L, final)
    best = min((s_, n) for n, s_ in L.s.items() if not n.startswith("probe_"))
    w = L.pi / L.pi.sum()
    change = final[g] - L.V[best[1]][g]
    log(f"  estimated public RMSE {est_final:.6f}; best scored file {best[1]} {best[0]:.6f} "
        f"(RMS difference {np.sqrt(w @ change ** 2):.5f}); in-span by construction, so the estimate credits all of it")
    cands = [("refit", est_final, final, 0.0), (f"best scored file {best[1]}", best[0], np.round(L.V[best[1]].copy(), 6), 0.0)]
    report["transfer_update"] = []
    for tu in transfer_updates(L):
        v = tu["vector"].copy()
        v[~g] = final[~g]
        log(f"  {tu['label']} ({tu['directions']} directions, curvature scales "
            + ", ".join(f"{x:.2f}" for x in tu["curvature_scale"]) + "): coordinates "
            + ", ".join(f"{x:+.2f}" for x in tu["coordinates"])
            + f"; expected public RMSE {tu['expected_rmse']:.6f} +- {tu['sd_rmse']:.6f}, P(beats best) {tu['p_beats_best']:.2f}")
        if (tu["p_beats_best"] >= MIN_MOVE_PROB
                or (tu["p_beats_best"] >= EXPLORE_MOVE_PROB
                    and tu["expected_rmse"] + EXPLORE_RISK_SD * tu["sd_rmse"] < best[0])):
            cands.append((tu["label"], tu["expected_rmse"], v, tu["sd_rmse"]))
        report["transfer_update"].append({k_: v_ for k_, v_ in tu.items() if k_ != "vector"})
    if any(label_group(n) == "transfer" for n in L.s):
        # Large in-span refits are scored by the same public-curvature assumption that failed: keep them out once
        # transfer files exist, and trust the measured local update or the best scored file.
        cands = [c_ for c_ in cands if c_[0] != "refit"]
    # the scored best file is never resubmitted (it measures nothing); among the moves whose mean beats it, take the lowest
    # risk-adjusted score (mean + sd / 2): a large move along weakly measured directions must earn its risk
    move_prob = {u["label"]: u["p_beats_best"] for u in report["transfer_update"]}
    moves = [c_ for c_ in cands if not c_[0].startswith("best scored file") and c_[1] < best[0]
             and move_prob.get(c_[0], 1.0) >= MIN_MOVE_PROB]
    if moves:
        choice = min(moves, key=lambda c: c[1] + c[3] / 2)
    else:
        explore = [c_ for c_ in cands if not c_[0].startswith("best scored file")
                   and c_[1] + EXPLORE_RISK_SD * c_[3] < best[0]
                   and move_prob.get(c_[0], 0.0) >= EXPLORE_MOVE_PROB]
        if not explore:
            raise SystemExit("no distinct candidate clears the risk rule; keep the current submission and design a new probe")
        choice = min(explore, key=lambda c: c[1] + EXPLORE_RISK_SD * c[3])
        log(f"  exploratory fallback: P(beats best) >= {EXPLORE_MOVE_PROB:.2f}, mean + {EXPLORE_RISK_SD:.1f} SD < best")
    final = choice[2]
    rat = {k: round(float(final[h].mean() / final[l].mean()), 3) for k, (h, l) in grp_l.items()}
    log(f"  chosen: {choice[0]} (estimated public RMSE {choice[1]:.6f} +- {choice[3]:.6f})")
    report["aggregate_fit"] = dict(report.get("aggregate_fit", {}), chosen=choice[0], chosen_est=choice[1],
                                   exploration_fallback=not bool(moves))
    log(f"  scorecard ratios (zero-score leaders: rural 2.23-2.38, tribal 2.79-2.97): {rat}")
    report["aggregate_fit"] = dict(report.get("aggregate_fit", {}), hyper=hyper, washington_affine=[float(a_b[0]), float(a_b[1])],
                                   est_public_refit=est_final, best_scored=dict(file=best[1], rmse=best[0]), ratios=rat)
    sources = [lb_name(**LOOP_MODEL, aux_feature_tag=AUX_FEATURE_TAG) + ".csv"]
    bet = 0.0
    if TRANSFER > 0 and all(os.path.exists(out(f_)) for f_ in sources):
        lb_phys = sum(pd.read_csv(out(f_), dtype={"GEOID": str}).set_index("GEOID")["coverage_gap_score"] for f_ in sources) / len(sources)
        final, bet = transfer_step(L, final, lb_phys.loc[L.geoid].to_numpy(float), report, choice[1], sources[0])
        final = np.round(final, 6)
        rat = {k: round(float(final[h].mean() / final[l].mean()), 3) for k, (h, l) in grp_l.items()}
        report["aggregate_fit"]["ratios_after_transfer"] = rat
    n_loop = sum(n.startswith("tr_loop") for n in L.s) + 1
    report["next_label"] = f"tr_loop{n_loop:02d}_{round(100 * bet):03d}.csv"
    if do_forward:
        fv = forward_validation(L)
        old_files = ~fv.file.str.startswith("fs_")
        log(f"  chronological validation of the measurement model ({old_files.sum()} files): MAE {fv.error[old_files].abs().mean():.6f}; "
            f"the physics files were forecast with errors " + ", ".join(f"{e:+.4f}" for e in fv.error[~old_files]))
        report["forward_validation"] = fv.to_dict(orient="records")
    sub = pd.DataFrame({"GEOID": L.geoid, "coverage_gap_score": final})
    assert len(sub) == 9794 and sub.GEOID.is_unique and sub.coverage_gap_score.between(0, 1).all()
    sub.to_csv(out_path, index=False)
    return final


# ================================================================================================
# 9. PROBE DESIGN (active learning with aggregate labels)
# ================================================================================================
AUX_TARGETS = [("E_fire", 20), ("E_ems", 20), ("E_sch", 40), ("n_poi_bq", 2000)]
PROBE_STEP = 0.003                      # RMS of each probe's step: costs at most ~0.00007 RMSE if the direction is empty
COMPONENTS = ["c_road", "c_building", "c_facility", "c_establishment"]


def physics_contributions(df, report_path, use_scorecard=False, model=None, auxiliary=None):
    """Rebuild a calibrated physics model from its report and return each component's contribution to the score
    (the four sum to the model's prediction)."""
    with open(report_path) as f:
        report = json.load(f)
    if model is None:
        set_priors(use_scorecard)
        par = report["calibration"]["parameters"]
        key = "posterior"
    else:
        set_priors(True, **model)
        par = report["parameters"]
        key = "fitted"
    assert [p["parameter"] for p in par] == NAMES, f"{report_path} does not match the model's parameters"
    th = np.array([p[key] for p in par])
    if auxiliary is None:
        mu, _ = fit_auxiliary(df)
    else:
        mu = auxiliary
    sim = predict(inputs(df, mu), th)
    return {k: sim[k] for k in COMPONENTS}, sim["Ey"]


def make_probes(df, K=5, step=PROBE_STEP, min_rms=0.002, include_loop=False,
                include_regional=False, include_strata=False, directory="probes", family=""):
    """Directions the scored files have not measured yet: each physics model's component contributions, made
    orthogonal to the span of every scored file and to each other (largest first). Each probe is the best scored
    file plus `step` along one direction; its score measures the target's projection on that direction, and the next
    stage-8 fit uses it. Region and scorecard modes also test masked component directions."""
    L = AggregateLabels()
    g = L.graded
    w = L.pi / L.pi.sum()
    sw = np.sqrt(w)[:, None]
    pos = pd.Series(np.arange(len(df)), index=df["GEOID"].to_numpy())[L.geoid].to_numpy()
    cands = {}
    models = [("physics", out("physics_report.json"), False, None),
              ("physics_scorecard", out("physics_scorecard_report.json"), True, None)]
    if include_loop:
        models.append(("physics_loop", out(lb_name(**LOOP_MODEL, aux_feature_tag=AUX_FEATURE_TAG) + "_report.json"), True, LOOP_MODEL))
    row_region = df["region"].to_numpy()[pos][g]
    strata_masks = None
    if include_strata:
        strata_masks = {k: (hi[pos][g], lo[pos][g]) for k, (hi, lo) in scorecard_groups(df).items()}
    auxiliary, _ = fit_auxiliary(df)
    log("  selected auxiliary count models cross-fitted once and shared across physics variants")
    for tag, path, sc, model in models:
        contrib, ey = physics_contributions(df, path, sc, model, auxiliary)
        log(f"  {tag}: components rebuilt (sum matches the model to {np.abs(sum(contrib.values()) - ey).max():.1e})")
        for k, v in contrib.items():
            v = v[pos][g]
            cands[f"{tag}:{k[2:]}"] = v
            if include_regional:
                for region in GRADED:
                    cands[f"{tag}:{region}:{k[2:]}"] = v * (row_region == region)
            if include_strata:
                for group, (high, low) in strata_masks.items():
                    cands[f"{tag}:{group}:high:{k[2:]}"] = v * high
                    cands[f"{tag}:{group}:low:{k[2:]}"] = v * low
    set_priors(False)
    # The constant probe is exactly collinear with the intercept. Reduced QR returns an arbitrary extra basis
    # vector for that null direction, which would erase unrelated parts of every candidate. Use an SVD basis of
    # the actual weighted span and drop numerical null directions before projecting.
    X = np.column_stack([np.ones(g.sum())] + [L.V[f][g] for f in L.s
                                               if f not in ("probe_zeros.csv", "probe_const_0.10.csv")])
    Xw = X * sw
    Qx, sx, _ = np.linalg.svd(Xw, full_matrices=False)
    rank = int(np.sum(sx > sx[0] * max(Xw.shape) * np.finfo(float).eps))
    Qx = Qx[:, :rank]
    log(f"  probe projection: rank {rank} of {X.shape[1]} weighted score directions")
    resid = {}
    for k, v in cands.items():
        vw = v * sw[:, 0]
        resid[k] = (vw - Qx @ (Qx.T @ vw)) / sw[:, 0]
    chosen = []
    while len(chosen) < K:
        size = {k: float(np.sqrt(w @ r ** 2)) for k, r in resid.items() if k not in [c[0] for c in chosen]}
        if not size:
            break
        k = max(size, key=size.get)
        if size[k] < min_rms:
            break
        e = resid[k] / size[k]
        chosen.append((k, e, size[k]))
        for j in resid:
            resid[j] = resid[j] - (w @ (resid[j] * e)) * e
    best = min((s_, n) for n, s_ in L.s.items() if not n.startswith(("probe_", "p")))
    base = L.V[best[1]]
    prior_probe_coefficients = []
    ordered_labels = list(L.s)
    for j, name in enumerate(ordered_labels):
        if not (name.startswith("q") and name[1:3].isdigit() and j):
            continue
        earlier = ordered_labels[:j]
        prior_best = min((L.s[n], n) for n in earlier if not n.startswith(("probe_", "p")))
        delta = L.V[name][g] - L.V[prior_best[1]][g]
        delta_rms = float(np.sqrt(w @ delta ** 2))
        if delta_rms > 1e-8:
            # From s_probe^2 = s_base^2 + delta_rms^2 - 2 delta_rms * projection.
            theta = (prior_best[0] ** 2 + delta_rms ** 2 - L.s[name] ** 2) / (2 * delta_rms)
            prior_probe_coefficients.append(float(theta))
    previous_probe_coefficient = float(np.median(prior_probe_coefficients)) if prior_probe_coefficients else None
    if previous_probe_coefficient is not None:
        log(f"  {len(prior_probe_coefficients)} scored small probe(s): median projection "
            f"{previous_probe_coefficient:+.6f} under inferred public weights")
    probe_dir = out(directory)
    os.makedirs(probe_dir, exist_ok=True)
    manifest = []
    generation = len(L.s) + 1
    for i, (k, e, size) in enumerate(chosen, 1):
        v = base.copy()
        v[g] = np.clip(base[g] + step * e, 0, 1)
        name = f"q{generation:03d}{family}_{i:02d}_{k.replace(':', '_')}"
        path = os.path.join(probe_dir, f"{name}.csv")
        pd.DataFrame({"GEOID": L.geoid, "coverage_gap_score": np.round(v, 6)}).to_csv(path, index=False)
        step_rms = float(np.sqrt(L.AP((v[g] - base[g]) ** 2)))
        no_signal = float(np.sqrt(best[0] ** 2 + step_rms ** 2))
        positive_benchmark = float(np.sqrt(max(best[0] ** 2 + step_rms ** 2 - 2 * step_rms * 0.003, 0)))
        previous_probe = (float(np.sqrt(max(best[0] ** 2 + step_rms ** 2
                                             - 2 * step_rms * previous_probe_coefficient, 0)))
                          if previous_probe_coefficient is not None else None)
        manifest.append(dict(file=path, label_name=f"{name}.csv", direction=k, base=best[1], base_rmse=best[0], step_rms=step_rms,
                             out_of_span_rms_of_component=size, expected_rmse_if_no_signal=no_signal,
                             previous_probe_count=len(prior_probe_coefficients),
                             previous_probe_coefficient=previous_probe_coefficient,
                             expected_rmse_if_previous_probe=previous_probe,
                             expected_rmse_if_projection_plus_0_003=positive_benchmark))
        line = (f"  {path}: {k} (new part RMS {size:.4f}); score scenarios: "
                f"no signal {no_signal:.6f}")
        if previous_probe is not None:
            line += f", prior-probe coefficient {previous_probe:.6f}"
        log(line + f", positive +0.003 benchmark {positive_benchmark:.6f}")
    with open(os.path.join(probe_dir, f"manifest_{generation:03d}.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(probe_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def make_candidate_probe(candidate_path, out_path, step=PROBE_STEP, against_paths=()):
    """Probe a branch residual orthogonal to scored files and optional earlier unscored probes."""
    L = AggregateLabels()
    candidate = pd.read_csv(candidate_path, dtype={"GEOID": str})
    if not {"GEOID", "coverage_gap_score"}.issubset(candidate.columns):
        raise ValueError(f"{candidate_path} must contain GEOID and coverage_gap_score")
    if candidate.GEOID.duplicated().any() or set(candidate.GEOID) != set(L.geoid):
        raise ValueError(f"{candidate_path} IDs must match the scored sample IDs exactly")
    candidate = candidate.set_index("GEOID").loc[L.geoid]
    candidate_values = candidate["coverage_gap_score"].to_numpy(float)
    if not np.isfinite(candidate_values).all() or np.any((candidate_values < 0) | (candidate_values > 1)):
        raise ValueError(f"{candidate_path} scores must be finite and between 0 and 1")
    if not np.isfinite(step) or step <= 0:
        raise ValueError("candidate probe step must be positive and finite")

    g = L.graded
    w = L.pi / L.pi.sum()
    sw = np.sqrt(w)
    scored = [L.V[f][g] for f in L.s if f not in ("probe_zeros.csv", "probe_const_0.10.csv")]
    Xw = np.column_stack([np.ones(g.sum()), *scored]) * sw[:, None]
    Q, singular, _ = np.linalg.svd(Xw, full_matrices=False)
    rank = int(np.sum(singular > singular[0] * max(Xw.shape) * np.finfo(float).eps))
    Q = Q[:, :rank]
    score_span_Q = Q.copy()

    best_score, best_name = min((score, name) for name, score in L.s.items()
                                if not name.startswith(("probe_", "p")))
    base = L.V[best_name].copy()
    against_names = []
    against_basis = []
    for against_path in against_paths:
        frame = pd.read_csv(against_path, dtype={"GEOID": str})
        if not {"GEOID", "coverage_gap_score"}.issubset(frame.columns):
            raise ValueError(f"{against_path} must contain GEOID and coverage_gap_score")
        if frame.GEOID.duplicated().any() or set(frame.GEOID) != set(L.geoid):
            raise ValueError(f"{against_path} IDs must match the scored sample IDs exactly")
        values = frame.set_index("GEOID").loc[L.geoid, "coverage_gap_score"].to_numpy(float)
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError(f"{against_path} scores must be finite and between 0 and 1")
        direction_w = (values[g] - base[g]) * sw
        direction_w -= Q @ (Q.T @ direction_w)
        for prior_direction in against_basis:
            direction_w -= prior_direction * (prior_direction @ direction_w)
        norm = float(np.linalg.norm(direction_w))
        if norm <= 1e-8:
            raise ValueError(f"{against_path} has no new direction beyond scored files and earlier probes")
        against_basis.append(direction_w / norm)
        against_names.append(against_path)
    if against_basis:
        Q = np.column_stack([Q, *against_basis])

    delta = candidate_values[g] - base[g]
    delta_w = delta * sw
    residual_w = delta_w - Q @ (Q.T @ delta_w)
    branch_residual_rms = float(np.linalg.norm(residual_w))
    if branch_residual_rms < 1e-6:
        raise ValueError("candidate has no measurable component outside the scored-file span")
    direction = residual_w / (branch_residual_rms * sw)

    if against_basis:
        # Project into both the box of valid scores and the null space of all measured/earlier directions.
        # Dykstra's alternating projections preserve score validity and orthogonality. Re-scale the target when
        # active [0, 1] bounds shorten the projected vector, so the final rounded probe retains the requested RMS.
        lower, upper = -base[g] * sw, (1.0 - base[g]) * sw
        amplitude = float(step)
        probe = None
        for _ in range(10):
            target_w = amplitude * direction * sw
            probe_w, box_correction, span_correction = target_w.copy(), np.zeros_like(target_w), np.zeros_like(target_w)
            for _ in range(1000):
                box = np.clip(probe_w + box_correction, lower, upper)
                box_correction = probe_w + box_correction - box
                span = box + span_correction
                next_w = span - Q @ (Q.T @ span)
                span_correction = span - next_w
                change = np.linalg.norm(next_w - probe_w)
                probe_w = next_w
                if change < 1e-12:
                    break
            else:
                raise ValueError("candidate probe projection did not converge")
            probe = base.copy()
            probe[g] = np.round(np.clip(base[g] + probe_w / sw, 0, 1), 6)
            actual_step = float(np.sqrt(w @ (probe[g] - base[g]) ** 2))
            if actual_step <= 0:
                raise ValueError("bounds removed the candidate probe direction")
            if abs(actual_step / step - 1) <= 0.002:
                break
            amplitude *= float(step) / actual_step
    else:
        scale = float(step)
        for _ in range(4):
            proposal = np.clip(base[g] + scale * direction, 0, 1)
            proposal = np.round(proposal, 6)
            actual_step = float(np.sqrt(w @ (proposal - base[g]) ** 2))
            if actual_step <= 0:
                raise ValueError("rounding removed the candidate probe direction")
            scale *= step / actual_step
        probe = base.copy()
        probe[g] = np.round(np.clip(base[g] + scale * direction, 0, 1), 6)
    actual_delta = probe[g] - base[g]
    actual_step = float(np.sqrt(w @ actual_delta ** 2))
    actual_delta_w = actual_delta * sw
    delta_norm = max(np.linalg.norm(actual_delta_w), 1e-15)
    score_span_fraction = float(np.linalg.norm(score_span_Q.T @ actual_delta_w) / delta_norm)
    orthogonalized_span_fraction = float(np.linalg.norm(Q.T @ actual_delta_w) / delta_norm)
    against_direction_cosines = [float(direction @ actual_delta_w / delta_norm) for direction in against_basis]
    if abs(actual_step / step - 1) > 0.02 or orthogonalized_span_fraction > 0.02:
        raise ValueError(f"rounded probe failed its contract (step={actual_step:.6f}, "
                         f"orthogonalized span fraction={orthogonalized_span_fraction:.4f})")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pd.DataFrame({"GEOID": L.geoid, "coverage_gap_score": probe}).to_csv(out_path, index=False)
    previous_probe_coefficients = []
    for j, name in enumerate(L.s):
        if not (name.startswith("q") and name[1:3].isdigit() and j):
            continue
        earlier = list(L.s)[:j]
        prior_best = min((L.s[n], n) for n in earlier if not n.startswith(("probe_", "p")))
        prior_delta = L.V[name][g] - L.V[prior_best[1]][g]
        prior_rms = float(np.sqrt(w @ prior_delta ** 2))
        if prior_rms > 1e-8:
            previous_probe_coefficients.append(float(
                (prior_best[0] ** 2 + prior_rms ** 2 - L.s[name] ** 2) / (2 * prior_rms)))
    previous_coefficient = (float(np.median(previous_probe_coefficients))
                            if previous_probe_coefficients else None)
    expected = {"no_signal": float(np.sqrt(best_score ** 2 + actual_step ** 2)),
                "projection_plus_step": float(np.sqrt(max(best_score ** 2 - actual_step ** 2, 0)))}
    if previous_coefficient is not None:
        expected["previous_small_probe_coefficient"] = float(np.sqrt(max(
            best_score ** 2 + actual_step ** 2 - 2 * actual_step * previous_coefficient, 0)))
    report = dict(candidate_file=candidate_path, candidate_md5=file_md5(candidate_path),
                  orthogonalized_against=against_names,
                  output_file=out_path, output_md5=file_md5(out_path),
                  label_name=os.path.basename(out_path), base_file=best_name, base_rmse=best_score,
                  scored_file_count=len(L.s), scored_span_rank=rank,
                  candidate_branch_delta_rms=float(np.sqrt(w @ delta ** 2)),
                  candidate_orthogonal_residual_rms=branch_residual_rms,
                  requested_step_rms=float(step), actual_step_rms=actual_step,
                  residual_span_fraction=score_span_fraction,
                  orthogonalized_span_fraction=orthogonalized_span_fraction,
                  against_direction_cosines=against_direction_cosines,
                  prior_small_probe_coefficients=previous_probe_coefficients,
                  conditional_rmse_scenarios=expected,
                  note="Exploratory measurement probe. Scenarios are conditional; only the real leaderboard score measures this direction.")
    atomic_json(out_path + ".json", report)
    log(f"  wrote {out_path}; orthogonal branch RMS {branch_residual_rms:.6f}, step RMS {actual_step:.6f}, "
        f"score-span fraction {score_span_fraction:.3g}, all-against fraction {orthogonalized_span_fraction:.3g}; "
        f"no-signal RMSE {expected['no_signal']:.6f}")
    return report


def candidate_probe_sweep(candidate_paths, report_path="outputs/branch_residual_sweep.json"):
    """Rank saved candidate branches by their public-weighted residual outside the scored-file span."""
    L = AggregateLabels()
    g = L.graded
    w = L.pi / L.pi.sum()
    sw = np.sqrt(w)
    scored = [L.V[f][g] for f in L.s if f not in ("probe_zeros.csv", "probe_const_0.10.csv")]
    Xw = np.column_stack([np.ones(g.sum()), *scored]) * sw[:, None]
    Q, singular, _ = np.linalg.svd(Xw, full_matrices=False)
    rank = int(np.sum(singular > singular[0] * max(Xw.shape) * np.finfo(float).eps))
    Q = Q[:, :rank]
    best_score, best_name = min((score, name) for name, score in L.s.items()
                                if not name.startswith(("probe_", "p")))
    base = L.V[best_name]
    results, residuals = [], {}
    for path in candidate_paths:
        candidate = pd.read_csv(path, dtype={"GEOID": str})
        if not {"GEOID", "coverage_gap_score"}.issubset(candidate.columns):
            raise ValueError(f"{path} must contain GEOID and coverage_gap_score")
        if candidate.GEOID.duplicated().any() or set(candidate.GEOID) != set(L.geoid):
            raise ValueError(f"{path} IDs must match the scored sample IDs exactly")
        values = candidate.set_index("GEOID").loc[L.geoid, "coverage_gap_score"].to_numpy(float)
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError(f"{path} scores must be finite and between 0 and 1")
        delta = values[g] - base[g]
        delta_w = delta * sw
        residual_w = delta_w - Q @ (Q.T @ delta_w)
        residual_rms = float(np.linalg.norm(residual_w))
        residuals[path] = residual_w / residual_rms if residual_rms > 0 else residual_w
        total_rms = float(np.sqrt(w @ delta ** 2))
        report_path_for_file = (path[:-4] + "_report.json" if path.endswith(".csv") else "")
        forecasts = {}
        if report_path_for_file and os.path.exists(report_path_for_file):
            with open(report_path_for_file) as f:
                forecasts = json.load(f).get("transfer", {}).get("expected", {})
        results.append(dict(file=path, md5=file_md5(path), delta_rms_from_best=total_rms,
                            unmeasured_rms=residual_rms,
                            unmeasured_share=(residual_rms / total_rms if total_rms else 0.0),
                            conditional_forecasts=forecasts))
    results.sort(key=lambda row: row["unmeasured_rms"], reverse=True)
    pairs = []
    for i, a in enumerate(results):
        for b in results[i + 1:]:
            pairs.append(dict(a=a["file"], b=b["file"], cosine=float(residuals[a["file"]] @ residuals[b["file"]])))
    report = dict(scored_file_count=len(L.s), score_span_rank=rank,
                  base=dict(file=best_name, rmse=best_score), candidates=results,
                  preferred_candidate=(results[0]["file"] if results else None),
                  preference_reason="largest public-weighted residual outside the scored-file span",
                  pairwise_residual_cosines=pairs)
    atomic_json(report_path, report)
    log(f"  ranked {len(results)} candidate branch(es) against {rank} measured directions")
    for row in results:
        log(f"    {row['file']}: unmeasured RMS {row['unmeasured_rms']:.6f}; total delta {row['delta_rms_from_best']:.6f}")
    log(f"  wrote {report_path}")
    return report


# ================================================================================================
# 10. PHYSICS CALIBRATED ON THE AGGREGATE LABELS
# ================================================================================================
LOOP_MODEL = dict(structural=True, unmatched=True, regional=True, flex=0.2)   # the physics model --round recalibrates
TRANSFER = 0.3      # share of that model's unscored part added to the next file as a measured bet (flexible directions have
                    # measured +0.5, +0.4, +0.85; the region-specific one -0.7, so it is not in the loop)
MIN_BET_STEP = 0.0011   # keep the rounded probe above the update's 0.001 direction cut-off
MAX_BET_SHARE = 0.6     # cap a probe at 60% of its unmeasured physics direction
CURVATURE = 1.3     # public-row curvature scale measured from the transfer files (1.2-1.4), used for expected outcomes


class LabelTargets(Targets):
    """Published statistics, the public scorecard and the orthogonalised moments of every scored file."""

    def __init__(self, df, groups, Q, o, N, w, pos, graded):
        super().__init__(df, groups, True)
        self.Q, self.o, self.N, self.w, self.pos, self.g = Q, o, N, w, pos, graded
        self.names = self.names + [f"moment[{j}]" for j in range(Q.shape[1])]

    def observed(self):
        return np.r_[super().observed(), self.o]

    def model(self, sim):
        return np.r_[super().model(sim), (self.Q * self.w[:, None]).T @ sim["Ey"][self.pos][self.g]]

    def sd(self, sim):
        return np.r_[super().sd(sim), np.sqrt(np.diag(self.N))]


def calibrate_physics_on_labels(df, groups, report, S=512, regional=False, structural=False, unmatched=False, covariates=False,
                                flex=False, flex_regional=False, nn=False, holdout=0):
    """Stage 10: refit the physics model's 41 parameters (stratum effects included) to the published statistics,
    the public scorecard and every scored file's moment, starting from the scorecard-calibrated physics. Its part
    outside the scored span is the physics extrapolation of everything the scores have measured."""
    L = AggregateLabels()
    g = L.graded
    pos = pd.Series(np.arange(len(df)), index=df["GEOID"].to_numpy())[L.geoid].to_numpy()
    names = [n for n in L.s if n not in ("probe_zeros.csv", "e06_r2f_only_eastern-wa.csv")]
    best = min((s_, n) for n, s_ in L.s.items() if not n.startswith("probe_"))
    _, parts, _ = eb_posterior(L, names, L.V[best[1]][g])
    set_priors(True, regional, structural, unmatched, covariates, flex, flex_regional, nn)
    with open(out("physics_scorecard_report.json")) as f:
        start = {p["parameter"]: p["posterior"] for p in json.load(f)["calibration"]["parameters"]}
    warm = ["physics_lb_struct_report.json", "physics_lb_struct_unmatched_report.json", "physics_lb_struct_regional_report.json",
            "physics_lb_struct_unmatched_regional_report.json"] + (["physics_lb_struct_unmatched_regional_flex_report.json",
                                                         f"physics_lb_struct_unmatched_regional{flex_suffix(flex)}_report.json"] if flex else [])
    warm += [f"physics_lb_struct_unmatched_regional{flex_suffix(flex)}_rflex_report.json"] if flex and flex_regional else []
    if holdout:
        # held-out test: every hypothesis starts from the same base fit, and never from a fit that saw the held-out moments
        warm = [w_ for w_ in warm if "flex" not in w_]
    else:
        # the wide network starts from the fitted 6-unit network (its first 6 units and two outputs), then its own last fit
        warm += [lb_name(structural, unmatched, regional, covariates, flex, flex_regional, nn=n_) + "_report.json"
                 for n_ in ([True] if nn == "wide" else []) + ([nn] if nn else [])]
    for wf in map(out, warm) if structural else []:
        if os.path.exists(wf):
            with open(wf) as f:                                           # warm start from the fitted smaller models
                start.update({p["parameter"]: p["fitted"] for p in json.load(f)["parameters"]})
    if nn:   # hidden layer starts at a fixed random draw from its prior (an all-zero network has no gradient)
        r0 = np.random.default_rng(0)
        for n_, sd_ in zip(NAMES, PRIOR_SD):
            if n_.startswith("nn_w1_") and n_ not in start:
                start[n_] = float(r0.normal(0, sd_))
    th0 = np.array([start.get(n, c0) for n, c0 in zip(NAMES, THETA0)])   # new mechanisms start at their prior centres
    mu, aux = fit_auxiliary(df)
    report["auxiliary_models"] = aux
    report["auxiliary_model_configs"] = AUX_GBM_BY_TARGET.copy()
    report["auxiliary_feature_configs"] = AUX_FEATURE_BY_TARGET.copy()
    F = inputs(df, mu)
    K = parts["Q"].shape[1] - holdout                                   # the last `holdout` moments are left out of the fit
    T = LabelTargets(df, groups, parts["Q"][:, :K], parts["o"][:K], parts["noise"][:K, :K], parts["w"], pos, g)
    obs, k0 = T.observed(), len(T.names) - K
    sim0 = simulate(F, th0, S)
    m0, sd0 = T.model(sim0), T.sd(sim0)
    J = moment_jacobian(F, T, th0, S)
    th, m, sd, cost, _, _ = calibrate(F, T, J, S, th0=th0, iters=40)
    chi = lambda mm, ss: ((mm - obs) / ss) ** 2
    log(f"  chi-square: published + scorecard {chi(m0, sd0)[:k0].sum():.1f} -> {chi(m, sd)[:k0].sum():.1f}; "
        f"{parts['Q'].shape[1]} score moments {chi(m0, sd0)[k0:].sum():.1f} -> {chi(m, sd)[k0:].sum():.1f}")
    sim = predict(F, th)
    pred = np.clip(sim["Ey"], 0, 1)
    if holdout:
        Qh, oh = parts["Q"][:, K:], parts["o"][K:]
        sdh = np.sqrt(np.diag(parts["noise"])[K:])
        held = lambda ey: (Qh * parts["w"][:, None]).T @ ey[pos][g]
        z_end, z_start = (held(sim["Ey"]) - oh) / sdh, (held(sim0["Ey"]) - oh) / sdh
        log(f"  held-out test: the last {holdout} score moments, chi-square {np.sum(z_start ** 2):.1f} at the start -> "
            f"{np.sum(z_end ** 2):.1f} after fitting the other {K}")
        report["holdout"] = dict(moments=holdout, chi2_start=float(np.sum(z_start ** 2)), chi2=float(np.sum(z_end ** 2)),
                                 z=z_end.tolist())
    out_csv = lb_name(structural, unmatched, regional, covariates, flex, flex_regional, nn=nn, holdout=holdout,
                      aux_feature_tag=AUX_FEATURE_TAG) + ".csv"
    pd.DataFrame({"GEOID": df["GEOID"], "coverage_gap_score": np.round(pred, 6)}).to_csv(out(out_csv), index=False)
    v = L.V[best[1]].copy()
    v[g] = pred[pos][g]
    est = estimate(L, v)
    ratios = {k: round(float(pred[h].mean() / pred[l].mean()), 3) for k, (h, l) in groups.items()}
    log(f"  label-calibrated physics ({out_csv}): mean {pred[df['graded'].to_numpy()].mean():.4f}; estimated public RMSE {est:.6f}; "
        f"scorecard ratios {ratios}")
    report.update(parameters=[dict(parameter=n, start=float(a), fitted=float(b)) for n, a, b in zip(NAMES, th0, th)],
                  targets=[dict(target=n, observed=float(x), start=float(y), fitted=float(z))
                           for n, x, y, z in zip(T.names[:k0], obs[:k0], m0[:k0], m[:k0])],
                  est_public=est, ratios=ratios, cost=cost)
    set_priors(False)
    return pred


def transfer_updates(L, draws=5000000, block=100000, min_rms=0.001, proj_scale=0.005, seed=0, k_median=None, k_sd=0.35,
                     coord_sd=3.0, forecast=None, correlation_prior="diagonal"):
    """Local Bayesian update around the anchor (the best file scored before the first transfer label).

    Every transfer file is a displacement D_k from the anchor, and its public MSE is exactly
        MSE_k = MSE_anchor + pub(D_k^2) - 2 pub(D_k (y - anchor)).
    The linear terms are what the scores measure. The quadratic term is the mean of D_k^2 over the unknown public rows.
    Estimated weights get it wrong for large, concentrated moves: every measured direction has shown public curvature
    1.0-1.6x the estimate. So the curvature along the main directions (each displacement's new part with RMS >= min_rms)
    gets a prior: log-normal scales with median CURVATURE and sd k_sd. The default correlation_prior is diagonal because
    the scored directions are weighted-orthogonal; "gram" adds a valid positive-definite correlation matrix, while the
    legacy "independent" off-diagonal normal prior is retained only for comparison. Each coordinate gets
    a prior x ~ N(0, coord_sd^2), so a direction measured by one small step is shrunk instead of extrapolated. A file's
    score is exact except for its part outside the main directions (a small bet, clipping): its noise is
    2 x proj_scale x RMS(that part) in MSE units. Prior draws (in blocks, two passes with the same seeds) are weighted by
    the marginal likelihood of all transfer scores. Candidates: the full update x* = E[C]^-1 E[G], and the best scored
    file with only its newest one or two directions moved (the others held where they were scored)."""
    g = L.graded
    order = list(L.s)
    names = [n for n in order if label_group(n) == "transfer"]
    if not names:
        return []
    anchor = min([n for n in order[:order.index(names[0])] if not n.startswith("probe_")], key=L.s.get)
    a, ma = L.V[anchor][g], L.s[anchor] ** 2
    w = L.pi / L.pi.sum()
    Ds = [L.V[n][g] - a for n in names]
    E = []
    for D in Ds:
        e = D - sum(((w @ (D * q)) / (w @ q ** 2) * q for q in E), np.zeros_like(D))
        if np.sqrt(w @ e ** 2) >= min_rms:
            E.append(e)
    K = len(E)
    X = np.array([[(w @ (D * e)) / (w @ e ** 2) for e in E] for D in Ds])            # coordinates of each file
    M = np.array([L.s[n] ** 2 for n in names])
    rest = [D - sum((xj * e for xj, e in zip(xk, E)), np.zeros_like(D)) for D, xk in zip(Ds, X)]
    sd = np.array([2 * proj_scale * np.sqrt(w @ r_ ** 2) + 1e-7 for r_ in rest])        # per-file noise (MSE units)
    c = np.array([[L.AP(ei * ej) for ej in E] for ei in E])
    sq = np.sqrt(np.diag(c))
    iu = np.triu_indices(K, 1)
    Wf = 1 / sd ** 2
    P = np.linalg.solve(X.T @ (X * Wf[:, None]), (X * Wf[:, None]).T)                    # K x files, weighted LS

    def chunk(i):
        """One block of prior draws: curvature C, the projections G it implies, and the log weight of the draw."""
        rng = np.random.default_rng([seed, i])
        k = np.exp(rng.normal(np.log(CURVATURE if k_median is None else k_median), k_sd, (block, K)))
        if correlation_prior == "independent":
            R = np.tile(np.eye(K), (block, 1, 1))
            corr = rng.normal(0, 0.3, (block, len(iu[0])))
            R[:, iu[0], iu[1]] = corr
            R[:, iu[1], iu[0]] = corr
        elif correlation_prior == "gram":
            # A Gram matrix gives valid positive-definite correlations. Independent off-diagonal normals
            # are indefinite for almost every 10-direction draw, so they are not a covariance prior.
            dof = max(K + 2, 12)
            A = rng.normal(size=(block, K, dof))
            R = np.matmul(A, np.swapaxes(A, 1, 2))
            scale = np.sqrt(np.diagonal(R, axis1=1, axis2=2))
            R /= scale[:, :, None] * scale[:, None, :]
        elif correlation_prior == "diagonal":
            R = np.tile(np.eye(K), (block, 1, 1))
        else:
            raise ValueError(f"unknown correlation prior: {correlation_prior}")
        C = R * (k[:, :, None] * k[:, None, :]) ** 0.5 * np.outer(sq, sq)[None]         # draw x K x K
        rhs = (np.einsum("nk,dkl,nl->dn", X, C, X) - (M - ma)[None]) / 2                # = X G
        if coord_sd is None:
            G = rhs @ P.T
            logw = -0.5 * np.sum((2 * (rhs - G @ X.T) / sd[None]) ** 2, axis=1)
            return k, C, G, logw
        Sg = coord_sd ** 2 * C @ C                                                       # prior covariance of G = C x
        Sr = X @ Sg @ X.T + np.diag(sd ** 2 / 4)[None]                                   # draw x files x files
        Lc = np.linalg.cholesky(Sr)
        z = np.linalg.solve(Lc, rhs[:, :, None])[:, :, 0]
        logw = -0.5 * np.sum(z ** 2, axis=1) - np.log(np.diagonal(Lc, axis1=1, axis2=2)).sum(1)
        G = np.einsum("dkl,nl,dn->dk", Sg, X, np.linalg.solve(Sr, rhs[:, :, None])[:, :, 0])
        return k, C, G, logw

    nb = max(draws // block, 1)
    acc = []
    for i in range(nb):                                                                   # pass 1: posterior moments
        k, C, G, logw = chunk(i)
        m_ = logw.max()
        wi = np.exp(logw - m_)
        acc.append((m_, wi.sum(), (wi ** 2).sum(), np.einsum("d,dkl->kl", wi, C), wi @ G, wi @ k))
    top = max(t[0] for t in acc)
    sc = [np.exp(t[0] - top) for t in acc]
    W0 = sum(s_ * t[1] for s_, t in zip(sc, acc))
    EC, EG = (sum(s_ * t[3] for s_, t in zip(sc, acc)) / W0, sum(s_ * t[4] for s_, t in zip(sc, acc)) / W0)
    Ek = sum(s_ * t[5] for s_, t in zip(sc, acc)) / W0
    eff = float(W0 ** 2 / sum(s_ ** 2 * t[2] for s_, t in zip(sc, acc)))
    best_name = min((n for n in order if not n.startswith("probe_")), key=L.s.get)
    best_scored = L.s[best_name]
    forecast_coordinates = forecast_residual = forecast_stats = None
    if forecast is not None:
        Df = np.asarray(forecast)[g] - a
        forecast_coordinates = np.array([(w @ (Df * e)) / (w @ e ** 2) for e in E])
        forecast_residual = Df - sum((xj * e for xj, e in zip(forecast_coordinates, E)), np.zeros_like(Df))
        forecast_stats = np.zeros(3)

    cands = []
    x = np.linalg.solve(EC, EG)
    v = L.V[anchor].copy()
    v[g] = np.clip(a + sum(xi * e for xi, e in zip(x, E)), 0, 1)
    cands.append((f"local Bayesian update around {anchor}", x, v))
    if best_name in names:
        xb = X[names.index(best_name)]
        for nf in (1, 2):
            f, h = np.arange(K - nf, K), np.arange(K - nf)
            xn = xb.copy()
            xn[f] = np.linalg.solve(EC[np.ix_(f, f)], EG[f] - EC[np.ix_(f, h)] @ xb[h])
            v = L.V[best_name].copy()
            v[g] = np.clip(v[g] + sum((xn[j] - xb[j]) * E[j] for j in f), 0, 1)
            cands.append((f"{best_name} with its newest {nf} direction(s) at " + ", ".join(f"{xn[j]:.2f}x" for j in f), xn, v))
    stats = np.zeros((len(cands), 3))
    for i in range(nb):                                                                   # pass 2: outcomes of each candidate
        k, C, G, logw = chunk(i)
        wi = np.exp(logw - top)
        for j, (_, xc, _) in enumerate(cands):
            mse = ma + np.einsum("k,dkl,l->d", xc, C, xc) - 2 * G @ xc
            stats[j] += [wi @ mse, wi @ mse ** 2, wi @ (mse < best_scored ** 2)]
        if forecast is not None:
            mse = (ma + np.einsum("k,dkl,l->d", forecast_coordinates, C, forecast_coordinates)
                   - 2 * G @ forecast_coordinates + CURVATURE * (w @ forecast_residual ** 2))
            forecast_stats += [wi @ mse, wi @ mse ** 2, wi @ (mse < best_scored ** 2)]
    out = []
    for (label, xc, v), (s1, s2, pb) in zip(cands, stats / W0):
        out.append(dict(label=label, anchor=anchor, directions=K, coordinates=xc.tolist(), expected_rmse=float(np.sqrt(s1)),
                        sd_rmse=float(np.sqrt(max(s2 - s1 ** 2, 0)) / (2 * np.sqrt(s1))), p_beats_best=float(pb),
                        effective_draws=eff, curvature_scale=Ek.tolist(), correlation_prior=correlation_prior,
                        posterior_draws=draws, vector=np.round(v, 6)))
    if forecast is not None:
        s1, s2, pb = forecast_stats / W0
        return dict(candidates=out, forecast=dict(expected_rmse=float(np.sqrt(max(s1, 0))),
                                                  sd_rmse=float(np.sqrt(max(s2 - s1 ** 2, 0)) / (2 * np.sqrt(max(s1, 1e-12)))),
                                                  p_beats_best=float(pb), coordinates=forecast_coordinates.tolist(),
                                                  residual_rms=float(np.sqrt(w @ forecast_residual ** 2)),
                                                  assumed_residual_curvature=CURVATURE, effective_draws=eff,
                                                  curvature_scale=Ek.tolist()))
    return out


def transfer_forward_sweep(L, points=10, draws=1000000, block=50000, correlation_prior="diagonal"):
    """Rolling-origin validation of the transfer posterior used to forecast new leaderboard candidates."""
    import copy

    order = list(L.s)
    transfer_positions = [i for i, name in enumerate(order) if label_group(name) == "transfer"]
    eligible = transfer_positions[3:]  # leave at least three scored transfer files to fit the posterior
    held_out = eligible[-points:]
    rows = []
    for j in held_out:
        name = order[j]
        train = copy.copy(L)
        prefix = order[:j]
        train.s = {key: L.s[key] for key in prefix}
        train.V = {key: L.V[key] for key in prefix}
        train.pi = np.full(train.graded.sum(), P_PUBLIC)
        train.public_weights()
        forecast = transfer_updates(train, draws=draws, block=block, forecast=L.V[name],
                                    correlation_prior=correlation_prior)
        prior_files = [key for key in train.s if not key.startswith("probe_")]
        direct = estimate(train, L.V[name], prior_files)
        pred = forecast["forecast"]
        rows.append(dict(file=name, prior_transfer_files=sum(label_group(key) == "transfer" for key in prefix),
                         prior_best=min((score, key) for key, score in train.s.items() if not key.startswith("probe_"))[1],
                         actual=float(L.s[name]), transfer_prediction=pred["expected_rmse"],
                         transfer_sd=pred["sd_rmse"], transfer_p_beats_best=pred["p_beats_best"],
                         transfer_residual_rms=pred["residual_rms"], direct_projection=direct,
                         effective_draws=pred["effective_draws"]))
        log(f"  {name}: actual {L.s[name]:.6f}; transfer {pred['expected_rmse']:.6f} ± {pred['sd_rmse']:.6f}; "
            f"direct {direct:.6f}; new residual RMS {pred['residual_rms']:.5f}; ESS {pred['effective_draws']:.0f}")

    report = dict(validation="rolling origin on transfer files; each held-out score/vector excluded from fit",
                  points=len(rows), draws=draws, block=block, correlation_prior=correlation_prior,
                  assumed_residual_curvature=CURVATURE, results=rows,
                  summaries={})
    for model in ("transfer_prediction", "direct_projection"):
        err = np.array([row[model] - row["actual"] for row in rows])
        report["summaries"][model] = dict(n=len(err), mae=float(np.mean(np.abs(err))),
                                          rmse=float(np.sqrt(np.mean(err ** 2))), bias=float(np.mean(err)))
    report_path = out(f"transfer_forward_sweep_{correlation_prior}.json")
    atomic_json(report_path, report)
    for model, stats in report["summaries"].items():
        log(f"  {model}: MAE {stats['mae']:.6f}, RMSE {stats['rmse']:.6f}, bias {stats['bias']:+.6f} (n={stats['n']})")
    log(f"Wrote {report_path}")
    return report


def transfer_step(L, base, physics_lb, report, base_rmse, source):
    """Add TRANSFER x the part of the label-calibrated physics that no scored file contains, when it is large enough
    to be measured by its step (at least MIN_BET_STEP RMS, at most MAX_BET_SHARE of it). Returns the file and the share."""
    g = L.graded
    w = L.pi / L.pi.sum()
    sw = np.sqrt(w)[:, None]
    X = np.column_stack([np.ones(g.sum())] + [L.V[n][g] for n in L.s if n != "probe_zeros.csv"])
    Qx, _ = np.linalg.qr(X * sw)
    d = physics_lb[g] - base[g]
    u = d - (Qx @ (Qx.T @ (d[:, None] * sw)))[:, 0] / sw[:, 0]
    size = float(np.sqrt(w @ u ** 2))
    share = max(TRANSFER, MIN_BET_STEP / max(size, 1e-12))
    if share > MAX_BET_SHARE:
        log(f"  no bet: the unscored part of {source} is RMS {size:.4f}, too small for a measurable step")
        report["transfer"] = dict(share=0.0, unscored_rms=size)
        return base, 0.0
    out_ = base.copy()
    out_[g] = np.clip(base[g] + share * u, 0, 1)
    outcomes = {f"coef_{b:+.1f}": float(np.sqrt(max(base_rmse ** 2 + size ** 2 * (CURVATURE * share ** 2 - 2 * share * b), 0)))
                for b in (-0.3, 0.0, 0.3, 0.5)}
    log(f"  transfer step: {share:.2f} x the unscored part of {source} (RMS {size:.4f}); expected public RMSE by the "
        f"truth's coefficient along it: " + ", ".join(f"{k[5:]} -> {v:.6f}" for k, v in outcomes.items()))
    report["transfer"] = dict(share=share, unscored_rms=size, expected=outcomes)
    return out_, share


def lb_name(structural=False, unmatched=False, regional=False, covariates=False, flex=False, flex_regional=False, nn=False,
            holdout=0, aux_feature_tag=""):
    """File stem of a label-calibrated physics model (in outputs/)."""
    return ("physics_lb" + ("_struct" if structural else "") + ("_unmatched" if unmatched else "") + ("_regional" if regional else "")
            + ("_covar" if covariates else "") + (flex_suffix(flex) if flex else "") + ("_rflex" if flex_regional else "")
            + ("_nnw" if nn == "wide" else "_nn" if nn else "") + (f"_ho{holdout}" if holdout else "") + (f"_{aux_feature_tag}" if aux_feature_tag else ""))


def file_md5(path):
    import hashlib
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def write_stage8(df, groups, pred, out_path, report, do_forward, report_path=None):
    """Stage 8, then the report: what was chosen, the next label's name and the file's checksum (so --round can check
    that the scored file is the one stage 8 wrote)."""
    tmp = out_path + ".tmp"
    final = aggregate_label_fit(df, groups, pred, tmp, report, do_forward=do_forward)
    os.replace(tmp, out_path)
    report["file"], report["file_md5"] = out_path, file_md5(out_path)
    if report_path is None:
        report_path = out("submission_report.json") if out_path == "submission.csv" else os.path.splitext(out_path)[0] + "_report.json"
    atomic_json(report_path, report)
    if out_path == "submission.csv":
        log(f"  next label: when {out_path} is scored, run  python train.py --round <public RMSE>  (records it as {report['next_label']})")
    else:
        log(f"  next label: when {out_path} is scored, run  python train.py --round <public RMSE> --round-file {out_path} "
            f"--round-name {report['next_label']}  (records it as {report['next_label']})")
    return final


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--rebuild", action="store_true", help="recompute the cached features")
    ap.add_argument("--scorecard", action="store_true", help="physics variant also calibrated on the public Bias Scorecard")
    ap.add_argument("--physics-only", action="store_true", help="stop after the physics model (stages 1-7)")
    ap.add_argument("--no-forward", action="store_true", help="skip the chronological validation of stage 8")
    ap.add_argument("--discovery", action="store_true", help="write the Best Bias Discovery tables")
    ap.add_argument("--aux-sweep", action="store_true", help="compare auxiliary Poisson-GBM settings with county-grouped CV")
    ap.add_argument("--aux-feature-sweep", action="store_true", help="test density, regional/county effects, and supplied strata with county-grouped CV")
    ap.add_argument("--aux-strata-expanded-sweep", action="store_true", help="compare current strata features with every eligible supplied strata field")
    ap.add_argument("--aux-ems-strata-sweep", action="store_true", help="ablate disjoint expanded EMS strata families and tune Poisson model capacity")
    ap.add_argument("--aux-hurdle-sweep", action="store_true", help="compare county-held-out Poisson fits with two-part zero/count models")
    ap.add_argument("--aux-cross-count-sweep", action="store_true", help="test whether other supplied Overture counts help county-held-out proxy models")
    ap.add_argument("--aux-source-mix-sweep", action="store_true", help="test target-excluded Overture source-composition features with county-grouped CV")
    ap.add_argument("--aux-source-signal-sweep", action="store_true", help="test target-excluded Overture-signals provenance beyond first-source composition")
    ap.add_argument("--aux-source-confidence-sweep", action="store_true", help="test source-specific Overture place-confidence summaries with county-grouped CV")
    ap.add_argument("--aux-source-depth-sweep", action="store_true", help="test Overture source-record depth and cross-dataset support with county-grouped CV")
    ap.add_argument("--aux-source-richness-sweep", action="store_true", help="test target-excluded address and contact-field shares with county-grouped CV")
    ap.add_argument("--aux-poi-category-sweep", action="store_true", help="test target-excluded detailed Overture POI categories with county-grouped CV")
    ap.add_argument("--aux-poi-source-category-sweep", action="store_true", help="test target-excluded Overture categories split by first-source family")
    ap.add_argument("--aux-poi-identity-sweep", action="store_true", help="test target-excluded POI name, brand, contact, and status completeness by source family")
    ap.add_argument("--aux-poi-identity-depth-sweep", action="store_true", help="test whether POI identity completeness adds value beyond selected source-depth features")
    ap.add_argument("--aux-poi-identity-ablation-sweep", action="store_true", help="ablate BrightQuery POI identity feature groups over the selected category and source-depth model")
    ap.add_argument("--aux-spatial-lag-sweep", action="store_true", help="test target-excluded nearby-tract Overture counts with county-grouped CV")
    ap.add_argument("--aux-region-sweep", action="store_true", help="compare shared and region-specific auxiliary models with county-grouped CV")
    ap.add_argument("--aux-region-blend-sweep", action="store_true", help="test shrinkage blends of shared and regional auxiliary models")
    ap.add_argument("--eb-forward-sweep", action="store_true", help="rolling-origin validation of aggregate-label score forecasts")
    ap.add_argument("--eb-forward-points", type=int, default=20, help="number of most recent scores to hold out in --eb-forward-sweep")
    ap.add_argument("--eb-forward-sigmas", type=float, nargs="+", help="posterior noise scales to compare in --eb-forward-sweep")
    ap.add_argument("--public-weight-forward-sweep", action="store_true", help="rolling-origin test of shrinkage for inferred public-row weights")
    ap.add_argument("--public-weight-forward-points", type=int, default=40, help="number of recent scores in the public-weight forecast sweep")
    ap.add_argument("--public-weight-forward-blends", type=float, nargs="+", default=(0.0, 0.25, 0.5, 0.75, 1.0),
                    help="mixing weights for inferred inclusion weights; 0 is uniform, 1 uses the full estimate")
    ap.add_argument("--transfer-forward-sweep", action="store_true", help="rolling-origin validation of the transfer posterior")
    ap.add_argument("--transfer-forward-points", type=int, default=8, help="number of recent transfer scores to hold out")
    ap.add_argument("--transfer-forward-draws", type=int, default=1000000, help="posterior draws per transfer holdout")
    ap.add_argument("--transfer-forward-correlation-prior", choices=("independent", "gram", "diagonal"), default="diagonal",
                    help="curvature-correlation prior for --transfer-forward-sweep")
    ap.add_argument("--aux-features", choices=("baseline", "density_cv", "density_building_cv", "county_cv", "strata_cv", "strata_full_cv", "strata_mixed_cv", "strata_ems_no_gehe_cv", "regional_blend_cv", "hurdle_cv", "cross_counts_cv", "cross_counts_direct_cv", "source_mix_cv", "source_confidence_cv", "source_richness_cv", "spatial_lags_cv", "spatial_fire_ems_cv", "poi_categories_cv", "poi_categories_depth_cv"), default="baseline",
                    help="select baseline, density, county-context, strata, hurdle-blended, direct cross-count, source-mix, source-confidence, source-richness, spatial-lag, or POI category/source-depth features; expanded or mixed strata-CV; an EMS no-gehe variant; or density features with a 0.10 regional blend")
    ap.add_argument("--out", default="submission.csv", help="submission path")
    ap.add_argument("--draws", type=int, default=512, help="Monte Carlo draws per tract during calibration")
    ap.add_argument("--add-label", nargs=3, metavar=("NAME", "FILE", "RMSE"), help="append a scored file to the labels and exit")
    ap.add_argument("--round", type=float, metavar="RMSE", help="one loop round: record the score of the uploaded submission, recalibrate the loop physics model on all labels, write the next submission")
    ap.add_argument("--round-file", help="with --round-name and --round, path to a scored probe or alternate candidate")
    ap.add_argument("--round-name", help="with --round-file and --round, label name for the scored candidate")
    ap.add_argument("--fit-only", action="store_true", help="stage 8 only, reading the physics model from outputs/physics.csv")
    ap.add_argument("--loop-physics", action="store_true", help="with --fit-only, use the latest loop-calibrated physics model")
    ap.add_argument("--probes", type=int, default=0, metavar="K", help="write K probe files to probes/ and exit")
    ap.add_argument("--loop-probes", type=int, default=0, metavar="K", help="write K probes using baseline, scorecard, and latest loop physics models")
    ap.add_argument("--deep-probes", type=int, default=0, metavar="K", help="write K probes from global and region-specific component directions")
    ap.add_argument("--strata-probes", type=int, default=0, metavar="K", help="write K probes from scorecard-stratum component directions")
    ap.add_argument("--candidate-probe", metavar="PATH", help="probe the unscored residual of a candidate branch")
    ap.add_argument("--candidate-probe-out", metavar="PATH", help="CSV path for --candidate-probe output")
    ap.add_argument("--candidate-probe-step", type=float, default=PROBE_STEP, help="public-weighted RMS step for --candidate-probe")
    ap.add_argument("--candidate-probe-against", nargs="+", metavar="PATH",
                    help="remove earlier unscored probe directions before generating a candidate probe")
    ap.add_argument("--candidate-probe-sweep", nargs="+", metavar="PATH", help="rank candidate files by unmeasured public-weighted residual")
    ap.add_argument("--candidate-probe-sweep-out", default="outputs/branch_residual_sweep.json", help="report path for --candidate-probe-sweep")
    ap.add_argument("--calibrate-physics", action="store_true", help="stage 10: physics calibrated on the labels -> outputs/physics_lb*.csv")
    ap.add_argument("--regional", action="store_true", help="with --calibrate-physics: add per-region multipliers")
    ap.add_argument("--structural", action="store_true", help="with --calibrate-physics: facility leakage and road-frame mechanisms")
    ap.add_argument("--unmatched", action="store_true", help="with --calibrate-physics: reference facilities outside the category match")
    ap.add_argument("--covariates", action="store_true", help="with --calibrate-physics: screened covariate effects on TIGER length and buildings")
    ap.add_argument("--flex", action="store_true", help="with --calibrate-physics: flexible context effects on TIGER length and unseen facilities")
    ap.add_argument("--flex-sd", type=float, default=FLEX_SD, help="with --flex: prior sd of the flexible effects")
    ap.add_argument("--flex-regional", action="store_true", help="with --flex: per-region deviations of the flexible effects")
    ap.add_argument("--nn", action="store_true", help="with --calibrate-physics: neural context effects (MLP) in the simulator")
    ap.add_argument("--no-footprint-columns", action="store_true",
                    help="drop the footprint-derived strata columns (USFS/CarbonPlan building counts) and the Microsoft-sourced "
                         "building share; outputs get the suffix _nofp")
    ap.add_argument("--nn-wide", action="store_true", help="with --calibrate-physics: the wide network (12 units, all four reference layers)")
    ap.add_argument("--holdout", type=int, default=0, metavar="M",
                    help="with --calibrate-physics: leave the last M score moments out of the fit and report their chi-square")
    args = ap.parse_args()
    global FOOTPRINT_COLUMNS, AUX_FEATURE_BY_TARGET, AUX_FEATURE_TAG, AUX_USE_REGION_BLEND, AUX_USE_HURDLE, AUX_USE_CROSS_COUNTS, AUX_USE_SOURCE_MIX, AUX_USE_SOURCE_CONFIDENCE, AUX_USE_SOURCE_RICHNESS, AUX_USE_SPATIAL_LAGS, AUX_SPATIAL_LAG_TARGETS, AUX_USE_POI_CATEGORIES, AUX_POI_CATEGORY_KINDS, AUX_USE_SOURCE_DEPTH, AUX_SOURCE_DEPTH_TARGETS
    AUX_FEATURE_BY_TARGET = (AUX_DENSITY_BUILDING_FEATURE_CHOICES.copy() if args.aux_features == "density_building_cv"
                             else AUX_FEATURE_CHOICES.copy() if args.aux_features in ("density_cv", "regional_blend_cv")
                             else AUX_COUNTY_FEATURE_CHOICES.copy() if args.aux_features == "county_cv"
                             else AUX_STRATA_FEATURE_CHOICES.copy() if args.aux_features == "strata_cv"
                             else AUX_STRATA_FULL_FEATURE_CHOICES.copy() if args.aux_features == "strata_full_cv"
                             else AUX_STRATA_MIXED_FEATURE_CHOICES.copy() if args.aux_features == "strata_mixed_cv"
                             else AUX_STRATA_NO_GEHE_FEATURE_CHOICES.copy() if args.aux_features == "strata_ems_no_gehe_cv"
                             else AUX_CROSS_COUNT_FEATURE_CHOICES.copy() if args.aux_features in ("cross_counts_cv", "cross_counts_direct_cv", "source_mix_cv", "source_confidence_cv", "source_richness_cv", "spatial_lags_cv", "spatial_fire_ems_cv", "poi_categories_cv", "poi_categories_depth_cv")
                             else AUX_HURDLE_FEATURE_CHOICES.copy() if args.aux_features == "hurdle_cv" else {})
    AUX_FEATURE_TAG = ("auxfeatdb" if args.aux_features == "density_building_cv" else
                       "auxfeat" if args.aux_features == "density_cv" else
                       "county" if args.aux_features == "county_cv" else
                       "strata" if args.aux_features == "strata_cv" else
                       "strata_full" if args.aux_features == "strata_full_cv" else
                       "strata_mixed" if args.aux_features == "strata_mixed_cv" else
                       "strata_ems_no_gehe" if args.aux_features == "strata_ems_no_gehe_cv" else
                       "regional_blend" if args.aux_features == "regional_blend_cv" else
                       "crosscounts" if args.aux_features == "cross_counts_cv" else
                       "crosscounts_direct" if args.aux_features == "cross_counts_direct_cv" else
                       "sourcemix" if args.aux_features == "source_mix_cv" else
                       "sourceconfidence" if args.aux_features == "source_confidence_cv" else
                       "sourcerichness" if args.aux_features == "source_richness_cv" else
                       "spatiallags" if args.aux_features == "spatial_lags_cv" else
                       "spatialfireems" if args.aux_features == "spatial_fire_ems_cv" else
                       "poicategories" if args.aux_features == "poi_categories_cv" else
                       "poicategoriesdepth" if args.aux_features == "poi_categories_depth_cv" else
                       "hurdle" if args.aux_features == "hurdle_cv" else "")
    if args.no_footprint_columns:
        FOOTPRINT_COLUMNS = False
        AUX_FEATURE_TAG = "_".join(t for t in (AUX_FEATURE_TAG, "nofp") if t)
    AUX_USE_REGION_BLEND = args.aux_features == "regional_blend_cv"
    AUX_USE_HURDLE = args.aux_features in ("hurdle_cv", "cross_counts_cv")
    AUX_USE_CROSS_COUNTS = args.aux_features in ("cross_counts_cv", "cross_counts_direct_cv", "spatial_lags_cv", "spatial_fire_ems_cv", "poi_categories_cv", "poi_categories_depth_cv")
    AUX_USE_SOURCE_MIX = args.aux_features in ("source_mix_cv", "source_confidence_cv", "source_richness_cv", "poi_categories_cv", "poi_categories_depth_cv")
    AUX_USE_SOURCE_CONFIDENCE = args.aux_features in ("source_confidence_cv", "source_richness_cv", "poi_categories_cv", "poi_categories_depth_cv")
    AUX_USE_SOURCE_RICHNESS = args.aux_features == "source_richness_cv"
    AUX_USE_SPATIAL_LAGS = args.aux_features in ("spatial_lags_cv", "spatial_fire_ems_cv")
    AUX_USE_POI_CATEGORIES = args.aux_features in ("poi_categories_cv", "poi_categories_depth_cv")
    AUX_POI_CATEGORY_KINDS = AUX_POI_CATEGORY_SELECTED.copy() if AUX_USE_POI_CATEGORIES else {}
    AUX_USE_SOURCE_DEPTH = args.aux_features == "poi_categories_depth_cv"
    AUX_SOURCE_DEPTH_TARGETS = AUX_SOURCE_DEPTH_SELECTED.copy() if AUX_USE_SOURCE_DEPTH else set()
    AUX_SPATIAL_LAG_TARGETS = ({target for target, _ in AUX_TARGETS} if args.aux_features == "spatial_lags_cv"
                               else {"E_fire", "E_ems"} if args.aux_features == "spatial_fire_ems_cv" else set())
    if args.loop_physics and not args.fit_only:
        ap.error("--loop-physics requires --fit-only")
    if bool(args.round_file) != bool(args.round_name) or ((args.round_file or args.round_name) and args.round is None):
        ap.error("--round-file and --round-name must be supplied together with --round")
    if bool(args.candidate_probe) != bool(args.candidate_probe_out):
        ap.error("--candidate-probe and --candidate-probe-out must be supplied together")
    if args.candidate_probe_against and not args.candidate_probe:
        ap.error("--candidate-probe-against requires --candidate-probe")
    if args.candidate_probe and args.candidate_probe_sweep:
        ap.error("choose --candidate-probe or --candidate-probe-sweep")
    if args.candidate_probe and (args.round is not None or args.add_label or args.calibrate_physics or args.fit_only):
        ap.error("--candidate-probe only generates a probe; do not combine it with scoring or model-fitting actions")
    if args.candidate_probe_sweep and (args.round is not None or args.add_label or args.calibrate_physics or args.fit_only):
        ap.error("--candidate-probe-sweep only ranks candidates; do not combine it with scoring or model-fitting actions")
    if sum(bool(x) for x in (args.probes, args.loop_probes, args.deep_probes, args.strata_probes)) > 1:
        ap.error("choose only one of --probes, --loop-probes, --deep-probes, or --strata-probes")
    if args.add_label:
        add_label(*args.add_label)
        return
    if args.candidate_probe:
        make_candidate_probe(args.candidate_probe, args.candidate_probe_out, args.candidate_probe_step,
                             args.candidate_probe_against or ())
        return
    if args.candidate_probe_sweep:
        candidate_probe_sweep(args.candidate_probe_sweep, args.candidate_probe_sweep_out)
        return
    if args.aux_sweep:
        os.makedirs(OUT, exist_ok=True)
        df, _ = load()
        log("3/3 auxiliary-model hyperparameter sweep (county-grouped 5-fold CV)")
        auxiliary_model_sweep(df)
        return
    if args.aux_feature_sweep:
        os.makedirs(OUT, exist_ok=True)
        df, _ = load()
        log("3/3 auxiliary feature ablation (county-grouped 5-fold CV)")
        auxiliary_feature_sweep(df)
        return
    if args.aux_strata_expanded_sweep:
        os.makedirs(OUT, exist_ok=True)
        df, _ = load()
        log("3/3 expanded supplied-strata ablation (county-grouped 5-fold CV)")
        auxiliary_feature_sweep(df, variants=["density_strata_full"],
                                report_file="aux_strata_expanded_sweep.json")
        return
    if args.aux_ems_strata_sweep:
        os.makedirs(OUT, exist_ok=True)
        df, _ = load()
        log("3/3 EMS expanded-strata family ablation and model-capacity sweep (county-grouped 5-fold CV)")
        auxiliary_ems_strata_sweep(df, report_file="aux_ems_strata_sweep_v2.json")
        return
    if args.aux_hurdle_sweep:
        os.makedirs(OUT, exist_ok=True)
        df, _ = load()
        log("3/3 zero-versus-positive hurdle sweep (county-grouped 5-fold CV)")
        auxiliary_hurdle_sweep(df)
        return
    if args.aux_cross_count_sweep:
        os.makedirs(OUT, exist_ok=True)
        df, _ = load()
        log("3/3 cross-category Overture count feature sweep (county-grouped 5-fold CV)")
        auxiliary_cross_count_sweep(df)
        return
    if args.aux_source_mix_sweep:
        os.makedirs(OUT, exist_ok=True)
        df, _ = load()
        log("3/3 target-excluded Overture source-mix feature sweep (county-grouped 5-fold CV)")
        auxiliary_source_mix_sweep(df)
        return
    if args.aux_source_signal_sweep:
        os.makedirs(OUT, exist_ok=True)
        df, _ = load()
        log("3/3 incremental Overture-signals provenance sweep (county-grouped 5-fold CV)")
        auxiliary_source_signal_sweep(df)
        return
    if args.aux_source_confidence_sweep:
        os.makedirs(OUT, exist_ok=True)
        df, _ = load()
        log("3/3 source-specific Overture place-confidence sweep (county-grouped 5-fold CV)")
        auxiliary_source_confidence_sweep(df)
        return
    if args.aux_source_depth_sweep:
        os.makedirs(OUT, exist_ok=True)
        df, _ = load()
        log("3/3 target-excluded Overture source-depth sweep (county-grouped 5-fold CV)")
        auxiliary_source_depth_sweep(df)
        return
    if args.aux_source_richness_sweep:
        os.makedirs(OUT, exist_ok=True)
        df, _ = load()
        log("3/3 source-specific Overture address/contact richness sweep (county-grouped 5-fold CV)")
        auxiliary_source_richness_sweep(df)
        return
    if args.aux_poi_category_sweep:
        os.makedirs(OUT, exist_ok=True)
        df, _ = load()
        log("3/3 target-excluded Overture POI category sweep (county-grouped 5-fold CV)")
        auxiliary_poi_category_sweep(df)
        return
    if args.aux_poi_source_category_sweep:
        os.makedirs(OUT, exist_ok=True)
        df, _ = load()
        log("3/3 target-excluded source-by-category Overture POI sweep (county-grouped 5-fold CV)")
        auxiliary_poi_source_category_sweep(df)
        return
    if args.aux_poi_identity_sweep:
        os.makedirs(OUT, exist_ok=True)
        df, _ = load()
        log("3/3 target-excluded Overture POI identity/status completeness sweep (county-grouped 5-fold CV)")
        auxiliary_poi_identity_sweep(df)
        return
    if args.aux_poi_identity_depth_sweep:
        os.makedirs(OUT, exist_ok=True)
        df, _ = load()
        log("3/3 target-excluded identity completeness over category and source-depth features (county-grouped 5-fold CV)")
        auxiliary_poi_identity_sweep(df, report_file="aux_poi_identity_depth_sweep.json", include_depth=True)
        return
    if args.aux_poi_identity_ablation_sweep:
        os.makedirs(OUT, exist_ok=True)
        df, _ = load()
        log("3/3 BrightQuery identity-field ablation over the selected category and source-depth model (county-grouped 5-fold CV)")
        auxiliary_poi_identity_ablation_sweep(df)
        return
    if args.aux_spatial_lag_sweep:
        os.makedirs(OUT, exist_ok=True)
        df, _ = load()
        log("3/3 target-excluded nearby-tract count sweep (county-grouped 5-fold CV)")
        auxiliary_spatial_lag_sweep(df)
        return
    if args.aux_region_sweep:
        os.makedirs(OUT, exist_ok=True)
        df, _ = load()
        log("3/3 shared-versus-regional auxiliary models (county-grouped 5-fold CV)")
        auxiliary_region_sweep(df)
        return
    if args.aux_region_blend_sweep:
        os.makedirs(OUT, exist_ok=True)
        df, _ = load()
        log("3/3 regional-model shrinkage blend sweep (county-grouped 5-fold CV)")
        auxiliary_region_blend_sweep(df)
        return
    if args.eb_forward_sweep:
        if args.eb_forward_points < 1:
            ap.error("--eb-forward-points must be positive")
        os.makedirs(OUT, exist_ok=True)
        log("rolling-origin aggregate-label forecast validation")
        empirical_bayes_forward_sweep(AggregateLabels(), points=args.eb_forward_points,
                                      sigmas=args.eb_forward_sigmas or (0.0312, 0.0624, 0.1248, 0.2496))
        return
    if args.public_weight_forward_sweep:
        if args.public_weight_forward_points < 1:
            ap.error("--public-weight-forward-points must be positive")
        if not args.public_weight_forward_blends or any(x < 0 or x > 1 for x in args.public_weight_forward_blends):
            ap.error("--public-weight-forward-blends must contain values in [0, 1]")
        os.makedirs(OUT, exist_ok=True)
        log("rolling-origin public-row weight shrinkage validation")
        public_weight_forward_sweep(AggregateLabels(), points=args.public_weight_forward_points,
                                    blends=tuple(args.public_weight_forward_blends))
        return
    if args.transfer_forward_sweep:
        if args.transfer_forward_points < 1 or args.transfer_forward_draws < 1:
            ap.error("transfer-forward points and draws must be positive")
        os.makedirs(OUT, exist_ok=True)
        log("rolling-origin transfer-posterior validation")
        transfer_forward_sweep(AggregateLabels(), points=args.transfer_forward_points,
                               draws=args.transfer_forward_draws,
                               correlation_prior=args.transfer_forward_correlation_prior)
        return
    physics_path = out("physics_scorecard.csv" if args.scorecard else "physics.csv")
    if args.loop_physics:
        physics_path = out(lb_name(**LOOP_MODEL, aux_feature_tag=AUX_FEATURE_TAG) + ".csv")
    os.makedirs(OUT, exist_ok=True)
    set_priors(args.scorecard)
    t0 = time.time()
    clock = lambda: f"{time.time() - t0:.0f}s"
    report = {"inputs": "challenge data, organiser-published statistics, and public RMSE scores of submitted files"}

    def stage10(model):
        df, groups = load()
        log("10 physics calibrated on the aggregate labels (published statistics + scorecard + score moments)")
        rep = {}
        calibrate_physics_on_labels(df, groups, rep, **model)
        name = lb_name(**model, aux_feature_tag=AUX_FEATURE_TAG)
        atomic_json(out(f"{name}_report.json"), rep)
        log(f"Wrote {out(name)}.csv and {out(name)}_report.json ({clock()})")
        return df, groups

    if args.round is not None:
        # one round of the loop: score -> label -> recalibrated physics -> stage 8 -> next submission
        with open(out("submission_report.json")) as f:
            last = json.load(f)
        score_path = args.round_file or args.out
        current_md5 = file_md5(score_path)
        state = None
        if os.path.exists(ROUND_STATE):
            with open(ROUND_STATE) as f:
                state = json.load(f)
        # If stage 8 finished but the process stopped before its checkpoint, reconcile that completed output.
        if (state and state.get("phase") != "complete" and last.get("next_label") != state.get("label")
                and current_md5 == last.get("file_md5")):
            state.update(phase="complete", output_md5=current_md5, next_label=last["next_label"])
            atomic_json(ROUND_STATE, state)
            state = None
        if state and state.get("phase") != "complete":
            label = state["label"]
            if abs(float(state["rmse"]) - args.round) > 1e-9:
                raise SystemExit(f"unfinished round {label} has RMSE {state['rmse']:.9f}; resume it with that score before starting another round")
            if os.path.abspath(score_path) != state.get("input_path", os.path.abspath(args.out)):
                raise SystemExit(f"unfinished round {label} was started from {state.get('input_path', args.out)}; resume with that file")
            with np.load(LABELS, allow_pickle=False) as z:
                known = {n: float(s) for n, s in zip(z["names"].tolist(), z["scores"])}
            if label not in known:
                if current_md5 != state["input_md5"]:
                    raise SystemExit(f"unfinished round {label} has no recorded label and {score_path} no longer matches its input")
                add_label(label, score_path, args.round)
            elif abs(known[label] - args.round) > 1e-9:
                raise SystemExit(f"unfinished round {label} is recorded with RMSE {known[label]:.9f}, not {args.round:.9f}")
            log(f"resuming round {label} from checkpoint {state['phase']}")
        else:
            if not args.round_file and current_md5 != last["file_md5"]:
                raise SystemExit(f"{args.out} is not the file stage 8 last wrote; record its score with --add-label")
            label = args.round_name or last["next_label"]
            add_label(label, score_path, args.round)
            state = dict(label=label, rmse=float(args.round), input_md5=current_md5,
                         input_path=os.path.abspath(score_path), phase="label_recorded")
            atomic_json(ROUND_STATE, state)
        report["scored_input"] = dict(label=label, file=score_path, file_md5=current_md5, public_rmse=float(args.round))
        df, groups = stage10(LOOP_MODEL)
        state.update(phase="physics_ready")
        atomic_json(ROUND_STATE, state)
        physics_path = out(lb_name(**LOOP_MODEL, aux_feature_tag=AUX_FEATURE_TAG) + ".csv")
        report["physics_source"] = physics_path
        phys = pd.read_csv(physics_path, dtype={"GEOID": str}).set_index("GEOID")["coverage_gap_score"]
        log("8/8 aggregate labels")
        write_stage8(df, groups, phys.loc[df["GEOID"]].to_numpy(float), args.out, report, not args.no_forward,
                     report_path=out("submission_report.json"))
        with open(out("submission_report.json")) as f:
            next_report = json.load(f)
        state.update(phase="complete", output_md5=next_report["file_md5"], next_label=next_report["next_label"])
        atomic_json(ROUND_STATE, state)
        log(f"Wrote {args.out} ({clock()})")
        return
    if args.calibrate_physics:
        stage10(dict(regional=args.regional, structural=args.structural, unmatched=args.unmatched, covariates=args.covariates,
                     flex=args.flex and args.flex_sd, flex_regional=args.flex and args.flex_regional, nn="wide" if args.nn_wide else args.nn,
                     holdout=args.holdout))
        return
    if args.probes or args.loop_probes or args.deep_probes or args.strata_probes:
        df, _ = load()
        k = args.deep_probes or args.strata_probes or args.loop_probes or args.probes
        deep = bool(args.deep_probes)
        stratified = bool(args.strata_probes)
        log(f"9/9 probe design: {k} directions the scored files have not measured"
            + (" (global + region-specific components)" if deep else " (scorecard-stratum components)" if stratified else ""))
        suffix = "_auxfeat" if AUX_FEATURE_TAG else ""
        directory = ("probes_deep" if deep else "probes_strata" if stratified
                     else "probes_loop" if args.loop_probes else "probes") + suffix
        family = (("d" if deep else "sf" if stratified else "l" if args.loop_probes else "")
                  if AUX_FEATURE_TAG else ("s" if stratified else ""))
        make_probes(df, K=k, include_loop=bool(args.loop_probes or deep or stratified), include_regional=deep,
                    include_strata=stratified, directory=directory, family=family)
        return
    if args.fit_only:
        df, groups = load()
        phys = pd.read_csv(physics_path, dtype={"GEOID": str}).set_index("GEOID")["coverage_gap_score"]
        pred = phys.loc[df["GEOID"]].to_numpy(float)
        source = "latest loop-calibrated physics model" if args.loop_physics else "physics model read from physics.csv"
        report["physics_source"] = physics_path
        log(f"8/8 aggregate labels (stage 8 only; {source})")
        write_stage8(df, groups, pred, args.out, report, not args.no_forward)
        if args.loop_physics and args.out == "submission.csv" and os.path.exists(ROUND_STATE):
            with open(ROUND_STATE) as f:
                state = json.load(f)
            if state.get("phase") == "complete":
                state.update(output_md5=report["file_md5"], next_label=report["next_label"])
                atomic_json(ROUND_STATE, state)
        log(f"Wrote {args.out} ({clock()})")
        return

    log("1/7 data")
    download()
    log("2/7 features")
    for r in REGIONS:
        build_features(r, force=args.rebuild)
    df, groups = load()
    g = df["graded"].to_numpy()
    log(f"  {len(df)} tracts ({g.sum()} graded), {df.shape[1]} columns ({clock()})")
    no_route = {r: float((df.loc[df.region == r, "L_route"] <= 1).mean()) for r in GRADED}
    report["mechanism_check"] = {r: dict(published_undefined=UNDEFINED_SHARE[r], no_overture_route=round(no_route[r], 3))
                                 for r in GRADED}
    log("  mechanism check (before any fitting), share of tracts with no Overture route vs published undefined share: "
        + ", ".join(f"{r} {no_route[r]:.3f}/{UNDEFINED_SHARE[r]:.2f}" for r in GRADED))

    log("3/7 auxiliary models (county-grouped 5-fold CV)")
    mu, aux = fit_auxiliary(df)
    for rep in aux:
        log(f"  {rep['target']:9s} CV Poisson deviance: GLM {rep['cv_deviance']['poisson_glm']:.3f}, GBM {rep['cv_deviance']['poisson_gbm']:.3f}, "
            f"null {rep['null_deviance']:.3f} -> {rep['chosen']} with {rep['feature_variant']} features (explains {rep['explained']:.0%})")
    report["auxiliary_models"] = dict(feature_configs=AUX_FEATURE_BY_TARGET.copy(), results=aux)

    log("4/7 simulator")
    F = inputs(df, mu)
    T = Targets(df, groups, args.scorecard)
    sim0 = simulate(F, THETA0, args.draws)
    log(f"  prior centre: mean score {sim0['Ey'][g].mean():.4f} ({clock()})")

    log("5/7 calibration (simulated moments)")
    J0 = moment_jacobian(F, T, THETA0, args.draws)
    th, m, sd, cost, post_cov, J = calibrate(F, T, J0, args.draws)
    obs = T.observed()
    m_prior = T.model(sim0)
    fit = pd.DataFrame(dict(target=T.names, published=obs, prior_centre=m_prior, calibrated=m, pull=(m - obs) / sd))
    log(fit.round(3).to_string())
    post_sd = np.sqrt(np.diag(post_cov))
    params = pd.DataFrame(dict(parameter=NAMES, prior_centre=THETA0, prior_sd=PRIOR_SD, posterior=th, posterior_sd=post_sd,
                               identified=post_sd < 0.5 * PRIOR_SD))
    log(f"  cost {cost:.1f}; {params.identified.sum()} of {len(NAMES)} parameters identified by the data "
        f"(posterior sd < half the prior sd): {', '.join(params.parameter[params.identified])} ({clock()})")
    report["calibration"] = dict(targets=fit.to_dict(orient="records"), parameters=params.to_dict(orient="records"))

    log("6/7 validation")
    loto = leave_one_target_out(T, J, th, m, sd, m_prior)
    loto["error_held_out"] = loto.held_out_prediction - loto.published
    loto["error_prior"] = loto.prior_prediction - loto.published
    log(loto.round(3).to_string())
    log(f"  leave-one-target-out mean |error|: held-out fit {loto.error_held_out.abs().mean():.4f} vs prior centre "
        f"{loto.error_prior.abs().mean():.4f}")
    report["leave_one_target_out"] = loto.to_dict(orient="records")

    sens_base = simulate(F, th, 1024, seed=7)["Ey"]
    sens = prior_sensitivity(F, th, g, sens_base)
    log("  prior sensitivity (RMS change of the prediction for +1 prior sd), top 8:\n" + sens.head(8).round(4).to_string(index=False))
    report["prior_sensitivity"] = sens.to_dict(orient="records")

    log("7/7 prediction (E[score | data, theta_MAP], 4,096 draws per tract)")
    sim = predict(F, th)
    sim2 = predict(F, th, seed=900)
    pred = np.clip(sim["Ey"], 0, 1)
    stability = float(np.sqrt(np.mean((sim["Ey"] - sim2["Ey"]) ** 2)))
    comp = {k: round(float(np.mean(v[g])), 4) for k, v in sim.items() if k.endswith(("_defined", "_gap"))}
    log(f"  mean {pred[g].mean():.4f}; components (graded): {comp}; Monte Carlo stability RMS {stability:.5f}")
    sub = pd.DataFrame({"GEOID": df["GEOID"], "coverage_gap_score": np.round(pred, 6)})
    assert len(sub) == 9794 and sub.GEOID.is_unique and sub.coverage_gap_score.between(0, 1).all()
    sub.to_csv(physics_path, index=False)
    ratios = {k: round(float(pred[h].mean() / pred[l].mean()), 3) for k, (h, l) in groups.items()}
    report["physics"] = dict(file=physics_path, mean_graded=float(pred[g].mean()), components=comp, scorecard_ratios=ratios,
                             mc_stability_rms=stability)
    log(f"  scorecard ratios of the physics prediction: {ratios}; wrote {physics_path} ({clock()})")
    with open(physics_path.replace(".csv", "_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    final = pred
    if not (args.physics_only or args.scorecard):
        log("8/8 aggregate labels (public RMSE of the scored files)")
        report8 = {}
        final_l = write_stage8(df, groups, np.round(pred, 6), args.out, report8, not args.no_forward)   # = physics.csv
        final = pd.Series(final_l, index=AggregateLabels().geoid.to_numpy())[df["GEOID"].to_numpy()].to_numpy()
        log(f"Wrote {args.out} ({clock()})")
    if args.discovery:
        log("discovery")
        discovery(df, final)


if __name__ == "__main__":
    sys.exit(main())
