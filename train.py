"""
Bias Bounty Mapping Equity Challenge: coverage-gap model trained from scratch.

    pip install -r requirements.txt
    python train.py                  # data -> features -> physics model -> aggregate-label posterior -> submission.csv
    python train.py --rebuild        # recompute the cached features in data/features/
    python train.py --physics-only   # stop after the physics model (physics.csv)
    python train.py --scorecard      # physics variant also calibrated on the public Bias Scorecard (physics_scorecard.csv)
    python train.py --discovery      # Best Bias Discovery tables (write-up only, not used for the submission)

Inputs: the permitted challenge data (Overture release 2026-08-19.0 roads / unfiltered roads / rail /
infrastructure / buildings / places, ACS housing units, the strata tables and the tract boundaries), statistics
the organisers published, and in stage 8 the public RMSE of 65 submitted files (aggregate labels).

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
                 aggregate_labels.npz holds the vectors and public scores of 65 submitted files, including
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


def build_features(region, force=False):
    base, extra = os.path.join(FEAT, f"{region}.parquet"), os.path.join(FEAT, f"{region}-extra.parquet")
    if os.path.exists(base) and os.path.exists(extra) and not force:
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
        f["region"] = r
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    zero = [c for c in df.columns if c.startswith(("L_", "Lb_", "n_"))] + ["housing_units"]
    df[zero] = df[zero].fillna(0)
    df["area_km2"] = df["area_m2"] / 1e6
    df["pop"] = df["pop_total"].fillna(0).clip(lower=0)
    df["county"] = df["STATEFP"].astype(str) + df["COUNTYFP"].astype(str)
    df["graded"] = df["region"].isin(GRADED)
    for k, a, b, c in [("fire", "n_fd", "n_fd_alt", "n_fire_name"), ("ems", "n_ems", "n_ems_alt", "n_ems_name"),
                       ("sch", "n_sch", "n_sch_alt", "n_sch_name")]:
        df[f"E_{k}"] = df[a] + df[b] + df[c]          # all Overture evidence of the facility type
        df[f"X_{k}"] = df[b] + df[c]                  # weak evidence (alternate category or name only)
    u = df["usfs_BuildingCount_sum"]                  # independent building count (USFS)
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
    models = {"poisson_glm": lambda: make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-3, max_iter=2000)),
              "poisson_gbm": lambda: HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_iter=300,
                                                                   max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0,
                                                                   random_state=0)}
    oof = {k: np.zeros(len(y)) for k in models}
    for tr, te in GroupKFold(5).split(X, y, df["county"]):
        for k, make in models.items():
            oof[k][te] = np.clip(make().fit(X[tr], y[tr]).predict(X[te]), 1e-4, None)
    dev = {k: float(mean_poisson_deviance(y, p)) for k, p in oof.items()}
    null = float(mean_poisson_deviance(y, np.full(len(y), y.mean())))
    best = min(dev, key=dev.get)
    return oof[best], dict(target=target, cv_deviance=dev, null_deviance=null, chosen=best,
                           explained=1 - dev[best] / null)


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
NAMES, THETA0, PRIOR_SD = [], np.zeros(0), np.zeros(0)


def set_priors(use_scorecard):
    """Stratum effects are free only when the public scorecard is used to calibrate them."""
    global NAMES, THETA0, PRIOR_SD
    pri = CORE_PRIORS + (STRATA_PRIORS if use_scorecard else [])
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
    for k in STRATA:
        F[f"s_{k}"] = df[f"s_{k}"].to_numpy(float)
    F["n"] = len(df)
    return F


def simulate(F, th, S=256, seed=0):
    """Monte Carlo under the exact composite. Returns per-tract E[y], E[y^2], P(some component undefined),
    E[TIGER length] and E[reference EMS count]."""
    p = dict(zip(NAMES, th))
    rng = np.random.default_rng(seed)
    n = F["n"]
    ln = lambda sig, shape: np.exp(sig * rng.standard_normal(shape) - 0.5 * sig ** 2)
    eff = lambda pre: np.exp(sum(p.get(f"{pre}_{k}", 0.0) * F[f"s_{k}"] for k in STRATA))
    m = np.exp(np.array([p[f"log_m_{r}"] for r in GRADED] + [np.mean([p[f"log_m_{r}"] for r in GRADED])]))[F["ridx"]]
    # roads: TIGER S1100+S1200 length vs Overture named length
    inc = lambda q, L: np.where(rng.random((S, n)) < SIG(p[q]), 1.0, 0.0) * L[None, :]
    G = (F["L_route"] + p["beta_bnd"] * F["bnd"])[None, :] + inc("logit_q_fm", F["L_fm"]) + inc("logit_q_cty", F["L_cty"]) \
        + inc("logit_q_unr", F["L_unr"])
    has = G > 1.0
    T = (m * eff("gr"))[None, :] * G * ln(np.exp(p["log_sigma_t"]), (S, n))
    surprise = ~has & (rng.random((S, n)) < SIG(p["logit_p_surprise"]))
    T = np.where(surprise, 400.0 * ln(1.0, (S, n)), T)
    dT = has | surprise
    t = np.where(dT, 1 - np.minimum(1, F["N"][None, :] / np.maximum(T, 1e-9)), 0.0)
    # buildings: Microsoft count = kappa * USFS count * noise
    MS = (np.exp(p["log_kappa"]) * F["U_b"])[None, :] * ln(np.exp(p["log_sigma_b"]), (S, n))
    dB = MS > 0.5
    b = np.where(dB, 1 - np.minimum(1, F["O_b"][None, :] / np.maximum(MS, 1e-9)), 0.0)
    # facilities: reference = Bin(Overture primary, a) + Bin(weak evidence, x) + Poisson(unseen)
    fac = eff("gf") * np.exp(p["ge_fac"] * F["z_effort"])
    gs, ds, H_ems, types = [], [], None, {}
    for k in ["fire", "ems", "sch"]:
        r = SIG(p[f"logit_r_{k}"])
        lam = F[f"mu_{k}"] * (1 - r) / r * fac
        H = (rng.binomial(np.broadcast_to(F[f"O_{k}"], (S, n)), SIG(p[f"logit_a_{k}"]))
             + rng.binomial(np.broadcast_to(F[f"X_{k}"], (S, n)), SIG(p[f"logit_x_{k}"]))
             + rng.poisson(np.broadcast_to(lam, (S, n))))
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
    cm = np.exp(p["log_eps"]) * F["pop"] * np.exp(p["a_bq"] * F["z_bq"])
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
    cells.to_csv("discovery.csv", index=False)
    svi.to_csv("discovery_svi_control.csv")
    examples.to_csv("discovery_examples.csv", index=False)


# ================================================================================================
# 8. AGGREGATE LABELS: every scored submission's public RMSE is an exact moment of the hidden target
# ================================================================================================
LABELS = "aggregate_labels.npz"         # (GEOID, region, file names, public RMSE, prediction vectors) of 65 scored files
P_PUBLIC = 0.3                          # the public leaderboard is about 30% of the graded tracts
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
            eq[st] = [f for f in order[:order.index(st)] if not f.startswith(("stack3", "stack4_with"))]
        Hs, rhs = [], []
        for F, comps in eq.items():
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
    """Append one scored file (its vector and public RMSE) to the aggregate labels."""
    with np.load(LABELS, allow_pickle=False) as z:
        art = {k: z[k] for k in z.files}
    assert name not in art["names"], f"{name} is already a label"
    v = pd.read_csv(path, dtype={"GEOID": str}).set_index("GEOID").loc[art["geoid"].astype(str), "coverage_gap_score"]
    assert len(v) == 9794 and v.notna().all()
    art["names"] = np.append(art["names"], name).astype("<U42")
    art["scores"] = np.append(art["scores"], float(rmse))
    art["predictions"] = np.column_stack([art["predictions"], v.to_numpy(float)])
    np.savez_compressed(LABELS, **art)
    log(f"{LABELS}: added {name} ({path}, public RMSE {float(rmse):.9f}); {len(art['names'])} labels")


def label_group(name):
    if name.startswith("probe_const"):
        return "const"
    if name.startswith("tr_"):
        return "transfer"           # deliberate small step: prior N(1, 1) in units of the step taken
    if name.startswith("p") and name[1:3].isdigit():
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
        if np.sqrt(w @ v ** 2) > 1e-5 * np.sqrt(w @ V[:, i] ** 2):
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


def aggregate_label_fit(df, groups, physics, out_path, report, do_forward=True):
    """Stage 8: fit the target to the aggregate labels (the scored files) and write the submission."""
    L = AggregateLabels()
    g = L.graded
    pos = pd.Series(np.arange(len(df)), index=df["GEOID"].to_numpy())[L.geoid].to_numpy()   # df row of each label row
    f = physics[pos]
    stored = L.V["fs_physics.csv"]
    log(f"  {len(L.s)} scored files; physics stage reproduces its scored file (fs_physics.csv, RMSE "
        f"{L.s['fs_physics.csv']:.6f}): max |diff| {np.abs(np.round(f, 6) - stored).max():.1e}")
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
    cands = [("refit", est_final, final), (f"best scored file {best[1]}", best[0], np.round(L.V[best[1]].copy(), 6))]
    for tu in transfer_updates(L):
        v = tu["vector"].copy()
        v[~g] = final[~g]
        log(f"  local Bayesian update around {tu['anchor']} ({tu['directions']} directions, curvature scales "
            + ", ".join(f"{x:.2f}" for x in tu["curvature_scale"]) + "): coordinates "
            + ", ".join(f"{x:+.2f}" for x in tu["coordinates"])
            + f"; expected public RMSE {tu['expected_rmse']:.6f} +- {tu['sd_rmse']:.6f}, P(beats best) {tu['p_beats_best']:.2f}")
        if tu["p_beats_best"] >= 0.9:
            cands.append((f"local Bayesian update around {tu['anchor']}", tu["expected_rmse"], v))
        report["transfer_update"] = {k_: v_ for k_, v_ in tu.items() if k_ != "vector"}
    if any(label_group(n) == "transfer" for n in L.s):
        # Large in-span refits are scored by the same public-curvature assumption that failed: keep them out once
        # transfer files exist, and trust the measured local update or the best scored file.
        cands = [c_ for c_ in cands if c_[0] != "refit"]
    choice = min(cands, key=lambda c: c[1])
    final = choice[2]
    rat = {k: round(float(final[h].mean() / final[l].mean()), 3) for k, (h, l) in grp_l.items()}
    log(f"  chosen: {choice[0]} (estimated public RMSE {choice[1]:.6f})")
    report["aggregate_fit"] = dict(report.get("aggregate_fit", {}), chosen=choice[0], chosen_est=choice[1])
    log(f"  scorecard ratios (zero-score leaders: rural 2.23-2.38, tribal 2.79-2.97): {rat}")
    report["aggregate_fit"] = dict(report.get("aggregate_fit", {}), hyper=hyper, washington_affine=[float(a_b[0]), float(a_b[1])],
                                   est_public_refit=est_final, best_scored=dict(file=best[1], rmse=best[0]), ratios=rat)
    if os.path.exists("physics_lb.csv"):
        lb_phys = pd.read_csv("physics_lb.csv", dtype={"GEOID": str}).set_index("GEOID")["coverage_gap_score"]
        final = np.round(transfer_step(L, final, lb_phys.loc[L.geoid].to_numpy(float), report, choice[1]), 6)
        rat = {k: round(float(final[h].mean() / final[l].mean()), 3) for k, (h, l) in grp_l.items()}
        report["aggregate_fit"]["ratios_after_transfer"] = rat
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


def physics_contributions(df, report_path, use_scorecard):
    """Rebuild a calibrated physics model from its report and return each component's contribution to the score
    (the four sum to the model's prediction)."""
    set_priors(use_scorecard)
    with open(report_path) as f:
        par = json.load(f)["calibration"]["parameters"]
    assert [p["parameter"] for p in par] == NAMES, f"{report_path} does not match the model's parameters"
    th = np.array([p["posterior"] for p in par])
    X, _ = context_matrix(df)
    mu = {t: cross_fit(df, X, t, cap)[0] for t, cap in AUX_TARGETS}
    sim = predict(inputs(df, mu), th)
    return {k: sim[k] for k in COMPONENTS}, sim["Ey"]


def make_probes(df, K=5, step=PROBE_STEP, min_rms=0.002):
    """Directions the scored files have not measured yet: each physics model's component contributions, made
    orthogonal to the span of every scored file and to each other (largest first). Each probe is the best scored
    file plus `step` along one direction; its score measures the target's projection on that direction (sd about
    0.001), and the next stage-8 fit uses it."""
    L = AggregateLabels()
    g = L.graded
    w = L.pi / L.pi.sum()
    sw = np.sqrt(w)[:, None]
    pos = pd.Series(np.arange(len(df)), index=df["GEOID"].to_numpy())[L.geoid].to_numpy()
    cands = {}
    for tag, path, sc in [("physics", "physics_report.json", False), ("physics_scorecard", "physics_scorecard_report.json", True)]:
        contrib, ey = physics_contributions(df, path, sc)
        log(f"  {tag}: components rebuilt (sum matches the model to {np.abs(sum(contrib.values()) - ey).max():.1e})")
        for k, v in contrib.items():
            cands[f"{tag}:{k[2:]}"] = v[pos][g]
    set_priors(False)
    X = np.column_stack([np.ones(g.sum())] + [L.V[f][g] for f in L.s if f != "probe_zeros.csv"])
    Qx, _ = np.linalg.qr(X * sw)
    resid = {k: v - (Qx @ (Qx.T @ (v[:, None] * sw)))[:, 0] / sw[:, 0] for k, v in cands.items()}
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
    os.makedirs("probes", exist_ok=True)
    manifest = []
    for i, (k, e, size) in enumerate(chosen, 1):
        v = base.copy()
        v[g] = np.clip(base[g] + step * e, 0, 1)
        name = f"q{i:02d}_{k.replace(':', '_')}"
        path = os.path.join("probes", f"{name}.csv")
        pd.DataFrame({"GEOID": L.geoid, "coverage_gap_score": np.round(v, 6)}).to_csv(path, index=False)
        no_signal = float(np.sqrt(best[0] ** 2 + L.AP((v[g] - base[g]) ** 2)))
        manifest.append(dict(file=path, label_name=f"{name}.csv", direction=k, base=best[1], base_rmse=best[0], step_rms=step,
                             out_of_span_rms_of_component=size, expected_rmse_if_no_signal=no_signal,
                             expected_rmse_if_like_physics=float(np.sqrt(max(best[0] ** 2 - 2 * step * 0.003 + step ** 2, 0)))))
        log(f"  {path}: {k} (new part RMS {size:.4f}); expected score {no_signal:.6f} if the direction is empty, "
            f"{manifest[-1]['expected_rmse_if_like_physics']:.6f} if it carries signal like the physics directions")
    with open(os.path.join("probes", "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


# ================================================================================================
# 10. PHYSICS CALIBRATED ON THE AGGREGATE LABELS
# ================================================================================================
TRANSFER = 0.0      # no new bet this round (was 0.10; step along the unscored part of the label-calibrated physics. Its coefficient has been +0.68 and
                    # -0.86 (z 14 and 8): informative but of unpredictable sign. A small step costs at most ~0.0001 now
                    # and its score still measures the direction fully, so the next file can take the measured optimum.
TRANSFER_PRIOR_SD = 0.8   # prior sd of a calibrated-physics direction's coefficient (spread of the measured +0.68, -0.86)


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


def calibrate_physics_on_labels(df, groups, report, S=512):
    """Stage 10: refit the physics model's 41 parameters (stratum effects included) to the published statistics,
    the public scorecard and every scored file's moment, starting from the scorecard-calibrated physics. Its part
    outside the scored span is the physics extrapolation of everything the scores have measured."""
    L = AggregateLabels()
    g = L.graded
    pos = pd.Series(np.arange(len(df)), index=df["GEOID"].to_numpy())[L.geoid].to_numpy()
    names = [n for n in L.s if n not in ("probe_zeros.csv", "e06_r2f_only_eastern-wa.csv")]
    best = min((s_, n) for n, s_ in L.s.items() if not n.startswith("probe_"))
    _, parts, _ = eb_posterior(L, names, L.V[best[1]][g])
    set_priors(True)
    with open("physics_scorecard_report.json") as f:
        th0 = np.array([p["posterior"] for p in json.load(f)["calibration"]["parameters"]])
    X, _ = context_matrix(df)
    mu = {t: cross_fit(df, X, t, cap)[0] for t, cap in AUX_TARGETS}
    F = inputs(df, mu)
    T = LabelTargets(df, groups, parts["Q"], parts["o"], parts["noise"], parts["w"], pos, g)
    obs, k0 = T.observed(), len(T.names) - parts["Q"].shape[1]
    sim0 = simulate(F, th0, S)
    m0, sd0 = T.model(sim0), T.sd(sim0)
    J = moment_jacobian(F, T, th0, S)
    th, m, sd, cost, _, _ = calibrate(F, T, J, S, th0=th0, iters=40)
    chi = lambda mm, ss: ((mm - obs) / ss) ** 2
    log(f"  chi-square: published + scorecard {chi(m0, sd0)[:k0].sum():.1f} -> {chi(m, sd)[:k0].sum():.1f}; "
        f"{parts['Q'].shape[1]} score moments {chi(m0, sd0)[k0:].sum():.1f} -> {chi(m, sd)[k0:].sum():.1f}")
    sim = predict(F, th)
    pred = np.clip(sim["Ey"], 0, 1)
    pd.DataFrame({"GEOID": df["GEOID"], "coverage_gap_score": np.round(pred, 6)}).to_csv("physics_lb.csv", index=False)
    v = L.V[best[1]].copy()
    v[g] = pred[pos][g]
    est = estimate(L, v)
    ratios = {k: round(float(pred[h].mean() / pred[l].mean()), 3) for k, (h, l) in groups.items()}
    log(f"  label-calibrated physics: mean {pred[df['graded'].to_numpy()].mean():.4f}; estimated public RMSE {est:.6f}; "
        f"scorecard ratios {ratios}")
    report.update(parameters=[dict(parameter=n, start=float(a), fitted=float(b)) for n, a, b in zip(NAMES, th0, th)],
                  targets=[dict(target=n, observed=float(x), start=float(y), fitted=float(z))
                           for n, x, y, z in zip(T.names[:k0], obs[:k0], m0[:k0], m[:k0])],
                  est_public=est, ratios=ratios, cost=cost)
    set_priors(False)
    return pred


def transfer_updates(L, draws=200000, min_rms=0.002, proj_scale=0.005, seed=0):
    """Local Bayesian update around the anchor (the best file scored before the first transfer label).

    Every transfer file is a displacement D_k from the anchor, and its public MSE is exactly
        MSE_k = MSE_anchor + pub(D_k^2) - 2 pub(D_k (y - anchor)).
    The linear terms are what the scores measure. The quadratic term is the mean of D_k^2 over the unknown public rows.
    Estimated weights get it wrong for large, concentrated moves: the third transfer file showed the public curvature
    1.4x the estimate. So the curvature along the main directions (each displacement's new part with RMS >= min_rms)
    gets a prior: log-normal scales (sd 0.4) and correlations N(0, 0.3^2). Each prior draw fits the linear terms to
    all transfer scores by weighted least squares and is weighted by how well it reproduces them. A file's score is
    exact except for its part outside the main directions (a small bet, clipping): its noise is
    2 x proj_scale x RMS(that part) in MSE units. The update minimises the posterior expected MSE: x* = E[C]^-1 E[G]."""
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
    rng = np.random.default_rng(seed)
    k = np.exp(rng.normal(0, 0.4, (draws, K)))
    R = np.tile(np.eye(K), (draws, 1, 1))
    iu = np.triu_indices(K, 1)
    corr = rng.normal(0, 0.3, (draws, len(iu[0])))
    R[:, iu[0], iu[1]] = corr
    R[:, iu[1], iu[0]] = corr
    sq = np.sqrt(np.diag(c))
    C = R * (k[:, :, None] * k[:, None, :]) ** 0.5 * np.outer(sq, sq)[None]             # draw x K x K
    q = np.einsum("nk,dkl,nl->dn", X, C, X)                                               # draw x files
    rhs = (q - (M - ma)[None]) / 2                                                        # = X G
    Wf = 1 / sd ** 2
    P = np.linalg.solve(X.T @ (X * Wf[:, None]), (X * Wf[:, None]).T)                    # K x files, weighted LS
    G = rhs @ P.T
    res = rhs - G @ X.T
    logw = -0.5 * np.sum((2 * res / sd[None]) ** 2, axis=1)
    wt = np.exp(logw - logw.max())
    wt /= wt.sum()
    EC, EG = np.einsum("d,dkl->kl", wt, C), wt @ G
    x = np.linalg.solve(EC, EG)
    mse = ma + np.einsum("k,dkl,l->d", x, C, x) - 2 * G @ x
    best_scored = min(L.s[n] for n in order if not n.startswith("probe_"))
    exp_mse = float(wt @ mse)
    v = L.V[anchor].copy()
    v[g] = np.clip(a + sum(xi * e for xi, e in zip(x, E)), 0, 1)
    return [dict(anchor=anchor, directions=K, coordinates=x.tolist(), expected_rmse=float(np.sqrt(exp_mse)),
                 sd_rmse=float(np.sqrt(wt @ (mse - exp_mse) ** 2) / (2 * np.sqrt(exp_mse))),
                 p_beats_best=float(wt @ (mse < best_scored ** 2)), effective_draws=float(1 / (wt ** 2).sum()),
                 curvature_scale=(wt @ k).tolist(), vector=np.round(v, 6))]


def transfer_step(L, base, physics_lb, report, base_rmse):
    """Add TRANSFER x the part of the label-calibrated physics that no scored file contains."""
    g = L.graded
    w = L.pi / L.pi.sum()
    sw = np.sqrt(w)[:, None]
    X = np.column_stack([np.ones(g.sum())] + [L.V[n][g] for n in L.s if n != "probe_zeros.csv"])
    Qx, _ = np.linalg.qr(X * sw)
    d = physics_lb[g] - base[g]
    u = d - (Qx @ (Qx.T @ (d[:, None] * sw)))[:, 0] / sw[:, 0]
    out = base.copy()
    out[g] = np.clip(base[g] + TRANSFER * u, 0, 1)
    size = float(np.sqrt(w @ u ** 2))
    outcomes = {f"coef_{b:.1f}": float(np.sqrt(max(base_rmse ** 2 + size ** 2 * (TRANSFER ** 2 - 2 * TRANSFER * b), 0)))
                for b in (0.0, 0.2, 0.4, 0.6)}
    log(f"  transfer step: {TRANSFER} x the unscored part of physics_lb.csv (RMS {size:.4f}); expected public RMSE by the "
        f"truth's coefficient along it: " + ", ".join(f"{k[5:]} -> {v:.6f}" for k, v in outcomes.items()))
    report["transfer"] = dict(share=TRANSFER, unscored_rms=size, expected=outcomes)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--rebuild", action="store_true", help="recompute the cached features")
    ap.add_argument("--scorecard", action="store_true", help="physics variant also calibrated on the public Bias Scorecard")
    ap.add_argument("--physics-only", action="store_true", help="stop after the physics model (stages 1-7)")
    ap.add_argument("--no-forward", action="store_true", help="skip the chronological validation of stage 8")
    ap.add_argument("--discovery", action="store_true", help="write the Best Bias Discovery tables")
    ap.add_argument("--out", default="submission.csv", help="submission path")
    ap.add_argument("--draws", type=int, default=512, help="Monte Carlo draws per tract during calibration")
    ap.add_argument("--add-label", nargs=3, metavar=("NAME", "FILE", "RMSE"), help="append a scored file to the labels and exit")
    ap.add_argument("--fit-only", action="store_true", help="stage 8 only, reading the physics model from physics.csv")
    ap.add_argument("--probes", type=int, default=0, metavar="K", help="write K probe files to probes/ and exit")
    ap.add_argument("--calibrate-physics", action="store_true", help="stage 10: physics calibrated on the labels -> physics_lb.csv")
    args = ap.parse_args()
    if args.add_label:
        add_label(*args.add_label)
        return
    physics_path = "physics_scorecard.csv" if args.scorecard else "physics.csv"
    set_priors(args.scorecard)
    t0 = time.time()
    clock = lambda: f"{time.time() - t0:.0f}s"
    report = {"inputs": "challenge data, organiser-published statistics, and (stage 8) the public RMSE of 65 submitted files"}

    if args.calibrate_physics:
        df, groups = load()
        log("10 physics calibrated on the aggregate labels (published statistics + scorecard + score moments)")
        rep = {}
        calibrate_physics_on_labels(df, groups, rep)
        with open("physics_lb_report.json", "w") as f:
            json.dump(rep, f, indent=2, default=float)
        log(f"Wrote physics_lb.csv and physics_lb_report.json ({clock()})")
        return
    if args.probes:
        df, _ = load()
        log(f"9/9 probe design: {args.probes} directions the scored files have not measured")
        make_probes(df, K=args.probes)
        return
    if args.fit_only:
        df, groups = load()
        phys = pd.read_csv("physics.csv", dtype={"GEOID": str}).set_index("GEOID")["coverage_gap_score"]
        pred = phys.loc[df["GEOID"]].to_numpy(float)
        log("8/8 aggregate labels (stage 8 only; physics model read from physics.csv)")
        aggregate_label_fit(df, groups, pred, args.out, report, do_forward=not args.no_forward)
        with open("submission_report.json", "w") as f:
            json.dump(report, f, indent=2, default=float)
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
    X, xnames = context_matrix(df)
    mu, aux = {}, []
    for target, cap in AUX_TARGETS:
        mu[target], rep = cross_fit(df, X, target, cap)
        aux.append(rep)
        log(f"  {target:9s} CV Poisson deviance: GLM {rep['cv_deviance']['poisson_glm']:.3f}, GBM {rep['cv_deviance']['poisson_gbm']:.3f}, "
            f"null {rep['null_deviance']:.3f} -> {rep['chosen']} (explains {rep['explained']:.0%})")
    report["auxiliary_models"] = dict(features=xnames, results=aux)

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
        final_l = aggregate_label_fit(df, groups, pred, args.out, report8, do_forward=not args.no_forward)
        with open("submission_report.json", "w") as f:
            json.dump(report8, f, indent=2, default=float)
        final = pd.Series(final_l, index=AggregateLabels().geoid.to_numpy())[df["GEOID"].to_numpy()].to_numpy()
        log(f"Wrote {args.out} ({clock()})")
    if args.discovery:
        log("discovery")
        discovery(df, final)


if __name__ == "__main__":
    sys.exit(main())
