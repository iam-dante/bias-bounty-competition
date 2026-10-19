"""
Bias Bounty Mapping Equity Challenge: data-only model (no leaderboard feedback).

    pip install -r requirements.txt duckdb scikit-learn
    python data_only.py              # data -> features -> physics simulator -> submission_data_only.csv
    python data_only.py --rebuild    # recompute the cached features in data/features/

This script uses only the permitted challenge data (Overture extracts, ACS housing, strata tables)
and the five Paradise scores the organisers published. It never reads leaderboard scores, scored
submission files or any other team's scorecard.

Technique (each step prints its progress):
  1. DATA. Downloads the permitted files from the public Source Cooperative bucket (no account).
  2. FEATURES (DuckDB spatial). The Overture side of every coverage ratio is computed exactly per
     tract: named-highway length, building count, places and facility evidence. The reference side
     (TIGER highways, Microsoft buildings, HIFLD facilities, CBP) was withdrawn, so the features
     that predict it are measured instead: Overture route designations (TIGER S1100/S1200 is the
     Interstate/US/State system), highways running along tract boundaries, facility names and
     alternate categories, USFS building counts, population.
  3. PHYSICS. A generative model of each withdrawn reference count (log-normal TIGER length,
     Microsoft count, binomial + Poisson facilities, CBP establishments) is pushed through the
     exact official composite (mean of the defined road / building / POI components) by Monte
     Carlo. The average over draws is E[score | permitted data], the RMSE-optimal prediction under
     the model. Four calibrated parameter sets (global and per-region) are averaged.
  4. OUTPUT. Clipped to [0, 1]; the five published Paradise tracts are set to their scores.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
import warnings
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
DATA = os.environ.get("BB_DATA", "data")
FEAT = os.path.join(DATA, "features")
REGIONS = ["eastern-ok", "maricopa-az", "northern-ca", "south-central-tx", "eastern-wa"]
STATE = {"eastern-ok": "OK", "maricopa-az": "AZ", "northern-ca": "CA", "south-central-tx": "TX", "eastern-wa": "WA"}
CALIB_REGIONS = REGIONS[:4]            # eastern-wa is in the file but not graded
RIDX = {r: i for i, r in enumerate(REGIONS)}
DATA_URL = "https://data.source.coop/humane-intelligence/bias-bounty-mapping-equity-challenge"
# five Paradise tracts with published coverage_gap_score (public discussion 34910)
PUBLISHED = {"06007002300": 0.252, "06007001602": 0.254, "06007001703": 0.191, "06007002200": 0.249,
             "06007001601": 0.323}


# ================================================================================================
# 1. DATA: the permitted challenge files, from the public Source Cooperative bucket
# ================================================================================================
def data_files():
    """The files the pipeline reads, relative to data/ and to the bucket root."""
    out = []
    for r in REGIONS:
        out += [f"reference/{r}/{r}-{x}" for x in ("overture-roads.parquet", "overture-buildings.parquet",
                                                    "overture-pois.parquet", "census-acs-housing.parquet",
                                                    "sample-submission.csv")]
        out += [f"strata/{r}/{r}-{x}" for x in ("census-tracts.parquet", "strata-tract-table.parquet")]
    return out


def _http(url, method="GET"):
    # the proxy rejects Python's default User-Agent with 403
    return urllib.request.urlopen(urllib.request.Request(url, method=method, headers={"User-Agent": "bias-bounty-solution"}),
                                  timeout=60)


def download():
    """Fetch every missing or incomplete file (size checked against the bucket). No credentials."""
    files = data_files()
    for i, rel in enumerate(files, 1):
        path, url = os.path.join(DATA, rel), f"{DATA_URL}/{rel}"
        try:
            size = int(_http(url, "HEAD").headers["Content-Length"])
        except Exception as e:
            if os.path.exists(path):
                print(f"  [{i:2d}/{len(files)}] {rel}: bucket unreachable ({e}), using the local copy")
                continue
            raise
        if os.path.exists(path) and os.path.getsize(path) == size:
            print(f"  [{i:2d}/{len(files)}] {rel}: up to date ({size / 1e6:.1f} MB)")
            continue
        print(f"  [{i:2d}/{len(files)}] {rel}: downloading {size / 1e6:.1f} MB ...", flush=True)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with _http(url) as r, open(path + ".part", "wb") as f:
            while chunk := r.read(1 << 22):
                f.write(chunk)
        assert os.path.getsize(path + ".part") == size, f"{rel}: incomplete download, rerun to resume"
        os.replace(path + ".part", path)


# ================================================================================================
# 2. FEATURES
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
# Overture's historic / future route designations ride on ordinary streets in Arizona (they are not
# TIGER S1100/S1200); elsewhere dropping them breaks the official road-undefined bound.
HISTORIC_NOT_TIGER = {"AZ"}


def _con():
    import duckdb
    con = duckdb.connect()
    con.sql("INSTALL spatial; LOAD spatial; SET threads=10; SET memory_limit='10GB';")
    con.sql("SET preserve_insertion_order=false; SET enable_progress_bar=false;")
    return con


def _t(g):  # CRS84 -> CONUS Albers metres; always_xy is required (challenge README)
    return f"ST_Transform({g}, 'EPSG:4326', 'EPSG:5070', always_xy := true)"


def _sql_list(xs):
    return "[" + ",".join(f"'{x}'" for x in xs) + "]"


def _tracts(con, region):
    con.sql(f"""CREATE OR REPLACE TABLE tr AS SELECT GEOID, geometry g4326, {_t('geometry')} g
                FROM '{DATA}/strata/{region}/{region}-census-tracts.parquet'""")
    con.sql("CREATE OR REPLACE TABLE trb AS SELECT GEOID, ST_Boundary(g) bd FROM tr")


def _route_sql(st):
    """Route networks that are TIGER S1100/S1200: Interstate, US and State systems (with their
    loops / spurs / business routes and state toll roads); Texas FM/RM roads and county roads kept apart."""
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
          sum(L) FILTER (class = 'tertiary')      AS L_tertiary,       -- roads TIGER may call S1200
          -- TIGER S1200 is "usually in the U.S., State or County Highway system": county routes and
          -- unrouted arterials are length TIGER may hold that the route-based estimate leaves out
          sum(L) FILTER (is_county AND NOT is_route AND NOT is_fm
                         AND class IN ('primary','secondary','tertiary')) AS L_county,
          sum(L) FILTER (NOT is_route AND NOT is_fm AND NOT is_county AND class IN ('primary','secondary')
                         AND coalesce(subclass, '') <> 'link') AS L_unrouted
        FROM rl GROUP BY GEOID""").df()
    # Overture route length within 30 m of the tract boundary
    nb = con.sql("""
        WITH bb AS (SELECT GEOID, ST_Buffer(bd, 30) bz FROM trb)
        SELECT rl.GEOID, sum(ST_Length(ST_Intersection(rl.gi, bb.bz))) FILTER (rl.is_route) AS L_route_nearB
        FROM rl JOIN bb USING (GEOID)
        WHERE (rl.is_named OR rl.is_route OR rl.is_fm) AND ST_Intersects(rl.gi, bb.bz)
        GROUP BY rl.GEOID""").df()
    # tract boundary running ALONG a route highway: TIGER puts such a highway in both tracts,
    # Overture's copy lands in one (parallel pieces only; crossings are < 4 x buffer and dropped)
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
          count(*) FILTER (ds = 'meta') n_poi_meta,               -- upstream source of each place
          count(*) FILTER (ds = 'BrightQuery') n_poi_bq,          -- business registry
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


def _facility_points(con, region):
    """One row per (Overture facility point, tract within 500 m): d > 0 is the distance to the
    boundary of the containing tract, d < 0 minus the distance to a neighbouring tract. The two
    maps place a station tens of metres apart, so reference copies near a boundary leak next door."""
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


def build_features(region, force=False):
    """data/features/<region>.parquet (tract features + strata) and <region>-facpts.parquet."""
    os.makedirs(FEAT, exist_ok=True)
    out, pts_out = os.path.join(FEAT, f"{region}.parquet"), os.path.join(FEAT, f"{region}-facpts.parquet")
    if os.path.exists(out) and os.path.exists(pts_out) and not force:
        print(f"  features [{region}] cached")
        return
    t0 = time.time()
    con = _con()
    _tracts(con, region)
    df = _road_features(con, region, STATE[region])
    for part in (_building_features(con, region), _poi_features(con, region)):
        df = df.merge(part, on="GEOID", how="left")
    acs = pd.read_parquet(f"{DATA}/reference/{region}/{region}-census-acs-housing.parquet", columns=["GEOID", "housing_units"])
    strata = pd.read_parquet(f"{DATA}/strata/{region}/{region}-strata-tract-table.parquet")
    df = df.merge(acs, on="GEOID", how="left").merge(strata, on="GEOID", how="left")
    df.insert(1, "region", region)
    df.to_parquet(out, index=False)
    _facility_points(con, region).to_parquet(pts_out, index=False)
    print(f"  features [{region}] {df.shape} in {time.time() - t0:.0f}s", flush=True)


# ================================================================================================
# 3. PHYSICS: generative model of the withdrawn reference counts
# ================================================================================================
def load():
    frames = []
    for r in REGIONS:
        f = pd.read_parquet(os.path.join(FEAT, f"{r}.parquet"))
        ss = pd.read_csv(f"{DATA}/reference/{r}/{r}-sample-submission.csv", dtype={"GEOID": str})
        f = ss[["GEOID"]].merge(f.drop(columns=["region"]), on="GEOID", how="left")  # authoritative scored list & order
        f["region"] = r
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    zero = [c for c in df.columns if c.startswith(("L_", "Lb_", "n_"))] + ["housing_units"]
    df[zero] = df[zero].fillna(0)
    df["area_km2"] = df["area_m2"] / 1e6
    df["rural"] = (df["ruca_primary"] >= 4).astype(int)
    df["burned"] = df["mtbs_wildfire_ever"].fillna(False).astype(bool).astype(int)
    above = lambda c: (df[c] > df[c].median()).astype(int)
    df["g_rural"] = (df["pct_urban"] < 0.5).astype(int)
    df["g_tribal"] = df["tribal_any"].fillna(False).astype(bool).astype(int)
    df["g_svi"], df["g_cvi"] = above("svi_overall"), above("cvi_overall")
    df["g_drs"], df["g_drw"] = above("usdm_summer_dsci"), above("usdm_winter_dsci")
    df["g_wild"], df["g_heat"] = above("usfs_WHP_mean"), above("epht_heat_days_summer")
    df["ridx"] = df["region"].map(RIDX)
    df["pop"] = df["pop_total"].fillna(0).clip(lower=0)
    # Overture evidence of each facility type; "X_" is weak evidence (alternate category / name)
    for k, a, b, c in [("fire", "n_fd", "n_fd_alt", "n_fire_name"), ("ems", "n_ems", "n_ems_alt", "n_ems_name"),
                       ("sch", "n_sch", "n_sch_alt", "n_sch_name")]:
        df[f"E_{k}"] = df[a] + df[b] + df[c]
        df[f"X_{k}"] = df[b] + df[c]
    u = df["usfs_BuildingCount_sum"]          # independent building count (USFS)
    df["U_bld"] = u.where(u > 0, df["n_bld"]).fillna(df["n_bld"])
    return df


def prior_rates(df):
    """Expected Overture-visible facility evidence (Poisson GLM); drives the stations Overture never shows."""
    from sklearn.linear_model import PoissonRegressor
    X = np.column_stack([np.log1p(df["pop"]), np.log1p(df["area_km2"]), np.log1p(df["n_poi"]), np.log1p(df["n_bld"]),
                         df["rural"], np.log1p(df["housing_units"]), *[(df["ridx"] == i).astype(float) for i in range(len(REGIONS))]])
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return pd.DataFrame({f"mu_{k[2:]}": PoissonRegressor(alpha=1e-3, max_iter=1000).fit(X, df[k].clip(upper=20)).predict(X)
                         for k in ["E_fire", "E_ems", "E_sch"]}, index=df.index)


@dataclass
class Params:
    m: tuple = (1.7, 1.7, 2.0, 1.7)   # TIGER records per highway edge, per region
    w_fm: float = 0.0                  # share of Texas FM/RM length TIGER calls S1200
    beta_bnd: float = 0.5              # boundary highway whose Overture copy is next door
    sigma_t: float = 0.25              # log-noise of the TIGER length
    p_def0: float = 0.03               # P(TIGER highway | no route evidence)
    kappa_b: float = 0.97              # Microsoft count = kappa * USFS count * noise
    sigma_b: float = 0.10
    alpha_fire: float = 0.90           # reference facilities = Bin(Overture primary, alpha) + Bin(weak, alpha_x)
    alpha_fire_x: float = 0.60         #   + Poisson(mu (1 - r) / r)   [never shown by Overture]
    r_fire: float = 0.60
    alpha_ems: float = 0.90
    alpha_ems_x: float = 0.05
    r_ems: float = 0.90
    alpha_sch: float = 0.85
    alpha_sch_x: float = 0.10
    r_sch: float = 0.95
    eps_cbp: float = 0.025             # business establishments = eps * population * noise
    sigma_c: float = 0.35
    w_cty: float = 0.0                 # share of county-route length TIGER calls S1200
    w_unr: float = 0.0                 # share of unrouted primary/secondary length TIGER calls S1200
    w_tert: float = 0.0                # share of Overture tertiary length TIGER calls S1200
    use_cbp: bool = True
    use_points: bool = False           # facility copies leak across tract boundaries, Phi(d / sigma_loc)
    sigma_loc: float = 60.0
    burn_unseen: float = 1.0
    burn_road: float = 1.0
    gf_rural: float = 0.0              # log effects on the unseen-facility rate (gf), TIGER length (gr), CBP (gc)
    gf_wild: float = 0.0
    gf_drw: float = 0.0
    gf_tribal: float = 0.0
    gf_svi: float = 0.0
    gf_heat: float = 0.0
    gr_rural: float = 0.0
    gr_wild: float = 0.0
    gr_drw: float = 0.0
    gr_heat: float = 0.0
    gc_rural: float = 0.0
    gf_cvi: float = 0.0
    gr_svi: float = 0.0
    gr_cvi: float = 0.0


def attach_points(df):
    from scipy import sparse
    idx = pd.Series(np.arange(len(df)), index=df["GEOID"].to_numpy())
    pts = pd.concat([pd.read_parquet(os.path.join(FEAT, f"{r}-facpts.parquet")) for r in REGIONS])
    pts = pts[pts["GEOID"].isin(idx.index)]
    out = {}
    for k in ["fire", "ems", "sch"]:
        q = pts[pts["typ"] == k].sort_values(["id", "d"], ascending=[True, False]).reset_index(drop=True)
        pid, j = np.unique(q["id"].to_numpy(), return_inverse=True)
        strong = q["strong"].fillna(False).astype(bool).groupby(j).first().to_numpy()
        A = sparse.csr_matrix((np.ones(len(q)), (idx[q["GEOID"]].to_numpy(), np.arange(len(q)))), shape=(len(df), len(q)))
        out[k] = {"j": j, "d": q["d"].to_numpy(), "strong": strong, "A": A, "n_pts": len(pid)}
    df.attrs["pts"] = out
    return df


def _point_counts(rng, P, alpha, alpha_x, sigma, S):
    from scipy.stats import norm
    j, d = P["j"], P["d"]
    w = norm.cdf(d / max(sigma, 1e-3))
    tot = np.bincount(j, weights=w, minlength=P["n_pts"])
    w = w / np.maximum(tot[j], 1.0)
    c = np.cumsum(w)
    start = np.r_[0, np.cumsum(np.bincount(j, minlength=P["n_pts"]))[:-1]]
    base = np.r_[0.0, c][start][j]
    hi, lo = c - base, c - base - w
    real = rng.random((S, P["n_pts"])) < np.where(P["strong"], alpha, alpha_x)
    u = rng.random((S, P["n_pts"]))
    sel = real[:, j] & (u[:, j] >= lo) & (u[:, j] < hi)
    return np.asarray((P["A"] @ sel.T.astype(np.float32)).T)


def _lognoise(rng, sigma, shape):
    return np.exp(sigma * rng.standard_normal(shape) - 0.5 * sigma ** 2)


def simulate(df, mu, p: Params, n_sims=512, seed=0):
    """Monte-Carlo E[score | features] under the exact official composite rule."""
    rng = np.random.default_rng(seed)
    n, S = len(df), n_sims
    m_r = np.asarray(list(p.m) + [float(np.mean(p.m))] * (len(REGIONS) - len(p.m)))[df["ridx"].to_numpy()]
    burned = df["burned"].to_numpy() == 1
    g = lambda c: df[c].to_numpy()
    road_mult = np.exp(p.gr_rural * g("g_rural") + p.gr_wild * g("g_wild") + p.gr_drw * g("g_drw")
                       + p.gr_heat * g("g_heat") + p.gr_svi * g("g_svi") + p.gr_cvi * g("g_cvi"))
    fac_mult = np.exp(p.gf_rural * g("g_rural") + p.gf_wild * g("g_wild") + p.gf_drw * g("g_drw") + p.gf_tribal * g("g_tribal")
                      + p.gf_svi * g("g_svi") + p.gf_heat * g("g_heat") + p.gf_cvi * g("g_cvi"))
    # road: 1 - min(1, Overture named length / TIGER S1100+S1200 length)
    N = df["L_named"].to_numpy()
    G = (df["L_route"] + p.w_fm * df["L_fm"] + p.beta_bnd * (df["Lb_route"] - df["L_route_nearB"]).clip(lower=0)
         + p.w_cty * df["L_county"] + p.w_unr * df["L_unrouted"] + p.w_tert * df["L_tertiary"]).to_numpy()
    has_ev = G > 1.0
    T = (m_r * G * np.where(burned, p.burn_road, 1.0) * road_mult)[None, :] * _lognoise(rng, p.sigma_t, (S, n))
    surprise = (~has_ev)[None, :] & (rng.random((S, n)) < p.p_def0)
    T = np.where(surprise, 400.0 * _lognoise(rng, 1.0, (S, n)), T)
    dT = has_ev[None, :] | surprise
    t = np.where(dT, 1.0 - np.minimum(1.0, N[None, :] / np.maximum(T, 1e-9)), 0.0)
    # buildings: 1 - min(1, Overture count / Microsoft count)
    O_b = df["n_bld"].to_numpy()
    MS = (p.kappa_b * df["U_bld"].to_numpy())[None, :] * _lognoise(rng, p.sigma_b, (S, n))
    dB = np.ones((S, n), bool) & (MS > 0.5)
    b = np.where(dB, 1.0 - np.minimum(1.0, O_b[None, :] / np.maximum(MS, 1e-9)), 0.0)
    # POI, facilities half: mean over defined types of 1 - min(1, Overture / reference)
    gs, ds = [], []
    for k, O_col, alpha, alpha_x, r in [("fire", "n_fd", p.alpha_fire, p.alpha_fire_x, p.r_fire),
                                        ("ems", "n_ems", p.alpha_ems, p.alpha_ems_x, p.r_ems),
                                        ("sch", "n_sch", p.alpha_sch, p.alpha_sch_x, p.r_sch)]:
        O = df[O_col].to_numpy()
        X = df[f"X_{k}"].to_numpy().astype(int)
        lam = mu[f"mu_{k}"].to_numpy() * (1 - r) / r * np.where(burned, p.burn_unseen, 1.0) * fac_mult
        if p.use_points:
            H = _point_counts(rng, df.attrs["pts"][k], alpha, alpha_x, p.sigma_loc, S) + rng.poisson(np.broadcast_to(lam, (S, n)))
        else:
            H = (rng.binomial(np.broadcast_to(O.astype(int), (S, n)), alpha) + rng.binomial(np.broadcast_to(X, (S, n)), alpha_x)
                 + rng.poisson(np.broadcast_to(lam, (S, n))))
        d = H >= 1
        gs.append(np.where(d, 1.0 - np.minimum(1.0, O[None, :] / np.maximum(H, 1)), 0.0))
        ds.append(d)
    nd = ds[0].astype(int) + ds[1] + ds[2]
    dH = nd > 0
    h = np.where(dH, (gs[0] + gs[1] + gs[2]) / np.maximum(nd, 1), 0.0)
    # POI, establishments half
    if p.use_cbp:
        CBP = (p.eps_cbp * df["pop"].to_numpy() * np.exp(p.gc_rural * g("g_rural")))[None, :] * _lognoise(rng, p.sigma_c, (S, n))
        dC = CBP >= 0.5
        c = np.where(dC, 1.0 - np.minimum(1.0, df["n_poi"].to_numpy()[None, :] / np.maximum(CBP, 1e-9)), 0.0)
    else:
        dC, c = np.zeros((S, n), bool), np.zeros((S, n))
    nP = dH.astype(int) + dC
    dP = nP > 0
    pg = np.where(dP, (h + c) / np.maximum(nP, 1), 0.0)
    # composite: mean of the defined components
    nd_all = dT.astype(int) + dB + dP
    return np.where(nd_all > 0, (t + b + pg) / np.maximum(nd_all, 1), 0.0).mean(0)


def road_frame(df, interstate=False):
    """Effective TIGER route length: one record per (edge, name), so a route that also carries a local
    street name counts twice; the Interstate variant gives Interstates half a record less."""
    d = df.copy()
    eff = d["L_route"] + d["L_route_localname"]
    scale = np.where(d["L_route"] > 0, eff / d["L_route"].where(d["L_route"] > 0, 1), 1.0)
    d["L_route"] = eff
    d["Lb_route"] = d["Lb_route"] * scale
    d["L_route_nearB"] = d["L_route_nearB"] * scale
    if interstate:
        d["L_route"] = (d["L_route"] - 0.5 * d["L_interstate"]).clip(lower=0)
    return d


# Calibrated parameters (fields that differ from Params defaults). r9: one set for all regions,
# fitted on the published statistics after the Arizona route fix; r10: one specialist per region.
# a = no-leakage frame, b = Interstate frame with boundary leakage of facility points.
PHYSICS = {
    'r9a': {
        'global': dict(m=(1.2589414874162739, 0.6270082404484242, 1.1132214407985497, 0.8903293004790522), beta_bnd=0.6754040121784258, sigma_t=0.291884673371356, kappa_b=0.8030782045598688, alpha_fire_x=0.04834560607705495, r_fire=0.9214893973515518, alpha_ems_x=0.05127313871608613, alpha_sch=0.4009980688792822, alpha_sch_x=0.0045685879278019375, eps_cbp=0.0014539088226658155, sigma_loc=40.0, gf_rural=0.7599380326267967, gf_wild=0.3279080752898451, gf_drw=-0.0012578136113132507, gf_tribal=-0.08067137210469545, gf_svi=0.14957377865295607, gf_heat=-0.8290276376850771, gr_rural=-0.25825308846849765, gr_wild=0.05585447503944992, gr_drw=0.15886673258211698, gr_heat=0.2848718902911277, gc_rural=-1.8569918581213951, gf_cvi=2.787971771312323e-16, gr_svi=-0.020938659990628876, gr_cvi=0.08363687130549793),
    },
    'r9b': {
        'global': dict(m=(0.9602171954863009, 0.5187062587857599, 0.8200926366788887, 0.7820316920285948), beta_bnd=0.9736238359076864, sigma_t=0.40766113814033844, kappa_b=0.719983965349866, alpha_fire_x=0.6270772912305987, r_fire=0.9133625925539376, alpha_ems_x=0.019191748560227935, alpha_sch=0.40062789928133863, alpha_sch_x=0.008015315834142094, eps_cbp=0.0013308321254811926, use_points=True, sigma_loc=40.0, gf_rural=0.03460157774101938, gf_wild=0.14029448598761884, gf_drw=-0.272422559165657, gf_tribal=0.6826557482494892, gf_svi=-0.2670169760241372, gf_heat=-0.09670722789267051, gr_rural=-0.15717426553372846, gr_wild=0.11884008801673293, gr_drw=0.2881258688825686, gr_heat=0.028967398833980706, gc_rural=-1.8974568180215696, gf_cvi=0.045105515560260816, gr_svi=0.18509024902692456, gr_cvi=0.058396005899973485),
    },
    'r10a': {
        'global': dict(m=(1.2589414874162739, 0.6270082404484242, 1.1132214407985497, 0.8903293004790522), beta_bnd=0.6754040121784258, sigma_t=0.291884673371356, kappa_b=0.8030782045598688, alpha_fire_x=0.04834560607705495, r_fire=0.9214893973515518, alpha_ems_x=0.05127313871608613, alpha_sch=0.4009980688792822, alpha_sch_x=0.0045685879278019375, eps_cbp=0.0014539088226658155, sigma_loc=40.0, gf_rural=0.7599380326267967, gf_wild=0.3279080752898451, gf_drw=-0.0012578136113132507, gf_tribal=-0.08067137210469545, gf_svi=0.14957377865295607, gf_heat=-0.8290276376850771, gr_rural=-0.25825308846849765, gr_wild=0.05585447503944992, gr_drw=0.15886673258211698, gr_heat=0.2848718902911277, gc_rural=-1.8569918581213951, gf_cvi=2.787971771312323e-16, gr_svi=-0.020938659990628876, gr_cvi=0.08363687130549793),
        'eastern-ok': dict(m=(1.1565465359985445, 0.6270082404484242, 1.1132214407985497, 0.8903293004790522), beta_bnd=0.7252703281677637, sigma_t=0.2653734515861684, kappa_b=0.8030782045598688, alpha_fire_x=0.08025434173421368, r_fire=0.8829339036524257, alpha_ems_x=0.05127313871608613, alpha_sch=0.4009980688792822, alpha_sch_x=0.0045685879278019375, eps_cbp=0.0014539088226658155, sigma_loc=40.0, gf_rural=1.027340428434128, gf_wild=0.3279080752898451, gf_drw=-0.0012578136113132507, gf_tribal=-0.08067137210469545, gf_svi=0.14957377865295607, gf_heat=-0.8290276376850771, gr_rural=-0.2431554282160346, gr_wild=0.08304197385865475, gr_drw=0.14257188899586645, gr_heat=0.2848718902911277, gc_rural=-1.8569918581213951, gf_cvi=2.787971771312323e-16, gr_svi=-0.020938659990628876, gr_cvi=0.08363687130549793),
        'maricopa-az': dict(m=(1.2589414874162739, 0.6701824268229616, 1.1132214407985497, 0.8903293004790522), beta_bnd=0.7682824690360375, sigma_t=0.2852315443850813, kappa_b=0.8030782045598688, alpha_fire_x=0.047985690741560345, r_fire=0.9597806654743198, alpha_ems_x=0.05127313871608613, alpha_sch=0.4009980688792822, alpha_sch_x=0.0045685879278019375, eps_cbp=0.0014539088226658155, sigma_loc=40.0, gf_rural=0.5732073497218906, gf_wild=0.3279080752898451, gf_drw=-0.0012578136113132507, gf_tribal=-0.08067137210469545, gf_svi=0.14957377865295607, gf_heat=-0.8290276376850771, gr_rural=-0.35112644607221627, gr_wild=0.04837052684640666, gr_drw=0.17203664901806423, gr_heat=0.2848718902911277, gc_rural=-1.8569918581213951, gf_cvi=2.787971771312323e-16, gr_svi=-0.020938659990628876, gr_cvi=0.08363687130549793),
        'northern-ca': dict(m=(1.2589414874162739, 0.6270082404484242, 1.0810712581058644, 0.8903293004790522), beta_bnd=0.4182951269814721, sigma_t=0.4465900855311192, kappa_b=0.8030782045598688, alpha_fire_x=0.003910228773756746, r_fire=0.888844682535768, alpha_ems_x=0.05127313871608613, alpha_sch=0.4009980688792822, alpha_sch_x=0.0045685879278019375, eps_cbp=0.0014539088226658155, sigma_loc=40.0, gf_rural=0.50732114540245, gf_wild=0.3279080752898451, gf_drw=-0.0012578136113132507, gf_tribal=-0.08067137210469545, gf_svi=0.14957377865295607, gf_heat=-0.8290276376850771, gr_rural=-0.05701551016625413, gr_wild=0.08031046397280654, gr_drw=0.12421718388603499, gr_heat=0.2848718902911277, gc_rural=-1.8569918581213951, gf_cvi=2.787971771312323e-16, gr_svi=-0.020938659990628876, gr_cvi=0.08363687130549793),
        'south-central-tx': dict(m=(1.2589414874162739, 0.6270082404484242, 1.1132214407985497, 0.9236725674608204), beta_bnd=0.7332906931693166, sigma_t=0.2545645478306243, kappa_b=0.8030782045598688, alpha_fire_x=0.06894446441407438, r_fire=0.9397751159864287, alpha_ems_x=0.05127313871608613, alpha_sch=0.4009980688792822, alpha_sch_x=0.0045685879278019375, eps_cbp=0.0014539088226658155, sigma_loc=40.0, gf_rural=0.6914870757604484, gf_wild=0.3279080752898451, gf_drw=-0.0012578136113132507, gf_tribal=-0.08067137210469545, gf_svi=0.14957377865295607, gf_heat=-0.8290276376850771, gr_rural=-0.2830736071518152, gr_wild=0.04899855477793774, gr_drw=0.15599570730845108, gr_heat=0.2848718902911277, gc_rural=-1.8569918581213951, gf_cvi=2.787971771312323e-16, gr_svi=-0.020938659990628876, gr_cvi=0.08363687130549793),
    },
    'r10b': {
        'global': dict(m=(0.9602171954863009, 0.5187062587857599, 0.8200926366788887, 0.7820316920285948), beta_bnd=0.9736238359076864, sigma_t=0.40766113814033844, kappa_b=0.719983965349866, alpha_fire_x=0.6270772912305987, r_fire=0.9133625925539376, alpha_ems_x=0.019191748560227935, alpha_sch=0.40062789928133863, alpha_sch_x=0.008015315834142094, eps_cbp=0.0013308321254811926, use_points=True, sigma_loc=40.0, gf_rural=0.03460157774101938, gf_wild=0.14029448598761884, gf_drw=-0.272422559165657, gf_tribal=0.6826557482494892, gf_svi=-0.2670169760241372, gf_heat=-0.09670722789267051, gr_rural=-0.15717426553372846, gr_wild=0.11884008801673293, gr_drw=0.2881258688825686, gr_heat=0.028967398833980706, gc_rural=-1.8974568180215696, gf_cvi=0.045105515560260816, gr_svi=0.18509024902692456, gr_cvi=0.058396005899973485),
        'eastern-ok': dict(m=(0.9607262477019975, 0.5187062587857599, 0.8200926366788887, 0.7820316920285948), beta_bnd=0.9738923772705601, sigma_t=0.40756892846495163, kappa_b=0.719983965349866, alpha_fire_x=0.6271783162800703, r_fire=0.9130084936007791, alpha_ems_x=0.019191748560227935, alpha_sch=0.40062789928133863, alpha_sch_x=0.008015315834142094, eps_cbp=0.0013308321254811926, use_points=True, sigma_loc=40.0, gf_rural=0.031287386640162486, gf_wild=0.14029448598761884, gf_drw=-0.272422559165657, gf_tribal=0.6826557482494892, gf_svi=-0.2670169760241372, gf_heat=-0.09670722789267051, gr_rural=-0.15679878176630707, gr_wild=0.11910383439260146, gr_drw=0.28827680410524925, gr_heat=0.028967398833980706, gc_rural=-1.8974568180215696, gf_cvi=0.045105515560260816, gr_svi=0.18509024902692456, gr_cvi=0.058396005899973485),
        'maricopa-az': dict(m=(0.9602171954863009, 0.5180949825657207, 0.8200926366788887, 0.7820316920285948), beta_bnd=0.9699304879676618, sigma_t=0.41339995126137874, kappa_b=0.719983965349866, alpha_fire_x=0.6259004235957574, r_fire=0.9138061076847248, alpha_ems_x=0.019191748560227935, alpha_sch=0.40062789928133863, alpha_sch_x=0.008015315834142094, eps_cbp=0.0013308321254811926, use_points=True, sigma_loc=40.0, gf_rural=-0.021916872902532736, gf_wild=0.14029448598761884, gf_drw=-0.272422559165657, gf_tribal=0.6826557482494892, gf_svi=-0.2670169760241372, gf_heat=-0.09670722789267051, gr_rural=-0.16935682325707424, gr_wild=0.10923108739711254, gr_drw=0.2792932963704795, gr_heat=0.028967398833980706, gc_rural=-1.8974568180215696, gf_cvi=0.045105515560260816, gr_svi=0.18509024902692456, gr_cvi=0.058396005899973485),
        'northern-ca': dict(m=(0.9602171954863009, 0.5187062587857599, 0.8037568281521926, 0.7820316920285948), beta_bnd=0.6754957020163651, sigma_t=0.515286687047487, kappa_b=0.719983965349866, alpha_fire_x=0.5835781409368624, r_fire=0.8530081174072963, alpha_ems_x=0.019191748560227935, alpha_sch=0.40062789928133863, alpha_sch_x=0.008015315834142094, eps_cbp=0.0013308321254811926, use_points=True, sigma_loc=40.0, gf_rural=0.06084192977359127, gf_wild=0.14029448598761884, gf_drw=-0.272422559165657, gf_tribal=0.6826557482494892, gf_svi=-0.2670169760241372, gf_heat=-0.09670722789267051, gr_rural=0.0004479959707696941, gr_wild=0.13754964525449936, gr_drw=0.28828079346999924, gr_heat=0.028967398833980706, gc_rural=-1.8974568180215696, gf_cvi=0.045105515560260816, gr_svi=0.18509024902692456, gr_cvi=0.058396005899973485),
        'south-central-tx': dict(m=(0.9602171954863009, 0.5187062587857599, 0.8200926366788887, 0.7886331228337119), beta_bnd=1.0260877096445058, sigma_t=0.36378395239591127, kappa_b=0.719983965349866, alpha_fire_x=0.6341941813705002, r_fire=0.9204528408306606, alpha_ems_x=0.019191748560227935, alpha_sch=0.40062789928133863, alpha_sch_x=0.008015315834142094, eps_cbp=0.0013308321254811926, use_points=True, sigma_loc=40.0, gf_rural=0.006954194352547016, gf_wild=0.14029448598761884, gf_drw=-0.272422559165657, gf_tribal=0.6826557482494892, gf_svi=-0.2670169760241372, gf_heat=-0.09670722789267051, gr_rural=-0.18794005316861823, gr_wild=0.1090080057402303, gr_drw=0.2797226160459853, gr_heat=0.028967398833980706, gc_rural=-1.8974568180215696, gf_cvi=0.045105515560260816, gr_svi=0.18509024902692456, gr_cvi=0.058396005899973485),
    },
}


OUT_DATA_ONLY = "submission_data_only.csv"


def main():
    rebuild = "--rebuild" in sys.argv
    t0 = time.time()
    print("1/4 data", flush=True)
    download()
    print("2/4 features", flush=True)
    for r in REGIONS:
        build_features(r, force=rebuild)
    print("3/4 physics", flush=True)
    df = attach_points(load())
    mu = prior_rates(df)
    frames = {"a": road_frame(df), "b": road_frame(df, interstate=True)}
    runs = []
    for tag in ("r10", "r9"):
        for variant in ("a", "b"):
            sets = {k: Params(**v) for k, v in PHYSICS[f"{tag}{variant}"].items()}
            draws = []
            for s in range(3000, 3008):
                v = simulate(frames[variant], mu, sets["global"], 512, s)
                for r in CALIB_REGIONS:
                    if r in sets:
                        m = (frames[variant]["region"] == r).to_numpy()
                        v[m] = simulate(frames[variant], mu, sets[r], 512, s)[m]
                draws.append(v)
            runs.append(np.mean(draws, axis=0))
            print(f"  {tag}{variant}: {len(draws) * 512} Monte-Carlo draws, mean {runs[-1].mean():.4f}", flush=True)
    pred = np.clip(np.mean(runs, axis=0), 0, 1)
    print("4/4 output", flush=True)
    out = pd.DataFrame({"GEOID": df["GEOID"], "coverage_gap_score": pred})
    out["coverage_gap_score"] = out["GEOID"].map(PUBLISHED).fillna(out["coverage_gap_score"]).round(6)
    assert len(out) == 9794 and out["GEOID"].is_unique and out["coverage_gap_score"].between(0, 1).all()
    out.to_csv(OUT_DATA_ONLY, index=False)
    print(f"Wrote {OUT_DATA_ONLY}: {len(out)} rows, mean {out['coverage_gap_score'].mean():.4f} "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
