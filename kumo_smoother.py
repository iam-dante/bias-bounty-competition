# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "pyarrow", "duckdb", "torch>=2.7", "structured-data-models==0.2.1"]
# ///
"""Rebuild the Kumo Tabular smoother stored in dataset/kumo_inputs.npz (key kumo_smooth).

Run: uv run kumo_smoother.py [--data data] [--out kumo_large_smooth.csv]

1. DATA      downloads the provided layers it needs from the public Source Cooperative bucket (about 360 MB, no
             account): Overture POIs, infrastructure and rail, ACS housing, census tracts, strata tables.
2. FEATURES  per-tract counts and shares (POI sources, categories, attribute completeness, infrastructure, rail
             length) plus the permitted numeric strata columns; footprint columns (usfs_BuildingCount,
             usfs_HUCount, carbonplan_buildings) and metadata columns are excluded. 96 columns are kept: every
             Overture-side column, then strata columns by |Spearman r| with U1, skipping near-duplicates.
3. KUMO      NVIDIA Kumo Tabular (large, open weights nvidia/Kumo-Tabular) in-context regression of U1 on the
             features, cross-fitted over 10 folds so no tract sees its own U1 value. Output = predictive mean.

The stored output came from a Kaggle Tesla T4 run in float16 (cross-fit R2 0.567 against U1, 49 s). Kumo's
ensemble sampling is unseeded, so a rerun gives the same direction but not the same bytes: a full Kaggle GPU rerun
of this script from the raw layers had correlation 0.996 with the stored vector and cosine 0.981 on the part
outside all measured directions (the part q094 steps along); a CPU run of the medium model had cosine 0.969.
train.py therefore uses the stored vector, checked by SHA256.
"""
import argparse
import io
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DATA_URL = "https://data.source.coop/humane-intelligence/bias-bounty-mapping-equity-challenge"
REGIONS = ["eastern-ok", "maricopa-az", "northern-ca", "south-central-tx", "eastern-wa"]
LAYERS = [("reference", "sample-submission.csv"), ("reference", "census-acs-housing.parquet"),
          ("reference", "overture-pois.parquet"), ("reference", "overture-infrastructure.parquet"),
          ("reference", "overture-rail.parquet"), ("strata", "strata-tract-table.parquet"),
          ("strata", "census-tracts.parquet")]
FOOTPRINT = ("usfs_BuildingCount", "usfs_HUCount", "carbonplan_buildings")
META = ("_year", "_year_min", "_year_max", "_vintage", "_version", "_edition", "_extent", "_metric", "_threshold",
        "_window_start", "_window_end", "_record_start", "_record_end", "_calibration")
ALB = "ST_Transform({}, 'EPSG:4326', 'EPSG:5070', always_xy := true)"
FOLDS, MAX_COLS, CHUNK = 10, 96, 128


def log(message):
    print(message, flush=True)


def download(data):
    for r in REGIONS:
        for kind, name in LAYERS:
            path = Path(data) / kind / r / f"{r}-{name}"
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            # the proxy rejects Python's default User-Agent with 403
            request = urllib.request.Request(f"{DATA_URL}/{kind}/{r}/{r}-{name}", headers={"User-Agent": "kumo-smoother"})
            with urllib.request.urlopen(request, timeout=120) as source, open(f"{path}.part", "wb") as target:
                while chunk := source.read(1 << 22):
                    target.write(chunk)
            os.replace(f"{path}.part", path)
            log(f"  downloaded {path}")


def region_table(con, data, r):
    d = f"{data}/reference/{r}/{r}"
    con.sql(f"""CREATE OR REPLACE TABLE tr AS SELECT GEOID, geometry g, ST_Area({ALB.format('geometry')}) area_m2
                FROM '{data}/strata/{r}/{r}-census-tracts.parquet'""")
    poi = con.sql(f"""
        WITH p AS (SELECT geometry pt, categories.primary cat, basic_category bc, confidence conf, sources[1].dataset ds,
                          brand IS NOT NULL has_brand, websites IS NOT NULL has_web, phones IS NOT NULL has_phone,
                          addresses IS NOT NULL has_addr, operating_status os
                   FROM '{d}-overture-pois.parquet')
        SELECT tr.GEOID, count(*) n_poi,
          count(*) FILTER (ds = 'meta') n_poi_meta, count(*) FILTER (ds = 'Microsoft') n_poi_msft,
          count(*) FILTER (ds = 'BrightQuery') n_poi_bq,
          avg(conf) poi_conf_mean, count(*) FILTER (conf < 0.5) n_poi_lowconf,
          count(*) FILTER (has_brand) n_poi_brand, count(*) FILTER (has_web) n_poi_web,
          count(*) FILTER (has_phone) n_poi_phone, count(*) FILTER (has_addr) n_poi_addr,
          count(*) FILTER (os IS NOT NULL AND os <> 'open') n_poi_closed,
          count(*) FILTER (cat = 'fire_department') n_fire, count(*) FILTER (cat = 'ambulance_and_ems_services') n_ems,
          count(*) FILTER (cat LIKE '%school%') n_school, count(*) FILTER (cat LIKE '%hospital%') n_hospital,
          count(*) FILTER (cat LIKE '%church%' OR cat LIKE '%religious%') n_church,
          count(*) FILTER (cat LIKE '%restaurant%') n_restaurant,
          count(*) FILTER (cat LIKE '%government%' OR cat LIKE '%police%') n_gov,
          count(DISTINCT bc) n_poi_kinds
        FROM tr JOIN p ON ST_Contains(tr.g, p.pt) GROUP BY 1""").df()
    infra = con.sql(f"""
        WITH s AS (SELECT subtype, ST_PointOnSurface(geometry) pt FROM '{d}-overture-infrastructure.parquet')
        SELECT tr.GEOID, count(*) n_infra, count(*) FILTER (subtype = 'transit') n_transit,
               count(*) FILTER (subtype = 'power') n_power, count(*) FILTER (subtype = 'communication') n_comm
        FROM tr JOIN s ON ST_Contains(tr.g, s.pt) GROUP BY 1""").df()
    rail = con.sql(f"""
        WITH s AS (SELECT ST_PointOnSurface(geometry) pt, ST_Length({ALB.format('geometry')}) L
                   FROM '{d}-overture-rail.parquet')
        SELECT tr.GEOID, sum(L) L_rail FROM tr JOIN s ON ST_Contains(tr.g, s.pt) GROUP BY 1""").df()
    area = con.sql("SELECT GEOID, area_m2 FROM tr").df()
    acs = pd.read_parquet(f"{d}-census-acs-housing.parquet", columns=["GEOID", "housing_units"])
    strata = pd.read_parquet(f"{data}/strata/{r}/{r}-strata-tract-table.parquet")
    f = pd.read_csv(f"{d}-sample-submission.csv", dtype={"GEOID": str})[["GEOID"]]
    for part in (area, poi, infra, rail, acs, strata):
        f = f.merge(part, on="GEOID", how="left")
    f["region"] = r
    return f


def build_features(data):
    import duckdb
    con = duckdb.connect()
    con.sql("INSTALL spatial; LOAD spatial; SET threads=8; SET memory_limit='8GB';")
    df = pd.concat([region_table(con, data, r) for r in REGIONS], ignore_index=True)
    counts = [c for c in df.columns if c.startswith("n_")] + ["L_rail", "housing_units"]
    df[counts] = df[counts].fillna(0)
    km2 = df["area_m2"].to_numpy(float) / 1e6
    pop = df["pop_total"].fillna(0).clip(lower=0).to_numpy(float)
    X = {"GEOID": df["GEOID"], "region": df["region"],
         "log_area_km2": np.log(km2 + 0.01), "log_pop": np.log1p(pop),
         "log_pop_density": np.log1p(pop / (km2 + 0.01)), "log_housing": np.log1p(df["housing_units"])}
    for c in counts:
        if c != "housing_units":
            v = df[c].to_numpy(float) / (1000.0 if c == "L_rail" else 1.0)
            X[f"log_{c}"] = np.log1p(v)
            X[f"log_{c}_per_kpop"] = np.log1p(1000 * v / (pop + 1))
    n = df["n_poi"].to_numpy(float)
    for c in ("n_poi_meta", "n_poi_msft", "n_poi_bq", "n_poi_brand", "n_poi_web", "n_poi_phone", "n_poi_addr",
              "n_poi_lowconf"):
        X[f"share_{c[2:]}"] = df[c] / (n + 1)
    X["poi_conf_mean"] = df["poi_conf_mean"]
    X["urban"] = df["ur_class"].eq("Urban").astype(float)
    for k, r in enumerate(REGIONS[:4]):
        X[f"region_{k}"] = df["region"].eq(r).astype(float)
    for col in df.columns[df.columns.get_loc("STATEFP"):]:
        if col in X or col in {"ALAND", "pop_total"} or col.startswith(FOOTPRINT) or col.endswith(META) \
                or col.endswith("_covered") or col.startswith(("n_", "L_", "poi_", "area_", "housing")):
            continue
        v = df[col]
        if pd.api.types.is_bool_dtype(v):
            v = v.astype(float)
        if not pd.api.types.is_numeric_dtype(v):
            continue
        v = pd.to_numeric(v, errors="coerce").astype(float)
        if v.notna().sum() >= 100 and v.std() >= 1e-12:
            X[f"s_{col}"] = v
    return pd.DataFrame(X)


def select_columns(feat, y):
    rank = feat.rank(pct=True)
    yr = pd.Series(y).rank(pct=True)
    ovr = [c for c in feat.columns if not c.startswith("s_")]
    strata = sorted((c for c in feat.columns if c.startswith("s_")), key=lambda c: -abs(np.nan_to_num(rank[c].corr(yr))))
    keep = list(ovr)
    for c in strata:
        if len(keep) >= MAX_COLS:
            break
        if all(abs(np.nan_to_num(rank[c].corr(rank[k]))) <= 0.95 for k in keep):
            keep.append(c)
    return keep


def cross_fit(frame, size):
    import torch
    import sdm
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    table = sdm.TableTensor.from_pandas(frame, {c: "numerical" for c in frame.columns}, device=device)
    model = sdm.models.KumoTabular(task="regression", size=size, device=device)
    n = len(frame)
    fold = np.random.default_rng(0).permutation(n) % FOLDS
    pred = np.full(n, np.nan)
    t0 = time.time()
    for k in range(FOLDS):
        q, c = np.flatnonzero(fold == k), np.flatnonzero(fold != k)
        qi, ci = torch.as_tensor(q, device=device), torch.as_tensor(c, device=device)
        with torch.no_grad(), torch.autocast(device, dtype=torch.float16, enabled=device != "cpu"):
            model.fit(table[ci].drop_columns("target"), table[ci, "target"])
            for b in range(0, len(q), CHUNK):
                out = model.predict(table[qi[b:b + CHUNK]].drop_columns("target")).numerical.float().cpu().numpy()
                # [E, R, 999 quantiles] -> predictive mean
                pred[q[b:b + CHUNK]] = out.reshape(-1, len(q[b:b + CHUNK]), out.shape[-1]).mean(axis=(0, 2))
        model.clear()
        log(f"  [{size}] fold {k + 1}/{FOLDS}: {time.time() - t0:.0f}s")
    return pred


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", type=Path, default=Path("kumo_large_smooth.csv"))
    parser.add_argument("--size", default="large", choices=["small", "medium", "large"])
    args = parser.parse_args()
    log("[1/3] Downloading the provided layers")
    download(args.data)
    log("[2/3] Building per-tract features")
    X = build_features(args.data)
    with np.load(Path(__file__).resolve().parent / "dataset" / "kumo_inputs.npz", allow_pickle=False) as z:
        names = z["span_names"].tolist()
        u1 = pd.Series(z["span_predictions"][:, names.index("tr_u1_000.csv")], index=z["geoid"])
    y = u1.loc[X["GEOID"]].to_numpy(float)
    feat = X.drop(columns=["GEOID", "region"])
    keep = select_columns(feat, y)
    log(f"  {len(keep)} columns from {feat.shape[1]} candidates")
    log(f"[3/3] Cross-fitting Kumo Tabular ({args.size})")
    frame = feat[keep].astype("float32").copy()
    frame["target"] = y.astype("float32")
    pred = cross_fit(frame, args.size)
    resid = y - pred
    log(f"  cross-fit R2 against U1 {1 - resid.var() / y.var():.4f}")
    pd.DataFrame({"GEOID": X["GEOID"], "coverage_gap_score": np.round(np.clip(pred, 0, 1), 6)}).to_csv(args.out, index=False)
    log(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
