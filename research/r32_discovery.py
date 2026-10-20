"""R32 / T4: Best Bias Discovery evidence. A coverage pattern outside the scorecard grid:
which upstream feed supplies a tract's Overture places.

Run from the repo root: python research/r32_discovery.py

Inputs are challenge data only (Overture places with their `sources`, strata tables, tract
names) plus two prediction vectors: R29 (our best file, public RMSE 0.062095, reproduces the
9 truth scorecard ratios within 0.011) and the leaderboard-free physics model (data_only.py).
Writes research/r32/discovery_*.csv.
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import solution as S

REGIONS = ["eastern-ok", "maricopa-az", "northern-ca", "south-central-tx"]
OUT = ROOT / "research/r32"
REGISTRY = "('BrightQuery','AllThePlaces','DAC','PinMeTo')"
FIRE = "categories.primary = 'fire_department'"
EMS = "categories.primary = 'ambulance_and_ems_services'"
SCHOOL = "categories.primary IN ('elementary_school','middle_school','high_school','school','private_school','public_school')"
HEALTH = ("categories.primary IN ('health_and_medical','public_health_clinic','hospital','home_health_care','dialysis_clinic',"
          "'urgent_care_clinic','pharmacy','medical_center','emergency_room','doctor')")


def source_counts(con, region):
    d = ROOT / "data"
    return con.sql(f"""
        WITH tr AS (SELECT GEOID, NAMELSAD, COUNTYFP, STATEFP, geometry g FROM '{d}/strata/{region}/{region}-census-tracts.parquet'),
             p AS (SELECT geometry, sources[1].dataset ds, categories FROM '{d}/reference/{region}/{region}-overture-pois.parquet')
        SELECT tr.GEOID, any_value(tr.NAMELSAD) AS tract_name, any_value(tr.STATEFP || tr.COUNTYFP) AS county, count(p.ds) n_places,
               count(*) FILTER (p.ds = 'meta') n_meta, count(*) FILTER (p.ds IN {REGISTRY}) n_registry,
               count(*) FILTER (p.ds IN ('Microsoft','Foursquare')) n_msft_fsq,
               count(*) FILTER ({HEALTH.replace('categories', 'p.categories')}) n_health,
               count(*) FILTER ({HEALTH.replace('categories', 'p.categories')} AND p.ds IN {REGISTRY}) n_health_registry,
               count(*) FILTER ({FIRE.replace('categories', 'p.categories')}) n_fire,
               count(*) FILTER ({FIRE.replace('categories', 'p.categories')} AND p.ds = 'meta') n_fire_meta,
               count(*) FILTER ({EMS.replace('categories', 'p.categories')}) n_ems,
               count(*) FILTER ({EMS.replace('categories', 'p.categories')} AND p.ds = 'meta') n_ems_meta,
               count(*) FILTER ({SCHOOL.replace('categories', 'p.categories')}) n_school,
               count(*) FILTER ({SCHOOL.replace('categories', 'p.categories')} AND p.ds = 'meta') n_school_meta
        FROM tr LEFT JOIN p ON ST_Contains(tr.g, p.geometry) GROUP BY tr.GEOID""").df().assign(region=region)


def main():
    con = duckdb.connect()
    con.sql("INSTALL spatial; LOAD spatial; SET threads=4;")
    df = pd.concat([source_counts(con, r) for r in REGIONS], ignore_index=True)
    lb = S.Leaderboard()
    gid = lb.geoid.astype(str)
    pred = pd.DataFrame({"GEOID": gid, "r29": lb.vec("r29_correlated_scorecard.csv"), "graded": lb.graded})
    phys = pd.read_csv(ROOT / "submission_data_only.csv", dtype={"GEOID": str}).rename(columns={"coverage_gap_score": "physics"})
    st = pd.concat([pd.read_parquet(ROOT / f"data/strata/{r}/{r}-strata-tract-table.parquet",
                                    columns=["GEOID", "ur_class", "pop_total", "svi_overall", "tribal_any"]) for r in REGIONS])
    df = df.merge(pred, on="GEOID").merge(phys, on="GEOID").merge(st.drop_duplicates("GEOID"), on="GEOID")
    df = df[df.graded].copy()
    df["meta_share"] = df.n_meta / df.n_places.clip(lower=1)
    df["registry_per_1k"] = 1000 * df.n_registry / df.pop_total.clip(lower=100)
    df.to_csv(OUT / "discovery_tracts.csv", index=False)

    # 1. Facility categories by upstream source (exact Overture counts, no model)
    fac = pd.DataFrame({k: [df[f"n_{k}"].sum(), df[f"n_{k}_meta"].sum()] for k in ["fire", "ems", "school"]},
                       index=["places", "from_meta"]).T
    fac["meta_share"] = fac.from_meta / fac.places
    fac.loc["health"] = [df.n_health.sum(), np.nan, np.nan]
    fac.loc["health", "registry_share"] = df.n_health_registry.sum() / df.n_health.sum()
    print("Facility categories by upstream source (graded tracts):\n" + fac.round(3).to_string())

    # 2. Within region x urban/rural cells: top vs bottom quartile of the Meta-only share
    rows = []
    for (r, u), c in df.groupby(["region", "ur_class"]):
        q1, q3 = c.meta_share.quantile([0.25, 0.75])
        hi, lo = c[c.meta_share >= q3], c[c.meta_share <= q1]
        rows.append(dict(region=r, ur_class=u, n=len(c), meta_share_q1=q1, meta_share_q3=q3,
                         r29_gap_hi=hi.r29.mean(), r29_gap_lo=lo.r29.mean(), r29_ratio=hi.r29.mean() / lo.r29.mean(),
                         physics_ratio=hi.physics.mean() / lo.physics.mean(),
                         registry_per_1k_hi=hi.registry_per_1k.median(), registry_per_1k_lo=lo.registry_per_1k.median(),
                         health_per_1k_hi=1000 * hi.n_health.sum() / hi.pop_total.sum(),
                         health_per_1k_lo=1000 * lo.n_health.sum() / lo.pop_total.sum(),
                         svi_hi=hi.svi_overall.mean(), svi_lo=lo.svi_overall.mean()))
    cells = pd.DataFrame(rows)
    cells.to_csv(OUT / "discovery_cells.csv", index=False)
    print("\nRegion x urban/rural: tracts in the top vs bottom quartile of Meta-only place share\n"
          + cells.round(3).to_string())

    # 3. Same comparison inside SVI halves (urban tracts): not a restatement of the SVI stratum
    u = df[df.ur_class.eq("Urban")].copy()
    u["svi_high"] = u.svi_overall >= df.svi_overall.median()
    u["meta_q"] = u.groupby("region").meta_share.transform(lambda s: pd.qcut(s, 4, labels=False, duplicates="drop"))
    svi = u.groupby(["svi_high", "meta_q"]).r29.mean().unstack()
    print("\nUrban tracts, mean estimated gap by SVI half (rows) and within-region Meta-share quartile (cols):\n"
          + svi.round(4).to_string())
    svi.to_csv(OUT / "discovery_svi_control.csv")

    # 4. Named examples: urban, top-quartile Meta share, largest estimated gap
    top = u[u.meta_q == 3].nlargest(12, "r29")[["GEOID", "tract_name", "county", "region", "pop_total", "n_places", "meta_share",
                                               "n_registry", "n_health", "r29", "physics"]]
    top.to_csv(OUT / "discovery_examples.csv", index=False)
    print("\nExamples:\n" + top.round(3).to_string())


if __name__ == "__main__":
    main()
