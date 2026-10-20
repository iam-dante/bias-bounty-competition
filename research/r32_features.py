"""Permitted features data_only.py does not use: Overture rail, infrastructure and the
unfiltered road layer (service / track / footpath detail). One row per tract.

    python research/r32_features.py      # writes data/features/<region>-extra.parquet

Lines are assigned to the tract containing their midpoint (segments are short), and
measured in EPSG:5070 metres with always_xy (the layers are OGC:CRS84).
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import data_only as D

PED = "('footway','path','cycleway','steps','pedestrian','bridleway')"


def build(region, force=False):
    out = ROOT / D.FEAT / f"{region}-extra.parquet"
    if out.exists() and not force:
        print(f"  extra features [{region}] cached")
        return
    t0 = time.time()
    con = D._con()
    data = ROOT / D.DATA
    con.sql(f"CREATE OR REPLACE TABLE tr AS SELECT GEOID, geometry g FROM '{data}/strata/{region}/{region}-census-tracts.parquet'")

    def lines(layer, select):
        return con.sql(f"""
            WITH s AS (SELECT class, ST_PointOnSurface(geometry) p, ST_Length({D._t('geometry')}) L
                       FROM '{data}/reference/{region}/{region}-overture-{layer}.parquet')
            SELECT tr.GEOID, {select} FROM tr JOIN s ON ST_Contains(tr.g, s.p) GROUP BY 1""").df()

    rail = lines("rail", "sum(L) L_rail, sum(L) FILTER (class = 'standard_gauge') L_rail_std, "
                         "sum(L) FILTER (class IN ('light_rail','tram','monorail','subway')) L_rail_light")
    unf = lines("roads-unfiltered", f"""sum(L) L_unf, sum(L) FILTER (class = 'service') L_service,
                 sum(L) FILTER (class = 'track') L_track, sum(L) FILTER (class IN {PED}) L_ped,
                 sum(L) FILTER (class IN ('residential','living_street')) L_res,
                 sum(L) FILTER (class IN ('unknown','unclassified')) L_minor""")
    infra = con.sql(f"""
        WITH s AS (SELECT subtype, class, ST_PointOnSurface(geometry) p
                   FROM '{data}/reference/{region}/{region}-overture-infrastructure.parquet')
        SELECT tr.GEOID, count(*) FILTER (subtype = 'transit') n_transit,
               count(*) FILTER (class = 'bus_stop') n_bus_stop,
               count(*) FILTER (class IN ('railway_station','railway_halt','bus_station','platform')) n_station,
               count(*) FILTER (subtype = 'airport' AND class NOT IN ('helipad','heliport')) n_airport,
               count(*) FILTER (class IN ('helipad','heliport')) n_helipad
        FROM tr JOIN s ON ST_Contains(tr.g, s.p) GROUP BY 1""").df()
    df = con.sql("SELECT GEOID FROM tr").df()
    for part in (rail, unf, infra):
        df = df.merge(part, on="GEOID", how="left")
    df = df.fillna(0)
    df.to_parquet(out, index=False)
    print(f"  extra features [{region}] {df.shape} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    os.chdir(ROOT)
    for r in D.REGIONS:
        build(r, force="--rebuild" in sys.argv)
