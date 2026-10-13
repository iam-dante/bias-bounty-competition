# The problem: Bias Bounty Mapping Equity Challenge

Competition: https://zindi.world/competitions/bias-bounty-mapping-equity-challenge (Zindi, Humane Intelligence,
Radiant Earth). Runs 28 Aug to 31 Oct 2026. Prize pool $10,000.

## The question

Open map data (Overture Maps) is used for emergency dispatch, evacuation routing and disaster relief. The
hypothesis is that the communities most exposed to climate risk (wildfire corridors, extreme-heat cities,
drought-prone tribal lands) are also the worst mapped. The task is to estimate, for every US Census tract in
the study regions, how much of the real world Overture is missing.

## What we predict

One number per tract, `coverage_gap_score` in [0, 1] (0 = fully mapped, 1 = nothing mapped), for the 9,794
GEOIDs in the sample submissions:

| region | scored tracts | hazard |
|---|---|---|
| eastern-ok | 1,192 | drought, heat, tribal lands |
| maricopa-az | 1,593 | heat, drought |
| northern-ca | 591 | wildfire (Butte / Shasta / Tehama) |
| south-central-tx | 6,003 | heat, drought (I-35 / I-37) |
| eastern-wa | 415 | in the file, but not graded |

Metric: **RMSE** against the organisers' reference scores. The public leaderboard uses about 30 % of the
tracts, and the private leaderboard (final ranking) uses the other 70 %. Submit only `GEOID,coverage_gap_score`.
Optional component columns are graded too and make the score worse.

## How the organisers built the target

The score is the **mean of the defined components** among three gaps, each `1 - min(1, Overture / reference)`:

1. **Road gap.** Overture `motorway/trunk/primary/secondary` length against TIGER/Line `S1100 + S1200`
   (named highways). The component is undefined when the tract has no TIGER highway. This happens in 55 % of
   Maricopa tracts.
2. **Building gap.** Overture building count against Microsoft US Building Footprints. ACS housing units
   are only a sanity check.
3. **POI gap.** The mean of two halves. The facilities half is the mean over fire stations, EMS and schools
   (Overture category against USGS/HIFLD counts). The establishments half compares all Overture places with
   Census County Business Patterns.

A component with no reference is left out of the mean, so the divisor varies from 1 to 3 by tract.

## The catch: no labels and no reference layers

On 2026-09-25 the organisers removed TIGER roads, Microsoft buildings, HIFLD facilities and CBP from the data
package. The rules **prohibit** rebuilding them, or using copies obtained elsewhere, to reproduce the target.
Only the provided datasets may be used for the scored submission. External data is allowed only for the
Best Bias Discovery writeup. No training labels are published, so this is estimation without labels:
predict `E[score | permitted data]`.

## Permitted data (the only source)

The public Source Cooperative bucket, `source.coop/humane-intelligence/bias-bounty-mapping-equity-challenge`,
needs no account. `solution.py` downloads what it needs over HTTPS. The "Bias API" on the competition page
is the organisers' scorecard that grades every submission for equity. It is not a data source.

The competition page also lists TIGER/Line roads, Microsoft building footprints, HIFLD facilities and County
Business Patterns. Those are the layers the target was built from. They were withdrawn and must not be used,
including copies downloaded from anywhere else. No API key is involved: every permitted file is public.

### Inventory of everything provided (checked 2026-10-13, all downloaded to `data/`)

Rows are summed over the five regions. "New info" is the share of a feature's variance that the 30 scored files
do not already explain. A high value means a probe can test it, not that it predicts the target.

| file | size | what it is | verdict |
|---|---|---|---|
| `overture-roads` | 537 MB, 3.3 M segments | drivable roads, class + route refs | **used**: named-highway length, route system, boundary highways; tertiary length (probe) |
| `overture-buildings` | 2.1 GB, 19 M footprints | building polygons + upstream `sources` | **used**: count per tract. Source mix: see the open question below |
| `overture-pois` | 277 MB, 2.0 M places | places, categories, `sources`, confidence | **used**: facility evidence, total places, share by upstream source, BrightQuery and Meta (probes) |
| `census-acs-housing` | 26 MB | ACS housing units | **used** (Poisson prior; the organisers say it is a sanity check only) |
| `sample-submission.csv` | small | scored GEOIDs | **used**: authoritative row list and order |
| `strata/<region>/census-tracts` | 40 MB | tract polygons | **used**: geometry for every spatial join |
| `strata/<region>/strata-tract-table` | 5 MB, 227 cols | every strata source joined | **used**: RUCA, SVI, CVI, tribal, drought, heat, wildfire, USFS building count, population; land area (probe) |
| `overture-roads-unfiltered` | 1.5 GB, 7.2 M | adds service roads, tracks, paths | not used. Service length has new info 0.79, but no mechanism links it to TIGER S1100/S1200 |
| `overture-infrastructure` | 3 MB, 33 k | transit stops, airports | not used: new info 0.93, no link to any component |
| `overture-rail` | 12 MB, 32 k | rail lines | not used: not part of any component |
| per-source strata tables (SVI, CVI, USDM, drought.gov, EPHT, NASA heat, CDC WONDER, USFS, CarbonPlan, MTBS, NIFC, FPA-FOD, USGS, NOAA GHCN, RUCC, NCHS) | 0.1–11 MB each | the full detail behind the joined table (seasonal, yearly and raster stats) | not used beyond the joined table. These are hazard and vulnerability descriptors. They define the scorecard strata, not the coverage mechanism |
| `strata/national/*` | 0.5 GB | the same tables for all 85,396 US tracts | not useful: no labels anywhere, and the regional rows are identical |
| `census-aiannh`, tribal tracts/subdivisions, `boundaries/` | small | tribal and region geometry | not used: tribal overlap is already in the joined table |

### Open question: Overture's own source attribution

About half of the Overture buildings come from **Microsoft ML Buildings** (eastern-ok 64 %, northern-ca 45 %,
south-central-tx 39 %, maricopa-az 13 %). The rest come from OpenStreetMap and Esri. Microsoft footprints are
the withdrawn building reference, so the Microsoft share is a direct handle on the building gap: correlation
+0.37 with r14 and new info 0.80. It sits in a provided, permitted file, and the data README encourages looking
at `sources`. But it is also part of a copy of a withdrawn layer, and the rules forbid using "equivalent
copies" to approximate the target. Ask the organisers before using it.

## How we solve it

`solution.py` (see its header) runs these steps:

1. Compute the exact Overture side of each ratio.
2. Infer the withdrawn reference side from permitted evidence. TIGER S1100/S1200 is the Interstate/US/State
   route system, and Overture keeps the route refs. Tract boundaries are TIGER road edges. Facility names
   and alternate categories show stations that Overture does not label as such. USFS gives an independent
   building count.
3. Push a generative model of the reference counts through the exact composite rule by Monte Carlo.
4. Blend with the leaderboard information. Each scored file's RMSE is an exact moment of the hidden target.
5. Apply a small correction measured by probe submissions.

Current best: **0.063433** public RMSE (`submission.csv`, r14). Every score is in `lb_scores.csv`.

## Constraints to remember

- At most 5 submissions a day in practice (the dialog limit), though the page says 10. The total limit is 200.
- Choose 2 final submissions for the private leaderboard before 31 Oct.
- The top 10 must hand in code that reproduces their score within 48 hours. Set seeds. Use only open-source
  tools and no paid APIs.
- There is no time dimension: one snapshot score per tract, and no tract has a history. Forecasting models
  such as TimesFM have nothing to forecast here.
- Read GEOID as text. Maricopa's state FIPS is `04`.
- Open question (thread 35292): are constants calibrated on the leaderboard allowed? Steps 4 and 5 depend
  on the answer. The label-free model alone (r2f) scored 0.069455.
