# Best Bias Discovery: the map is only as complete as its upstream feed

Zindi user: **iam-dante**. Associated leaderboard submission: R29 (`submission_r29.csv`, public RMSE 0.062094756).
Reproduce with `python research/r32_discovery.py`. It uses challenge data only: Overture places with their
`sources` field, the strata tables and the tract names. Outputs are in `research/r32/discovery_*.csv`.

## 1. The pattern (outside the scorecard grid)

Overture's places layer is a merge of upstream feeds. In the four scored regions, Meta (Facebook pages)
supplies most places, followed by Microsoft, BrightQuery (a business registry), Foursquare and
AllThePlaces. The feed matters for the categories that emergency services rely on:

| Overture category (graded tracts) | Places | Share from Meta pages |
|---|---:|---:|
| Fire stations (`fire_department`) | 3,462 | **85%** |
| Schools (six school categories) | 23,723 | **88%** |
| EMS (`ambulance_and_ems_services`) | 793 | 60% |
| Health places (clinics, pharmacies, home health, dialysis, ...) | 77,908 | 35% (**65% from registries**) |

**Finding.** Among *urban* tracts, those whose places come mostly from Meta alone have about **twice the
estimated coverage gap** of urban tracts with a diverse source mix. This holds in every region. It is a
pattern within one scorecard stratum (urban), so the API's urban/rural ratio cannot show it. In rural
tracts it disappears (ratio 0.85–1.36), because there almost every tract is Meta-dominated.

## 2. Evidence

Each row compares the top and bottom quartiles of the Meta-only place share, within a region and urban class
(`discovery_cells.csv`). "Model gap" is our best submission R29. "Physics" is an independent
leaderboard-free model (`data_only.py`).

| Region | Urban tracts | Model gap, top / bottom quartile | Ratio | Physics ratio | Registry places per 1k residents, top / bottom | Health places per 1k, top / bottom |
|---|---:|---|---:|---:|---|---|
| Eastern Oklahoma | 780 | 0.129 / 0.055 | **2.35** | 2.05 | 3.3 / 13.6 | 1.19 / 5.50 |
| Maricopa | 1,419 | 0.050 / 0.027 | **1.88** | 1.85 | 1.6 / 11.5 | 0.45 / 4.73 |
| Northern California | 444 | 0.062 / 0.039 | **1.61** | 1.43 | 1.5 / 10.3 | 0.67 / 5.54 |
| South-Central Texas | 4,998 | 0.060 / 0.028 | **2.18** | 1.88 | 2.4 / 11.6 | 0.83 / 5.22 |

- **It is not social vulnerability restated** (`discovery_svi_control.csv`). Within urban tracts the gap rises
  steadily across Meta-share quartiles in both SVI halves:
  - low SVI: 0.029, 0.036, 0.043, 0.053;
  - high SVI: 0.034, 0.042, 0.056, 0.073.

  High SVI widens the effect (2.15× vs 1.8×), so the burden compounds.
- **Measured inputs vs model estimates.** The source mix and the facility counts are exact Overture
  measurements, with no model involved. The gap values are model estimates, since no tract labels are published.
  - R29 was calibrated on the 60 submissions scored before it.
  - It reproduces the organisers' true Bias Scorecard on all 9 strata within 0.011. These are the ratios of the
    zero-error entry at rank 1, read from the public leaderboard API to 3 decimals.
  - The independent physics model, which uses no leaderboard feedback, shows the same pattern (ratios 1.43–2.05).
- **Named tracts** (`discovery_examples.csv`): urban, top-quartile Meta share, and the largest estimated gaps:

  | Tract | County (FIPS) | Estimated gap |
  |---|---|---:|
  | 40113940014 | Osage County, OK (40113) | 0.38 |
  | 40131050103 | Rogers County, OK (40131) | 0.34 |
  | 40065968800 | FIPS 40065, OK | 0.34 |
  | 06101050504 | Sutter County, CA (06101) | 0.32 |
  | 06007003100 | Butte County, CA (06007) | 0.30 |
  | 04021000903 | Pinal County, AZ (04021) | 0.30 |
  | 48215020211 | Hidalgo County, TX (48215) | 0.29 |
  | 48215022301 | Hidalgo County, TX (48215) | 0.29 |

  Tract 48215020211 has 4,383 residents, 52 mapped places, 85% from Meta, 5 from registries and 1 health place.

## 3. Who is affected and why it matters

- **Emergency dispatch.** Fire stations and EMS posts enter Overture mainly through their own social media
  pages. Volunteer and small municipal departments without an active page drop out of the map. Dispatch
  or routing tools that geocode "nearest station" against Overture then point to a farther station.
- **Evacuation and relief.** Clinics, pharmacies, dialysis and home-health providers come mostly from
  business registries. Where registry coverage is thin, these places are missing at 5–10× the rate. That is
  the case in urban tracts with high SVI in Eastern Oklahoma's tribal service areas and in Hidalgo County's
  border neighbourhoods. Relief planners who use the map to find medically dependent residents' care
  points miss them.
- **Why the scorecard cannot see it.** The API averages gaps by urban/rural and hazard strata. This bias
  lives *inside* the urban stratum and depends on a property of the map's supply chain, not of the community.
  A useful remedy: add registry and authoritative facility feeds (state fire-marshal lists, school directories)
  where the Meta-only share is high. The share is computable from Overture alone, so it can serve as a
  screening layer for any open-map release.

## Data and limitations

- Only challenge data are used. Overture release 2026-08-19.0 places (`sources[1].dataset`, `categories.primary`),
  the strata tables and the census tract names. No withdrawn reference layer is used.
- Gap values are estimates (public RMSE 0.062). The source-mix measurements are exact.
- The leaderboard measured the place-source directions during calibration (probes `p15_04`, `p15_05`).
- Retrieval: challenge bucket `source.coop/humane-intelligence/bias-bounty-mapping-equity-challenge`, files dated 2026-08-26.
