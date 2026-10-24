# Best Bias Discovery: the map is only as complete as its upstream feed

Zindi user: **iam-dante**. Reproduce with `python train.py --discovery`, which writes `outputs/discovery.csv`,
`outputs/discovery_svi_control.csv` and `outputs/discovery_examples.csv`. It uses challenge data only: Overture places with
their `sources` and `categories`, the strata tables and the tract boundaries. The gap estimates come from the
final model in [METHOD.md](METHOD.md) (estimated public RMSE 0.0616). That model reproduces the official
scorecard ratios within about 0.1.

## 1. The pattern (outside the scorecard grid)

Overture's places layer merges several upstream feeds: Meta (Facebook pages), Microsoft, BrightQuery (a
business registry), Foursquare and AllThePlaces. Which feed supplies a category matters for emergency services.
These are exact counts over the 9,379 graded tracts, with no model involved:

| Overture category | Share from Meta pages | Share from business registries |
|---|---:|---:|
| Fire stations (`fire_department`) | **85%** | – |
| Schools (six school categories) | **88%** | – |
| Health places (clinics, pharmacies, home health, dialysis, urgent care, ...) | – | **65%** |

**Finding.** Inside the *urban* stratum, tracts whose places come mostly from Meta pages have 3.6–6× fewer
registry businesses and 4–10× fewer health places per resident. They also carry a higher estimated coverage
gap, in every region. The scorecard reports a single urban/rural ratio, so it averages this away. In rural
tracts the pattern disappears, because almost every rural tract is Meta-dominated.

## 2. Evidence

Each row compares the top and bottom quartiles of the Meta-only place share, within a region's urban tracts.
The places-per-resident counts are exact; the gap is the model estimate.

| Region (urban tracts) | n | Registry places per 1k residents, top / bottom | Health places per 1k, top / bottom | Estimated gap, top / bottom | Ratio |
|---|---:|---|---|---|---:|
| Eastern Oklahoma | 780 | 4.8 / 17.3 | 1.19 / 5.50 | 0.128 / 0.055 | **2.32** |
| Maricopa | 1,419 | 3.1 / 17.4 | 0.45 / 4.73 | 0.053 / 0.028 | **1.92** |
| Northern California | 444 | 3.0 / 18.3 | 0.67 / 5.54 | 0.066 / 0.040 | **1.65** |
| South-Central Texas | 4,998 | 3.6 / 17.6 | 0.83 / 5.22 | 0.060 / 0.029 | **2.04** |

- **Rural tracts: no effect** (ratios 0.93–1.34).
- **Not social vulnerability restated** (`discovery_svi_control.csv`). Within urban tracts the estimated gap
  rises steadily with the Meta-only share in both SVI halves:
  - low SVI: 0.030, 0.035, 0.041, 0.052 across the four quartiles;
  - high SVI: 0.036, 0.043, 0.057, 0.074.

  High SVI widens the effect (2.0× vs 1.75×), so the burden compounds.
- **Named tracts** (`discovery_examples.csv`): urban, top-quartile Meta share, and the largest estimated gaps:

  | Tract | County (FIPS) | Residents | Mapped places | Share from Meta | From registries | Estimated gap |
  |---|---|---:|---:|---:|---:|---:|
  | 40113940014 | Osage County, OK (40113), tribal | 3,914 | 141 | 75% | 4 | 0.39 |
  | 40131050103 | Rogers County, OK (40131), tribal | 2,482 | 34 | 74% | 3 | 0.37 |
  | 40065968800 | FIPS 40065, OK, tribal | 3,436 | 54 | 76% | 3 | 0.35 |
  | 40125500400 | Pottawatomie County, OK (40125), tribal | 3,386 | 161 | 70% | 29 | 0.33 |
  | 48215020211 | Hidalgo County, TX (48215) | 4,383 | 52 | 85% | 5 | 0.32 |
  | 06101050504 | Sutter County, CA (06101) | 7,395 | 71 | 70% | 3 | 0.31 |
  | 48215022301 | Hidalgo County, TX (48215) | 6,111 | 325 | 77% | 35 | 0.31 |

## 3. Who is affected and why it matters

- **Emergency dispatch.** Fire stations enter Overture mainly through their own social-media pages. Volunteer
  and small municipal departments without an active page drop out. A dispatch or routing tool that geocodes
  the "nearest station" against Overture then points to a farther one.
- **Evacuation and relief.** Clinics, pharmacies, dialysis and home-health providers come mostly from business
  registries. Where registry coverage is thin, health places per resident fall to a fifth or a tenth of
  comparable urban tracts. Relief planners who use the map to find medically dependent residents' care points
  miss them. This happens in the tribal tracts of Eastern Oklahoma and in Hidalgo County on the Texas border.
- **Why the scorecard cannot see it.** The bias sits inside the urban stratum. It depends on a property of the
  map's supply chain, not of the community.
- **Remedy.** Add registry and authoritative facility feeds where the Meta-only share is high. The share is
  computable from Overture alone, so it can screen any open-map release.

## Data and limitations

- Overture release 2026-08-19.0 places (`sources[1].dataset`, `categories.primary`), the strata tables and the
  census tract boundaries, from the challenge bucket. No external or withdrawn data.
- The source mix and the places per resident are exact measurements. Gap values are model estimates; their
  limits are in [METHOD.md](METHOD.md) §6 and §8.
