# Best Bias Discovery: the map is only as complete as its upstream feed

Zindi user: **iam-dante**. Reproduce with `python train.py --discovery`, which writes `discovery.csv`,
`discovery_svi_control.csv` and `discovery_examples.csv`. It uses challenge data only: Overture places with
their `sources` and `categories`, the strata tables and the tract boundaries. The gap estimates come from
the from-scratch model in [METHOD.md](METHOD.md); no leaderboard information is involved.

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
| Eastern Oklahoma | 780 | 4.8 / 17.3 | 1.19 / 5.50 | 0.143 / 0.077 | **1.86** |
| Maricopa | 1,419 | 3.1 / 17.4 | 0.45 / 4.73 | 0.140 / 0.083 | **1.69** |
| Northern California | 444 | 3.0 / 18.3 | 0.67 / 5.54 | 0.128 / 0.100 | **1.28** |
| South-Central Texas | 4,998 | 3.6 / 17.6 | 0.83 / 5.22 | 0.096 / 0.079 | **1.23** |

- **Not social vulnerability restated** (`discovery_svi_control.csv`). Within urban tracts the estimated gap
  rises with the Meta-only share in both SVI halves:
  - low SVI: 0.087, 0.086, 0.098, 0.134 across the four quartiles;
  - high SVI: 0.070, 0.075, 0.081, 0.098.
- **Not rurality:** the comparison is made inside the urban stratum of each region.
- **Named tracts** (`discovery_examples.csv`): urban, top-quartile Meta share, and the largest estimated gaps:

  | Tract | County (FIPS) | Residents | Mapped places | Share from Meta | From registries | Estimated gap |
  |---|---|---:|---:|---:|---:|---:|
  | 04019980400 | Pima County, AZ (04019) | 1,646 | 7 | 100% | 0 | 0.64 |
  | 48113013630 | Dallas County, TX (48113) | 2,154 | 15 | 73% | 0 | 0.60 |
  | 04021000903 | Pinal County, AZ (04021) | 6,544 | 17 | 77% | 1 | 0.56 |
  | 48061010602 | Cameron County, TX (48061) | 1,748 | 26 | 77% | 0 | 0.48 |
  | 40131050103 | Rogers County, OK (40131), tribal | 2,482 | 34 | 74% | 3 | 0.45 |
  | 40065968800 | FIPS 40065, OK, tribal | 3,436 | 54 | 76% | 3 | 0.44 |

## 3. Who is affected and why it matters

- **Emergency dispatch.** Fire stations enter Overture mainly through their own social-media pages. Volunteer
  and small municipal departments without an active page drop out. A dispatch or routing tool that geocodes
  the "nearest station" against Overture then points to a farther one.
- **Evacuation and relief.** Clinics, pharmacies, dialysis and home-health providers come mostly from business
  registries. Where registry coverage is thin, health places per resident fall to a fifth or a tenth of
  comparable urban tracts. Relief planners who use the map to find medically dependent residents' care points
  miss them. This happens in tribal tracts of Eastern Oklahoma, in Cameron County on the Texas border and in
  fast-growing Arizona suburbs.
- **Why the scorecard cannot see it.** The bias sits inside the urban stratum. It depends on a property of the
  map's supply chain, not of the community.
- **Remedy.** Add registry and authoritative facility feeds where the Meta-only share is high. The share is
  computable from Overture alone, so it can screen any open-map release.

## Data and limitations

- Overture release 2026-08-19.0 places (`sources[1].dataset`, `categories.primary`), the strata tables and the
  census tract boundaries, from the challenge bucket. No external or withdrawn data.
- The source mix and the places per resident are exact measurements. Gap values are model estimates from a
  label-free model; its limits are in [METHOD.md](METHOD.md) §6.
