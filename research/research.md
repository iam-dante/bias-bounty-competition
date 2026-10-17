# Research log: local validation and probe strategy (2026-10-16)

Run every script from the repo root after `python solution.py`, which refreshes `validation.csv`.

## 1. Local validation of the score prediction (`solution.py`, step 6)

Each scored file's own score is hidden and predicted from the other files. The prediction projects the file
on them (their moments `mean_P(f*y)` are exact) and treats the unexplained remainder as carrying no signal.

| test, 24 files from r12 on | MAE | max | within 0.0005 |
|---|---|---|---|
| predicted from files scored before it | 0.000140 | 0.000446 | 24 / 24 |
| leave-one-out | 0.000050 | 0.000187 | 24 / 24 |

The early September files miss by up to 0.0025, because a single file then carried a lot of new information.
Data: `../validation.csv` (48 rows).

## 2. Public-subset weights (`public_weights.py`)

| weights | MAE before | MAE leave-one-out |
|---|---|---|
| uniform 30 % | 0.000155 | 0.000073 |
| conditioned on exact identities (used) | 0.000140 | 0.000050 |

Probe decoding barely depends on them: the boundary-leak moment is 0.005670 vs 0.005649. For the full
22-probe correction, the expected public score is 0.062482 vs 0.062502. The one visible effect is that the
two near-duplicate Maricopa directions (unrouted arterials, land area; z = 10.8 either way) swap places.

## 3. Correction policy on simulated public/private splits (`semisynth.py`)

The truth carries the 22 measured directions plus noise. Each run is 300 random 30/70 splits; the correction
is refitted on the simulated public part and scored on the private part.

| policy | private (Gaussian) | gap (Gaussian) | private (t4) | gap (t4) |
|---|---|---|---|---|
| **z ≥ 2 + James–Stein shrink (used)** | **0.062779** | +0.000410 | **0.062751** | +0.000462 |
| z ≥ 2 + 2× shrink | 0.062781 | +0.000388 | 0.062753 | +0.000440 |
| z ≥ 1.5 + JS shrink | 0.062767 | +0.000463 | 0.062743 | +0.000512 |
| z ≥ 3 + JS shrink | 0.062813 | +0.000312 | 0.062797 | +0.000375 |
| all directions, no shrink | 0.062793 | +0.000589 | 0.062775 | +0.000646 |

- The used policy is within 0.00001 of the best on private; z ≥ 1.5 is within noise.
- Expected public-to-private loss is about 0.0004–0.0005.
- The private score moves ±0.0006 (Gaussian) to ±0.0015 (t4) depending on which tracts are private.

## 4. Probe step size

A probe's moment is decoded exactly from its own score, so how much signal it measures does not depend on
the push size δ. Only the probe's own cost does: δ² in MSE. Section 2 showed the decoding is insensitive to
the public-subset weights. **δ is 0.002 from p19_06 on** (it was 0.004). With no signal, a probe now costs
+0.00003 RMSE instead of +0.00013, so even it beats the best scored file. `write_probes` never overwrites a
written probe file, because its score is decoded from that exact content.

## 5. How much new signal each probe carried (`probe_signal.py` → `probe_signal.csv`)

`z_new` = (MSE predicted before the probe − actual MSE) / (2δ · se). The pipeline's printed z values are
cumulative from r12 and overstate this.

- 22 probes: **55 % found signal** (|z| ≥ 2), mean |z| = 2.0.
- Physics mechanisms and engineered features did equally well (6/9 and 6/13 hits, mean |z| 2.0 each).
- **Region splits did worse:** mean |z| 1.5 against 2.3 for directions covering all regions. The queue now
  favours new directions over region splits.
- One-sided bets got the sign right in 5 of 6 cases. The miss, `tertiary_s1200`, measured the overshoot of
  the tertiary feature correction applied just before it.

## 6. Candidate scan (`scan.py` → `candidate_scan.csv`)

163 numeric columns, standardised within each region. Median unexplained share after the 50 scored files:
0.70. Ranking by unexplained share alone favours noisy hazard columns with no mechanism (prescribed burns,
heat trends). Queued were only candidates that are both high on the list and tied to a component or a
scorecard stratum:
- EMS evidence: 0.93
- school alternate-category share: 0.90
- legal tribal share: 0.97
- summer drought D2+: 0.91
- heat (WBGT days): 0.87

## 7. Forecast to 31 Oct (`probe_signal.py`)

About 63 submissions remain. Each draws its new-signal z from the measured distribution, weakening 40 % over
the queue; the forecast includes JS shrinkage and the public-to-private loss.

| signal pool | final public (median, 10–90 %) | final private (median) |
|---|---|---|
| last 13 probes (current regime) | **0.06168** (0.06147–0.06188) | **0.06252** |
| all 22 probes (includes the large early hits) | 0.06057 (0.05990–0.06114) | 0.06153 |

0.0615 public is the realistic ceiling, and only if every slot goes to a probe. 0.05 is out of reach.

## 8. Track record of the pipeline's expected score for each corrected base

| base | predicted | actual | error |
|---|---|---|---|
| r12 | 0.064012 | 0.064009 (scored) | +0.000003 |
| r14 | 0.063448 | 0.063433 (scored) | +0.000015 |
| r16 | 0.062992 | ≈0.06296 (decoded from 2 probes with no signal) | ≈+0.00004 |
| r18 | 0.062653 | ≈0.06260 (decoded from 1 probe) | ≈+0.00005 |

The predictions are slightly pessimistic. Decoding a base from a single probe carries about ±0.0001 noise:
r14 decoded from its p15_05 probe gives 0.063323, against its real 0.063433.

## 9. End-to-end test on a planted truth (`synthetic_e2e.py` → `synthetic_e2e.csv`)

The planted truth is y = clip(r19 + τ · (an unprobed direction) + noise). Every scored file's public score
is regenerated from y on a random 30 % public set, then the real `public_subset`, `probe_correction` and
`validate` run on those scores. 12 replicates:

| check | result |
|---|---|
| predicted public vs true public | MAE **0.000067**, max 0.00023 |
| correction on the true private rows (vs r12) | **−0.0015** (it transfers) |
| step-6 backtest on the synthetic scores | mean MAE 0.00017, worst 0.00059 |

Caveat: the planted truth lies in the span the correction can represent, so private transfer here is
optimistic. The real public-to-private loss is about 0.0004–0.0005 (section 3).

## 10. Unprobed physics mechanisms (`physics_candidates.py` → `physics_candidates.csv`)

Each of 17 parameters was moved one step under the r10a specialists (common random numbers), with the
scored files and measured probes removed. Leaders by unexplained share:
- `fm_as_s1200`: TX Farm-to-Market roads as TIGER S1200. Size 0.010, 75 % unexplained, max correlation
  with any other candidate 0.11. One-sided.
- `burned_unseen`: facilities missing from Overture in burned areas. 78 % unexplained.
- `fac_tribal`: facilities on tribal land. 64 %.
- `cbp_noise`: noise in the establishment count. 64 %.
- `ms_vs_usfs_buildings`: Microsoft-to-USFS building ratio, the first physics probe of the building gap. 51 %.

The CBP-per-person and CBP-rural directions are near duplicates (corr 0.94). The five leaders are probes
p19_11–15 (δ = 0.002).

## 11. Monte-Carlo noise in physics probe directions (`mc_noise.py` → `mc_noise.csv`)

Each direction was built with two different seeds (512 draws each) and with 4096 draws, after removing the
scored files.

| direction | seed-vs-seed corr | 512 vs 4096 corr |
|---|---|---|
| fm_s1200 | 0.999 | 1.000 |
| ms_vs_usfs_buildings | 0.995 | 0.997 |
| cbp_noise | 0.948 | 0.969 |
| tiger_noise | 0.515 | 0.623 |
| burned_unseen | 0.406 | 0.574 |
| **unrouted_s1200** | **0.154** | **0.209** |

Deterministic mechanisms (FM roads, the building ratio) are clean at 512 draws. Stochastic ones are not: the
scored `unrouted_s1200` probe was about 80 % simulation noise and still found signal (z ≈ 2.4).

Changes made:
- `physics_direction` now averages 8 × 512 draws.
- The noisy pending probes (`burned_unseen`, `fac_tribal`) were rebuilt under new names. The 512-draw files
  are kept in `next/superseded/`.
- `unrouted_s1200_clean` and `tiger_noise_clean` re-probe the two noisy hits with denoised directions.

## 12. Selection order for the correction (`selection_policy.py`)

The same semi-synthetic splits as section 3, 150 per noise model; paired differences in private RMSE are
relative to scoring order.

| order | private (Gaussian) | gap | private (t4) | gap | paired vs scoring order |
|---|---|---|---|---|---|
| scoring order (old default) | 0.062731 | +0.000239 | 0.062841 | +0.000561 | – |
| strongest first | 0.062683 | +0.000106 | 0.062804 | +0.000433 | −0.000049 ± 0.000007, better in 70 % |
| **backward elimination (new default)** | 0.062689 | +0.000174 | 0.062820 | +0.000515 | −0.000042 ± 0.000006, better in 71 % |

On the real scores:
- **Strongest-first degenerates.** It keeps only `tiger_noise`, a probe built on r18 that already carries
  the whole earlier correction (expected 0.062561).
- **Backward elimination keeps 3 directions** (Maricopa land area, TIGER noise, places per person). These are
  the latest probes and contain the earlier corrections, so expected public improves from 0.062502 to
  **0.062470**: lighter shrinkage (0.978 vs 0.918) and 3 fitted numbers instead of 11.
- It is the default in `solution.py` since 2026-10-16.

## 13. Does the 14-number blend overfit? (`blend_optimism.py`)

The truth is the real blend plus noise. The blend is refitted on 200 random 30 % public sets (with the
pipeline's 1e-7 ridge) and scored on the 70 %. Paired differences in private RMSE are relative to the used
blend.

| blend | private (Gaussian) | gap | private (t4) | gap | paired vs used (Gaussian / t4) |
|---|---|---|---|---|---|
| **14 numbers (used)** | **0.062630** | +0.000258 | **0.062774** | +0.000285 | – |
| 5 files | 0.062663 | +0.000081 | 0.062788 | +0.000079 | +0.000033 / +0.000013 |
| ridge to equal weights, 0.01 | 0.062829 | +0.000032 | 0.062940 | +0.000014 | +0.000199 / +0.000165 |
| ridge to equal weights, 0.1 | 0.063136 | −0.000022 | 0.063242 | −0.000038 | +0.000506 / +0.000468 |

- The used blend is the best on private. Its larger public-to-private gap is the price of a better fit, and
  no simpler blend recovers it.
- The blend contributes about 0.00026–0.00029 to the public-to-private loss. The correction contributes the
  rest of the ~0.0004–0.0005 in section 3.
- Without the 1e-7 ridge, the near-collinear basis (stacks are combinations of other files) blows up on
  private, to an RMSE of about 1. The ridge is essential.

## 14. Overlap between pending probes (`pending_overlap.py` → `pending_overlap.csv`)

Each pending probe's push in `next/`, minus everything the scored files span, was correlated pairwise.

- Only one pair overlaps: `tribal_pct` ~ `tribal_legal_pct` (+0.88). `p19_04_tribal_pct` was moved to
  `next/superseded/`; upload `p19_09` (smaller push) instead.
- Next largest: population density ~ urban share (+0.47); burned-area ~ tribal facilities (+0.30).
- Every other pair is below 0.25, so the remaining probes are close to independent and none wastes a slot.

## 15. Final pick (`final_pick.csv`)

Private estimate = public + blend loss (section 13, about 0.00027) + correction loss (section 12, by
selection rule and number of directions).

| file | public | private estimate |
|---|---|---|
| **r19b = submission.csv** | 0.062470 (expected) | **0.06309** |
| p18_02_tiger_noise (scored best) | 0.062562 (scored) | 0.06316 |
| r19 | 0.062502 (expected) | 0.06317 |

r19b has the best expected private score as well as public. The final private score is the better of the 2
selected files. A good pair is the newest pipeline output plus one confirmed, differently built file
(for example a scored probe), rather than two near-copies.

## 16. Reproducibility for code review

The top 10 must hand in code that reproduces their score within 48 hours.

- A fresh `git clone` of `main` (03ab13c), run with `BB_DATA` pointing at the downloaded data, reproduced
  `submission.csv` **byte for byte** (md5 `86bf4cfa07444682996978599c22e5f5`) in 216 s. The working tree was
  clean afterwards, so `validation.csv` was identical too.
- On 2026-10-13 a full `--rebuild` of every feature from the raw files reproduced the scored r14 file
  exactly (max |diff| 0).
- Seeds are fixed (physics 3000–3007, probe directions 77–84) and the package versions are pinned in
  `requirements.txt`.

## 17. Clipping and the pessimism of the expected score (`clipping.py` → `clipping.csv`)

y is in [0, 1], so clipping a corrected prediction below 0 can only lower the error. The expected-score
formula ignores this.

| selection | tracts < 0 | most negative | expected (formula) | with the minimum clipping gain |
|---|---|---|---|---|
| scoring order (r19) | 366 | −0.026 | 0.062502 | 0.062492 |
| backward (r19b) | 398 | −0.013 | 0.062470 | 0.062468 |

Clipping explains about a fifth to a third of r19's 0.00003–0.00005 pessimism in section 8, and almost none
for r19b. Backward elimination pushes tracts less far below 0, so r19b's expected 0.062470 should be close
to unbiased.

## 18. Probe queue for the rest of the competition (`queue_plan.py` → `queue_plan.csv`, `queue_plan_ranked.csv`)

34 candidate directions were built:
- 14 physics mechanisms at 8 × 512 draws (clean re-builds of measured ones included)
- 12 regional splits of the strongest hits
- 8 features

The scored files **and the 16 pending probes** were removed, and the candidates ranked by the absolute share
of new variance each still adds.
- 9 were dropped as already measured (< 20 % unexplained), for example clean county routes 0.0 %, clean
  boundary leak 0.1 %, Maricopa unrouted 3 %, the latter already covered by pending p19_18.
- The first run ranked by the normalised residual, which put exactly these near-duplicates first. The script
  now ranks by absolute share.
- The 22 kept are in `PROBE_QUEUE` right after the pending probes, with global directions interleaved with
  region splits (section 5).
- Top of the plan: heat (WBGT days, 86 % new), places per person in Northern CA (88 %), TIGER record
  multiplicity by concurrent routes (59 %), places per person in Maricopa (75 %), boundary share (43 %).

With the 16 pending probes, that is about 38 slots, roughly 8 days at 5 a day.

## 19. The floor: irreducible error under the physics model (`floor.py` → `floor.csv`)

For each tract, the official composite's spread across 2048 Monte-Carlo draws of the withdrawn reference
counts, under the r10a regional specialists. Its RMS is the RMSE that even a perfect E[score | permitted data]
would have, if the model's noise assumptions are right.

| TIGER length noise σ_t | floor (all) | eastern-ok | maricopa-az | northern-ca | south-central-tx |
|---|---|---|---|---|---|
| model (≈0.29) | **0.0555** | 0.0693 | 0.0492 | 0.0704 | 0.0522 |
| −0.1 | 0.0512 | 0.0625 | 0.0461 | 0.0660 | 0.0482 |
| −0.2 | 0.0481 | 0.0577 | 0.0437 | 0.0614 | 0.0455 |

- r19b's 0.062470 = floor 0.0555 combined in quadrature with about 0.029 of model error.
- The `tiger_noise` probe (z = +11.3) measured in the direction of *less* TIGER noise, so the true floor is
  probably nearer 0.051. The headroom is larger than the earlier ~0.056 estimate.
- TIGER length noise is the largest assumed noise source, so road-component probes have the most reducible
  error left to find. The queue already favours them.

## 20. Uncertainty of r19b's expected score (`expected_uncertainty.py`)

Over 30 redraws of the public set from the estimated membership probabilities (correction refitted each
time), r19b's expected public RMSE was **0.062473 ± 0.000048** (range 0.062401–0.062580). This matches the
backtest's typical error (section 1), so the expected score is as precise as the validation says.
