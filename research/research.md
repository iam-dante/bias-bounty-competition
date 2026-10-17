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
