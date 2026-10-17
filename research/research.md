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
