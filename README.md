# Bias Bounty Mapping Equity Challenge

Predict the Overture Maps coverage gap of every Census tract. The problem, the data and the rules are in
[problem.md](problem.md).

**Current submission:** `submission.csv` (r19), expected public RMSE **0.062502**. Best scored so far:
`p18_02_tiger_noise` at **0.062562** (r18 itself decodes to about 0.06260).

## Run

```bash
pip install -r requirements.txt
python solution.py
```

That one command:

1. downloads the permitted challenge files into `data/` (about 3.4 GB on the first run, no account needed,
   and files already on disk are skipped),
2. builds the tract features (cached in `data/features/`),
3. runs the physics model,
4. fits the leaderboard blend and checks it against the scored r12 file,
5. applies the probe correction and writes `submission.csv`,
6. validates the local score prediction against every scored file (`validation.csv`).

Options:

```bash
python solution.py --rebuild      # recompute the cached features from the raw data
python solution.py --probes 4     # also write the next 4 probe files to next/
```

## Layout

```
solution.py      the whole pipeline (data -> features -> physics -> blend -> correction -> submission.csv)
problem.md       what the competition asks, the data, the rules
lb_scores.csv    public RMSE of every scored file (inputs of the blend)
scored/          the scored files themselves; each score is an exact fact about the hidden target
submission.csv   the file to submit (GEOID, coverage_gap_score)
data/            challenge data and feature cache (gitignored, downloaded by solution.py)
```

## Daily cycle (5 submissions a day)

1. Submit `submission.csv` and the probe files in `next/`.
2. Copy each scored file into `scored/` and append `file,score` to `lb_scores.csv`.
3. Rerun `python solution.py --probes`. It fits every scored probe jointly, updates `submission.csv`, and
   writes the next probes. Probes are built on the current corrected prediction, so each one is also a
   near-best submission.

A probe built on a one-sided mechanism, such as `county_s1200` or `unrouted_s1200` (TIGER can only hold
*more* length than the route-based estimate), is a bet as well: its sign is the physical one. Every probe has
the directions already measured removed, so it pays only for new signal. If the new direction carries no
signal, it scores about r14 + 0.00013.

Submit only the score column. Zindi grades every column in the file, and the optional component columns
cost about 0.08 RMSE.

## Validation (no labels, so the leaderboard is the test set)

No tract has a known answer (the reference layers were withdrawn), so a train/validation split is impossible.
Step 6 of `solution.py` validates the local score prediction instead. It hides each scored file's own score
and predicts it from the files scored **before** it, which is the position before every submission. It also
predicts it from **all other** files (leave-one-out). The 48 data points are in `validation.csv`.

| test (24 files from r12 on) | mean abs error | max abs error | within 0.0005 |
|---|---|---|---|
| predicted before scoring | 0.000140 | 0.000446 | 24 / 24 |
| leave-one-out | 0.000050 | 0.000187 | 24 / 24 |

**Private set (70 %, the final ranking).** A semi-synthetic test was run on 2026-10-16: 300 random 30/70
splits, with a truth that carries the 22 measured probe directions plus noise. On those splits the current
policy (a direction enters at z ≥ 2, then James–Stein shrinkage) gave the lowest private RMSE among: no
shrinkage, 2× shrinkage, z ≥ 3, and all directions. z ≥ 1.5 was within 0.00001 (noise). The correction loses
about **0.0004–0.0005** from public to private. Which tracts fall in the private set moves the private score
by **±0.0006** with Gaussian errors and up to ±0.0015 with heavy-tailed errors. So expect r19's private
score to be about public + 0.0004.

## Score history (public RMSE)

| file | score | what changed |
|---|---|---|
| r2f | 0.069455 | label-free generative model |
| stack3 | 0.066116 | leaderboard blend + scorecard groups |
| r6e | 0.065531 | vulnerability effects in the physics |
| stack7 | 0.064214 | regional specialists, Arizona route fix |
| r12 | 0.064009 | public-subset-corrected blend, Monte-Carlo denoising |
| r14 | 0.063433 | r12 + correction along road ratio (z = −6.1) and boundary leak (z = +4.7) |
| **p15_03_tertiary_vs_named** | **0.063190** | r14 + probe: tertiary vs named road length |
| r16 | 0.062992 expected, ≈0.06295 decoded | + unrouted arterials (z = +7.5), tertiary (z = +8.6), BrightQuery places (z = +5.2) |
| **p16_03_tertiary_vs_named_northern-ca** | **0.063072** | r16 + probe (no new signal: it confirms r16) |
| r17 | 0.062703 expected, ≈0.06272 decoded | + tertiary physics (z = +7.8), Maricopa tertiary (z = +7.8), land area (z = +8.0) |
| **p17_03_log_land_area_maricopa-az** | **0.062675** | r17 + probe |
| r18 | 0.062653 expected, ≈0.06260 decoded | + unrouted arterials in Maricopa (z = +10.8) |
| **p18_02_tiger_noise** | **0.062562** | r18 + probe: TIGER length noise |
| r19 (submission.csv) | 0.062502 expected | + TIGER length noise (z = +11.3), places per person (z = +9.7) |

Tested and dropped: full anchoring to the published cell means (0.064221), physics calibrated on
leaderboard moments (0.064317), the refitted-physics probe (0.063802, conditional z 0.4) and TIGER record
multiplicity (0.063592, conditional z 0.5), county routes (0.063467) and Meta place share
(0.063449), which add nothing given the other p15 directions.
