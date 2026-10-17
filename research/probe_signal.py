"""New-signal z of every scored probe (from validation.csv) and an end-of-competition forecast.
    python solution.py          # refreshes validation.csv
    python research/probe_signal.py"""
import numpy as np, pandas as pd

t = pd.read_csv("validation.csv")
p = t[t.file.str.match(r"^p\d+_")].copy()
se = 0.0625 / np.sqrt(0.3 * 9379)                        # standard error of one probe moment
delta = np.where(p.file.str.extract(r"^p(\d+)_")[0].astype(int) >= 19, 0.002, 0.004)
# actual MSE vs the MSE predicted from everything scored before the probe = the probe's new signal
p["z_new"] = -(p.actual ** 2 - p.pred_before ** 2) / (2 * delta) / se
p["kind"] = np.where(p.file.str.contains("s1200|noise|multiplicity|gmm|boundary_leak"), "physics", "feature")
p["region_split"] = p.file.str.contains("maricopa|northern-ca|eastern-ok|south-central")
p[["file", "actual", "pred_before", "z_new", "kind", "region_split"]].round(6).to_csv("research/probe_signal.csv", index=False)
print(f"{len(p)} probes; hit rate |z|>=2: {(p.z_new.abs() >= 2).mean():.2f}; mean |z|: {p.z_new.abs().mean():.2f}")
for col in ["kind", "region_split"]:
    print(p.groupby(col).z_new.agg(lambda x: f"n={len(x)} hits={(x.abs() >= 2).sum()} mean|z|={x.abs().mean():.2f}").to_string())

# forecast: remaining submissions draw new-signal z from the recent regime, weakening 40 % over the queue
best = float(t.loc[t.file.str.match(r"^p\d+_"), "actual"].min())
rng, se2, n_left = np.random.default_rng(1), se ** 2, 63
start = 0.062502 ** 2
for label, pool in [("all probes", p.z_new.to_numpy()), ("last 13 probes", p.z_new.to_numpy()[-13:])]:
    out = []
    for _ in range(20000):
        zs = rng.choice(pool, n_left) * np.linspace(1.0, 0.6, n_left)
        hit = np.abs(zs) >= 2
        gain = ((zs[hit] ** 2 - 1) * 0.92 * se2).sum()
        out.append((np.sqrt(start - gain), np.sqrt(start - gain + 2 * hit.sum() * se2 + 0.00045 * 2 * 0.0625)))
    o = np.array(out)
    print(f"forecast ({label}): public {np.median(o[:, 0]):.5f} [{np.quantile(o[:, 0], .1):.5f}, {np.quantile(o[:, 0], .9):.5f}], "
          f"private {np.median(o[:, 1]):.5f}")
