"""
E03 / E04: feature analysis.

E03  what drives the current best (stack7): surrogate gradient-boosting model on grouped features,
     grouped permutation importance (held-out R^2 drop).
E04  what the physics model is MISSING: the leaderboard-optimal blend of scored single models minus
     the best single model (r10d) is the correction the hidden target asks for. Regress that
     correction on feature groups, including strata never used by the model.
"""
import numpy as np
import pandas as pd
import harness as H
from sklearn.model_selection import KFold
import lightgbm as lgb

D = H.data(); df = D["df"]; lb = H.LB()
g = lb.graded
w, _ = lb.blend()
names = list(w)
B = sum(w[n] * (lb.vec(n) if n != "__const__" else np.ones(len(df))) for n in names)
r10 = lb.vec("r10d_regional_affine.csv")
s7 = lb.vec("stack7_r10.csv")
corr = B - r10
print(f"blend-vs-r10d correction: rms {np.sqrt(np.mean(corr[g]**2)):.4f}")

lg = lambda c: np.log1p(df[c].clip(lower=0))
F = pd.DataFrame(index=df.index)
GROUPS = {
  "road_overture": ["L_named", "L_motorway", "L_trunk", "L_primary", "L_secondary", "L_tertiary", "L_residential", "L_all", "n_named_seg"],
  "road_route_evidence": ["L_route", "L_route_localname", "L_route_multiref", "L_interstate", "L_fm", "L_county", "L_route_lowclass", "L_named_noroute"],
  "road_boundary": ["Lb_route", "Lb_named", "Lb_fm", "Lb_interstate", "Lb_tertiary", "L_route_nearB", "L_named_nearB", "perim_m"],
  "buildings": ["n_bld", "n_bld_ms", "n_bld_osm", "n_bld_google", "U_bld", "bld_area_med", "carbonplan_buildings"],
  "facilities_fire": ["n_fd", "n_fd_alt", "n_fire_name", "n_fire_prot", "n_fd_closed"],
  "facilities_ems_school": ["n_ems", "n_ems_alt", "n_ems_name", "n_sch", "n_sch_alt", "n_sch_name", "n_sch_like"],
  "pois_business": ["n_poi", "n_poi_open", "n_poi_c80", "n_poi_brand", "n_food", "n_shop", "n_relig", "n_poi_meta", "n_poi_fsq", "n_poi_msft"],
  "population_size": ["pop", "housing_units", "area_km2", "ruca_pop_density", "pct_urban"],
  "rurality_class": ["ruca_primary", "rucc_2023", "nchs_2013"],
  "social_vuln": ["svi_overall", "svi_socioeconomic", "svi_household", "svi_minority", "svi_housing_transport"],
  "climate_vuln": ["cvi_overall", "cvi_baseline", "cvi_climate", "cvi_baseline_infrastructure", "cvi_climate_extreme_events"],
  "tribal": ["tribal_pct", "tribal_legal_pct"],
  "wildfire": ["usfs_WHP_mean", "usfs_BP_mean", "usfs_Exposure_mean", "usfs_HUExposure_mean", "mtbs_wildfire_burned_pct_land", "nifc_wildfire_fires", "fod_fires", "fod_ignition_density"],
  "drought_heat": ["usdm_summer_dsci", "usdm_winter_dsci", "epht_heat_days_summer", "hwd_maxtemp_mean", "gehe_wbgt30_days_mean", "pmdi_mean", "spi12_mean"],
  "transit_rail": ["n_infra", "n_infra_bus", "L_rail"],
  "weather_stations": ["ghcn_any_nearest_km", "ghcn_stations_in_tract"],
}
COUNTS = set(GROUPS["road_overture"] + GROUPS["road_route_evidence"] + GROUPS["road_boundary"] + GROUPS["buildings"]
             + GROUPS["facilities_fire"] + GROUPS["facilities_ems_school"] + GROUPS["pois_business"]
             + ["pop", "housing_units", "area_km2", "ruca_pop_density", "n_infra", "n_infra_bus", "L_rail", "fod_fires", "nifc_wildfire_fires"])
for gname, cols in GROUPS.items():
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce").astype(float)
        F[c] = np.log1p(x.clip(lower=0)) if c in COUNTS else x
F["region"] = df["ridx"]
F["route_ratio"] = np.log((df["L_named"] + 100) / (df["L_route"] + 0.5 * (df["Lb_route"] - df["L_route_nearB"]).clip(lower=0) + 100))
GROUPS["engineered_ratio"] = ["route_ratio"]
GROUPS["region"] = ["region"]

def grouped_importance(y, mask, label):
    X = F[mask].reset_index(drop=True); yy = y[mask]
    kf = KFold(5, shuffle=True, random_state=0); preds = np.zeros(len(yy)); imp = {k: [] for k in GROUPS}
    rng = np.random.default_rng(0)
    for tr, te in kf.split(X):
        m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1).fit(X.iloc[tr], yy[tr])
        p = m.predict(X.iloc[te]); preds[te] = p
        base = np.mean((yy[te] - p) ** 2)
        for k, cols in GROUPS.items():
            Xp = X.iloc[te].copy()
            perm = rng.permutation(len(Xp))
            Xp[cols] = Xp[cols].to_numpy()[perm]
            imp[k].append(np.mean((yy[te] - m.predict(Xp)) ** 2) - base)
    r2 = 1 - np.mean((yy - preds) ** 2) / np.var(yy)
    out = pd.Series({k: np.mean(v) / np.var(yy) for k, v in imp.items()}).sort_values(ascending=False)
    print(f"\n[{label}] surrogate CV R^2 = {r2:.3f}; grouped permutation importance (share of variance lost):")
    print(out.round(4).to_string())
    return out, r2

imp_best, r2_best = grouped_importance(s7, g, "E03 what drives stack7")
imp_corr, r2_corr = grouped_importance(corr, g, "E04 what explains the leaderboard's correction to r10d")
res = pd.DataFrame({"drives_best": imp_best, "explains_correction": imp_corr}).fillna(0)
res.to_csv("exp11/feature_importance.csv")
# univariate: which single (incl. never-used) strata columns track the correction
strata = pd.read_parquet("data/features/northern-ca.parquet").columns
used = set(sum(GROUPS.values(), []))
rows = []
for c in df.columns:
    if c in used or df[c].dtype == object or c.startswith(("GEOID", "g_", "E_", "X_")):
        continue
    x = pd.to_numeric(df[c], errors="coerce")
    if x[g].notna().mean() < 0.8 or x[g].nunique() < 5:
        continue
    xr = x[g].rank(pct=True).fillna(0.5)
    rows.append((c, np.corrcoef(xr, pd.Series(corr[g]).rank(pct=True))[0, 1]))
u = pd.DataFrame(rows, columns=["col", "spearman_with_correction"]).sort_values("spearman_with_correction", key=abs, ascending=False)
print("\nunused columns most associated with the correction:\n", u.head(15).round(3).to_string(index=False))
u.to_csv("exp11/unused_feature_screen.csv", index=False)
H.log("E03", "surrogate importance on stack7", "feature-analysis", {"surrogate_r2": r2_best})
H.log("E04", "features explaining LB correction", "feature-analysis", {"surrogate_r2": r2_corr})
