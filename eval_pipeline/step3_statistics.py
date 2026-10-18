"""Step 3: boxplots, mixed models (delta ~ emotion + (1|sujeto)), Holm, c-sweep.

Run:  python eval_pipeline/step3_statistics.py
Needs: pandas, numpy, matplotlib, scipy, statsmodels
Output: results/box_*.png, results/mixedmodels.csv, results/csweep.csv, results/csweep_r2.png,
        results/SUMMARY.md (Step 3 section; Step 4 appends to it)
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

SEED = 42
np.random.seed(SEED)
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
PHI = (1 + 5 ** 0.5) / 2
EMOTIONS = ["enojado", "feliz", "triste", "sorprendido"]
P_IDS = ["P1", "P2", "P3a", "P3b", "P4", "P5", "P6"]
S_IDS = ["S1", "S2", "S3", "S4", "S5", "S6"]
C_GRID = np.round(np.arange(1.20, 2.2001, 0.01), 2)

df = pd.read_csv(OUT / "features.csv")
df["clase"] = pd.Categorical(df["clase"], EMOTIONS)


def fit_r2(d, col):
    """Mixed model col ~ emotion + (1|sujeto). Returns marginal R2, LRT p (emotion), n."""
    d = d[["sujeto", "clase", col]].rename(columns={col: "y"}).dropna()
    full = smf.mixedlm("y ~ C(clase)", d, groups=d["sujeto"]).fit(reml=False)
    null = smf.mixedlm("y ~ 1", d, groups=d["sujeto"]).fit(reml=False)
    fixed = np.asarray(full.model.exog @ full.fe_params.values)
    var_f = np.var(fixed)
    var_u = float(full.cov_re.iloc[0, 0])
    var_e = float(full.scale)
    r2m = var_f / (var_f + var_u + var_e)
    lrt = max(2 * (full.llf - null.llf), 0.0)
    return r2m, stats.chi2.sf(lrt, df=len(EMOTIONS) - 1), len(d)


# 1. boxplots -----------------------------------------------------------------
def boxgrid(prefix, ids, title, fname):
    n = len(ids)
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.4 * rows), squeeze=False)
    for ax, k in zip(axes.ravel(), ids):
        data = [df.loc[df.clase == e, f"{prefix}_{k}"].dropna() for e in EMOTIONS]
        ax.boxplot(data, labels=[e[:5] for e in EMOTIONS], showfliers=True)
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set_title(f"{prefix}_{k}")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)


boxgrid("dr", P_IDS, "Delta r (P set) by emotion", "box_dr_P.png")
boxgrid("ddphi", P_IDS, "Delta d_phi (P set) by emotion", "box_ddphi_P.png")
boxgrid("dr", S_IDS, "Delta (S set) by emotion", "box_d_S.png")

# 2. mixed models -------------------------------------------------------------
tests = [("dr", k) for k in P_IDS + S_IDS] + [("ddphi", k) for k in P_IDS]
res = []
for pre, k in tests:
    r2, p, n = fit_r2(df, f"{pre}_{k}")
    res.append({"feature": k, "delta": "d_phi" if pre == "ddphi" else "dr", "n": n,
                "marginal_R2": r2, "p_LRT": p})
mm = pd.DataFrame(res)
order = np.argsort(mm.p_LRT.values)
m = len(mm)
holm = np.empty(m)
running = 0.0
for rank, i in enumerate(order):
    running = max(running, (m - rank) * mm.p_LRT.values[i])
    holm[i] = min(running, 1.0)
mm["p_holm"] = holm
mm.to_csv(OUT / "mixedmodels.csv", index=False)
print(mm.round(4).to_string(index=False))

# 3. c-sweep ------------------------------------------------------------------
r2_ref = {k: mm[(mm.feature == k) & (mm.delta == "dr")].marginal_R2.iloc[0] for k in P_IDS}
sweep = []
for k in P_IDS:
    r, rn = df[f"r_{k}"], df[f"rneu_{k}"]
    for c in C_GRID:
        d = df.assign(_y=(r - c).abs() - (rn - c).abs())
        r2, p, _ = fit_r2(d, "_y")
        sweep.append({"feature": k, "c": c, "marginal_R2": r2, "p_LRT": p, "source": "grid"})
    # data-derived controls (design doc): midpoint of class means, median of r_neutro
    class_means = df.groupby("clase", observed=True)[f"r_{k}"].mean()
    derived = {"midpoint_class_means": (class_means.min() + class_means.max()) / 2,
               "median_rneu": float(rn.median())}
    for name, c in derived.items():
        d = df.assign(_y=(r - c).abs() - (rn - c).abs())
        r2, p, _ = fit_r2(d, "_y")
        sweep.append({"feature": k, "c": c, "marginal_R2": r2, "p_LRT": p, "source": name})
sw = pd.DataFrame(sweep)
sw.to_csv(OUT / "csweep.csv", index=False)

fig, axes = plt.subplots(2, 4, figsize=(17, 7), squeeze=False)
for ax, k in zip(axes.ravel(), P_IDS):
    s = sw[(sw.feature == k) & (sw.source == "grid")]
    ax.plot(s.c, s.marginal_R2, color="C0", label="Δd_c")
    ax.axhline(r2_ref[k], color="k", ls="--", lw=1, label="Δr")
    ax.axvline(PHI, color="C3", lw=1, label="ϕ")
    ax.set_title(k)
    ax.set_xlabel("c")
    ax.set_ylabel("marginal R²")
axes.ravel()[0].legend(fontsize=8)
for ax in axes.ravel()[len(P_IDS):]:
    ax.axis("off")
fig.tight_layout()
fig.savefig(OUT / "csweep_r2.png", dpi=150)
plt.close(fig)

# 4. SUMMARY.md (Step 3 section) ---------------------------------------------
phi_row = sw[(sw.source == "grid") & (np.isclose(sw.c, round(PHI, 2)))].set_index("feature")
sweep_tab = []
for k in P_IDS:
    s = sw[(sw.feature == k) & (sw.source == "grid")]
    best = s.loc[s.marginal_R2.idxmax()]
    sweep_tab.append({
        "feature": k, "R2_dr": r2_ref[k], "R2_dphi(c=1.62)": phi_row.loc[k, "marginal_R2"],
        "best_c": best.c, "R2_best_c": best.marginal_R2,
        "R2_midpoint": sw[(sw.feature == k) & (sw.source == "midpoint_class_means")].marginal_R2.iloc[0],
        "R2_median_rneu": sw[(sw.feature == k) & (sw.source == "median_rneu")].marginal_R2.iloc[0],
    })
sweep_tab = pd.DataFrame(sweep_tab)

with open(OUT / "SUMMARY.md", "w", encoding="utf-8") as f:
    f.write("# SUMMARY\n\n## Step 3 — Statistics\n\n")
    f.write("Mixed model `delta ~ emotion + (1|sujeto)` (ML). Marginal R² (Nakagawa). "
            "p = likelihood-ratio test of the emotion term (3 df); Holm over "
            f"{m} tests.\n\n")
    f.write(mm.round(4).to_markdown(index=False) + "\n\n")
    f.write("### c-sweep (marginal R², P set)\n\n")
    f.write(sweep_tab.round(4).to_markdown(index=False) + "\n\n")
    f.write("![csweep](csweep_r2.png)\n\n")
    f.write("### Boxplots\n\n![dr P](box_dr_P.png)\n\n![ddphi P](box_ddphi_P.png)\n\n"
            "![d S](box_d_S.png)\n")
print(sweep_tab.round(4).to_string(index=False))
print("Saved to", OUT)
