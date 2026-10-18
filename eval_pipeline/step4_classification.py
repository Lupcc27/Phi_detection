"""Step 4: LOSO nearest-centroid classification on z-scores, conditions C1, C2, C4, C5, C6.

Run:  python eval_pipeline/step4_classification.py
Needs: pandas, numpy, matplotlib, scikit-learn, tabulate
Output: results/classification_f1.csv, results/confusion_<cond>.csv, results/confusion_all.png,
        results/bootstrap_diffs.csv, Step 4 section appended to results/SUMMARY.md
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
N_BOOT = 2000
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
EMOTIONS = ["enojado", "feliz", "triste", "sorprendido"]
P_IDS = ["P1", "P2", "P3a", "P3b", "P4", "P5", "P6"]
S_IDS = ["S1", "S2", "S3", "S4", "S5", "S6"]

df = pd.read_csv(OUT / "features.csv").reset_index(drop=True)
y = df["clase"].map({e: i for i, e in enumerate(EMOTIONS)}).values
groups = df["sujeto"].values

dr_P = [f"dr_{k}" for k in P_IDS]
dphi_P = [f"ddphi_{k}" for k in P_IDS]
dr_S = [f"dr_{k}" for k in S_IDS]
CONDITIONS = {
    "C1": dr_P,            # P as delta r
    "C2": dphi_P,          # P as delta d_phi
    "C4": dr_S,            # S as delta
    "C5": dr_S + dr_P,     # S + P(delta r)
    "C6": dr_S + dphi_P,   # S + P(delta d_phi)
}

# same LOSO partitions for every condition
splits = list(LeaveOneGroupOut().split(df, y, groups))
preds = {}
for name, cols in CONDITIONS.items():
    X = df[cols].values
    pred = np.empty(len(df), dtype=int)
    for tr, te in splits:
        clf = make_pipeline(StandardScaler(), NearestCentroid())  # scaler fitted on train fold only
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    preds[name] = pred

# report ------------------------------------------------------------------------
rows = []
cms = {}
for name, pred in preds.items():
    f1_cls = f1_score(y, pred, average=None, labels=range(4))
    rows.append({"condition": name, "n_features": len(CONDITIONS[name]),
                 "macro_F1": f1_score(y, pred, average="macro", labels=range(4)),
                 **{f"F1_{e}": v for e, v in zip(EMOTIONS, f1_cls)}})
    cm = pd.DataFrame(confusion_matrix(y, pred, labels=range(4)),
                      index=[f"true_{e}" for e in EMOTIONS], columns=[f"pred_{e}" for e in EMOTIONS])
    cm.to_csv(OUT / f"confusion_{name}.csv")
    cms[name] = cm
f1_tab = pd.DataFrame(rows)
f1_tab.to_csv(OUT / "classification_f1.csv", index=False)
print(f1_tab.round(3).to_string(index=False))

fig, axes = plt.subplots(1, len(cms), figsize=(4.2 * len(cms), 4), squeeze=False)
for ax, (name, cm) in zip(axes.ravel(), cms.items()):
    norm = cm.values / cm.values.sum(axis=1, keepdims=True)
    ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm.values[i, j]}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(4), [e[:5] for e in EMOTIONS])
    ax.set_yticks(range(4), [e[:5] for e in EMOTIONS])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{name} (F1={f1_tab.set_index('condition').loc[name, 'macro_F1']:.3f})")
fig.tight_layout()
fig.savefig(OUT / "confusion_all.png", dpi=150)
plt.close(fig)

# subject-level paired bootstrap -------------------------------------------------
rng = np.random.default_rng(SEED)
subjects = np.unique(groups)
idx_of = {s: np.where(groups == s)[0] for s in subjects}
PAIRS = [("C2", "C1"), ("C2", "C4"), ("C6", "C5")]
diffs = {f"{a}-{b}": [] for a, b in PAIRS}
for _ in range(N_BOOT):
    samp = rng.choice(subjects, size=len(subjects), replace=True)
    ii = np.concatenate([idx_of[s] for s in samp])
    f = {n: f1_score(y[ii], p[ii], average="macro", labels=range(4)) for n, p in preds.items()}
    for a, b in PAIRS:
        diffs[f"{a}-{b}"].append(f[a] - f[b])
obs = f1_tab.set_index("condition")["macro_F1"]
boot = pd.DataFrame([{
    "comparison": k, "observed_diff": obs[k.split("-")[0]] - obs[k.split("-")[1]],
    "ci95_low": np.percentile(v, 2.5), "ci95_high": np.percentile(v, 97.5),
    "boot_mean": np.mean(v), "n_boot": N_BOOT, "n_subjects": len(subjects),
} for k, v in diffs.items()])
boot.to_csv(OUT / "bootstrap_diffs.csv", index=False)
print(boot.round(4).to_string(index=False))

# append to SUMMARY.md (replace an earlier Step 4 section if re-run) --------------
summary = OUT / "SUMMARY.md"
text = summary.read_text(encoding="utf-8") if summary.exists() else "# SUMMARY\n"
text = text.split("\n## Step 4")[0].rstrip() + "\n"
sec = ["\n## Step 4 — Classification\n",
       "Leave-one-subject-out, 4 classes, nearest centroid (Euclidean) on z-scores fitted on "
       f"training folds only. Bootstrap: {N_BOOT} subject-level resamples, same predictions.\n",
       "### Macro-F1\n", f1_tab.round(4).to_markdown(index=False) + "\n",
       "### Bootstrap 95% CI of macro-F1 difference\n", boot.round(4).to_markdown(index=False) + "\n",
       "### Confusion matrices (counts; rows = true)\n", "![confusion](confusion_all.png)\n"]
for name, cm in cms.items():
    sec += [f"\n**{name}**\n", cm.to_markdown() + "\n"]
summary.write_text(text + "".join(s if s.endswith("\n") else s + "\n" for s in sec), encoding="utf-8")
print("Saved to", OUT)
