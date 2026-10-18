"""Step 2: P (phi) and S (standard) features, per-subject neutral reference, deltas, crossings.

Run:  python eval_pipeline/step2_features.py
Output: results/features.csv, results/crossings.csv, results/neutral_reference.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRE = ROOT / "ck_dataset" / "preprocesado"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

PHI = (1 + 5 ** 0.5) / 2
EMOTIONS = ["enojado", "feliz", "triste", "sorprendido"]

# MediaPipe indices (R = image-left side idx, L = image-right side idx; edit here after Step 1 review)
CH = (61, 291)
ST = (13, 14)
AL = (129, 358)
LC = (33, 263)
IC = (133, 362)
ME = 152
BR_IN = (55, 285)
BR_OUT = (70, 300)
LID_UP = (159, 386)
LID_LO = (145, 374)

P_IDS = ["P1", "P2", "P3a", "P3b", "P4", "P5", "P6"]
S_IDS = ["S1", "S2", "S3", "S4", "S5", "S6"]


def roll_correct(pts):
    """Rotate (478,2) points about the LC midpoint so the LC-LC line is horizontal."""
    a, b = pts[LC[0]], pts[LC[1]]
    ang = np.arctan2(b[1] - a[1], b[0] - a[0])
    c, s = np.cos(-ang), np.sin(-ang)
    R = np.array([[c, -s], [s, c]])
    center = (a + b) / 2
    return (pts - center) @ R.T + center


def features(pts):
    p = roll_correct(pts[:, :2].astype(float))
    x, y = p[:, 0], p[:, 1]

    ch_w = abs(x[CH[1]] - x[CH[0]])
    al_w = abs(x[AL[1]] - x[AL[0]])
    lc_w = abs(x[LC[1]] - x[LC[0]])
    ic_w = abs(x[IC[1]] - x[IC[0]])

    # vertical levels (image y grows downward, so "below" = larger y)
    y_lc = y[list(LC)].mean()
    y_st = (y[ST[0]] + y[ST[1]]) / 2
    y_ch = y[list(CH)].mean()
    y_ln = y[list(AL)].mean()
    y_me = y[ME]

    f = {
        "P1": ch_w / al_w,
        "P2": lc_w / ch_w,
        "P3a": (y_st - y_lc) / (y_me - y_st),
        "P3b": (y_ch - y_lc) / (y_me - y_ch),
        "P4": (y_me - y_ln) / (y_ln - y_lc),
        "P5": (y_ln - y_lc) / (y_ch - y_ln),
        "P6": (y_me - y_ch) / (y_ch - y_ln),
    }

    # S set (normalized by IC-IC unless the table says otherwise)
    brow_in_h = np.mean([y[IC[i]] - y[BR_IN[i]] for i in range(2)]) / ic_w
    brow_out_h = np.mean([y[LC[i]] - y[BR_OUT[i]] for i in range(2)]) / ic_w
    brow_gap = abs(x[BR_IN[1]] - x[BR_IN[0]]) / ic_w
    eye_open = np.mean([
        (y[LID_LO[i]] - y[LID_UP[i]]) / np.hypot(x[LC[i]] - x[IC[i]], y[LC[i]] - y[IC[i]])
        for i in range(2)
    ])
    mouth_open = abs(y[ST[1]] - y[ST[0]]) / ch_w
    corner_h = (y_st - y_ch) / ic_w  # >0: corners above stomion (smile)
    f.update({"S1": brow_in_h, "S2": brow_out_h, "S3": brow_gap,
              "S4": eye_open, "S5": mouth_open, "S6": corner_h})
    return f


meta = pd.read_csv(PRE / "metadata.csv")
npz = np.load(PRE / "landmarks.npz", allow_pickle=True)
ids = [str(i) for i in npz["img_id"]]
lm = npz["landmarks"]

feat = pd.DataFrame([features(l) for l in lm], index=ids)
feat.index.name = "img_id"

# neutral reference per subject: mean over the subject's neutrals that have landmarks
nps = pd.read_csv(PRE / "neutros_por_sujeto.csv")
ref_rows, ref_feat = [], {}
for r in nps.itertuples():
    listed = r.neutros.split(";")
    avail = [i for i in listed if i in feat.index]
    if r.neutro_ref not in avail and r.neutro_ref in feat.index:
        avail.append(r.neutro_ref)
    ref_feat[r.sujeto] = feat.loc[avail].mean()
    ref_rows.append({"sujeto": r.sujeto, "n_listed": len(listed), "n_used": len(avail)})
ref = pd.DataFrame(ref_rows)
ref.to_csv(OUT / "neutral_reference.csv", index=False)
n_multi = (ref.n_used > 1).sum()
print(f"Neutral reference: {n_multi}/{len(ref)} subjects average >1 neutral "
      f"(landmarks.npz only holds images that were preprocessed).")
if n_multi < len(ref) // 2:
    print("WARNING: most subjects fall back to the single medoid neutral because the other "
          "neutrals' landmarks are not in landmarks.npz.")

emo = meta[meta["clase"].isin(EMOTIONS) & meta["img_id"].isin(feat.index)].copy()
rows = []
for r in emo.itertuples():
    peak = feat.loc[r.img_id]
    neu = ref_feat[r.sujeto]
    row = {"img_id": r.img_id, "sujeto": r.sujeto, "clase": r.clase,
           "n_neutros_used": int(ref.loc[ref.sujeto == r.sujeto, "n_used"].iloc[0])}
    for k in P_IDS + S_IDS:
        row[f"r_{k}"] = peak[k]
        row[f"rneu_{k}"] = neu[k]
        row[f"dr_{k}"] = peak[k] - neu[k]
    for k in P_IDS:
        row[f"ddphi_{k}"] = abs(peak[k] - PHI) - abs(neu[k] - PHI)
    rows.append(row)
out = pd.DataFrame(rows)
out.to_csv(OUT / "features.csv", index=False)

# crossings: % of images per emotion x ratio with r_pico, r_neutro on opposite sides of phi
cr = {}
for e in EMOTIONS:
    sub = out[out.clase == e]
    cr[e] = {k: 100 * ((sub[f"r_{k}"] - PHI) * (sub[f"rneu_{k}"] - PHI) < 0).mean() for k in P_IDS}
crossings = pd.DataFrame(cr).T
crossings.index.name = "clase"
crossings["n"] = out.groupby("clase").size().reindex(crossings.index)
crossings.round(1).to_csv(OUT / "crossings.csv")

print(f"features.csv: {len(out)} rows")
print(out.groupby("clase")[[f"r_{k}" for k in P_IDS]].mean().round(2))
print(crossings.round(1))
print("Saved to", OUT)
