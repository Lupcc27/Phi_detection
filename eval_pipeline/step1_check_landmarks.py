"""Step 1: draw candidate landmark indices on 8 random images (2 per emotion), 2x zoom.

Run from anywhere:  python eval_pipeline/step1_check_landmarks.py
Output: results/landmark_check_*.png
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
PRE = ROOT / "ck_dataset" / "preprocesado"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

EMOTIONS = ["enojado", "feliz", "triste", "sorprendido"]
ZOOM = 2

# name -> index (BGR color per group)
GROUPS = {
    "mouth":  ((0, 0, 255),   {"CH_R": 61, "CH_L": 291, "ST_up": 13, "ST_lo": 14}),
    "nose":   ((0, 200, 0),   {"AL_R": 129, "AL_L": 358}),
    "eyes":   ((255, 0, 0),   {"LC_R": 33, "LC_L": 263, "IC_R": 133, "IC_L": 362}),
    "chin":   ((255, 0, 255), {"ME": 152}),
    "brows":  ((0, 165, 255), {"brIn_R": 55, "brIn_L": 285, "brMid_R": 105, "brMid_L": 334,
                               "brOut_R": 70, "brOut_L": 300}),
    "lids":   ((255, 255, 0), {"lidU_R": 159, "lidL_R": 145, "lidU_L": 386, "lidL_L": 374}),
}

meta = pd.read_csv(PRE / "metadata.csv")
npz = np.load(PRE / "landmarks.npz", allow_pickle=True)
ids = [str(i) for i in npz["img_id"]]
lm = npz["landmarks"]  # (N, 478, 3), pixels of the 288x288 image
row_of = {i: k for k, i in enumerate(ids)}

print(meta["clase"].value_counts())
missing = [e for e in EMOTIONS if e not in set(meta["clase"])]
if missing:
    raise SystemExit(f"Class names not found in metadata.clase: {missing}. Present: {sorted(set(meta['clase']))}")

rng = np.random.default_rng(SEED)
picked = []
for e in EMOTIONS:
    sub = meta[(meta["clase"] == e) & meta["rostro_detectado"] & meta["img_id"].isin(row_of)]
    picked += [(e, r) for r in sub.iloc[rng.choice(len(sub), 2, replace=False)].itertuples()]


def draw(img_bgr, pts):
    img = cv2.resize(img_bgr, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_CUBIC)
    for color, d in GROUPS.values():
        for name, idx in d.items():
            x, y = pts[idx, :2] * ZOOM
            cv2.circle(img, (int(round(x)), int(round(y))), 3, color, -1, cv2.LINE_AA)
            # right-side points label to the left, left-side to the right, to reduce overlap
            dx = -6 - 8 * len(f"{idx}") if name.endswith("_R") else 6
            cv2.putText(img, str(idx), (int(x) + dx, int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, color, 1, cv2.LINE_AA)
    return img


tiles = []
for e, r in picked:
    img = cv2.imread(str(ROOT / r.ruta), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Cannot read {ROOT / r.ruta}")
    pts = lm[row_of[r.img_id]]
    # also save each image on its own
    tile = draw(img, pts)
    cv2.putText(tile, f"{e} {r.img_id} {r.sujeto}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT / f"landmark_check_{e}_{r.img_id}.png"), tile)
    tiles.append(tile)

# overview: 2 rows x 4 cols (columns = emotions, rows = the two samples)
grid = np.vstack([np.hstack(tiles[0::2]), np.hstack(tiles[1::2])])
cv2.imwrite(str(OUT / "landmark_check_overview.png"), grid)

# legend
leg = np.full((130, 576, 3), 255, np.uint8)
y = 18
for g, (color, d) in GROUPS.items():
    cv2.putText(leg, f"{g}: " + ", ".join(f"{n}={i}" for n, i in d.items()), (6, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    y += 20
cv2.imwrite(str(OUT / "landmark_check_legend.png"), leg)
print("Saved to", OUT)
