# coco_person_to_no_device.py
# COCO 2017: download images train2017/val2017 and annotations instances_{split}2017.json

import json, random, os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ======== CONFIG ========
COCO_IMG_DIR   = r"/path/coco/train2017"     # or val2017 (you can run twice)
COCO_ANN_JSON  = r"/path/coco/annotations/instances_train2017.json"
OUT_ROOT       = r"/path/person_attr_ds"     # will write to train/val/no_device
TARGET_COUNT   = 5700                        # ≈ 3 * min(crutches, wheelchair)
SPLIT_RATIO    = 0.85                        # train/val
# Taqsimotni senga o‘xshatish uchun konservativ filtrlash:
MIN_W, MIN_H   = 64, 96                      # juda kichik bboxlarni tashla
CENTER_XY_WIN  = (0.2, 0.8)                  # markaz (x,y) 20%~80% oralig‘ida
W_FRAC_RANGE   = (0.10, 0.55)                # bbox_width/ImgWidth
H_FRAC_RANGE   = (0.20, 0.85)                # bbox_height/ImgHeight
SEED           = 42

# Agar stratifikatsiya xohlasang, bu binlar bilan teng ulushda tanlaymiz
WIDTH_BINS  = [0.10, 0.18, 0.26, 0.34, 0.42, 0.55]
HEIGHT_BINS = [0.20, 0.35, 0.50, 0.65, 0.85]

# ======== CODE ========
random.seed(SEED)
Path(OUT_ROOT, "train", "no_device").mkdir(parents=True, exist_ok=True)
Path(OUT_ROOT, "val",   "no_device").mkdir(parents=True, exist_ok=True)

print("[INFO] Loading COCO annotations...")
with open(COCO_ANN_JSON, "r") as f:
    coco = json.load(f)

imgs_by_id = {im["id"]: im for im in coco["images"]}
# COCO category id for 'person' is 1 in 2017 annotations
PERSON_CAT_ID = next((c["id"] for c in coco["categories"] if c["name"]=="person"), 1)

candidates = []  # (img_path, (x1,y1,x2,y2), (wfrac,hfrac,xc,yc))
for ann in tqdm(coco["annotations"], desc="Scan anns"):
    if ann.get("iscrowd", 0) != 0: 
        continue
    if ann["category_id"] != PERSON_CAT_ID:
        continue
    img = imgs_by_id.get(ann["image_id"])
    if img is None: 
        continue
    img_path = Path(COCO_IMG_DIR, img["file_name"])
    if not img_path.exists(): 
        continue

    x, y, w, h = ann["bbox"]  # COCO bbox: top-left + width/height (absolute)
    W, H = img["width"], img["height"]
    if w < MIN_W or h < MIN_H:
        continue

    # frac va markaz
    wfrac = w / W
    hfrac = h / H
    xc = (x + w/2) / W
    yc = (y + h/2) / H

    if not (W_FRAC_RANGE[0] <= wfrac <= W_FRAC_RANGE[1]):
        continue
    if not (H_FRAC_RANGE[0] <= hfrac <= H_FRAC_RANGE[1]):
        continue
    if not (CENTER_XY_WIN[0] <= xc <= CENTER_XY_WIN[1]):
        continue
    if not (CENTER_XY_WIN[0] <= yc <= CENTER_XY_WIN[1]):
        continue

    # bbox to xyxy with small padding clamp
    pad = 0
    x1 = max(0, int(x - pad)); y1 = max(0, int(y - pad))
    x2 = min(W-1, int(x + w + pad)); y2 = min(H-1, int(y + h + pad))

    candidates.append((img_path, (x1,y1,x2,y2), (wfrac,hfrac,xc,yc)))

if not candidates:
    raise SystemExit("No candidates after filters. Relax ranges.")

# ---- Stratified sample by (width,height) bins so distribution looks like yours ----
def bin_index(val, edges):
    for i in range(len(edges)-1):
        if edges[i] <= val < edges[i+1]:
            return i
    return len(edges)-2  # clamp to last

from collections import defaultdict
groups = defaultdict(list)
for it in candidates:
    _, _, (wf,hf,_,_) = it
    wb = bin_index(wf, WIDTH_BINS)
    hb = bin_index(hf, HEIGHT_BINS)
    groups[(wb,hb)].append(it)

# How many per group?
cells = list(groups.keys())
cells.sort()
per_cell = max(1, TARGET_COUNT // max(1, len(cells)))
picked = []
for c in cells:
    bucket = groups[c]
    random.shuffle(bucket)
    take = min(per_cell, len(bucket))
    picked.extend(bucket[:take])
# If still short, fill randomly
if len(picked) < TARGET_COUNT:
    remain = [it for it in candidates if it not in picked]
    random.shuffle(remain)
    picked.extend(remain[:TARGET_COUNT-len(picked)])
# If too many, trim
picked = picked[:TARGET_COUNT]
print(f"[INFO] Selected {len(picked)} no_device crops.")

# ---- Save crops with train/val split ----
n_train = int(len(picked) * SPLIT_RATIO)
for idx, (img_path, (x1,y1,x2,y2), _) in enumerate(tqdm(picked, desc="Save crops")):
    split = "train" if idx < n_train else "val"
    try:
        im = Image.open(img_path).convert("RGB")
        crop = im.crop((x1,y1,x2,y2))
        outp = Path(OUT_ROOT, split, "no_device", f"coco_nd_{idx:06d}.jpg")
        crop.save(outp, quality=95)
    except Exception as e:
        # skip unreadable images
        continue

print("[DONE] COCO -> no_device mining finished.")
print(f"Saved: train≈{n_train}, val≈{len(picked)-n_train}  (target={TARGET_COUNT})")
