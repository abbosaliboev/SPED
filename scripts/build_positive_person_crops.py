# build_positive_person_crops_wheelchair_fix_log.py
import cv2, math
from pathlib import Path
from ultralytics import YOLO

# ============== CONFIG ==============
TRAIN_DIR = r"C:/Users/dalab/Desktop/Abbos/SmartLight/dataset/train"
VAL_DIR   = r"C:/Users/dalab/Desktop/Abbos/SmartLight/dataset/valid"
OUT_DS    = r"C:/Users/dalab/Desktop/Abbos/SmartLight/person_attr_ds"

PERSON_MODEL_PATH = "yolov8s.pt"   # COCO person modeli
DEVICE_ID2NAME = {0: "uses_crutches", 1: "wheelchair_user"}  # YOLO id -> chiqish klass

# Person deteksiya
PERSON_CONF = 0.45
MIN_PERS_AREA = 40 * 40

# Matching mezonlari (umumiy + crutches)
IOU_TH = 0.20
BOTTOM_FRAC = 0.60  # device markazi person pastki 60% ichida bo'lsa ham qabul

# Wheelchair uchun maxsus mezonlar
WHEEL_LWR_IOU_MIN  = 0.08
WHEEL_SCORE_TH     = 0.55
WHEEL_CENTER_BONUS = 0.6
WHEEL_LWR_IOU_W    = 1.2
WHEEL_IOU_W        = 0.6
WHEEL_CLOSE_W      = 0.4
WHEEL_STAND_PEN    = 0.4

# Logging
LOG_EVERY = 200       # har 200-rasmda progress chiqaradi
VERBOSE_SAVE = False  # True qilsangiz har bir saqlangan crop yo'lini ham chop etadi

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ============== UTILS ==============
def area(b):
    x1,y1,x2,y2 = b; return max(0,x2-x1) * max(0,y2-y1)

def iou(a,b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw,ih = max(0,ix2-ix1), max(0,iy2-iy1)
    inter = iw*ih
    if inter <= 0: return 0.0
    ua = area(a) + area(b) - inter
    return inter/ua if ua>0 else 0.0

def yolo_to_xyxy(cx,cy,w,h,W,H):
    x1 = int((cx - w/2) * W); y1 = int((cy - h/2) * H)
    x2 = int((cx + w/2) * W); y2 = int((cy + h/2) * H)
    return max(0,x1), max(0,y1), min(W-1,x2), min(H-1,y2)

def box_center(b):
    x1,y1,x2,y2 = b
    return (0.5*(x1+x2), 0.5*(y1+y2))

def center_in_box(cx, cy, b):
    x1,y1,x2,y2 = b
    return (x1 <= cx <= x2) and (y1 <= cy <= y2)

def person_lower_half(pbox, frac=0.5):
    x1,y1,x2,y2 = pbox
    h = y2 - y1
    cut = int(y1 + (1.0 - frac) * h)
    return (x1, cut, x2, y2)

def center_in_bottom_fraction(cx, cy, pbox, frac=0.60):
    x1,y1,x2,y2 = pbox
    y_cut = y1 + (y2 - y1)*(1.0 - frac)
    return (x1 <= cx <= x2) and (cy >= y_cut)

def ensure_out_dirs(root: Path):
    for sp in ["train","val"]:
        for c in ["uses_crutches","wheelchair_user"]:
            (root/sp/c).mkdir(parents=True, exist_ok=True)

def list_images_and_roots(split_dir: Path):
    """images/ bo'lsa o'shani, bo'lmasa ildizni qidiradi; labels/ni qaytaradi."""
    images_dir = split_dir / "images"
    search_root = images_dir if images_dir.exists() else split_dir
    imgs = [p for p in search_root.rglob("*") if p.suffix.lower() in IMG_EXTS]
    labels_dir = split_dir / "labels"
    return imgs, labels_dir, search_root

# -------- wheelchair uchun mos PERSON tanlash skori --------
def wheelchair_match_score(pbox, dbox):
    lwr  = iou(person_lower_half(pbox, 0.5), dbox)
    full = iou(pbox, dbox)
    pcx,pcy = box_center(pbox)
    dcx,dcy = box_center(dbox)

    dw, dh = dbox[2]-dbox[0], dbox[3]-dbox[1]
    diag = max(1.0, math.hypot(dw, dh))
    dist = math.hypot(pcx-dcx, pcy-dcy) / diag
    close = max(0.0, 1.0 - min(1.5, dist))

    score = 0.0
    score += WHEEL_LWR_IOU_W * lwr
    score += WHEEL_IOU_W * full
    score += WHEEL_CLOSE_W * close
    if center_in_box(pcx, pcy, dbox):
        score += WHEEL_CENTER_BONUS

    px1,py1,px2,py2 = pbox
    dx1,dy1,dx2,dy2 = dbox
    ph, pw = (py2-py1), (px2-px1)
    if py1 < dy1 - 0.12*ph and (ph/max(pw,1)) > 1.8:
        score -= WHEEL_STAND_PEN

    return score, lwr, full, close

# ============== CORE ==============
def process_split(split_dir: str, split_name: str, person_model: YOLO, out_root: Path):
    print(f"\n=== [{split_name.upper()}] START ===")
    imgs, labels_dir, images_root = list_images_and_roots(Path(split_dir))
    print(f"[{split_name}] images_root: {images_root}")
    print(f"[{split_name}] labels_dir : {labels_dir}")
    print(f"[{split_name}] images_found: {len(imgs)}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels folder topilmadi: {labels_dir}")

    saved = {"uses_crutches":0, "wheelchair_user":0}
    processed = 0

    for ip in imgs:
        processed += 1
        if processed % LOG_EVERY == 0:
            print(f"[{split_name}] processing #{processed}: {ip}")

        lp = labels_dir / (ip.stem + ".txt")
        if not lp.exists():
            continue

        im = cv2.imread(str(ip))
        if im is None:
            continue
        H, W = im.shape[:2]

        # 1) device bbox'lari
        devices = []
        with open(lp, "r", encoding="utf-8") as f:
            for line in f:
                ss = line.strip().split()
                if len(ss) < 5: 
                    continue
                cls = int(float(ss[0]))
                if cls not in DEVICE_ID2NAME:
                    continue
                cx,cy,w,h = map(float, ss[1:5])
                dbox = yolo_to_xyxy(cx,cy,w,h,W,H)
                devices.append((dbox, DEVICE_ID2NAME[cls]))

        if not devices:
            continue

        # 2) PERSON deteksiya
        pres = person_model.predict(im, verbose=False, conf=PERSON_CONF)[0]
        persons = []
        if pres.boxes is not None:
            for b in pres.boxes:
                if int(b.cls.item()) == 0:
                    x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                    pbox = (x1,y1,x2,y2)
                    if area(pbox) >= MIN_PERS_AREA:
                        persons.append(pbox)
        if not persons:
            continue

        # 3) DEVICE -> mos PERSON (wheelchair uchun maxsus)
        for dbox, dname in devices:
            chosen_p = None

            if dname == "wheelchair_user":
                best = (-1e9, 0, 0, 0, None)  # score, lwr, full, close, pbox
                for pbox in persons:
                    score, lwr, full, close = wheelchair_match_score(pbox, dbox)
                    if score > best[0]:
                        best = (score, lwr, full, close, pbox)
                score, lwr, full, close, pbox = best
                if pbox is not None:
                    pcx,pcy = box_center(pbox)
                    ok_geom = (lwr >= WHEEL_LWR_IOU_MIN) or center_in_box(pcx, pcy, dbox)
                    if ok_geom and score >= WHEEL_SCORE_TH:
                        chosen_p = pbox

            else:  # uses_crutches
                best_iou, best_p = 0.0, None
                for pbox in persons:
                    ov = iou(pbox, dbox)
                    if ov > best_iou:
                        best_iou, best_p = ov, pbox
                if best_iou >= IOU_TH and best_p is not None:
                    chosen_p = best_p
                else:
                    dcx,dcy = box_center(dbox)
                    for pbox in persons:
                        if center_in_bottom_fraction(dcx, dcy, pbox, frac=BOTTOM_FRAC):
                            chosen_p = pbox
                            break

            if chosen_p is None:
                continue

            x1,y1,x2,y2 = chosen_p
            crop = im[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            outp = Path(out_root, split_name, dname, f"{ip.stem}_{x1}_{y1}.jpg")
            cv2.imwrite(str(outp), crop)
            saved[dname] += 1
            if VERBOSE_SAVE:
                print(f"[{split_name}] SAVED -> {dname}: {outp}")

    print(f"[{split_name}] DONE | processed_imgs: {processed} | uses_crutches: {saved['uses_crutches']} | wheelchair_user: {saved['wheelchair_user']}")
    print(f"=== [{split_name.upper()}] END ===\n")

def main():
    out_root = Path(OUT_DS)
    ensure_out_dirs(out_root)
    person_model = YOLO(PERSON_MODEL_PATH)

    process_split(TRAIN_DIR, "train", person_model, out_root)
    process_split(VAL_DIR,   "val",   person_model, out_root)

    print(f"[ALL] Crops saved under: {OUT_DS}")

if __name__ == "__main__":
    main()
