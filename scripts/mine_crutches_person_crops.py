# mine_crutches_person_crops.py
# Video -> PERSON croplar (faqat crutches)
# pip install ultralytics opencv-python

import cv2, os, math
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict

# ======== CONFIG ========
VIDEO = "IMG_4923.mp4"   # 0 = webcam yoki video yo'li
OUT   = r"C:/Users/dalab/Desktop/Abbos/SmartLight/crops/uses_crutches"
STEP  = 3                 # har 3-kadrda ishlash
PERSON_CONF = 0.40
DEVICE_CONF = 0.30
MIN_AREA    = 40*40
NMS_IOU     = 0.45
BOTTOM_FRAC = 0.70        # person pastki 70% hududi
EXPAND_PBOX = 0.10        # person bbox'ni tekshiruv uchun 10% kengaytirish
TARGET_DEVICE = "crutches"  # faqat shu klass saqlanadi
VALID_DEVICES = {"crutches", "wheelchair"}

person_model = YOLO("yolov8s.pt")   # COCO person
device_model = YOLO("best2.pt")     # custom: crutches/wheelchair

def area(b):
    x1,y1,x2,y2 = b; return max(0,x2-x1)*max(0,y2-y1)

def iou(a,b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw,ih = max(0,ix2-ix1), max(0,iy2-iy1)
    inter = iw*ih
    if inter<=0: return 0.0
    ua = area(a)+area(b)-inter
    return inter/ua if ua>0 else 0.0

def cxcy(box):
    x1,y1,x2,y2 = box
    return (0.5*(x1+x2), 0.5*(y1+y2))

def expand_box(box, W, H, r=0.10):
    x1,y1,x2,y2 = box
    w,h = x2-x1, y2-y1
    dx, dy = w*r, h*r
    nx1 = max(0, int(x1-dx))
    ny1 = max(0, int(y1-dy))
    nx2 = min(W-1, int(x2+dx))
    ny2 = min(H-1, int(y2+dy))
    return (nx1,ny1,nx2,ny2)

def center_in_box(cx, cy, box):
    x1,y1,x2,y2 = box
    return (x1 <= cx <= x2) and (y1 <= cy <= y2)

def center_in_bottom_fraction(cx, cy, pbox, frac=0.70):
    x1,y1,x2,y2 = pbox
    cut = y1 + (y2-y1)*(1.0-frac)  # pastki frac
    return (x1 <= cx <= x2) and (cy >= cut)

def classwise_nms(dets, iou_thr=0.45):
    # dets: list[(box, conf, name)]
    out = []
    byc = defaultdict(list)
    for i,(b,c,n) in enumerate(dets):
        byc[n].append((i,b,c))
    for n, items in byc.items():
        items.sort(key=lambda x: x[2], reverse=True)
        used = [False]*len(items)
        for i,(idx,bi,ci) in enumerate(items):
            if used[i]: continue
            out.append((bi,ci,n))
            for j,(idxj,bj,cj) in enumerate(items[i+1:], start=i+1):
                if used[j]: continue
                if iou(bi,bj) >= iou_thr:
                    used[j] = True
    return out

def device_matches_person(pbox, dbox):
    """
    Conservative moslik:
    - IoU >= 0.05  YOKI
    - device markazi person ichida VA person pastki 70% hududida
    """
    if iou(pbox, dbox) >= 0.05:
        return True
    dcx, dcy = cxcy(dbox)
    if center_in_box(dcx, dcy, pbox) and center_in_bottom_fraction(dcx, dcy, pbox, BOTTOM_FRAC):
        return True
    return False

def main():
    Path(OUT).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {VIDEO}")

    # device nomlari
    dev_names = {int(k): str(v).lower() for k,v in device_model.names.items()}

    idx = 0
    saved_crutches = 0
    matched_but_small = 0   # ixtiyoriy statistika
    while True:
        ret, frame = cap.read()
        if not ret: break
        idx += 1
        if idx % STEP != 0:
            continue

        H, W = frame.shape[:2]

        # 1) PERSON
        pres = person_model.predict(frame, verbose=False, conf=PERSON_CONF)[0]
        persons = []
        if pres.boxes is not None:
            for b in pres.boxes:
                if int(b.cls.item()) == 0:  # COCO person
                    x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                    pbox = (x1,y1,x2,y2)
                    if area(pbox) >= MIN_AREA:
                        persons.append(pbox)
        if not persons:
            continue

        # 2) DEVICE + NMS
        dres = device_model.predict(frame, verbose=False, conf=DEVICE_CONF)[0]
        raw_devs = []
        if dres.boxes is not None:
            for b in dres.boxes:
                cls = int(b.cls.item())
                name = dev_names.get(cls, str(cls)).lower()
                if name not in VALID_DEVICES:
                    continue
                x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                raw_devs.append(((x1,y1,x2,y2), float(b.conf.item()), name))
        devs = classwise_nms(raw_devs, iou_thr=NMS_IOU)

        # 3) Har bir PERSON uchun faqat CRUTCHES mosligini qidiramiz
        for i, pbox in enumerate(persons):
            pbox_chk = expand_box(pbox, W, H, r=EXPAND_PBOX)

            best_cr_box = None
            best_score = -1.0
            for dbox, dconf, dname in devs:
                if dname != TARGET_DEVICE:
                    continue
                if device_matches_person(pbox_chk, dbox):
                    # eng yaxshi moslik sifatida IoU'ni ishlatamiz
                    ov = iou(pbox, dbox)
                    if ov > best_score:
                        best_score = ov
                        best_cr_box = dbox

            if best_cr_box is None:
                continue  # bu person crutches emas

            # 4) CRUTCHES PERSON crop saqlaymiz
            x1,y1,x2,y2 = pbox
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            fname = f"cr5_{idx:06d}_{i}.jpg"
            cv2.imwrite(str(Path(OUT, fname)), crop)
            saved_crutches += 1

    cap.release()
    print(f"[DONE] Saved CRUTCHES person crops: {saved_crutches}")

if __name__ == "__main__":
    main()
