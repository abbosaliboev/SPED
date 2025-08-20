import cv2
import json
import time
import math
import numpy as np
from pathlib import Path
from collections import deque, defaultdict, Counter
from ultralytics import YOLO

# =========================
# =======  CONFIG  ========
# =========================

# 1) Model yo'llari
PERSON_MODEL_PATH = "yolov8s.pt"        # COCO (person=0)
DEVICE_MODEL_PATH = "best2.pt"           # Sizning custom model

# 2) Video manbasi
VIDEO_SOURCE = "new_test.mp4"  # 0=webcam, yoki "C:/path/to/video.mp4"

# 3) Klass nomlari (yaml bilan mos)
DEVICE_CLASS_NAMES = ["crutches", "wheelchair"]

# 4) Threshold va barqarorlik sozlamalari
PERSON_CONF_TH = 0.45
DEVICE_CONF_TH = 0.35

CLASS_CONF = {
    "crutches": 0.45,      # pramda chalkashish ko'proq bo'lishi mumkin, biroz yuqoriroq qildik
    "wheelchair": 0.50
}

MIN_DEVICE_AREA = 24 * 24    # juda kichik device bboxlar tashlab yuboriladi
MIN_PERSON_AREA = 40 * 40    # juda kichik person bboxlar tashlab yuboriladi

MATCH_DIST_TH  = 110         # person markazi ↔ device markazi maksimal masofa (px)
MATCH_IOU_TH   = 0.05        # person-device bbox IoU gapi (nol bo'lmasin)

# Temporal histerezis
TEMP_WIN  = 18               # oynacha uzunligi (kadr)
ON_HITS   = 9                # yoqish uchun min True soni
OFF_MISS  = 10               # ON holatida ketma-ket qancha miss bo'lsa o'chadi
LABEL_VOTE_K = 8             # label ovoz berish oynasi

# Qo‘shimcha yashil sekundlarni hisoblash (demo)
EXTRA_SEC_MIN = 5
EXTRA_SEC_MAX = 15

# ROI fayli (ixtiyoriy). Agar mavjud bo'lsa, device markazi shu poligon ichida bo'lgandagina hisobga olinadi.
ROI_JSON = "roi.json"  # format: {"points": [[x1,y1], [x2,y2], ...]}

# Log va overlaylar
DECISION_COOLDOWN_SEC = 2.0  # har necha sekundda qaror chiqarishni konsolga yozamiz
SHOW_FPS = True


# =========================
# =====  UTILITIES  =======
# =========================

def xyxy_to_cxcywh(box):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return cx, cy, w, h

def area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = area(a) + area(b) - inter
    return inter / ua if ua > 0 else 0.0

def point_in_poly(pt, poly):
    # Ray casting algoritmi
    if poly is None or len(poly) < 3:
        return True
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / ( (y2 - y1) + 1e-6 ) + x1):
            inside = not inside
    return inside

def point_in_bottom_half(px, py, person_box):
    x1, y1, x2, y2 = person_box
    mid_y = y1 + (y2 - y1) * 0.5
    return (x1 <= px <= x2) and (mid_y <= py <= y2)

def euclidean(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# =========================
# ======  TRACKING  =======
# =========================

class SimplePersonTracker:
    """
    Juda sodda treker: kadrlar orasida person markazlari yaqinligiga qarab ID saqlaydi.
    Real loyihada ByteTrack/OC-SORT o'rnatish tavsiya etiladi, lekin bu demo uchun yetarli.
    """
    def __init__(self, max_dist=160, max_age=20):
        self.next_id = 1
        self.tracks = {}   # id -> {"box":(x1,y1,x2,y2), "cxcy":(cx,cy), "last_seen":frame_idx}
        self.max_dist = max_dist
        self.max_age = max_age

    def update(self, persons, frame_idx):
        live_ids = list(self.tracks.keys())
        used_tracks = set()
        out = []
        for box, conf in persons:
            cx, cy, _, _ = xyxy_to_cxcywh(box)
            best_id, best_d = None, 1e9
            for tid in live_ids:
                if tid in used_tracks:
                    continue
                pcx, pcy = self.tracks[tid]["cxcy"]
                d = euclidean((cx, cy), (pcx, pcy))
                if d < best_d:
                    best_d, best_id = d, tid
            if best_d <= self.max_dist:
                self.tracks[best_id].update({"box": box, "cxcy": (cx, cy), "last_seen": frame_idx})
                used_tracks.add(best_id)
                out.append((best_id, box, conf))
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"box": box, "cxcy": (cx, cy), "last_seen": frame_idx}
                used_tracks.add(tid)
                out.append((tid, box, conf))

        # eskirgan tracklarni tozalash
        to_del = [tid for tid, tk in self.tracks.items() if frame_idx - tk["last_seen"] > self.max_age]
        for tid in to_del:
            del self.tracks[tid]

        return out  # [(id, box, conf), ...]

class HysteresisConfirm:
    """
    Temporal smoothing kuchaytirilgan: ON bo'lish uchun "ko'p tasdiq",
    ON holatida o'chirish uchun "ketma-ket miss" talab qiladi.
    Label voting bilan qo'shimcha barqarorlik beradi.
    """
    def __init__(self, win=TEMP_WIN, on_hits=ON_HITS, off_miss=OFF_MISS, vote_k=LABEL_VOTE_K):
        self.win = win
        self.on_hits = on_hits
        self.off_miss = off_miss
        self.buf = defaultdict(lambda: deque(maxlen=self.win))  # True/False
        self.lbl = defaultdict(lambda: deque(maxlen=vote_k))    # label names
        self.state = {}        # pid -> bool (ON/OFF)
        self.miss = defaultdict(int)

    def update(self, pid, detected, label=None):
        self.buf[pid].append(bool(detected))
        if label:
            self.lbl[pid].append(label)

        st = self.state.get(pid, False)
        if not st:
            # OFF -> ON: oynadagi True soni kifoya bo'lsa
            if sum(self.buf[pid]) >= self.on_hits:
                self.state[pid] = True
                self.miss[pid] = 0
        else:
            # ON holatida miss hisoblash
            if not detected:
                self.miss[pid] += 1
            else:
                self.miss[pid] = 0
            if self.miss[pid] >= self.off_miss:
                self.state[pid] = False
                self.miss[pid] = 0

        return self.state.get(pid, False)

    def voted_label(self, pid):
        if len(self.lbl[pid]) == 0:
            return None
        return Counter(self.lbl[pid]).most_common(1)[0][0]


# =========================
# =====  EXTENSION  =======
# =========================

def estimate_extra_seconds(person_box, device_name):
    # Oddiy demo qoidasi: sinfga qarab asosiy baza
    base = 8
    if device_name == "wheelchair":
        base += 3
    elif device_name == "crutches":
        base += 2
    elif device_name == "pram":
        base += 1
    return clamp(base, EXTRA_SEC_MIN, EXTRA_SEC_MAX)


# =========================
# ========  MAIN  =========
# =========================

def load_roi_polygon(json_path):
    p = Path(json_path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        pts = data.get("points", [])
        poly = [(float(x), float(y)) for x, y in pts]
        return poly if len(poly) >= 3 else None
    except Exception:
        return None

def draw_polygon(frame, poly, color=(255, 0, 255), thickness=2):
    if poly is None or len(poly) < 2:
        return
    pts = np.array(poly, dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)

def main():
    # ROI poligonini (bo'lsa) yuklaymiz
    crosswalk_poly = load_roi_polygon(ROI_JSON)

    # Modellar
    person_model = YOLO(PERSON_MODEL_PATH)
    device_model = YOLO(DEVICE_MODEL_PATH)

    # Device klass nomlarini tayyorlaymiz
    device_names = {int(k): str(v).lower() for k, v in device_model.names.items()}
    valid_device_set = set(n.lower() for n in DEVICE_CLASS_NAMES)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {VIDEO_SOURCE}")

    tracker = SimplePersonTracker(max_dist=160, max_age=20)
    tconfirm = HysteresisConfirm(win=TEMP_WIN, on_hits=ON_HITS, off_miss=OFF_MISS, vote_k=LABEL_VOTE_K)

    last_decision_time = 0.0
    frame_idx = 0
    t0 = time.time()
    fps = 0.0

    print("[INFO] Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        H, W = frame.shape[:2]

        # ========== 1) PERSON DETECTION ==========
        pres = person_model.predict(frame, verbose=False, conf=PERSON_CONF_TH)[0]
        person_boxes = []
        if pres.boxes is not None and len(pres.boxes) > 0:
            for b in pres.boxes:
                cls = int(b.cls.item())
                if cls == 0:  # COCO 'person'
                    x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                    box = (x1, y1, x2, y2)
                    if area(box) < MIN_PERSON_AREA:
                        continue
                    conf = float(b.conf.item())
                    person_boxes.append((box, conf))

        # ========== 2) DEVICE DETECTION ==========
        dres = device_model.predict(frame, verbose=False, conf=DEVICE_CONF_TH)[0]
        device_dets = []  # (box, conf, name)
        if dres.boxes is not None and len(dres.boxes) > 0:
            for b in dres.boxes:
                cls = int(b.cls.item())
                name = device_names.get(cls, str(cls)).lower()
                if name not in valid_device_set:
                    continue
                conf = float(b.conf.item())
                min_conf = CLASS_CONF.get(name, DEVICE_CONF_TH)
                if conf < min_conf:
                    continue
                x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                box = (x1, y1, x2, y2)
                if area(box) < MIN_DEVICE_AREA:
                    continue
                # ROI: device markazi poligon ichida bo'lsin (agar poly mavjud bo'lsa)
                dcx, dcy, _, _ = xyxy_to_cxcywh(box)
                if not point_in_poly((dcx, dcy), crosswalk_poly):
                    continue
                device_dets.append((box, conf, name))

        # ========== 3) TRACKING ==========
        tracked_persons = tracker.update(person_boxes, frame_idx)  # [(pid, box, conf), ...]

        # ========== 4) MATCHING (device ↔ person) ==========
        pid_to_device = {}  # pid -> (device_name, conf)
        for (pid, pbox, pconf) in tracked_persons:
            pcx, pcy, _, _ = xyxy_to_cxcywh(pbox)
            best, best_score = None, -1e9
            for (dbox, dconf, dname) in device_dets:
                dcx, dcy, _, _ = xyxy_to_cxcywh(dbox)

                # device markazi person bbox pastki yarimida bo'lsin
                if not point_in_bottom_half(dcx, dcy, pbox):
                    continue

                # masofa va IoU filtrlash
                d = euclidean((pcx, pcy), (dcx, dcy))
                if d > MATCH_DIST_TH:
                    continue
                ov = iou(pbox, dbox)
                if ov < MATCH_IOU_TH:
                    continue

                # aralash skor: katta IoU va kichik masofa ijobiy
                score = (ov * 1.0) + (max(0.0, MATCH_DIST_TH - d) / MATCH_DIST_TH) * 0.5
                if score > best_score:
                    best_score = score
                    best = (dbox, dconf, dname)
            if best is not None:
                pid_to_device[pid] = (best[2], best[1])  # (name, conf)

        # ========== 5) TEMPORAL HYSTERESIS + VOTING ==========
        confirmed_pids = []
        for (pid, pbox, _) in tracked_persons:
            has_device = pid in pid_to_device
            cur_label = pid_to_device[pid][0] if has_device else None
            is_on = tconfirm.update(pid, has_device, label=cur_label)
            if is_on:
                confirmed_pids.append(pid)

        # ========== 6) DECISION (EXTEND GREEN) ==========
        now = time.time()
        to_extend = 0
        reason = []
        for pid in confirmed_pids:
            # voted label -> yanada barqaror sinf
            voted = tconfirm.voted_label(pid) or (pid_to_device[pid][0] if pid in pid_to_device else None)
            if not voted:
                continue
            pbox = next(b for (ppid, b, _) in tracked_persons if ppid == pid)
            ext = estimate_extra_seconds(pbox, voted)
            to_extend = max(to_extend, ext)
            reason.append(f"ID{pid}:{voted} -> +{ext}s")

        if to_extend > 0 and (now - last_decision_time) > DECISION_COOLDOWN_SEC:
            last_decision_time = now
            print(f"[DECISION] Extend green by +{to_extend}s | " + "; ".join(reason))

        # ========== 7) OVERLAY ==========
        # ROI chizish
        draw_polygon(frame, crosswalk_poly, (255, 0, 255), 2)

        # Person bbox
        for (pid, pbox, pconf) in tracked_persons:
            x1, y1, x2, y2 = map(int, pbox)
            color = (0, 255, 0) if pid in confirmed_pids else (255, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID{pid} person"
            if pid in pid_to_device:
                dname, dconf = pid_to_device[pid]
                label += f" + {dname}"
            cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        # Device bbox
        for (dbox, dconf, dname) in device_dets:
            x1, y1, x2, y2 = map(int, dbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 255), 2)
            cv2.putText(frame, f"{dname} {dconf:.2f}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 255), 2, cv2.LINE_AA)

        # Decision overlay
        if to_extend > 0:
            cv2.putText(frame, f"EXTEND GREEN: +{to_extend}s", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 0), 3, cv2.LINE_AA)

        # FPS
        if SHOW_FPS:
            dt = time.time() - t0
            if dt > 0:
                fps = 1.0 / dt
            t0 = time.time()
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 200, 255), 2, cv2.LINE_AA)

        cv2.imshow("SmartCare AI Light - Stable Demo", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
