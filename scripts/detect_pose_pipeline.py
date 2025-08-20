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
PERSON_MODEL_PATH = "yolov8s.pt"          # COCO person
DEVICE_MODEL_PATH = "best2.pt"             # custom: crutches/pram/wheelchair
POSE_MODEL_PATH   = "yolov8s-pose.pt"     # YOLO pose (17 keypoints), yohud "yolo11s-pose.pt"

VIDEO_SOURCE = 0  # 0=webcam, yoki "C:/path/to/video.mp4"
DEVICE_CLASS_NAMES = ["crutches", "wheelchair"]

# Deteksiya thresholdlari
PERSON_CONF_TH = 0.45
DEVICE_CONF_TH = 0.50
CLASS_CONF = {
    "crutches": 0.50,
    "wheelchair": 0.50
}
MIN_DEVICE_AREA = 24 * 24
MIN_PERSON_AREA = 40 * 40

# Matching
MATCH_DIST_TH  = 110
MATCH_IOU_TH   = 0.05

# Temporal histerezis + voting
TEMP_WIN  = 18
ON_HITS   = 9
OFF_MISS  = 10
LABEL_VOTE_K = 8

# Pose-ni qachon yoqamiz?
POSE_TRIGGER_UPPER = 0.45   # device_conf < 0.65 bo'lsa, pose ishlatamiz
ALPHA_DEVICE = 0.7
BETA_POSE    = 0.3

# Qo'shimcha yashil sekund
EXTRA_SEC_MIN = 5
EXTRA_SEC_MAX = 15

# ROI poligon (ixtiyoriy)
ROI_JSON = "roi.json"

DECISION_COOLDOWN_SEC = 2.0
SHOW_FPS = True

# =========================
# =====  UTILITIES  =======
# =========================
def xyxy_to_cxcywh(box):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return x1 + w/2.0, y1 + h/2.0, w, h

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
    if inter <= 0: return 0.0
    ua = area(a) + area(b) - inter
    return inter/ua if ua > 0 else 0.0

def point_in_poly(pt, poly):
    if poly is None or len(poly) < 3: return True
    x, y = pt; inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]; x2, y2 = poly[(i+1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / ((y2 - y1) + 1e-6) + x1):
            inside = not inside
    return inside

def point_in_bottom_half(px, py, person_box):
    x1, y1, x2, y2 = person_box
    mid_y = y1 + (y2 - y1) * 0.5
    return (x1 <= px <= x2) and (mid_y <= py <= y2)

def euclidean(p, q): return math.hypot(p[0]-q[0], p[1]-q[1])
def clamp(v, lo, hi): return max(lo, min(hi, v))

def load_roi_polygon(json_path):
    p = Path(json_path)
    if not p.exists(): return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        pts = data.get("points", [])
        poly = [(float(x), float(y)) for x, y in pts]
        return poly if len(poly) >= 3 else None
    except Exception:
        return None

def draw_polygon(frame, poly, color=(255, 0, 255), thickness=2):
    if poly is None or len(poly) < 2: return
    pts = np.array(poly, dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)

# =========================
# ======  TRACKING  =======
# =========================
class SimplePersonTracker:
    def __init__(self, max_dist=160, max_age=20):
        self.next_id = 1
        self.tracks = {}
        self.max_dist = max_dist
        self.max_age  = max_age

    def update(self, persons, frame_idx):
        live_ids = list(self.tracks.keys())
        used = set()
        out = []
        for box, conf in persons:
            cx, cy, _, _ = xyxy_to_cxcywh(box)
            best_id, best_d = None, 1e9
            for tid in live_ids:
                if tid in used: continue
                pcx, pcy = self.tracks[tid]["cxcy"]
                d = euclidean((cx,cy),(pcx,pcy))
                if d < best_d: best_d, best_id = d, tid
            if best_d <= self.max_dist:
                self.tracks[best_id].update({"box":box, "cxcy":(cx,cy), "last_seen":frame_idx})
                used.add(best_id)
                out.append((best_id, box, conf))
            else:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = {"box":box, "cxcy":(cx,cy), "last_seen":frame_idx}
                used.add(tid)
                out.append((tid, box, conf))
        # expire
        for tid in [tid for tid, tk in self.tracks.items() if frame_idx - tk["last_seen"] > self.max_age]:
            del self.tracks[tid]
        return out

class HysteresisConfirm:
    def __init__(self, win=TEMP_WIN, on_hits=ON_HITS, off_miss=OFF_MISS, vote_k=LABEL_VOTE_K):
        self.win = win; self.on_hits = on_hits; self.off_miss = off_miss
        self.buf = defaultdict(lambda: deque(maxlen=self.win))  # True/False
        self.lbl = defaultdict(lambda: deque(maxlen=vote_k))    # labels
        self.state = {}
        self.miss  = defaultdict(int)

    def update(self, pid, detected_bool, label=None):
        self.buf[pid].append(bool(detected_bool))
        if label: self.lbl[pid].append(label)

        st = self.state.get(pid, False)
        if not st:
            if sum(self.buf[pid]) >= self.on_hits:
                self.state[pid] = True
                self.miss[pid] = 0
        else:
            if not detected_bool:
                self.miss[pid] += 1
            else:
                self.miss[pid] = 0
            if self.miss[pid] >= self.off_miss:
                self.state[pid] = False
                self.miss[pid] = 0
        return self.state.get(pid, False)

    def voted_label(self, pid):
        if len(self.lbl[pid]) == 0: return None
        return Counter(self.lbl[pid]).most_common(1)[0][0]

# =========================
# ========  POSE  =========
# =========================
# COCO-17 keypoint indexlari (YOLOv8-pose tartibi)
KP_NOSE=0; KP_LEYE=1; KP_REYE=2; KP_LEAR=3; KP_REAR=4
KP_LSH=5; KP_RSH=6; KP_LELB=7; KP_RELB=8; KP_LWR=9; KP_RWR=10
KP_LHIP=11; KP_RHIP=12; KP_LKNE=13; KP_RKNE=14; KP_LANK=15; KP_RANK=16

def angle(a, b, c):
    # b markaz nuqta, ABC burchak (gradus)
    ax, ay = a; bx, by = b; cx, cy = c
    v1 = np.array([ax-bx, ay-by]); v2 = np.array([cx-bx, cy-by])
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1<1e-6 or n2<1e-6: return 180.0
    cosang = np.clip(np.dot(v1, v2)/(n1*n2), -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def torso_tilt_deg(kp):
    # yelka markazi vs son (hip) markazi chizig'i vertikalga nisbatan og'ish burchagi
    LSH, RSH = kp[KP_LSH], kp[KP_RSH]
    LHIP, RHIP = kp[KP_LHIP], kp[KP_RHIP]
    if LSH[2]<0.2 or RSH[2]<0.2 or LHIP[2]<0.2 or RHIP[2]<0.2:
        return 0.0
    sh = ((LSH[0]+RSH[0])/2.0, (LSH[1]+RSH[1])/2.0)
    hp = ((LHIP[0]+RHIP[0])/2.0, (LHIP[1]+RHIP[1])/2.0)
    vx, vy = sh[0]-hp[0], sh[1]-hp[1]
    # Vertikal (0,1) bilan burchak
    n = math.hypot(vx, vy); 
    if n<1e-6: return 0.0
    cosang = abs(vy)/n
    cosang = np.clip(cosang, -1.0, 1.0)
    deg = math.degrees(math.acos(cosang))
    return deg  # 0=to'g'ri tik, 90=yassi

def sitting_score(kp):
    # Wheelchair tasdiqlovchi: tizzalar va son burchaklari kichik bo'lsa (bukilgan), score↑
    LHIP, RHIP = kp[KP_LHIP], kp[KP_RHIP]
    LKNE, RKNE = kp[KP_LKNE], kp[KP_RKNE]
    LANK, RANK = kp[KP_LANK], kp[KP_RANK]
    ok = all([LHIP[2]>0.2, RHIP[2]>0.2, LKNE[2]>0.2, RKNE[2]>0.2, LANK[2]>0.2, RANK[2]>0.2])
    if not ok: return 0.0
    # Son burchagi: hip-(knee)-(ankle) -> tizza burchagi
    lknee = angle(LHIP[:2], LKNE[:2], LANK[:2])
    rknee = angle(RHIP[:2], RKNE[:2], RANK[:2])
    # Hip burchagi: shoulder-hip-knee (qo'pol)
    LSH, RSH = kp[KP_LSH], kp[KP_RSH]
    if LSH[2]<0.2 or RSH[2]<0.2: return 0.0
    lhip = angle(LSH[:2], LHIP[:2], LKNE[:2])
    rhip = angle(RSH[:2], RHIP[:2], RKNE[:2])

    # O'tirish: tizza ~90° atrofida, hip ham buralgan (<120°)
    s = 0.0
    for b in [lknee, rknee]:
        if 60 <= b <= 120: s += 0.25
    for b in [lhip, rhip]:
        if 60 <= b <= 130: s += 0.25
    return min(1.0, s)  # [0..1]

def crutch_score(kp):
    # Crutches tasdiqlovchi: torso tilt > ~10° bo'lsa score↑
    # (oddiy heuristika – haqiqiy hayotda "bir tomonga og'ish" ko'p uchraydi)
    deg = torso_tilt_deg(kp)
    # 10° dan boshlab 25° gacha chiziqli oshiramiz
    if deg < 10: return 0.0
    if deg > 25: return 1.0
    return (deg - 10) / (25 - 10)

def run_pose_on_person(pose_model, frame, pbox):
    x1, y1, x2, y2 = map(int, pbox)
    x1 = max(0, x1); y1 = max(0, y1)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    res = pose_model.predict(crop, verbose=False, conf=0.25)[0]
    # Hech kim topilmasa
    if res.keypoints is None or len(res.keypoints) == 0:
        return None

    # Bir nechta kishi bo'lsa eng ishonchli(eng katta conf) ni tanlaymiz
    idx = 0
    if res.boxes is not None and len(res.boxes) > 1:
        try:
            confs = res.boxes.conf.cpu().numpy()
            idx = int(np.argmax(confs))
        except Exception:
            idx = 0

    # (17,2) koordinatalar
    kps = res.keypoints.xy[idx].cpu().numpy()  # shape: (17, 2)

    # (17,) yoki (1,) yoki skalyar bo‘lishi mumkin — hammasini (17,) ga keltiramiz
    if res.keypoints.conf is not None:
        kpc_raw = res.keypoints.conf[idx].cpu().numpy()  # ko'pincha (17,) yoki (1,)
        if np.isscalar(kpc_raw):
            kpc = np.full((17,), float(kpc_raw), dtype=np.float32)
        else:
            kpc_raw = np.array(kpc_raw).reshape(-1)       # 1D ga tekisla
            if kpc_raw.shape[0] == 17:
                kpc = kpc_raw.astype(np.float32)
            elif kpc_raw.shape[0] == 1:
                kpc = np.full((17,), float(kpc_raw[0]), dtype=np.float32)
            else:
                # kutilmagan o'lcham — xavfsiz default
                kpc = np.ones((17,), dtype=np.float32)
    else:
        kpc = np.ones((17,), dtype=np.float32)

    # (17,3): [x, y, conf]
    kp = np.column_stack([kps, kpc])
    return kp


def pose_consistency_score(device_name, kp):
    if kp is None: return 0.0
    if device_name == "wheelchair":
        return sitting_score(kp)  # [0..1]
    elif device_name == "crutches":
        return crutch_score(kp)   # [0..1]
    else:
        # pram: pose'dan mustahkam signal olish qiyin, 0.2 bilan cheklaymiz (ixtiyoriy)
        return 0.2

# =========================
# =====  EXTENSION  =======
# =========================
def estimate_extra_seconds(person_box, device_name):
    base = 8
    if device_name == "wheelchair": base += 3
    elif device_name == "crutches": base += 2
    elif device_name == "pram":     base += 1
    return clamp(base, EXTRA_SEC_MIN, EXTRA_SEC_MAX)

# =========================
# ========= MAIN ==========
# =========================
def main():
    crosswalk_poly = load_roi_polygon(ROI_JSON)

    person_model = YOLO(PERSON_MODEL_PATH)
    device_model = YOLO(DEVICE_MODEL_PATH)
    pose_model   = YOLO(POSE_MODEL_PATH)

    device_names = {int(k): str(v).lower() for k, v in device_model.names.items()}
    valid_device_set = set(n.lower() for n in DEVICE_CLASS_NAMES)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {VIDEO_SOURCE}")

    tracker  = SimplePersonTracker(max_dist=160, max_age=20)
    tconfirm = HysteresisConfirm(win=TEMP_WIN, on_hits=ON_HITS, off_miss=OFF_MISS, vote_k=LABEL_VOTE_K)

    last_decision_time = 0.0
    frame_idx, t0 = 0, time.time()
    fps = 0.0

    print("[INFO] Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        # 1) PERSON
        pres = person_model.predict(frame, verbose=False, conf=PERSON_CONF_TH)[0]
        person_boxes = []
        if pres.boxes is not None and len(pres.boxes) > 0:
            for b in pres.boxes:
                if int(b.cls.item()) == 0:
                    x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                    box = (x1, y1, x2, y2)
                    if area(box) < MIN_PERSON_AREA: continue
                    person_boxes.append((box, float(b.conf.item())))

        # 2) DEVICE
        dres = device_model.predict(frame, verbose=False, conf=DEVICE_CONF_TH)[0]
        device_dets = []
        if dres.boxes is not None and len(dres.boxes) > 0:
            for b in dres.boxes:
                cls = int(b.cls.item())
                name = device_names.get(cls, str(cls)).lower()
                if name not in valid_device_set: continue
                conf = float(b.conf.item())
                if conf < CLASS_CONF.get(name, DEVICE_CONF_TH): continue
                x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                box = (x1, y1, x2, y2)
                if area(box) < MIN_DEVICE_AREA: continue
                dcx, dcy, _, _ = xyxy_to_cxcywh(box)
                if not point_in_poly((dcx, dcy), crosswalk_poly): continue
                device_dets.append((box, conf, name))

        # 3) TRACK
        tracked_persons = tracker.update(person_boxes, frame_idx)

        # 4) MATCH
        pid_to_device = {}  # pid -> (name, device_conf)
        for (pid, pbox, pconf) in tracked_persons:
            pcx, pcy, _, _ = xyxy_to_cxcywh(pbox)
            best, best_score = None, -1e9
            for (dbox, dconf, dname) in device_dets:
                dcx, dcy, _, _ = xyxy_to_cxcywh(dbox)
                if not point_in_bottom_half(dcx, dcy, pbox): continue
                d = euclidean((pcx,pcy),(dcx,dcy))
                if d > MATCH_DIST_TH: continue
                ov = iou(pbox, dbox)
                if ov < MATCH_IOU_TH: continue
                score = (ov * 1.0) + (max(0.0, MATCH_DIST_TH - d)/MATCH_DIST_TH) * 0.5
                if score > best_score:
                    best_score = score
                    best = (dbox, dconf, dname)
            if best is not None:
                pid_to_device[pid] = (best[2], best[1])  # (name, conf)

        # 5) POSE (shartli) + TEMPORAL
        confirmed_pids = []
        final_scores   = {}  # pid -> final_score (vizual uchun)
        for (pid, pbox, _) in tracked_persons:
            if pid in pid_to_device:
                dname, dconf = pid_to_device[pid]
                final_score = dconf
                # Pose trigger: faqat pastroq conf bo'lsa
                if dconf < POSE_TRIGGER_UPPER:
                    kp = run_pose_on_person(pose_model, frame, pbox)
                    pscore = pose_consistency_score(dname, kp)  # [0..1]
                    final_score = ALPHA_DEVICE * dconf + BETA_POSE * pscore
                # Temporalga "detected" sifatida final_score > threshold?? – ON_HITS hisobi binary
                detected_bool = final_score >= CLASS_CONF.get(dname, 0.5)
                is_on = tconfirm.update(pid, detected_bool, label=dname if detected_bool else None)
                final_scores[pid] = final_score
                if is_on:
                    confirmed_pids.append(pid)
            else:
                is_on = tconfirm.update(pid, False, label=None)
                final_scores[pid] = 0.0

        # 6) DECISION
        now = time.time()
        to_extend, reason = 0, []
        for pid in confirmed_pids:
            voted = tconfirm.voted_label(pid) or (pid_to_device[pid][0] if pid in pid_to_device else None)
            if not voted: continue
            pbox = next(b for (ppid, b, _) in tracked_persons if ppid == pid)
            ext = estimate_extra_seconds(pbox, voted)
            to_extend = max(to_extend, ext)
            reason.append(f"ID{pid}:{voted} -> +{ext}s")
        if to_extend > 0 and (now - last_decision_time) > DECISION_COOLDOWN_SEC:
            last_decision_time = now
            print(f"[DECISION] Extend green by +{to_extend}s | " + "; ".join(reason))

        # 7) OVERLAY
        draw_polygon(frame, crosswalk_poly, (255, 0, 255), 2)
        for (pid, pbox, _) in tracked_persons:
            x1,y1,x2,y2 = map(int, pbox)
            color = (0,255,0) if pid in confirmed_pids else (255,255,0)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            lab = f"ID{pid} person"
            if pid in pid_to_device:
                dname, dconf = pid_to_device[pid]
                lab += f" + {dname} ({dconf:.2f})"
                if pid in final_scores:
                    lab += f" | score:{final_scores[pid]:.2f}"
            cv2.putText(frame, lab, (x1, max(0, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        for (dbox, dconf, dname) in device_dets:
            x1,y1,x2,y2 = map(int, dbox)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (120,120,255), 2)
            cv2.putText(frame, f"{dname} {dconf:.2f}", (x1, max(0,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120,120,255), 2, cv2.LINE_AA)

        if to_extend > 0:
            cv2.putText(frame, f"EXTEND GREEN: +{to_extend}s", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,220,0), 3, cv2.LINE_AA)

        if SHOW_FPS:
            dt = time.time() - t0
            if dt > 0: fps = 1.0/dt
            t0 = time.time()
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,200,255), 2, cv2.LINE_AA)

        cv2.imshow("SmartCare AI Light - Pose Assisted", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
