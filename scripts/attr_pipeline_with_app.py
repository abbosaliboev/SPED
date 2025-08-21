# pip install ultralytics torch torchvision opencv-python pillow requests

import cv2, time, math, warnings
import numpy as np
from collections import deque, defaultdict, Counter
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO

warnings.filterwarnings("ignore", category=UserWarning)

# =========================
# =======  PATHS  =========
# =========================
# Loyihaning root'i: SMARTCARE_LIGHTS/
ROOT = Path(__file__).resolve().parents[1]

# MODEL YO'LLARI (sizdagi fayllar nomiga mos)
PERSON_MODEL_PATH   = str(ROOT / "yolov8s.pt")            # sizda bor
ATTR_WEIGHTS        = str(ROOT / "person_attr_mnv3s.pth") # sizda bor
DEVICE_MODEL_PATH   = str(ROOT / "best2.pt")              # sizda bor

# =========================
# =======  CONFIG  ========
# =========================
VIDEO_SOURCE = 0  # 0=webcam

# DETECTOR (person)
PERSON_CONF = 0.45
MIN_PERSON_AREA = 40 * 40
DETECT_IMGSZ = 640
USE_HALF = True

# CLASSIFIER (MNv3)
CLS_IMSIZE = 224
TH_CRUTCHES   = 0.70
TH_WHEELCHAIR = 0.65

# DEVICE-GATING
USE_DEVICE_GATING = True
DEVICE_CONF = 0.40
HARD_GATE = True
SOFT_GATE_ADD = 0.10
LOG_GATING = False

# TRACKING + TEMPORAL
MATCH_MAX_DIST = 160
TRACK_MAX_AGE  = 20
TEMP_WIN  = 18
ON_HITS   = 9
OFF_MISS  = 10
LABEL_VOTE_K = 8

# SPEED
CLASSIFY_EVERY = 4
MOVE_IOU_TH = 0.20
SHOW_FPS = True

# ALERT STATE
from scripts.alert_client import notify_event, clear_event
prev_on = defaultdict(bool)  # pid -> previous ON
no_on_frames = 0
CLEAR_AFTER = 12  # ~ 0.5-1s, FPS ga bog'liq
cleared_sent = True

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

def euclidean(p, q):
    return math.hypot(p[0]-q[0], p[1]-q[1])

def iou(a, b):
    ax1,ay1,ax2,ay2 = map(float,a); bx1,by1,bx2,by2 = map(float,b)
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw,ih = max(0,ix2-ix1), max(0,iy2-iy1)
    inter = iw*ih
    if inter <= 0: return 0.0
    ua = area(a)+area(b)-inter
    return inter/ua if ua>0 else 0.0

def box_center(b):
    x1,y1,x2,y2 = b
    return (0.5*(x1+x2), 0.5*(y1+y2))

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
        used = set(); out=[]
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
        for tid in [tid for tid, tk in list(self.tracks.items()) if frame_idx - tk["last_seen"] > self.max_age]:
            del self.tracks[tid]
        return out

class HysteresisConfirm:
    def __init__(self, win=TEMP_WIN, on_hits=ON_HITS, off_miss=OFF_MISS, vote_k=LABEL_VOTE_K):
        self.win = win; self.on_hits = on_hits; self.off_miss = off_miss
        self.buf = defaultdict(lambda: deque(maxlen=self.win))
        self.lbl = defaultdict(lambda: deque(maxlen=vote_k))
        self.state = {}
        self.miss  = defaultdict(int)
    def update(self, pid, detected_bool, label=None):
        self.buf[pid].append(bool(detected_bool))
        if label: self.lbl[pid].append(label)
        st = self.state.get(pid, False)
        if not st:
            if sum(self.buf[pid]) >= self.on_hits:
                self.state[pid] = True; self.miss[pid] = 0
        else:
            if not detected_bool: self.miss[pid] += 1
            else: self.miss[pid] = 0
            if self.miss[pid] >= self.off_miss:
                self.state[pid] = False; self.miss[pid] = 0
        return self.state.get(pid, False)
    def voted_label(self, pid):
        if len(self.lbl[pid]) == 0: return None
        return Counter(self.lbl[pid]).most_common(1)[0][0]

# =========================
# ===== CLASSIFIER  =======
# =========================
def _norm_name(s: str) -> str:
    s = s.lower().strip()
    return s.replace("-", "").replace("_", "").replace(" ", "")

def build_idx_map(classes):
    norm = [_norm_name(c) for c in classes]
    def find_any(cands):
        for i, n in enumerate(norm):
            if n in cands: return i
        return None
    no_dev_syn  = {"nodevice","no_device","none","normal","person","noprop","noproptype"}
    cr_syn      = {"crutches","usescrutches","withcrutches","crutch"}
    wh_syn      = {"wheelchair","wheelchairuser","inwheelchair","wheelchairperson"}
    idx_no = find_any(no_dev_syn)
    idx_cr = find_any(cr_syn)
    idx_wh = find_any(wh_syn)
    missing = []
    if idx_no is None: missing.append("no_device")
    if idx_cr is None: missing.append("uses_crutches")
    if idx_wh is None: missing.append("wheelchair_user")
    if missing:
        raise ValueError(f"classes={classes} -> missing: {', '.join(missing)}")
    idx_map = {"no_device": idx_no, "uses_crutches": idx_cr, "wheelchair_user": idx_wh}
    idx2canon = {idx_no:"no_device", idx_cr:"uses_crutches", idx_wh:"wheelchair_user"}
    return idx_map, idx2canon

def load_attr_model(weights_path, device):
    ckpt = torch.load(weights_path, map_location="cpu")
    classes = ckpt["classes"]
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    tfm = transforms.Compose([transforms.Resize((CLS_IMSIZE, CLS_IMSIZE)), transforms.ToTensor()])
    return model, classes, tfm

def classify_batch(attr_model, tfm, device, frame_bgr, items):
    if not items: return {}
    tensors, pids = [], []
    H,W = frame_bgr.shape[:2]
    for pid, pbox in items:
        x1,y1,x2,y2 = map(int, pbox)
        x1=max(0,x1); y1=max(0,y1); x2=min(W-1,x2); y2=min(H-1,y2)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0: continue
        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img)
        t = tfm(pil)
        tensors.append(t); pids.append(pid)
    if not tensors: return {}
    batch = torch.stack(tensors, dim=0).to(device)
    with torch.no_grad():
        out = attr_model(batch)
        probs = torch.softmax(out, dim=1).cpu().numpy()
    ans = {}
    for i, pid in enumerate(pids):
        p = probs[i]; idx = int(np.argmax(p)); conf = float(p[idx])
        ans[pid] = (idx, conf, p.tolist())
    return ans

# =========================
# ===== DEVICE-GATE  ======
# =========================
def device_boxes(frame, device_model, dev_names):
    dres = device_model.predict(frame, verbose=False, conf=DEVICE_CONF)[0]
    out = {"crutches":[], "wheelchair":[]}
    if dres.boxes is None: return out
    for b in dres.boxes:
        cls = int(b.cls.item())
        name = dev_names.get(cls, str(cls)).lower()
        if name in out:
            x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
            out[name].append((x1,y1,x2,y2))
    return out

def gate_match(pbox, dboxes):
    if not dboxes: return False
    pcx,pcy = box_center(pbox)
    x1,y1,x2,y2 = pbox
    y_cut = y1 + 0.4*(y2-y1)  # pastki 60%
    for d in dboxes:
        dx1,dy1,dx2,dy2 = d
        dcx, dcy = (0.5*(dx1+dx2), 0.5*(dy1+dy2))
        if (x1 <= dcx <= x2) and (dcy >= y_cut): return True
        if iou(pbox, d) >= 0.05: return True
    return False

# =========================
# ========= MAIN ==========
# =========================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # DETECTOR
    det_args = {"conf":PERSON_CONF, "imgsz":DETECT_IMGSZ, "verbose":False}
    if torch.cuda.is_available():
        det_args["device"] = 0
        det_args["half"] = USE_HALF
    person_model = YOLO(PERSON_MODEL_PATH)

    # CLASSIFIER
    print("[LOAD] attribute classifier:", ATTR_WEIGHTS)
    attr_model, classes, tfm = load_attr_model(ATTR_WEIGHTS, device)
    print("[INFO] checkpoint classes:", classes)
    idx_map, idx2canon = build_idx_map(classes)
    IDX_CR  = idx_map["uses_crutches"]
    IDX_WH  = idx_map["wheelchair_user"]

    # DEVICE GATE
    device_model = None; dev_names = {}
    if USE_DEVICE_GATING:
        print("[LOAD] device model (gate):", DEVICE_MODEL_PATH)
        device_model = YOLO(DEVICE_MODEL_PATH)
        dev_names = {int(k): str(v).lower() for k,v in device_model.names.items()}
        print("[device classes]", dev_names)

    # VIDEO
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {VIDEO_SOURCE}")
    print("[INFO] Press 'q' to quit.")

    tracker  = SimplePersonTracker(max_dist=MATCH_MAX_DIST, max_age=TRACK_MAX_AGE)
    tconfirm = HysteresisConfirm(win=TEMP_WIN, on_hits=ON_HITS, off_miss=OFF_MISS, vote_k=LABEL_VOTE_K)

    cache = {}
    frame_idx, t0 = 0, time.time()
    fps = 0.0

    global no_on_frames, cleared_sent

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1

        # PERSON DETECT
        pres = person_model.predict(frame, **det_args)[0]
        person_boxes = []
        if pres.boxes is not None and len(pres.boxes) > 0:
            for b in pres.boxes:
                if int(b.cls.item()) == 0:
                    x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                    box = (x1, y1, x2, y2)
                    if area(box) < MIN_PERSON_AREA: continue
                    person_boxes.append((box, float(b.conf.item())))

        # TRACK
        tracked_persons = tracker.update(person_boxes, frame_idx)

        # need classify?
        need = []
        for (pid, pbox, _) in tracked_persons:
            c = cache.get(pid)
            reclass = False
            if c is None: reclass = True
            else:
                if (frame_idx - c["last_frame"]) >= CLASSIFY_EVERY: reclass = True
                elif iou(c["last_box"], pbox) < (1.0 - MOVE_IOU_TH): reclass = True
            if reclass: need.append((pid, pbox))

        # classify
        results = classify_batch(attr_model, tfm, device, frame, need)

        # update cache
        for (pid, pbox, _) in tracked_persons:
            if pid in results:
                idx, conf, probs = results[pid]
                cache[pid] = {"last_idx": idx, "last_conf": conf, "last_probs": probs,
                              "last_box": pbox, "last_frame": frame_idx}
            elif pid in cache:
                cache[pid]["last_box"] = pbox
                cache[pid]["last_frame"] = frame_idx

        # device gate
        devs = {"crutches": [], "wheelchair": []}
        if USE_DEVICE_GATING:
            db = device_boxes(frame, device_model, dev_names)
            devs["crutches"]  = db.get("crutches", [])
            devs["wheelchair"] = db.get("wheelchair", [])

        # logic + draw
        any_on = False
        for (pid, pbox, _) in tracked_persons:
            c = cache.get(pid)
            if not c:
                tconfirm.update(pid, False, label=None)
                continue

            idx, conf, probs = c["last_idx"], c["last_conf"], c["last_probs"]
            label = idx2canon.get(idx, None)  # 'no_device' | 'uses_crutches' | 'wheelchair_user'

            th_cr = TH_CRUTCHES
            th_wh = TH_WHEELCHAIR
            gate_cr = gate_wh = True
            if USE_DEVICE_GATING:
                gate_cr = gate_match(pbox, devs.get("crutches", []))
                gate_wh = gate_match(pbox, devs.get("wheelchair", []))
                if not HARD_GATE:
                    if not gate_cr: th_cr += SOFT_GATE_ADD
                    if not gate_wh: th_wh += SOFT_GATE_ADD

            detected_bool = False
            post_label = None
            if label == "uses_crutches" and conf >= th_cr:
                if (not USE_DEVICE_GATING) or (HARD_GATE and gate_cr) or (not HARD_GATE):
                    detected_bool = True; post_label = "uses_crutches"
            elif label == "wheelchair_user" and conf >= th_wh:
                if (not USE_DEVICE_GATING) or (HARD_GATE and gate_wh) or (not HARD_GATE):
                    detected_bool = True; post_label = "wheelchair_user"

            is_on = tconfirm.update(pid, detected_bool, label=post_label if detected_bool else None)
            if is_on: any_on = True

            # ALERT: rising edge
            if is_on and not prev_on[pid] and post_label:
                evt = "crutches" if post_label == "uses_crutches" else "wheelchair"
                notify_event(event_type=evt, extend_sec=5, conf=conf)
                print(f"[ALERT] pid={pid} -> {evt} (conf={conf:.2f}) sent")
            prev_on[pid] = is_on

            # draw
            x1,y1,x2,y2 = map(int, pbox)
            color = (0,255,0) if is_on and post_label else (255,255,0)
            pr_cr = probs[IDX_CR] if IDX_CR < len(probs) else 0.0
            pr_wh = probs[IDX_WH] if IDX_WH < len(probs) else 0.0
            txt = f"ID{pid} | {label} {conf:.2f} [cr:{pr_cr:.2f} wh:{pr_wh:.2f}]"
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, txt, (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        # CLEAR when all off for a few frames
        if any_on:
            no_on_frames = 0
            cleared_sent = False
        else:
            no_on_frames += 1
            if no_on_frames >= CLEAR_AFTER and not cleared_sent:
                clear_event()
                print("[CLEAR] no active priority pedestrians")
                cleared_sent = True

        if SHOW_FPS:
            dt = time.time() - t0
            if dt > 0: fps = 1.0/dt
            t0 = time.time()
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,200,255), 2, cv2.LINE_AA)

        cv2.imshow("Person Attribute Pipeline (ACC)", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()