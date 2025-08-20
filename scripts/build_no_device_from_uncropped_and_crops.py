# build_no_device_from_uncropped_and_crops.py
# pip install ultralytics opencv-python
import hashlib, random, shutil
from pathlib import Path
import cv2
from ultralytics import YOLO

# ========= SOZLAMALAR =========
CROPPED_NO_DEVICE_DIR = r"C:/Users/dalab/Desktop/Abbos/SmartLight/crops/no_device"   # allaqachon crop bo'lgan oddiy odamlar
UNCR_ROBOFLOw_DIRS    = [
    r"C:/Users/dalab/Desktop/11talik/train/Images",
]
OUT_ROOT              = r"C:/Users/dalab/Desktop/Abbos/SmartLight/no_device_last"         # chiqish dataset ildizi

SPLIT                 = 0.85
PERSON_CONF           = 0.45
MIN_AREA              = 40*40
TOPK_PER_IMAGE        = 3        # har rasmdan ko'pi bilan nechta person crop olinsin
SEED                  = 42
EXTS                  = {".jpg",".jpeg",".png",".webp",".bmp"}

# ========= UTIL =========
def md5sum(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for ch in iter(lambda: f.read(8192), b""):
            h.update(ch)
    return h.hexdigest()

def ensure_dirs(root: Path):
    for sp in ["train","val"]:
        for cls in ["no_device","uses_crutches","wheelchair_user"]:
            (root/sp/cls).mkdir(parents=True, exist_ok=True)

def collect_images(root: Path):
    return [p for p in root.rglob("*") if p.suffix.lower() in EXTS]

def copy_unique(src_img: Path, dst_dir: Path, seen_md5: set, new_name: str|None=None):
    m = md5sum(src_img)
    if m in seen_md5:
        return False
    seen_md5.add(m)
    dst = dst_dir / (new_name if new_name else src_img.name)
    shutil.copy2(src_img, dst)
    return True

# ========= MAIN =========
def main():
    random.seed(SEED)
    ensure_dirs(Path(OUT_ROOT))
    person_model = YOLO("yolov8s.pt")  # COCO person

    seen = set()
    train_nd = Path(OUT_ROOT, "train", "no_device")
    val_nd   = Path(OUT_ROOT, "val", "no_device")

    # 1) Avval 1500 ta allaqachon crop bo'lganlarni qo'shamiz (tasodifiy bo'lib)
    cropped = collect_images(Path(CROPPED_NO_DEVICE_DIR))
    random.shuffle(cropped)
    n_tr = int(len(cropped)*SPLIT)
    for i,p in enumerate(cropped):
        split_dir = train_nd if i < n_tr else val_nd
        copy_unique(p, split_dir, seen)

    # 2) Roboflow uncropped rasmlaridan person crop olamiz
    uncropped = []
    for d in UNCR_ROBOFLOw_DIRS:
        path = Path(d)
        if path.exists():
            uncropped += collect_images(path)
    print(f"[INFO] Uncropped candidates: {len(uncropped)}")

    saved = 0
    for idx, ip in enumerate(uncropped, 1):
        im = cv2.imread(str(ip))
        if im is None: 
            continue
        H, W = im.shape[:2]

        # Person detect
        res = person_model.predict(im, verbose=False, conf=PERSON_CONF)[0]
        boxes = []
        if res.boxes is not None:
            for b in res.boxes:
                if int(b.cls.item()) != 0:  # not person
                    continue
                x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                if (x2-x1)*(y2-y1) < MIN_AREA:
                    continue
                boxes.append((x1,y1,x2,y2, float(b.conf.item())))

        if not boxes:
            continue

        # eng ishonchlilarni tanla
        boxes.sort(key=lambda x: x[4], reverse=True)
        boxes = boxes[:TOPK_PER_IMAGE]

        # splitni tasodifiy tanlaymiz (har rasm uchun)
        split_dir = train_nd if random.random() < SPLIT else val_nd

        # crop + md5 bilan dublikatsiz saqlash
        for j,(x1,y1,x2,y2,_) in enumerate(boxes):
            crop = im[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            # vaqtinchalik faylga yozib md5 hisoblaymiz
            tmp_path = Path(OUT_ROOT, f"__tmp_{idx}_{j}.jpg")
            cv2.imwrite(str(tmp_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            ok = copy_unique(tmp_path, split_dir, seen, new_name=f"nd_uc_{idx:06d}_{j}.jpg")
            tmp_path.unlink(missing_ok=True)
            if ok:
                saved += 1

    print(f"[DONE] Added cropped ND: train/val after step1, and +{saved} crops from uncropped.")
    print(f"Output root: {OUT_ROOT}")
    print("Classes created: no_device (filled), uses_crutches (empty), wheelchair_user (empty)")

if __name__ == "__main__":
    main()
