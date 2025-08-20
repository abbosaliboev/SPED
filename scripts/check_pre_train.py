from pathlib import Path
from PIL import Image

# --- Config ---
train_img_dir = r"C:/Users/dalab/Desktop/Abbos/SmartLight/dataset/train/images"
train_lbl_dir = r"C:/Users/dalab/Desktop/Abbos/SmartLight/dataset/train/labels"
val_img_dir   = r"C:/Users/dalab/Desktop/Abbos/SmartLight/dataset/valid/images"
val_lbl_dir   = r"C:/Users/dalab/Desktop/Abbos/SmartLight/dataset/valid/labels"

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
DRY_RUN = True   # avval TRUE -> faqat ro'yxat; yozish uchun False qiling
CLASS_MAP = {11: 1}  # ixtiyoriy: masalan 11->1 majburan
ALLOWED_CLASSES = None  # masalan {0,1} deb cheklamoqchi bo'lsangiz, shu yerga to'plam bering

def find_image(img_dir: Path, stem: str):
    for ext in IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None

def is_clean_yolo_file(txt_path: Path) -> bool:
    """Toza YOLO-detection faylmi? (har satr 5 token, [0..1], bbox ichkarida)"""
    raw = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not raw:
        return True  # bo'sh labelni toza deb hisoblaymiz
    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) != 5:
            return False
        try:
            cid = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
        except:
            return False
        if ALLOWED_CLASSES and cid not in ALLOWED_CLASSES:
            return False
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            return False
        if (x - w/2) < 0 or (x + w/2) > 1 or (y - h/2) < 0 or (y + h/2) > 1:
            return False
    return True

def clip01(v): 
    return min(1.0, max(0.0, v))

def norm_bbox_xywh(x, y, w, h, W, H):
    # piksel bo'lsa normalize
    if max(x, y, w, h) > 1.0 and W and H:
        x, w = x / W, w / W
        y, h = y / H, h / H
    # clip
    L = clip01(x - w/2); R = clip01(x + w/2)
    T = clip01(y - h/2); B = clip01(y + h/2)
    nw, nh = R - L, B - T
    if nw <= 0 or nh <= 0:
        return None
    nx, ny = (L + R)/2, (T + B)/2
    if not (0 <= nx <= 1 and 0 <= ny <= 1 and 0 < nw <= 1 and 0 < nh <= 1):
        return None
    return nx, ny, nw, nh

def rebuild_corrupt_file(txt_path: Path, img_dir: Path):
    """Bir nechta obyekt bir qatorda yopishib qolgan holatlarni tiklaydi."""
    raw = txt_path.read_text(encoding="utf-8", errors="ignore")
    tokens = raw.split()
    if not tokens:
        return False, 0, 0

    img_path = find_image(img_dir, txt_path.stem)
    W = H = None
    if img_path:
        try:
            from PIL import Image
            with Image.open(img_path) as im:
                W, H = im.size
        except:
            pass

    # tokenlarni: [cid, x, y, w, h] bloklarga ajratish
    objs = []
    i = 0
    n = len(tokens)
    def is_int_token(s):
        try:
            return float(s).is_integer()
        except:
            return False

    while i < n:
        if not is_int_token(tokens[i]):
            i += 1
            continue
        cid = int(float(tokens[i])); i += 1
        if cid in CLASS_MAP:
            cid = CLASS_MAP[cid]
        if ALLOWED_CLASSES and cid not in ALLOWED_CLASSES:
            # keyingi classgacha o'tkazib yuboramiz
            while i < n and not is_int_token(tokens[i]): i += 1
            continue

        nums = []
        while i < n and not is_int_token(tokens[i]):
            try:
                nums.append(float(tokens[i]))
            except:
                pass
            i += 1
        # faqat bbox'ni qo'llab-quvvatlaymiz (seg bo'lsa tashlaymiz)
        if len(nums) >= 4:
            objs.append((cid, nums[:4]))

    # obyektlarni norm/clip qilib yozish
    lines = []
    kept = dropped = 0
    for cid, arr in objs:
        x, y, w, h = arr[:4]
        nb = norm_bbox_xywh(x, y, w, h, W, H)
        if nb is None:
            dropped += 1
            continue
        x, y, w, h = nb
        lines.append(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        kept += 1

    if not lines:
        # hech narsa tiklanmadi -> yozmaymiz
        return False, kept, dropped

    if DRY_RUN:
        return True, kept, dropped

    bak = txt_path.with_suffix(txt_path.suffix + ".bak")
    if not bak.exists():
        bak.write_text(raw, encoding="utf-8")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True, kept, dropped

def fix_only_bad(lbl_dir, img_dir):
    lbl_dir = Path(lbl_dir); img_dir = Path(img_dir)
    files = list(lbl_dir.glob("*.txt"))
    bad_files = []
    for f in files:
        if not is_clean_yolo_file(f):
            bad_files.append(f)

    print(f"[{lbl_dir.name}] found_bad={len(bad_files)} / total={len(files)}")
    total_changed = total_kept = total_dropped = 0
    for f in bad_files:
        changed, kept, dropped = rebuild_corrupt_file(f, img_dir)
        if changed:
            total_changed += 1
            total_kept += kept
            total_dropped += dropped
            print(f"  fixed: {f.name}  (kept={kept}, dropped={dropped})")
        else:
            print(f"  (inspect) could not fix reliably: {f.name}")
    print(f"[{lbl_dir.name}] changed_files={total_changed}, kept_objs={total_kept}, dropped_objs={total_dropped}, dry_run={DRY_RUN}")

# --- Run ---
fix_only_bad(train_lbl_dir, train_img_dir)
fix_only_bad(val_lbl_dir,   val_img_dir)
print("DONE.  DRY_RUN=False qilib yozdiring va labels.cache fayllarini o'chiring.")
