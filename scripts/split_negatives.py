# split_negatives_to_yolo_valid.py
# Python 3.8+
# Negativ rasmlar: label YO'Q (yoki bo'sh .txt)
#
# Manba:  frame/   (faqat rasmlar)
# Maqsad: dataset/
#   images/train|valid|test
#   labels/train|valid|test   (negativlar uchun bo'sh .txt ixtiyoriy)

import shutil
import random
from pathlib import Path

# ====== CONFIG ======
SRC_DIR  = Path(r"C:\Users\dalab\Desktop\Abbos\SmartLight\frame")              # manba (negativ rasmlar)
DST_ROOT = Path(r"C:\Users\dalab\Desktop\Abbos\SmartLight\dataset") # dataset ildizi

# ulushlar (valid nomiga e'tibor bering)
SPLIT = {
    "train": 0.8,
    "valid": 0.1,   # val emas, VALID
    "test":  0.1,
}
RANDOM_SEED = 42

# qaysi fayl kengaytmalari rasm sifatida olinadi
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Negativlar uchun bo'sh .txt yaratilsinmi? (YOLOv8 bo'sh/yo'q labelni "no object" deb qabul qiladi)
CREATE_EMPTY_TXT = True

# Fayl nomi to'qnashuvi bo'lsa ustiga yozish (True) yoki `_1`, `_2` qo'shib saqlash (False)
OVERWRITE = False
# ====================


def ensure_dirs(root: Path):
    for split in ("train", "valid", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def unique_path(dst_path: Path) -> Path:
    """Agar OVERWRITE=False bo'lsa, mavjud bo'lsa nomiga _1, _2 ... qo'shib qaytaradi."""
    if OVERWRITE or not dst_path.exists():
        return dst_path
    stem, suf = dst_path.stem, dst_path.suffix
    k = 1
    while True:
        cand = dst_path.with_name(f"{stem}_{k}{suf}")
        if not cand.exists():
            return cand
        k += 1


def main():
    assert abs(sum(SPLIT.values()) - 1.0) < 1e-6, "SPLIT ulushlari yig'indisi 1 bo'lishi shart."

    ensure_dirs(DST_ROOT)

    # 1) Manbadagi rasmlarni yig'amiz
    imgs = [p for p in SRC_DIR.iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file()]
    if not imgs:
        print(f"[ERR] Rasm topilmadi: {SRC_DIR}")
        return

    # 2) Aralashtiramiz va bo'lamiz
    random.seed(RANDOM_SEED)
    random.shuffle(imgs)
    n = len(imgs)
    n_train = int(n * SPLIT["train"])
    n_valid = int(n * SPLIT["valid"])
    n_test  = n - n_train - n_valid

    parts = {
        "train": imgs[:n_train],
        "valid": imgs[n_train:n_train + n_valid],
        "test":  imgs[n_train + n_valid:],
    }

    # 3) Copy + (ixtiyoriy) bo'sh label
    moved = {"train": 0, "valid": 0, "test": 0}
    for split, files in parts.items():
        img_dst_dir = DST_ROOT / "images" / split
        lbl_dst_dir = DST_ROOT / "labels" / split

        for src in files:
            dst_img = unique_path(img_dst_dir / src.name)
            shutil.copy2(src, dst_img)

            if CREATE_EMPTY_TXT:
                # negativ uchun mos nomdagi .txt (bo'sh)
                dst_txt = lbl_dst_dir / (dst_img.stem + ".txt")
                if OVERWRITE or not dst_txt.exists():
                    dst_txt.touch()

        moved[split] = len(files)

    # 4) Yakuniy natija
    print("[DONE] Negativ rasmlar datasetga ko'chirildi (copy).")
    print(f"  train: {moved['train']}")
    print(f"  valid: {moved['valid']}")
    print(f"  test : {moved['test']}")
    print(f"Dataset ildizi: {DST_ROOT}")
    print("Struktura:")
    print("  images/train|valid|test  <-- negativ rasmlar")
    print("  labels/train|valid|test  <-- (bo'sh .txt fayllar, agar CREATE_EMPTY_TXT=True)")
    print("\nEslatma: YOLOv8 uchun negativlar labelSIZ ham ishlaydi; bo'sh .txt ixtiyoriy.")


if __name__ == "__main__":
    main()
