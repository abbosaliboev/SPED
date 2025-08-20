import os
from pathlib import Path

# --- Config ---
train_img_dir = r"C:/Users/dalab/Desktop/Abbos/SmartLight/dataset/train/images"
train_lbl_dir = r"C:/Users/dalab/Desktop/Abbos/SmartLight/dataset/train/labels"
val_img_dir   = r"C:/Users/dalab/Desktop/Abbos/SmartLight/dataset/valid/images"
val_lbl_dir   = r"C:/Users/dalab/Desktop/Abbos/SmartLight/dataset/valid/labels"

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
DRY_RUN = False  # Avval True: faqat chiqaradi, o‘chirmaydi

def remove_images_without_labels(img_dir, lbl_dir):
    img_dir = Path(img_dir)
    lbl_dir = Path(lbl_dir)

    removed = 0
    kept = 0

    for img_file in img_dir.glob("*"):
        if img_file.suffix.lower() not in IMG_EXTS:
            continue
        label_file = lbl_dir / f"{img_file.stem}.txt"
        if not label_file.exists():
            print(f"[REMOVE] {img_file}")
            removed += 1
            if not DRY_RUN:
                try:
                    img_file.unlink()
                except Exception as e:
                    print(f"  ! Error deleting {img_file}: {e}")
        else:
            kept += 1

    print(f"[{img_dir.name}] kept={kept}, removed={removed}, dry_run={DRY_RUN}")

# Run for both train and val
remove_images_without_labels(train_img_dir, train_lbl_dir)
remove_images_without_labels(val_img_dir, val_lbl_dir)
print("DONE.")
