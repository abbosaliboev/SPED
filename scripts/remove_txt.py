import os
from pathlib import Path

# Rasm va label papkalari
pairs = [
    (
        r"C:/Users/dalab/Desktop/Abbos/SmartLight/dataset/train/images",
        r"C:/Users/dalab/Desktop/Abbos/SmartLight/dataset/train/labels"
    ),
    (
        r"C:/Users/dalab/Desktop/Abbos/SmartLight/dataset/valid/images",
        r"C:/Users/dalab/Desktop/Abbos/SmartLight/dataset/valid/labels"
    )
]

deleted_count = 0

for img_dir, lbl_dir in pairs:
    img_path = Path(img_dir)
    lbl_path = Path(lbl_dir)
    if not img_path.exists() or not lbl_path.exists():
        print(f"[SKIP] {img_dir} yoki {lbl_dir} topilmadi")
        continue

    # mavjud rasm fayllarining nomlarini (suffixsiz) to‘plash
    img_stems = {f.stem for f in img_path.iterdir() if f.is_file()}

    # har bir label uchun mos rasm borligini tekshirish
    for txt_file in lbl_path.glob("*.txt"):
        if txt_file.stem not in img_stems:
            try:
                txt_file.unlink()
                deleted_count += 1
            except Exception as e:
                print(f"Xato o‘chirishda: {txt_file} -> {e}")

print(f"[DONE] O‘chirilgan ortiqcha .txt fayllar soni: {deleted_count}")
