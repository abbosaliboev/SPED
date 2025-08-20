import os
from pathlib import Path

root_dirs = [
    r"C:/Users/dalab/Desktop/Abbos/SmartLight/dataset2/train/labels",
]

deleted_count = 0

for root in root_dirs:
    p = Path(root)
    if not p.exists():
        continue
    for bak_file in p.glob("*.bak"):
        try:
            bak_file.unlink()
            deleted_count += 1
        except Exception as e:
            print(f"Xato: {bak_file} -> {e}")

print(f"DONE. O'chirilgan .bak fayllar soni: {deleted_count}")
