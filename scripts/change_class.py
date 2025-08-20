import os
from pathlib import Path

train_img_dir = r"C:/Users/dalab/Desktop/Abbos/SmartLight/dataset2/train/images"
train_lbl_dir = r"C:/Users/dalab/Desktop/Abbos/SmartLight/dataset2/train/labels"


TARGET_OLD = "4"
TARGET_NEW = "1"

def process_dir(lbl_dir: str):
    lbl_path = Path(lbl_dir)
    if not lbl_path.exists():
        print(f"[SKIP] Not found: {lbl_dir}")
        return 0, 0

    files = list(lbl_path.glob("*.txt"))
    changed_files = 0
    changed_lines = 0

    for f in files:
        with f.open("r", encoding="utf-8") as rf:
            lines = rf.readlines()

        modified = False
        new_lines = []
        for line in lines:
            s = line.strip()
            if not s:
                new_lines.append(line)
                continue

            parts = s.split()
            # class id is the FIRST token
            if parts and parts[0] == TARGET_OLD:
                parts[0] = TARGET_NEW
                new_line = " ".join(parts) + ("\n" if not line.endswith("\n") else "")
                new_lines.append(new_line)
                modified = True
                changed_lines += 1
            else:
                new_lines.append(line)

        if modified:
            # backup
            backup = f.with_suffix(f.suffix + ".bak")
            if not backup.exists():
                backup.write_text("".join(lines), encoding="utf-8")
            # write updated
            f.write_text("".join(new_lines), encoding="utf-8")
            changed_files += 1

    return changed_files, changed_lines

total_files = total_lines = 0
for d in [train_lbl_dir]:
    cf, cl = process_dir(d)
    total_files += cf
    total_lines += cl
    print(f"[{d}] changed_files={cf}, changed_lines={cl}")

print(f"\nDONE. Total changed files: {total_files}, total changed lines: {total_lines}")
