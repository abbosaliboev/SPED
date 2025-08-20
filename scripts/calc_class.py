# count_classes.py
import os, yaml, collections

# --- Sozlamalar ---
ROOT = r"C:/Users/dalab/Desktop/Abbos/SmartLight/dataset"  # train/valid/test shu yerda
DATA_YAML = os.path.join(ROOT, "data.yaml")               # names: {0: Bike, 1: Pedestrian, ...}
SPLITS = ["train", "valid", "test"]                       # qaysi papkalar bor bo'lsa o'shalar

def load_names(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    names = y.get("names", {})
    # Ultralytics: ba'zida ro'yxat, ba'zida dict bo'ladi
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}

def count_split(split_dir):
    labels_dir = os.path.join(split_dir, "labels")
    images_dir = os.path.join(split_dir, "images")
    per_class = collections.Counter()
    files_with_unknown = 0
    empty_files = 0
    label_files = 0
    images_total = 0

    # rasmlar soni
    if os.path.isdir(images_dir):
        images_total = sum(1 for x in os.listdir(images_dir)
                           if os.path.splitext(x)[1].lower() in [".jpg",".jpeg",".png",".bmp",".webp"])

    if not os.path.isdir(labels_dir):
        return per_class, {"images_total": images_total, "label_files": 0,
                           "empty_files": 0, "files_with_unknown": 0}

    for f in os.listdir(labels_dir):
        if not f.endswith(".txt"): continue
        label_files += 1
        p = os.path.join(labels_dir, f)
        lines = [ln.strip() for ln in open(p, "r", encoding="utf-8", errors="ignore") if ln.strip()]
        if not lines:
            empty_files += 1
            continue
        for ln in lines:
            parts = ln.split()
            try:
                cid = int(parts[0])
                per_class[cid] += 1
            except Exception:
                files_with_unknown += 1
                break

    stats = {
        "images_total": images_total,
        "label_files": label_files,
        "empty_files": empty_files,
        "files_with_unknown": files_with_unknown
    }
    return per_class, stats

def main():
    names = load_names(DATA_YAML)
    total = collections.Counter()
    split_details = {}

    for sp in SPLITS:
        sp_dir = os.path.join(ROOT, sp)
        if not os.path.isdir(sp_dir):
            continue
        per_class, stats = count_split(sp_dir)
        split_details[sp] = (per_class, stats)
        total.update(per_class)

    # Chiqish
    print("=== Umumiy class instanslari (barcha splitlar) ===")
    all_ids = sorted(set(total.keys()) | set(names.keys()))
    for cid in all_ids:
        nm = names.get(cid, f"cls_{cid}")
        print(f"{cid:>3} | {nm:<15} : {total.get(cid, 0)}")

    print("\n=== Splitlar bo'yicha tafsilotlar ===")
    for sp, (pc, st) in split_details.items():
        print(f"\n[{sp}]")
        for cid in sorted(set(pc.keys()) | set(names.keys())):
            nm = names.get(cid, f"cls_{cid}")
            print(f"  {cid:>3} | {nm:<15} : {pc.get(cid, 0)}")
        print(f"  Images total       : {st['images_total']}")
        print(f"  Label files        : {st['label_files']}")
        print(f"  Empty label files  : {st['empty_files']}")
        print(f"  Files w/ unknownID : {st['files_with_unknown']}")

if __name__ == "__main__":
    main()
