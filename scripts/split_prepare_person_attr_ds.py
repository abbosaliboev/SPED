# split_prepare_person_attr_ds_v2.py
import random, shutil
from pathlib import Path

# --- KIRISH papkalar (hozir sendagi) ---
IN = {
    "no_device":        r"C:\Users\dalab\Desktop\Abbos\SmartLight\nodevice",
    "uses_crutches":    r"C:\Users\dalab\Desktop\Abbos\SmartLight\crutches",
    "wheelchair_user":  r"C:\Users\dalab\Desktop\Abbos\SmartLight\wheelchair",
}

# --- CHIQISH dataset ildizi (yangi papka, eski xatolar aralashmasin) ---
OUT_ROOT = r"C:\Users\dalab\Desktop\Abbos\SmartLight\person_attr_fixed"

SPLIT = 0.85
SEED  = 42
EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def collect_images(root: Path):
    return [p for p in root.rglob("*") if p.suffix.lower() in EXTS]

def main():
    random.seed(SEED)

    # 0) chiqarish papkasini tozalaymiz
    out = Path(OUT_ROOT)
    if out.exists():
        shutil.rmtree(out)
    for sp in ["train","val"]:
        for cls in IN.keys():
            (out/sp/cls).mkdir(parents=True, exist_ok=True)

    # 1) har klassni alohida split qilib, to'g'ridan-to'g'ri o'z klass papkasiga nusxalaymiz
    for cls, folder in IN.items():
        src = Path(folder)
        files = collect_images(src)
        random.shuffle(files)
        n_tr = int(len(files)*SPLIT)

        for i, p in enumerate(files):
            split = "train" if i < n_tr else "val"
            dst = out / split / cls / f"{cls[:2]}_{i:06d}{p.suffix.lower()}"
            shutil.copy2(p, dst)

        print(f"[{cls}] total={len(files)}  -> train≈{n_tr}, val≈{len(files)-n_tr}")

    # 2) audit
    for sp in ["train","val"]:
        base = out / sp
        cnt = {c: len(list((base/c).glob("*"))) for c in IN.keys()}
        print(f"[{sp}] no_device={cnt['no_device']} | uses_crutches={cnt['uses_crutches']} | wheelchair_user={cnt['wheelchair_user']}")
    print(f"[DONE] Dataset ready at: {OUT_ROOT}")

if __name__ == "__main__":
    main()
