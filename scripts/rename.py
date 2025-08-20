import os
import glob

base_path = r"C:/Users/dalab/Desktop/Abbos/SmartLight/dataset3"
splits = ['train', 'valid', 'test']

for split in splits:
    img_dir = os.path.join(base_path, split, 'images')
    label_dir = os.path.join(base_path, split, 'labels')

    img_paths = sorted(glob.glob(os.path.join(img_dir, '*.*')))
    counter = 0

    for img_path in img_paths:
        if not os.path.exists(img_path):
            print(f"⛔️ Skipped missing image: {img_path}")
            continue

        ext = os.path.splitext(img_path)[-1].lower()
        while True:
            new_name = f"img3_{counter:04d}{ext}"
            new_img_path = os.path.join(img_dir, new_name)
            if not os.path.exists(new_img_path):
                break
            counter += 1

        new_label_name = f"img3_{counter:04d}.txt"
        old_label_path = os.path.join(label_dir, os.path.splitext(os.path.basename(img_path))[0] + '.txt')
        new_label_path = os.path.join(label_dir, new_label_name)

        os.rename(img_path, new_img_path)

        if os.path.exists(old_label_path):
            os.rename(old_label_path, new_label_path)

        print(f"✅ Renamed: {os.path.basename(img_path)} -> {new_name}")
        counter += 1

    print(f"\n🎉 Finished renaming in `{split}`. Total renamed: {counter} images.\n")
