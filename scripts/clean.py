import os

img_dir = "C:/Users/dalab/Desktop/Abbos/SmartLight/dataset/train/images"
label_dir = "C:/Users/dalab/Desktop/Abbos/SmartLight/dataset/train/labels"

deleted = 0

for txt_file in os.listdir(label_dir):
    if txt_file.endswith(".txt"):
        # Fayl nomidan ".jpg" so'zini ham olib tashlaymiz
        clean_name = txt_file.replace(".jpg", "").replace(".txt", "")
        img_file = f"{clean_name}.jpg"
        img_path = os.path.join(img_dir, img_file)

        if not os.path.exists(img_path):
            full_txt_path = os.path.join(label_dir, txt_file)
            if os.path.exists(full_txt_path):
                os.remove(full_txt_path)
                print(f"❌ Deleted label: {txt_file}")
                deleted += 1

print(f"\n✅ Finished! Deleted {deleted} orphan label files.")
