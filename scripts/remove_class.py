import os

# Class ID for 'pram'
PRAM_CLASS_ID = 1

def remove_images_with_class(img_dir, lbl_dir, target_class):
    for lbl_file in os.listdir(lbl_dir):
        if not lbl_file.endswith(".txt"):
            continue

        lbl_path = os.path.join(lbl_dir, lbl_file)
        img_path_jpg = os.path.join(img_dir, lbl_file.replace(".txt", ".jpg"))
        img_path_png = os.path.join(img_dir, lbl_file.replace(".txt", ".png"))

        # Read annotation
        with open(lbl_path, "r") as f:
            lines = f.readlines()

        # Check if target class exists
        has_target_class = any(int(line.split()[0]) == target_class for line in lines)

        if has_target_class:
            # Remove label file
            os.remove(lbl_path)
            # Remove corresponding image
            if os.path.exists(img_path_jpg):
                os.remove(img_path_jpg)
            elif os.path.exists(img_path_png):
                os.remove(img_path_png)

            print(f"Removed: {lbl_file} and its image (contains class {target_class})")

# Example usage
train_img_dir = r"C:/Users/dalab/Desktop/11talik/train/images"
train_lbl_dir = r"C:/Users/dalab/Desktop/11talik/train/labels"

remove_images_with_class(train_img_dir, train_lbl_dir, PRAM_CLASS_ID)


print("Removal process complete!")
