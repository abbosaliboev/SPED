from ultralytics import YOLO

DATA_YAML = r"C:/Users/dalab/Desktop/Abbos/SmartLight/dataset/data.yaml"
RUNS_DIR  = "runs2"
EXP_NAME  = "assistive-yolov8s_noaug"

if __name__ == "__main__":
    model = YOLO("yolov8s.pt")  # COCO pretrained

    model.train(
        data=DATA_YAML,
        epochs=120,
        imgsz=640,
        batch=-1,            # auto-batch
        device=0,
        project=RUNS_DIR,
        name=EXP_NAME,
        pretrained=True,
        workers=4,
        seed=42,
        amp=True,
        patience=30,

        # ===== Augmentations OFF (supported args only) =====
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0,
        perspective=0.0,
        fliplr=0.0, flipud=0.0,
        close_mosaic=0,

        # ===== Optimizer & LR =====
        optimizer="SGD",     # (xohlasangiz AdamW: lr0=0.003, weight_decay=0.01)
        lr0=0.01,
        lrf=0.05,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        cos_lr=True,

        # ===== Dataloader =====
        cache="disk",
        rect=False,
        verbose=False
    )
