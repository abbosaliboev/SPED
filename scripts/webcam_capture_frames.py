# webcam_capture_frames_mac_safe.py
# pip install opencv-python

import cv2
import time
import sys
from pathlib import Path

# ======= CONFIG =======
CAM_INDEX    = 0                   # 0 = default webcam
OUT_DIR      = "../frames_back_cam"     # POSIX yo'l (forward slash)
LIMIT        = 1000
STEP         = 3
SET_SIZE     = (1280, 720)            # None bo'lsa o'zgartirmaydi
JPEG_QUALITY = 95
SHOW_FPS     = True
BLUR_VAR_MIN = 0                      # masalan 80..120 sinab ko'ring; 0 = o'chirilgan

def is_blurry(bgr, thr):
    if thr <= 0:
        return False
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(g, cv2.CV_64F).var()
    return var < thr

def open_capture(index):
    # Mac: AVFoundation; Windows: DSHOW; Linux: V4L2 (default Yetarli)
    if sys.platform == "darwin":
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            # ba'zi Mac'larda backend ko'rsatmasdan ochiladi
            cap = cv2.VideoCapture(index)
    elif sys.platform.startswith("win"):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index)
    else:
        cap = cv2.VideoCapture(index)  # Linux
    return cap

def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    cap = open_capture(CAM_INDEX)

    if not cap or not cap.isOpened():
        raise RuntimeError(
            f"Kamerani ochib bo'lmadi (index={CAM_INDEX}).\n"
            "- Mac’da Terminal/VSCode uchun Camera ruxsatini tekshiring: "
            "System Settings → Privacy & Security → Camera.\n"
            "- Agar M1/M2 bo'lsa, CAP_AVFOUNDATION kerak.\n"
            "- Boshqa ilova kamerani band qilib turmaganini tekshiring."
        )

    if SET_SIZE:
        w, h = SET_SIZE
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    print("[INFO] 'q' bosib chiqish mumkin.")
    saved = 0
    frame_idx = 0
    t0 = time.time(); fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[WARN] Kadr o'qilmadi, davom etyapmiz...")
            continue

        frame_idx += 1
        if SHOW_FPS:
            t1 = time.time()
            dt = t1 - t0
            if dt > 0: fps = 1.0 / dt
            t0 = t1

        do_save = (frame_idx % STEP == 0) and (saved < LIMIT)
        if do_save and is_blurry(frame, BLUR_VAR_MIN):
            do_save = False

        if do_save:
            fname = f"frame55_{saved:04d}.jpg"
            out_path = str(Path(OUT_DIR) / fname)
            cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            saved += 1

        txt = f"saved: {saved}/{LIMIT} | step:{STEP}"
        if SHOW_FPS: txt += f" | FPS:{fps:.1f}"
        cv2.putText(frame, txt, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 230, 40), 2, cv2.LINE_AA)

        cv2.imshow("Webcam Capture (press 'q' to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or saved >= LIMIT:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[DONE] Saqlandi: {saved} ta rasm -> {OUT_DIR}")

if __name__ == "__main__":
    main()