from ultralytics import YOLO
import cv2


person_model = YOLO("yolov8s.pt")  # COCO model, person 
assistive_model = YOLO("best.pt")  # Crutches, pram, wheelchair 

# Video yoki kamera
source = "IMG_4924.mp4"  # 0 = webcam; "test/test1.mp4" = video fayl

cap = cv2.VideoCapture(source)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1-bosqich: Odamni aniqlash
    person_results = person_model.predict(frame, conf=0.5, classes=[0], verbose=False)
    person_boxes = person_results[0].boxes.xyxy.cpu().numpy().astype(int)

    # 2-bosqich: Yordamchi asboblarni aniqlash
    assistive_results = assistive_model.predict(frame, conf=0.7, verbose=False)
    assistive_boxes = assistive_results[0].boxes.xyxy.cpu().numpy().astype(int)
    assistive_names = assistive_model.names
    assistive_classes = assistive_results[0].boxes.cls.cpu().numpy().astype(int)

    # Odamlar va asboblarni chizish
    for box in person_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    for box, cls_id in zip(assistive_boxes, assistive_classes):
        x1, y1, x2, y2 = box
        label = assistive_names[cls_id]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

    # Natijani ko‘rsatish
    cv2.imshow("SmartCare AI Light", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
