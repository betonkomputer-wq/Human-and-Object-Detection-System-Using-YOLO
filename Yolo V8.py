import cv2
import numpy as np
from ultralytics import YOLO

# Load model YOLO
model = YOLO("yolov8n.pt")

# Buka kamera
cap = cv2.VideoCapture(0)

# Set fullscreen otomatis
cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "YOLO Detection",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference YOLO
    results = model(frame, stream=True)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # Gambar bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Tampilkan label dan confidence
            text = f"{label} {conf:.2f}"
            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow("YOLO Detection", frame)

    # Tekan Q untuk keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()