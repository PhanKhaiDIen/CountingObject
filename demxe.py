import cv2
import numpy as np
from ultralytics import YOLO

# ———— 1. Mở video và chuẩn bị ghi đầu ra ————
cap = cv2.VideoCapture("xe1.mp4")
if not cap.isOpened():
    raise IOError("Không thể mở file video")

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_2zones_with_bboxes.mp4", fourcc, fps, (w, h))

# ———— 2. Định nghĩa 2 vùng đa giác (dtype=int32) ————
zone1 = np.array([(413, 258), (99, 346), (631, 452), (715, 322)], dtype=np.int32)
zone2 = np.array([(752, 330), (694, 466), (1071, 512), (959, 352)], dtype=np.int32)
zones = [zone1, zone2]
zone_colors = [(0, 255, 0), (0, 0, 255)]  # BGR: xanh lá và đỏ

# ———— 3. Load model và lớp cần đếm ————
model = YOLO("yolo11n.pt")
classes_to_count = [2, 5]                # COCO: 2=car, 5=bus
class_names      = {2: "car", 5: "bus"}  # ánh xạ id → tên

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ———— 4. Phát hiện và lọc car/bus ————
    raw_results = model(frame, classes=classes_to_count, verbose=False)
    results     = raw_results[0]

    # ———— 5. Khởi tạo bộ đếm mỗi vùng ————
    count_per_zone = [{"car": 0, "bus": 0}, {"car": 0, "bus": 0}]

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        cls_id        = int(box.cls[0])
        center        = ((x1 + x2) // 2, (y1 + y2) // 2)
        name          = class_names.get(cls_id)
        if name is None:
            continue

        for i, zone in enumerate(zones):
            if cv2.pointPolygonTest(zone, center, False) >= 0:
                count_per_zone[i][name] += 1
                break

        # ———— 6. Vẽ khung phát hiện đối tượng ————
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Vẽ khung bao quanh đối tượng
        label = f"{name}: {int(box.conf[0]*100)}%"  # Hiển thị tên đối tượng và độ tin cậy
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # ———— 7. Tô màu bán trong suốt cho từng zone ————
    overlay = frame.copy()
    alpha = 0.3
    for i, zone in enumerate(zones):
        cv2.fillPoly(overlay, [zone], color=zone_colors[i])

    # Ghép overlay lên frame chính
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # ———— 8. Vẽ viền và text cho từng zone ————
    for i, zone in enumerate(zones):
        cv2.polylines(frame, [zone], isClosed=True, color=zone_colors[i], thickness=2)
        txt = f"Vung {i+1}: Car={count_per_zone[i]['car']}  Bus={count_per_zone[i]['bus']}"
        cv2.putText(frame, txt,
                    (zone[0][0] + 10, zone[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

    # ———— 9. Ghi frame và hiển thị ————
    out.write(frame)
    cv2.imshow("Detection with Colored Zones and Bounding Boxes", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ———— 10. Giải phóng ————
cap.release()
out.release()
cv2.destroyAllWindows()
