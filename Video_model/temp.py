import os
import cv2
import numpy as np
import time
from ultralytics import YOLO

# Load all models
models = {
    "General": YOLO("./yolov8n.pt"),
    "Bottle": YOLO("./Yolo_models/bottle_detection_model/weights/best.pt"),
    "Blood": YOLO("./Yolo_models/Blood/weights/best.pt"),
    "License_Plate": YOLO("./Yolo_models/License_Plate/weights/best.pt"),
    "Cigarette": YOLO("./Yolo_models/Smoke/weights/best.pt")
}

# Global list to store filtered detections
filtered_detections = []

# Define effect rules
blur_pixelate_classes = ['alcohol_bottlerotation', 'cigarette', 'smoking']
pixelate_only_classes = ['blood', 'License_Plate', 'Violent']

# Input and Output video paths
video_path = "./Inputs/videoplayback.mp4"
output_path = "./Outputs/processed_video1.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Error: Unable to open video file.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

print("🔄 Processing video...")

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    original_frame = frame.copy()
    resized_frame = cv2.resize(frame, (640, 640))  # For detection

    detections_this_frame = []

    for model_name, model in models.items():
        results = model.predict(source=resized_frame, conf=0.3)

        for result in results:
            timestamp_seconds = frame_count / fps
            timestamp = time.strftime('%H:%M:%S', time.gmtime(timestamp_seconds))
            milliseconds = int((timestamp_seconds % 1) * 1000)
            timestamp_with_ms = f"{timestamp}.{milliseconds:03d}"

            for i, box in enumerate(result.boxes.xyxy):
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                class_id = int(result.boxes.cls[i].item())
                class_name = model.names.get(class_id, "Unknown")

                detection = {
                    "model": model_name,
                    "timestamp": timestamp_with_ms,
                    "class": class_name,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                }

                if class_name in blur_pixelate_classes + pixelate_only_classes:
                    filtered_detections.append(detection)
                    detections_this_frame.append((x1, y1, x2, y2, class_name))

    # Apply pixelation/blur to original frame
    for x1, y1, x2, y2, cls in detections_this_frame:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_width, x2), min(frame_height, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        region = original_frame[y1:y2, x1:x2]
        if region.size == 0:
            continue

        # Case 1: Smoke — show warning only
        if cls == 'smoking' or cls == 'cigarette':
            cv2.putText(
                original_frame,
                "Smoking is injurious to health",
                (x1, min(y2 + 30, frame_height - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            print(f"{cls} detected at {timestamp_with_ms} → Warning shown")
            continue

        # Case 2: Alcohol — show double warning
        if cls == 'alcohol_bottlerotation':
            warnings = [
                "Don't drink Alcohol",
                "Don't drink and drive"
            ]
            for idx, warning in enumerate(warnings):
                y_offset = min(y2 + 30 + (idx * 25), frame_height - 10)
                cv2.putText(
                    original_frame,
                    warning,
                    (x1, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
            print(f"🍺 {cls} detected at {timestamp_with_ms} → Warnings shown")
            continue

        # Case 3: Apply strong pixelation

        strong_pixel_size = 4  # lower = more pixelated
        temp = cv2.resize(region, (strong_pixel_size, strong_pixel_size), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(temp, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

        # Optional blur
        if cls in blur_pixelate_classes:
            pixelated = cv2.GaussianBlur(pixelated, (9, 9), 0)

        original_frame[y1:y2, x1:x2] = pixelated
        print(f"🎯 {cls} detected at {timestamp_with_ms} → Effect applied at ({x1},{y1},{x2},{y2})")

    out.write(original_frame)
    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Video processing complete.")
print(f"🎬 Output video saved to: {output_path}")
print("📦 Filtered detections:")
for det in filtered_detections:
    print(det)
