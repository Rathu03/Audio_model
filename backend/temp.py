from flask import Flask, jsonify
from flask_socketio import SocketIO
import cv2
from ultralytics import YOLO
import time
import os
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load YOLO models
models = {
    "General": YOLO("../Video_model/yolov8n.pt"),
    "Bottle": YOLO("../Video_model/Yolo_models/bottle_detection_model/weights/best.pt"),
    "Blood": YOLO("../Video_model/Yolo_models/Blood/weights/best.pt"),
    "License_Plate": YOLO("../Video_model/Yolo_models/License_Plate/weights/best.pt"),
    "Cigarette": YOLO("../Video_model/Yolo_models/Smoke/weights/best.pt")
}

UPLOAD_FOLDER = "uploads1"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Store final filtered detections
filtered_detections = []

@socketio.on("upload_video_file")
def handle_video(data):
    """
    Handles video upload, processes it with YOLO models, and sends filtered detection results.
    """
    try:
        print("Video upload received!")

        # Decode the video bytes
        video_bytes = np.frombuffer(data["video"], dtype=np.uint8)
        video_path = os.path.join(UPLOAD_FOLDER, "uploaded_video.mp4")

        # Save the video
        with open(video_path, "wb") as f:
            f.write(video_bytes)
        print(f"Video saved at: {video_path}")

        # Process the video and get filtered detections
        filtered = process_video(video_path)

        # Emit only filtered detections
        socketio.emit("video_process_complete", {
            "message": "Video processing complete!",
            "detections": filtered
        })

    except Exception as e:
        print(f"Error processing video: {e}")
        socketio.emit("error", {"message": f"Error: {str(e)}"})


def process_video(video_path):
    """
    Processes the video with YOLO models and returns only filtered detections.
    """
    global filtered_detections
    filtered_detections = []

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(fps / 10))  # Process at 10 FPS

    print(f"Processing at 10 FPS, skipping every {frame_interval} frames")

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval != 0:
            frame_count += 1
            continue

        resized_frame = cv2.resize(frame, (640, 640))

        for model_name, model in models.items():
            results = model.predict(source=resized_frame, conf=0.3)

            for result in results:
                if len(result.boxes) > 0:
                    timestamp_seconds = frame_count / fps
                    timestamp_milliseconds = int((timestamp_seconds - int(timestamp_seconds)) * 1000)
            
                    # Proper timestamp formatting with milliseconds
                    timestamp = time.strftime('%H:%M:%S', time.gmtime(int(timestamp_seconds)))
                    timestamp_with_ms = f"{timestamp}.{timestamp_milliseconds:03d}"
                    for box in result.boxes.xyxy:
                        x1, y1, x2, y2 = map(int, box.cpu().numpy())
                        class_id = int(result.boxes.cls[0].item())
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

                        # Filter only the required classes
                        if (
                            detection["class"] in ['blood', 'alcohol_bottlerotation', 'cigarette','smoking']
                        ):
                            filtered_detections.append(detection)

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    # Return filtered detections
    return filtered_detections


if __name__ == "__main__":
    socketio.run(app, debug=True)
