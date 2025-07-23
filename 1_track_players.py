import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import csv
import os

# 🔷 Set video type: either "tacticam" or "broadcast"
video_type = 'tacticam'  # ✅ Change this to 'tacticam' or tacticam when needed

# === Auto-set paths ===
video_path = f'videos/{video_type}.mp4'
output_video_path = f'tracking_outputs/{video_type}_tracked.mp4'
output_csv_path = f'{video_type}_tracking_output.csv'

# === Load models ===
yolo_model = YOLO('best.pt')
tracker = DeepSort(max_age=30)

# === Open video ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"❌ Could not open video: {video_path}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# === Output writer ===
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# === Initialize CSV for saving tracking data ===
os.makedirs('tracking_outputs', exist_ok=True)
csv_file = open(os.path.join('tracking_outputs', output_csv_path), 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['frame', 'track_id', 'x1', 'y1', 'x2', 'y2'])

# === Tracking loop ===
frame_id = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 detection
    results = yolo_model(frame)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())

        if conf > 0.5:
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'player'))

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Process each track
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())

        # Save to CSV
        csv_writer.writerow([frame_id, track_id, x1, y1, x2, y2])

        # Draw box and ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)
    frame_id += 1
    print(f"Processed frame {frame_id}")

# === Cleanup ===
csv_file.close()
cap.release()
out.release()
cv2.destroyAllWindows()