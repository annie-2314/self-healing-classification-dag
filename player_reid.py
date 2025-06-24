import cv2
import numpy as np
from ultralytics import YOLO
from boxmot import ByteTrack
from scipy.spatial.distance import cosine
from collections import defaultdict
import os

# ------------------ Paths ------------------
model_path = r"D:\Projects\player_reid_project\yolov8n.pt"  # Temporary model
video_path = r"D:\Projects\player_reid_project\15sec_input_720p.mp4"
output_path = r"D:\Projects\player_reid_project\output.mp4"

# ------------------ Load Model ------------------
assert os.path.exists(model_path), "Model file not found!"
assert os.path.exists(video_path), "Video file not found!"
model = YOLO(model_path)

# ------------------ Initialize Tracker ------------------
tracker = ByteTrack(
    track_thresh=0.3,  # Low threshold for sparse detections
    match_thresh=0.6,  # Lenient matching
    track_buffer=90,   # Retain tracks for 3 seconds (at 30 FPS)
    frame_rate=30
)

# ------------------ Helper: Color Histogram ------------------
def extract_color_histogram(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros((3000,))
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# ------------------ Setup Video ------------------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

# ------------------ Tracking Loop ------------------
frame_num = 0
lost_players = defaultdict(list)  # {player_id: [(frame_num, histogram), ...]}
id_remap = {}  # {new_track_id: original_id}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ------------------ YOLO Detection ------------------
    results = model.predict(frame, conf=0.1, verbose=False)
    detections = []
    print(f"Frame {frame_num}: Detected boxes = {len(results[0].boxes)}")
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().numpy()
        cls = int(box.cls[0].cpu().numpy())
        area = (x2 - x1) * (y2 - y1)
        print(f"Raw detection: x1={x1}, y1={y1}, x2={x2}, y2={y2}, conf={conf}, cls={cls}, area={area}")
        if cls == 0 and area > 100:  # Class 0 is "person"
            detection = [x1, y1, x2, y2, conf, cls]  # 6 elements for ByteTrack
            detections.append(detection)
            print(f"Valid detection added: {detection}")

    if len(detections) == 0:
        print(f"Frame {frame_num}: No valid player detections.")
        out.write(frame)
        frame_num += 1
        continue

    detections = np.array(detections, dtype=np.float32)
    print(f"Detections shape: {detections.shape}, dtype: {detections.dtype}")

    # ------------------ ByteTrack ------------------
    try:
        tracks = tracker.update(detections, frame)
        print(f"Frame {frame_num}: Generated tracks = {len(tracks)}")
    except Exception as e:
        print(f"Tracking error: {e}")
        tracks = []

    current_ids = set()
    for track in tracks:
        print(f"Track raw output: {track}")  # Debug track contents
        # Unpack 8 elements: [x1, y1, x2, y2, track_id, conf, class_id, extra]
        x1, y1, x2, y2, track_id, conf, class_id, _ = track  # Ignore extra element
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        current_ids.add(track_id)

        # Handle re-ID
        final_id = id_remap.get(track_id, track_id)

        # Draw
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID: {int(final_id)}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )
        print(f"Track: ID={final_id}, x1={x1}, y1={y1}, x2={x2}, y2={y2}, conf={conf}")

        # Update lost_players for re-identification
        hist = extract_color_histogram(frame, [x1, y1, x2, y2])
        lost_players[final_id].append((frame_num, hist))

    # ------------------ Re-identification ------------------
    for lost_id in list(lost_players.keys()):
        if lost_id in id_remap.values():
            continue
        if len(lost_players[lost_id]) == 0:
            continue
        last_frame, last_hist = lost_players[lost_id][-1]
        if frame_num - last_frame > 30:  # Lost for >1 second
            for track_id, hist in [(t[0], extract_color_histogram(frame, [int(t[1]), int(t[2]), int(t[3]), int(t[4])])) for t in tracks]:
                if track_id in id_remap:
                    continue
                similarity = 1 - cosine(last_hist, hist)
                if similarity > 0.5:
                    id_remap[track_id] = lost_id
                    print(f"Reassigned ID {track_id} → {lost_id}")
                    lost_players[lost_id].append((frame_num, hist))
                    break

    # ------------------ Info Overlay ------------------
    cv2.putText(frame, f"Frame: {frame_num}  Players: {len(tracks)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    out.write(frame)
    frame_num += 1

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------ Cleanup ------------------
cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Re-identification completed. Output saved to:", output_path)