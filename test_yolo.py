from ultralytics import YOLO
import cv2

# Load the YOLOv11 model
model = YOLO("yolov11_model.pt")  # Replace with the actual path to your model file

# Open the video
cap = cv2.VideoCapture("15sec_input_720p.mp4")  # Replace with the actual path to your video file

# Verify video and model are loaded
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()
if model is None:
    print("Error: Could not load YOLO model.")
    exit()

# Process the first few frames
frame_count = 0
while cap.isOpened() and frame_count < 5:  # Limit to 5 frames for testing
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Detect objects (players and ball)
    results = model.predict(frame, classes=[0], conf=0.5)  # Class 0 = player
    
    # Draw bounding boxes (optional for visualization)
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Conf: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the frame (optional)
    cv2.imshow("YOLO Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_count += 1

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Test completed. Check the console and window for results.")