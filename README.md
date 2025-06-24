# Player Re-Identification Project

This project implements a player re-identification system using YOLO for object detection and `boxmot`/`ByteTrack` for multi-object tracking. It processes a 15-second video (`15sec_input_720p.mp4`), detects every player, assigns unique IDs, and maintains those IDs when players leave and re-enter the frame. The output is saved as `output.mp4` with annotated bounding boxes and IDs.

---
## Output Video
Below is the output video showcasing the player re-identification results:

<video controls width="640" height="360">
  <source src="output.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

---

## 🧠 Project Overview

- **Purpose**: Detect and track players in a video, ensuring consistent ID assignment for re-identification.
- **Tools**: YOLOv8 for detection, `boxmot`/`ByteTrack` for tracking, OpenCV for video processing.
- **Output**: Annotated video with player IDs.

---

## ✅ Prerequisites

- Python 3.8 or higher  
- Git (for cloning the repository)  
- pip (Python package manager)

---

## ⚙️ Installation

### 📁 Clone the Repository

```bash
git clone https://github.com/annie-2314/player-re-identification-in-a-single-feed.git
cd player-reid-project
```

### 🧪 Set Up a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
```

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

### 📥 Install `boxmot`

```bash
pip install git+https://github.com/mikel-brostrom/yolo_tracking.git
```

### 📂 Download the Model

Place the YOLO model file (`yolov8n.pt`) in the project directory (`D:\Projects\player_reid_project` or adjust the path in the script). If you have a different model (e.g., `yolov11_model.pt`), replace it accordingly. You can download `yolov8n.pt` from the Ultralytics YOLO repository or obtain the specific model from your instructor.

---

## 🎞 Prepare the Input Video

Place your input video (`15sec_input_720p.mp4`) in the project directory. Ensure the path in the script matches:

```python
video_path = r"D:\Projects\player_reid_project\15sec_input_720p.mp4"
```

---

## 🚀 Usage

1. Activate the virtual environment (if not already active):

   ```bash
   source venv/bin/activate     # On Windows: venv\Scripts\activate
   ```

2. Run the script:

   ```bash
   python player_reid.py
   ```

3. A video window will display the tracking process. Press `q` to quit.

4. The output video (`output.mp4`) will be saved in the project directory.

---

## 📁 Project Files

- `player_reid.py`: Main script for detection and tracking.
- `requirements.txt`: List of Python dependencies (content below).
- `yolov8n.pt`: YOLO model file (replace with your model if different).
- `15sec_input_720p.mp4`: Input video file.
- `output.mp4`: Output video with annotated player IDs.

---

## 🛠 Troubleshooting

- **Installation Errors**: Ensure Git and pip are installed. If `boxmot` fails, verify your internet connection and retry the Git install command.
- **Tracking Issues**: If no tracks are generated, check the `detections` shape in the console output. Reinstall dependencies if needed:

  ```bash
  pip install --upgrade ultralytics boxmot opencv-python numpy
  ```

- **Missed Players**: Lower the confidence threshold (`conf=0.1`) or area filter (`area > 100`) in the script. Request a better model (e.g., `yolov11_model.pt`) from your instructor if needed.
- **Unpacking Errors**: If a `ValueError: too many values to unpack` occurs, ensure the script matches the `ByteTrack` output format (currently set for 8 elements).

---

## 🧾 Development Notes

- The project was developed with assistance from Grok 3 (xAI) on June 23, 2025.
- Initial issues included missing `setup.py` in the `boxmot` repository, resolved by installing directly from GitHub.
- The script was updated to handle an 8-element `ByteTrack` output (`[x1, y1, x2, y2, track_id, conf, class_id, extra]`) due to a version mismatch.

---

## 📌 Additional Instructions

1. Save this entire content as `README.md` in your project directory (e.g., `D:\Projects\player_reid_project`).
2. Extract the `requirements.txt` section (starting with `ultralytics==8.0.196`) and save it as `requirements.txt` in the same directory.
3. Ensure `player_reid.py` (the script part) is saved as `player_reid.py` in the same directory, overwriting the previous version if needed.
4. Update the paths in `player_reid.py` (`model_path`, `video_path`, `output_path`) to match your local setup.
5. To share on GitHub, initialize a repository:

   ```bash
   git init
   git add .
   git commit -m "Initial commit with player re-id project"
   git remote add origin https://github.com/annie-2314/player-re-identification-in-a-single-feed.git
   git push -u origin main
   ```

6. Test the setup by cloning on another machine:

   ```bash
   git clone https://github.com/annie-2314/player-re-identification-in-a-single-feed.git
   cd player-reid-project
   python -m venv venv
   venv\Scripts\activate  # On Windows
   pip install -r requirements.txt
   pip install git+https://github.com/mikel-brostrom/yolo_tracking.git
   ```

   Add `yolov8n.pt` and `15sec_input_720p.mp4`, then run:

   ```bash
   python player_reid.py
   ```

---

## ✅ Confirmation

- Confirm the files are saved correctly.
- Run `pip install -r requirements.txt` and ensure no errors.
- Verify the script runs and produces `output.mp4` with player IDs.
- Let me know with “Done with Step 6” or share any issues!
