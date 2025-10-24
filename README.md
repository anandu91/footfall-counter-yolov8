# Footfall Counter using Computer Vision

Detect, track, and **count people** as they **cross a virtual line (ROI)** using **YOLOv8** (detector) + **ByteTrack** (tracker) with OpenCV overlays.

---

## 1) Brief Description of the Approach
- **Detection:** Ultralytics **YOLOv8** runs on each frame (restricted to **COCO class 0: person**).
- **Tracking:** **ByteTrack** (via `ultralytics.YOLO.track`) keeps **stable track IDs** across frames.
- **ROI Line:** A virtual line **A→B** is specified in **normalized coordinates** (`--line x1 y1 x2 y2`).
- **Robustness:** Inner/outer **hysteresis bands**, **minimum travel**, **EMA centroid smoothing**, and **per-ID cooldown** prevent jitter and double-counts.
- **Output:** Live HUD (Entry/Exit), bbox + ID overlays, line/bands visualization, and a saved processed video.

**Screenshots**

<p align="center">
  <img src="assets/screenshots/frame_001.png" width="48%" alt="Footfall counter – sample frame 1">
  <img src="assets/screenshots/frame_002.png" width="48%" alt="Footfall counter – sample frame 2">
</p>

---

## 2) Video Source Used (link or description)
- **Primary dataset:** **Oxford Town Centre** (fixed street CCTV; research-use).  
  Kaggle mirror: https://www.kaggle.com/datasets/almightyj/oxford-town-centre/data
- **This repo’s input file:** `assets/Video_Input.mp4`  
  *(You may replace this with any public people video or your own recording.)*

---

## 3) Explanation of the Counting Logic (ROI Line Crossing)
1. Compute each tracked person’s **centroid** every frame.
2. For line **A→B**, compute the centroid’s **signed distance** to the line (which side it lies on).
3. A **non-zero sign flip** that happens **outside the inner band** counts as **one crossing**.
4. Project motion **along the line’s normal** to classify **Entry vs Exit** according to `--direction` (`lr`, `rl`, `tb`, `bt`).
5. Apply **minimum travel** and a **per-ID cooldown** so a single person isn’t counted multiple times while lingering near the boundary.

---

## 4) Dependencies & Setup Instructions

### Environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

requirements.txt

```bash
ultralytics>=8.2.0
opencv-python>=4.8.0
numpy>=1.24.0

```bash
python footfall_counter.py \
  --source assets/Video_Input.mp4 \
  --count_mode line \
  --line 0.50 0.10 0.50 0.90 \
  --direction lr \
  --margin 0.03 --inner 0.015 --mintravel 12 --cooldown 25 \
  --save assets/example_output.mp4 --show

Output: The processed video is saved to assets/example_output.mp4 (with IDs, ROI line, and live Entry/Exit counters), and final totals are also printed in the console.
