# Stealth Player Tracking & Re-Identification

This project performs **cross-camera player re-identification** by tracking players in two different video perspectives: **Tacticam** and **Broadcast**. It combines object detection, tracking, feature extraction, and a re-ID matching system using deep learning techniques.

## Folder Structure
stealth-player-tracking/

├── 1_track_players.py

├── 2_extract_features_osnet_x1_0.py

├── 3_match_tracks_osnet_1.py

├── 4_visualize_matched_crops.py

├── 5_app_view_matches.py

├── jersey_number_recognizer.py

├── requirements.txt

├── README.md

├── tracking_outputs/ # Generated tracking data, crops, features

└── videos/ # Input videos: tacticam.mp4, broadcast.mp4

## 🔧 Setup Instructions

# 1️⃣ Create and activate a virtual environment

python -m venv venv

venv\Scripts\activate  # For Windows

# 2#️⃣ Install dependencies
pip install -r requirements.txt

# How to Run

# Step 1: Track Players in Both Videos

Please download best.pt shared by you and place it in the project folder before running the scripts.

Edit video_type to 'tacticam' and run

python 1_track_players.py

Then edit video_type to 'broadcast' and run again

# Step 2: Extract ReID Features and Crops

python 2_extract_features_osnet_x1_0.py

Then edit video_type to 'broadcast' and run again


# Step 3: Match Tracks Across Cameras

python 3_match_tracks_osnet_1.py

# Step 4: Optional Jersey Number Matching (Experimental)

Uses OCR with EasyOCR (experimental and not very accurate)

python jersey_number_recognizer.py

# Step 5: Visualize Matched Pairs (side-by-side)

python 4_visualize_matched_crops.py

Streamlit App (UI Viewer)

streamlit run 5_app_view_matches.py

# Techniques Used

YOLOv8 for player detection

DeepSORT for tracking

OSNet via torchreid for feature extraction

Hungarian algorithm with cosine similarity for matching

EasyOCR (optional) for jersey number OCR

# Notes
OCR was integrated as an additional enhancement, but did not yield useful results due to low jersey text clarity.

All intermediate results are saved in tracking_outputs/.

# Pending: 

Integrate Homography-based alignment for better spatial matching

Improve jersey recognition using CRNN or fine-tuned OCR


