import streamlit as st
import os
import pandas as pd
import cv2
import glob
import re
from PIL import Image
from jersey_number_recognizer import recognize_jersey_number  # ✅ NEW

# === Utility to find representative crop ===
def find_representative_crop(crop_dir, track_id):
    pattern = os.path.join(crop_dir, f"track_{int(track_id)}_frame_*.jpg")
    matched_files = glob.glob(pattern)

    if not matched_files:
        return None

    def extract_frame_number(filename):
        match = re.search(r'frame_(\d+)', filename)
        return int(match.group(1)) if match else float('inf')

    matched_files.sort(key=extract_frame_number)
    return matched_files[len(matched_files) // 2]

# === Set paths ===
matches_csv_path = 'tracking_outputs/id_mapping_with_spatial_temporal.csv'
tacticam_crops_dir = 'tracking_outputs/tacticam_crops'
broadcast_crops_dir = 'tracking_outputs/broadcast_crops'

# === Load mapping CSV ===
if not os.path.exists(matches_csv_path):
    st.error(f"❌ Mapping file not found: {matches_csv_path}")
    st.stop()

matches_df = pd.read_csv(matches_csv_path)
if matches_df.empty:
    st.warning("Mapping CSV is empty.")
    st.stop()

# === Streamlit UI ===
st.title("🎽 Cross Camera Player Mapping")
st.write("Player Mapping across Tacticam and Broadcast clips.")

# Slider to choose a match index
match_index = st.slider("Select Match Index", 0, len(matches_df)-1, 0)

# Get track IDs
tac_id = matches_df.iloc[match_index]['tacticam_track_id']
broad_id = matches_df.iloc[match_index]['broadcast_track_id']

# Find representative crops
tac_crop_path = find_representative_crop(tacticam_crops_dir, tac_id)
broad_crop_path = find_representative_crop(broadcast_crops_dir, broad_id)

col1, col2 = st.columns(2)

# Display Tacticam Crop
with col1:
    st.subheader(f"Tacticam: ID {int(tac_id)}")
    if tac_crop_path and os.path.exists(tac_crop_path):
        tac_img = Image.open(tac_crop_path)
        st.image(tac_img, use_container_width=True)
        jersey_tac = recognize_jersey_number(tac_crop_path)
        st.markdown(f"👕 Jersey Number: **{jersey_tac or 'Not detected'}**")
    else:
        st.error("Tacticam crop not found.")
        jersey_tac = None

# Display Broadcast Crop
with col2:
    st.subheader(f"Broadcast: ID {int(broad_id)}")
    if broad_crop_path and os.path.exists(broad_crop_path):
        broad_img = Image.open(broad_crop_path)
        st.image(broad_img, use_container_width=True)
        jersey_broad = recognize_jersey_number(broad_crop_path)
        st.markdown(f"👕 Jersey Number: **{jersey_broad or 'Not detected'}**")
    else:
        st.error("Broadcast crop not found.")
        jersey_broad = None

# Match verdict
if jersey_tac and jersey_broad:
    if jersey_tac == jersey_broad:
        st.success("✅ Jersey numbers match!")
    else:
        st.warning("⚠️ Jersey numbers do not match.")