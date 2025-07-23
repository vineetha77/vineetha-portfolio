import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import glob
import re

# === Function to find the most representative crop (middle frame) ===
def find_representative_crop(crop_dir, track_id):
    pattern = os.path.join(crop_dir, f"track_{int(track_id)}_frame_*.jpg")
    matched_files = glob.glob(pattern)

    if not matched_files:
        return None

    # Sort files by frame number
    def extract_frame_number(filename):
        match = re.search(r'frame_(\d+)', filename)
        return int(match.group(1)) if match else float('inf')

    matched_files.sort(key=extract_frame_number)
    return matched_files[len(matched_files) // 2]  # Middle frame

# === Paths ===
matches_csv_path = 'tracking_outputs/id_mapping_with_spatial_temporal.csv'
tacticam_crops_dir = 'tracking_outputs/tacticam_crops'
broadcast_crops_dir = 'tracking_outputs/broadcast_crops'
output_path = 'tracking_outputs/side_by_side_matches.jpg'

# === Load matched track IDs ===
matches_df = pd.read_csv(matches_csv_path)

# === Setup for visualization ===
num_matches = min(10, len(matches_df))
fig, axes = plt.subplots(num_matches, 2, figsize=(8, 2 * num_matches))

shown = 0  # to track successful displays

for idx in range(len(matches_df)):
    if shown >= num_matches:
        break

    tac_id = matches_df.iloc[idx]['tacticam_track_id']
    broad_id = matches_df.iloc[idx]['broadcast_track_id']

    # Find representative crop images
    tac_crop_path = find_representative_crop(tacticam_crops_dir, tac_id)
    broad_crop_path = find_representative_crop(broadcast_crops_dir, broad_id)

    if tac_crop_path is None or broad_crop_path is None:
        print(f"[!] Skipping pair {tac_id}, {broad_id} — image missing.")
        continue

    tac_img = cv2.imread(tac_crop_path)
    broad_img = cv2.imread(broad_crop_path)

    tac_img = cv2.cvtColor(tac_img, cv2.COLOR_BGR2RGB)
    broad_img = cv2.cvtColor(broad_img, cv2.COLOR_BGR2RGB)

    axes[shown, 0].imshow(tac_img)
    axes[shown, 0].set_title(f'Tacticam: {int(tac_id)}')
    axes[shown, 0].axis('off')

    axes[shown, 1].imshow(broad_img)
    axes[shown, 1].set_title(f'Broadcast: {int(broad_id)}')
    axes[shown, 1].axis('off')

    shown += 1

# === Finalize and save ===
plt.tight_layout()
plt.savefig(output_path)
print(f"✅ Saved side-by-side match image: {output_path}")
plt.show()