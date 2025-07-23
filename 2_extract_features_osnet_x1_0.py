import os
import cv2
import torch
import torchreid
import pandas as pd
from torchvision import transforms
from tqdm import tqdm

# === CONFIG ===
video_type = 'tacticam'  # change to 'broadcast'/tacticam if needed
video_path = f'videos/{video_type}.mp4'
track_csv = f'tracking_outputs/{video_type}_tracking_output.csv'
output_features = f'tracking_outputs/{video_type}_features.csv'
output_crops_dir = f'tracking_outputs/{video_type}_crops'
os.makedirs(output_crops_dir, exist_ok=True)

# === Torchreid setup ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torchreid.models.build_model(name='osnet_x1_0', num_classes=1000, pretrained=True)
model.eval()
model.to(device)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Load tracks ===
df = pd.read_csv(track_csv)

# === Read video ===
cap = cv2.VideoCapture(video_path)
frame_dict = {}
frame_id = 0
print("Caching video frames...")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_dict[frame_id] = frame
    frame_id += 1
cap.release()

# === Extract features with cx, cy ===
features_list = []
print("Extracting features...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    frame_id = row['frame']
    track_id = row['track_id']
    x1, y1, x2, y2 = map(int, [row['x1'], row['y1'], row['x2'], row['y2']])
    frame = frame_dict.get(frame_id)
    if frame is None:
        continue

    crop = frame[y1:y2, x1:x2]
    if crop.shape[0] < 10 or crop.shape[1] < 10:
        continue

    crop_resized = transform(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model(crop_resized)
        feature_np = feature.cpu().numpy().flatten()

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    norm_cx = center_x / frame.shape[1]
    norm_cy = center_y / frame.shape[0]

    crop_filename = f'{output_crops_dir}/track_{track_id}_frame_{frame_id}.jpg'
    cv2.imwrite(crop_filename, crop)

    features_list.append({
        'track_id': track_id,
        'frame_id': frame_id,
        'cx': norm_cx,
        'cy': norm_cy,
        **{f'f_{i}': v for i, v in enumerate(feature_np)}
    })

# === Aggregate features ===
print(f"Saving aggregated features to: {output_features}")
features_df = pd.DataFrame(features_list)
temporal_df = features_df.groupby('track_id')['frame_id'].agg(['min', 'max'])
temporal_df['duration'] = temporal_df['max'] - temporal_df['min']
temporal_df = temporal_df.rename(columns={'min': 'start_frame', 'max': 'end_frame'})
agg_feats = features_df.groupby('track_id').mean().reset_index()
agg_feats = agg_feats.merge(temporal_df, on='track_id', how='left')
agg_feats.to_csv(output_features, index=False)