import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import linear_sum_assignment

def load_features(file_path):
    df = pd.read_csv(file_path)
    track_ids = df['track_id'].values
    visual_feats = df[[col for col in df.columns if col.startswith('f_')]].values
    meta_feats = df[['cx', 'cy', 'duration']].values
    return track_ids, visual_feats, meta_feats

def compute_id_mapping(tacticam_ids, tacticam_visual, tacticam_meta,
                       broadcast_ids, broadcast_visual, broadcast_meta,
                       similarity_threshold=0.6, alpha=0.7):
    scaler = MinMaxScaler()
    tacticam_meta_scaled = scaler.fit_transform(tacticam_meta)
    broadcast_meta_scaled = scaler.transform(broadcast_meta)

    visual_sim = cosine_similarity(tacticam_visual, broadcast_visual)
    meta_sim = cosine_similarity(tacticam_meta_scaled, broadcast_meta_scaled)
    combined_sim = alpha * visual_sim + (1 - alpha) * meta_sim

    cost_matrix = -combined_sim
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    mapping = []
    used_tacticam = set()
    used_broadcast = set()

    for i, j in zip(row_ind, col_ind):
        sim = combined_sim[i][j]
        if sim >= similarity_threshold:
            tid = int(tacticam_ids[i])
            bid = int(broadcast_ids[j])
            if tid not in used_tacticam and bid not in used_broadcast:
                mapping.append({
                    "tacticam_track_id": tid,
                    "broadcast_track_id": bid,
                    "combined_similarity": round(sim, 4),
                    "visual_similarity": round(visual_sim[i][j], 4),
                    "spatial_temporal_similarity": round(meta_sim[i][j], 4)
                })
                used_tacticam.add(tid)
                used_broadcast.add(bid)

    return pd.DataFrame(mapping)

def main():
    tacticam_path = "tracking_outputs/tacticam_features.csv"
    broadcast_path = "tracking_outputs/broadcast_features.csv"
    output_path = "tracking_outputs/id_mapping_with_spatial_temporal.csv"

    print("Loading features...")
    t_ids, t_visual, t_meta = load_features(tacticam_path)
    b_ids, b_visual, b_meta = load_features(broadcast_path)

    print("Computing combined similarity...")
    mapping_df = compute_id_mapping(t_ids, t_visual, t_meta, b_ids, b_visual, b_meta,
                                    similarity_threshold=0.6, alpha=0.7)

    print("Checking for duplicates...")
    assert mapping_df['tacticam_track_id'].is_unique, "❌ Duplicate tacticam IDs!"
    assert mapping_df['broadcast_track_id'].is_unique, "❌ Duplicate broadcast IDs!"

    print(f"Saving to {output_path}...")
    mapping_df.to_csv(output_path, index=False)
    print("✅ Done.")

if __name__ == "__main__":
    main()