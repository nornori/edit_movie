"""
Create training data for cut selection model with enhanced features
1 VIDEO = 1 SAMPLE (no sequence splitting)

Uses temporal and contextual features added by add_temporal_features.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# Settings
TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_features_and_labels(source_features_dir, active_labels_dir):
    """Load enhanced source features and active labels, grouped by video"""
    source_features_dir = Path(source_features_dir)
    active_labels_dir = Path(active_labels_dir)
    
    # Find matching files
    feature_files = list(source_features_dir.glob('*_features.csv'))
    
    video_data = {}  # {video_name: dataframe}
    
    for feature_file in feature_files:
        video_name = feature_file.stem.replace('_features', '')
        active_file = active_labels_dir / f'{video_name}_active.csv'
        
        if not active_file.exists():
            print(f"  Skipping {video_name}: no active labels")
            continue
        
        # Load features
        df_features = pd.read_csv(feature_file, low_memory=False)
        
        # Load active labels
        df_active = pd.read_csv(active_file)
        
        # Round time to 1 decimal place for matching
        df_features['time'] = df_features['time'].round(1)
        df_active['time'] = df_active['time'].round(1)
        
        # Merge on time
        df_merged = pd.merge(df_features, df_active, on='time', how='inner')
        
        if len(df_merged) == 0:
            print(f"  Skipping {video_name}: no matching timestamps")
            continue
        
        video_data[video_name] = df_merged
        
        print(f"  {video_name}: {len(df_merged)} frames, active={df_merged['active'].mean()*100:.1f}%")
    
    if not video_data:
        raise ValueError("No data loaded!")
    
    return video_data

def extract_features(df):
    """Extract audio, visual, and temporal features from a single video dataframe"""
    
    # === AUDIO FEATURES ===
    audio_base = [
        'audio_energy_rms', 'audio_is_speaking', 'silence_duration_ms', 'speaker_id',
        'text_is_active', 'telop_active',
        'pitch_f0', 'pitch_std', 'spectral_centroid', 'zcr'
    ]
    
    # Audio embeddings
    audio_embeddings = [f'speaker_emb_{i}' for i in range(192)]
    audio_mfcc = [f'mfcc_{i}' for i in range(13)]
    
    # Audio temporal features (moving averages, changes)
    audio_temporal = []
    for col in ['audio_energy_rms', 'pitch_f0', 'spectral_centroid']:
        audio_temporal.extend([
            f'{col}_ma3', f'{col}_ma5', f'{col}_ma10',
            f'{col}_std5', f'{col}_diff1', f'{col}_diff2'
        ])
    
    # Audio change features
    audio_changes = [
        'audio_change_score', 'silence_to_speech', 'speech_to_silence',
        'speaker_change', 'pitch_change'
    ]
    
    audio_cols = audio_base + audio_embeddings + audio_mfcc + audio_temporal + audio_changes
    
    # === VISUAL FEATURES ===
    visual_base = [
        'scene_change', 'visual_motion', 'saliency_x', 'saliency_y',
        'face_count', 'face_center_x', 'face_center_y', 
        'face_size', 'face_mouth_open', 'face_eyebrow_raise'
    ]
    
    # CLIP embeddings
    visual_clip = [f'clip_{i}' for i in range(512)]
    
    # Visual temporal features
    visual_temporal = []
    for col in ['visual_motion', 'face_count', 'scene_change']:
        visual_temporal.extend([
            f'{col}_ma3', f'{col}_ma5', f'{col}_ma10',
            f'{col}_std5', f'{col}_diff1', f'{col}_diff2'
        ])
    
    # Visual change features
    visual_changes = [
        'visual_motion_change', 'face_count_change', 'saliency_movement'
    ]
    
    # CLIP similarity features
    visual_similarity = [
        'clip_sim_prev', 'clip_sim_next', 'clip_sim_mean5'
    ]
    
    visual_cols = visual_base + visual_clip + visual_temporal + visual_changes + visual_similarity
    
    # === TEMPORAL FEATURES ===
    temporal_cols = [
        'time_since_prev', 'time_to_next', 'cut_duration',
        'position_in_video', 'cut_density_10s',
        'cumulative_position', 'cumulative_adoption_rate'
    ]
    
    # === EXTRACT EXISTING COLUMNS ===
    audio_cols_exist = [c for c in audio_cols if c in df.columns]
    visual_cols_exist = [c for c in visual_cols if c in df.columns]
    temporal_cols_exist = [c for c in temporal_cols if c in df.columns]
    
    # Convert to numeric, coercing errors to NaN, then fill with 0
    audio_features = df[audio_cols_exist].apply(pd.to_numeric, errors='coerce').fillna(0).values
    visual_features = df[visual_cols_exist].apply(pd.to_numeric, errors='coerce').fillna(0).values
    temporal_features = df[temporal_cols_exist].apply(pd.to_numeric, errors='coerce').fillna(0).values
    active_labels = df['active'].values
    
    return audio_features, visual_features, temporal_features, active_labels, audio_cols_exist, visual_cols_exist, temporal_cols_exist

def main():
    print("=" * 60)
    print("Creating Cut Selection Training Data (Full Video)")
    print("1 VIDEO = 1 SAMPLE (no sequence splitting)")
    print("=" * 60)
    
    source_features_dir = 'data/processed/source_features_enhanced'
    active_labels_dir = 'data/processed/active_labels'
    output_dir = Path('preprocessed_data')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data grouped by video
    print("\n1. Loading enhanced features and labels (grouped by video)...")
    video_data = load_features_and_labels(source_features_dir, active_labels_dir)
    
    print(f"\nTotal videos: {len(video_data)}")
    
    # Extract features for each video
    print("\n2. Extracting features for each video...")
    
    all_audio_features = []
    all_visual_features = []
    all_temporal_features = []
    all_active_labels = []
    all_video_names = []
    all_video_lengths = []
    
    feature_dims = None
    
    for video_name, df in video_data.items():
        audio_feat, visual_feat, temporal_feat, active_lab, audio_cols, visual_cols, temporal_cols = extract_features(df)
        
        if feature_dims is None:
            feature_dims = {
                'audio': len(audio_cols),
                'visual': len(visual_cols),
                'temporal': len(temporal_cols)
            }
            print(f"\n  Feature dimensions:")
            print(f"    Audio: {feature_dims['audio']} columns")
            print(f"    Visual: {feature_dims['visual']} columns")
            print(f"    Temporal: {feature_dims['temporal']} columns")
            print(f"    Total: {sum(feature_dims.values())} columns")
        
        all_audio_features.append(audio_feat)
        all_visual_features.append(visual_feat)
        all_temporal_features.append(temporal_feat)
        all_active_labels.append(active_lab)
        all_video_names.append(video_name)
        all_video_lengths.append(len(audio_feat))
        
        print(f"  {video_name}: {len(audio_feat)} frames")
    
    print(f"\n  Video length statistics:")
    print(f"    Min: {min(all_video_lengths)} frames")
    print(f"    Max: {max(all_video_lengths)} frames")
    print(f"    Mean: {np.mean(all_video_lengths):.1f} frames")
    print(f"    Median: {np.median(all_video_lengths):.1f} frames")
    
    # Normalize features (fit on all data, then split)
    print("\n3. Normalizing features...")
    
    # Concatenate all videos for fitting scalers
    all_audio_concat = np.concatenate(all_audio_features, axis=0)
    all_visual_concat = np.concatenate(all_visual_features, axis=0)
    all_temporal_concat = np.concatenate(all_temporal_features, axis=0)
    
    audio_scaler = StandardScaler()
    visual_scaler = StandardScaler()
    temporal_scaler = StandardScaler()
    
    audio_scaler.fit(all_audio_concat)
    visual_scaler.fit(all_visual_concat)
    temporal_scaler.fit(all_temporal_concat)
    
    # Transform each video separately
    all_audio_features_scaled = [audio_scaler.transform(feat) for feat in all_audio_features]
    all_visual_features_scaled = [visual_scaler.transform(feat) for feat in all_visual_features]
    all_temporal_features_scaled = [temporal_scaler.transform(feat) for feat in all_temporal_features]
    
    # Save scalers
    with open(output_dir / 'audio_scaler_cut_selection_enhanced_fullvideo.pkl', 'wb') as f:
        pickle.dump(audio_scaler, f)
    with open(output_dir / 'visual_scaler_cut_selection_enhanced_fullvideo.pkl', 'wb') as f:
        pickle.dump(visual_scaler, f)
    with open(output_dir / 'temporal_scaler_cut_selection_enhanced_fullvideo.pkl', 'wb') as f:
        pickle.dump(temporal_scaler, f)
    
    print("  Scalers saved")
    
    # Split train/val by video (prevent data leakage)
    print(f"\n4. Splitting train/val by video (test_size={TEST_SIZE})...")
    
    video_names_list = list(video_data.keys())
    train_videos, val_videos = train_test_split(
        video_names_list, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )
    
    print(f"  Train videos: {len(train_videos)}")
    print(f"  Val videos: {len(val_videos)}")
    
    # Separate train/val data
    train_audio = []
    train_visual = []
    train_temporal = []
    train_active = []
    train_video_names = []
    
    val_audio = []
    val_visual = []
    val_temporal = []
    val_active = []
    val_video_names = []
    
    for i, video_name in enumerate(all_video_names):
        if video_name in train_videos:
            train_audio.append(all_audio_features_scaled[i])
            train_visual.append(all_visual_features_scaled[i])
            train_temporal.append(all_temporal_features_scaled[i])
            train_active.append(all_active_labels[i])
            train_video_names.append(video_name)
        else:
            val_audio.append(all_audio_features_scaled[i])
            val_visual.append(all_visual_features_scaled[i])
            val_temporal.append(all_temporal_features_scaled[i])
            val_active.append(all_active_labels[i])
            val_video_names.append(video_name)
    
    # Calculate statistics
    train_active_ratio = np.mean([np.mean(labels) for labels in train_active])
    val_active_ratio = np.mean([np.mean(labels) for labels in val_active])
    
    print(f"\nTrain active ratio: {train_active_ratio*100:.2f}%")
    print(f"Val active ratio: {val_active_ratio*100:.2f}%")
    
    # Save as numpy arrays (list of arrays with different lengths)
    print("\n5. Saving...")
    np.savez(
        output_dir / 'train_fullvideo_cut_selection_enhanced.npz',
        audio=np.array(train_audio, dtype=object),
        visual=np.array(train_visual, dtype=object),
        temporal=np.array(train_temporal, dtype=object),
        active=np.array(train_active, dtype=object),
        video_names=np.array(train_video_names, dtype=object)
    )
    
    np.savez(
        output_dir / 'val_fullvideo_cut_selection_enhanced.npz',
        audio=np.array(val_audio, dtype=object),
        visual=np.array(val_visual, dtype=object),
        temporal=np.array(val_temporal, dtype=object),
        active=np.array(val_active, dtype=object),
        video_names=np.array(val_video_names, dtype=object)
    )
    
    print(f"\n✅ Full video data saved to {output_dir}")
    print(f"  train_fullvideo_cut_selection_enhanced.npz: {len(train_audio)} videos")
    print(f"  val_fullvideo_cut_selection_enhanced.npz: {len(val_audio)} videos")
    
    print(f"\nNext steps:")
    print(f"  1. Update dataset class to handle variable-length videos")
    print(f"  2. Update training script for batch_size=1")
    print(f"  3. Apply 90s constraint per video (not per batch)")

if __name__ == '__main__':
    main()
