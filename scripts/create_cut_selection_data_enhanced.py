"""
Create training data for cut selection model with enhanced features

Uses temporal and contextual features added by add_temporal_features.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# Settings
SEQUENCE_LENGTH = 1000
OVERLAP = 500
TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_features_and_labels(source_features_dir, active_labels_dir):
    """Load enhanced source features and active labels"""
    source_features_dir = Path(source_features_dir)
    active_labels_dir = Path(active_labels_dir)
    
    # Find matching files
    feature_files = list(source_features_dir.glob('*_features.csv'))
    
    all_data = []
    
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
        
        df_merged['video_name'] = video_name
        all_data.append(df_merged)
        
        print(f"  {video_name}: {len(df_merged)} frames, active={df_merged['active'].mean()*100:.1f}%")
    
    if not all_data:
        raise ValueError("No data loaded!")
    
    df_all = pd.concat(all_data, ignore_index=True)
    
    return df_all

def extract_features(df):
    """Extract audio, visual, and temporal features"""
    
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
    
    print(f"\n  Feature dimensions:")
    print(f"    Audio: {len(audio_cols_exist)}/{len(audio_cols)} columns")
    print(f"    Visual: {len(visual_cols_exist)}/{len(visual_cols)} columns")
    print(f"    Temporal: {len(temporal_cols_exist)}/{len(temporal_cols)} columns")
    print(f"    Total: {len(audio_cols_exist) + len(visual_cols_exist) + len(temporal_cols_exist)} columns")
    
    return audio_features, visual_features, temporal_features, active_labels

def create_sequences(audio_features, visual_features, temporal_features, active_labels, video_names, seq_len, overlap):
    """Create sequences with sliding window"""
    sequences_audio = []
    sequences_visual = []
    sequences_temporal = []
    sequences_active = []
    sequences_video_names = []
    
    # Group by video
    unique_videos = np.unique(video_names)
    
    for video_name in unique_videos:
        mask = video_names == video_name
        video_audio = audio_features[mask]
        video_visual = visual_features[mask]
        video_temporal = temporal_features[mask]
        video_active = active_labels[mask]
        
        # Create sequences with sliding window
        step = seq_len - overlap
        for start in range(0, len(video_audio) - seq_len + 1, step):
            end = start + seq_len
            
            sequences_audio.append(video_audio[start:end])
            sequences_visual.append(video_visual[start:end])
            sequences_temporal.append(video_temporal[start:end])
            sequences_active.append(video_active[start:end])
            sequences_video_names.append(video_name)
    
    sequences_audio = np.array(sequences_audio)
    sequences_visual = np.array(sequences_visual)
    sequences_temporal = np.array(sequences_temporal)
    sequences_active = np.array(sequences_active)
    
    return sequences_audio, sequences_visual, sequences_temporal, sequences_active, sequences_video_names

def main():
    print("=" * 60)
    print("Creating Cut Selection Training Data (Enhanced Features)")
    print("=" * 60)
    
    source_features_dir = 'data/processed/source_features_enhanced'
    active_labels_dir = 'data/processed/active_labels'
    output_dir = Path('preprocessed_data')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    print("\n1. Loading enhanced features and labels...")
    df_all = load_features_and_labels(source_features_dir, active_labels_dir)
    
    print(f"\nTotal data: {len(df_all)} frames from {df_all['video_name'].nunique()} videos")
    print(f"Active ratio: {df_all['active'].mean()*100:.2f}%")
    
    # Extract features
    print("\n2. Extracting features...")
    audio_features, visual_features, temporal_features, active_labels = extract_features(df_all)
    video_names = df_all['video_name'].values
    
    print(f"\nFeature shapes:")
    print(f"  Audio: {audio_features.shape}")
    print(f"  Visual: {visual_features.shape}")
    print(f"  Temporal: {temporal_features.shape}")
    print(f"  Active labels: {active_labels.shape}")
    
    # Normalize features
    print("\n3. Normalizing features...")
    audio_scaler = StandardScaler()
    visual_scaler = StandardScaler()
    temporal_scaler = StandardScaler()
    
    audio_features = audio_scaler.fit_transform(audio_features)
    visual_features = visual_scaler.fit_transform(visual_features)
    temporal_features = temporal_scaler.fit_transform(temporal_features)
    
    # Save scalers
    with open(output_dir / 'audio_scaler_cut_selection_enhanced.pkl', 'wb') as f:
        pickle.dump(audio_scaler, f)
    with open(output_dir / 'visual_scaler_cut_selection_enhanced.pkl', 'wb') as f:
        pickle.dump(visual_scaler, f)
    with open(output_dir / 'temporal_scaler_cut_selection_enhanced.pkl', 'wb') as f:
        pickle.dump(temporal_scaler, f)
    
    print("  Scalers saved")
    
    # Create sequences
    print(f"\n4. Creating sequences (length={SEQUENCE_LENGTH}, overlap={OVERLAP})...")
    sequences_audio, sequences_visual, sequences_temporal, sequences_active, sequences_video_names = create_sequences(
        audio_features, visual_features, temporal_features, active_labels, video_names,
        SEQUENCE_LENGTH, OVERLAP
    )
    
    print(f"  Total sequences: {len(sequences_audio)}")
    
    # Split train/val by video (prevent data leakage)
    print(f"\n5. Splitting train/val by video (test_size={TEST_SIZE})...")
    
    # Get unique videos
    unique_videos = np.unique(sequences_video_names)
    print(f"  Total videos: {len(unique_videos)}")
    
    # Split videos into train/val
    train_videos, val_videos = train_test_split(
        unique_videos, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )
    
    print(f"  Train videos: {len(train_videos)}")
    print(f"  Val videos: {len(val_videos)}")
    
    # Assign sequences to train/val based on video
    train_mask = np.array([v in train_videos for v in sequences_video_names])
    val_mask = np.array([v in val_videos for v in sequences_video_names])
    
    train_audio = sequences_audio[train_mask]
    train_visual = sequences_visual[train_mask]
    train_temporal = sequences_temporal[train_mask]
    train_active = sequences_active[train_mask]
    train_video_names = [sequences_video_names[i] for i in np.where(train_mask)[0]]
    
    val_audio = sequences_audio[val_mask]
    val_visual = sequences_visual[val_mask]
    val_temporal = sequences_temporal[val_mask]
    val_active = sequences_active[val_mask]
    val_video_names = [sequences_video_names[i] for i in np.where(val_mask)[0]]
    
    print(f"  Train sequences: {len(train_audio)} from {len(set(train_video_names))} videos")
    print(f"  Val sequences: {len(val_audio)} from {len(set(val_video_names))} videos")
    
    # Calculate statistics
    train_active_ratio = train_active.mean()
    val_active_ratio = val_active.mean()
    
    print(f"\nTrain active ratio: {train_active_ratio*100:.2f}%")
    print(f"Val active ratio: {val_active_ratio*100:.2f}%")
    
    # Save
    print("\n6. Saving...")
    np.savez(
        output_dir / 'train_sequences_cut_selection_enhanced.npz',
        audio=train_audio,
        visual=train_visual,
        temporal=train_temporal,
        active=train_active,
        video_names=train_video_names
    )
    
    np.savez(
        output_dir / 'val_sequences_cut_selection_enhanced.npz',
        audio=val_audio,
        visual=val_visual,
        temporal=val_temporal,
        active=val_active,
        video_names=val_video_names
    )
    
    print(f"\n✅ Enhanced data saved to {output_dir}")
    print(f"  train_sequences_cut_selection_enhanced.npz: {len(train_audio)} sequences")
    print(f"  val_sequences_cut_selection_enhanced.npz: {len(val_audio)} sequences")
    
    print(f"\nNext steps:")
    print(f"  1. Update training script to load enhanced data")
    print(f"  2. Update model to handle 3 modalities (audio, visual, temporal)")
    print(f"  3. Retrain with enhanced features")

if __name__ == '__main__':
    main()
