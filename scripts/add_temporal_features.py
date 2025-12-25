"""
Add temporal and contextual features to existing feature CSVs

This script enhances the feature set with:
1. Temporal statistics (moving averages, variance, change rates)
2. Cut timing information (time since last cut, cut density)
3. Scene similarity (CLIP embedding similarities)
4. Audio change detection (speech transitions, speaker changes)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def add_moving_statistics(df, columns, windows=[5, 10, 30, 60, 120]):
    """
    Add moving average and variance for specified columns
    
    Args:
        df: DataFrame with features
        columns: List of column names to compute statistics for
        windows: List of window sizes (5, 10, 30, 60, 120 frames)
    
    Returns:
        DataFrame with added moving statistics
    """
    print(f"  Adding moving statistics for {len(columns)} columns with windows {windows}...")
    
    for col in columns:
        if col not in df.columns:
            continue
            
        # Moving averages for all window sizes
        for window in windows:
            df[f'{col}_ma{window}'] = df[col].rolling(
                window=window, center=True, min_periods=1
            ).mean()
        
        # Moving variance for short (5), medium (30), and long (120) windows
        for window in [5, 30, 120]:
            df[f'{col}_std{window}'] = df[col].rolling(
                window=window, center=True, min_periods=1
            ).std().fillna(0)
        
        # Change rate (difference from previous frame)
        df[f'{col}_diff1'] = df[col].diff(1).fillna(0)
        df[f'{col}_diff2'] = df[col].diff(2).fillna(0)
        
        # Long-term change (difference from 30 frames ago)
        df[f'{col}_diff30'] = df[col].diff(30).fillna(0)
    
    return df


def add_cut_timing_features(df):
    """
    Add cut timing information
    
    Args:
        df: DataFrame with 'time' column
    
    Returns:
        DataFrame with added timing features
    """
    print("  Adding cut timing features...")
    
    # Time differences
    df['time_since_prev'] = df['time'].diff(1).fillna(0)
    df['time_to_next'] = -df['time'].diff(-1).fillna(0)
    
    # Cut duration (average of time_since_prev and time_to_next)
    df['cut_duration'] = (df['time_since_prev'] + df['time_to_next']) / 2
    
    # Position in video (normalized 0-1)
    df['position_in_video'] = (df['time'] - df['time'].min()) / (df['time'].max() - df['time'].min() + 1e-6)
    
    # Cut density (number of cuts in surrounding window)
    window_sec = 10.0  # 10 second window
    df['cut_density_10s'] = 0.0
    
    for idx in range(len(df)):
        current_time = df.loc[idx, 'time']
        # Count frames within ±10 seconds
        mask = (df['time'] >= current_time - window_sec) & (df['time'] <= current_time + window_sec)
        df.loc[idx, 'cut_density_10s'] = mask.sum()
    
    # Normalize cut density
    df['cut_density_10s'] = df['cut_density_10s'] / df['cut_density_10s'].max()
    
    return df


def add_clip_similarity_features(df):
    """
    Add CLIP embedding similarity features
    
    Args:
        df: DataFrame with clip_0 to clip_511 columns
    
    Returns:
        DataFrame with added similarity features
    """
    print("  Adding CLIP similarity features...")
    
    # Extract CLIP embeddings
    clip_cols = [f'clip_{i}' for i in range(512)]
    clip_cols_exist = [c for c in clip_cols if c in df.columns]
    
    if len(clip_cols_exist) < 512:
        print(f"    Warning: Only {len(clip_cols_exist)}/512 CLIP columns found")
        # Fill missing columns with zeros
        for col in clip_cols:
            if col not in df.columns:
                df[col] = 0.0
    
    clip_embeddings = df[clip_cols].values
    
    # Compute similarities
    df['clip_sim_prev'] = 0.0
    df['clip_sim_next'] = 0.0
    df['clip_sim_mean5'] = 0.0
    
    for idx in range(len(df)):
        current_emb = clip_embeddings[idx:idx+1]
        
        # Similarity with previous frame
        if idx > 0:
            prev_emb = clip_embeddings[idx-1:idx]
            df.loc[idx, 'clip_sim_prev'] = cosine_similarity(current_emb, prev_emb)[0, 0]
        
        # Similarity with next frame
        if idx < len(df) - 1:
            next_emb = clip_embeddings[idx+1:idx+2]
            df.loc[idx, 'clip_sim_next'] = cosine_similarity(current_emb, next_emb)[0, 0]
        
        # Mean similarity with surrounding 5 frames
        start = max(0, idx - 2)
        end = min(len(df), idx + 3)
        if end - start > 1:
            surrounding_embs = clip_embeddings[start:end]
            sims = cosine_similarity(current_emb, surrounding_embs)[0]
            # Exclude self-similarity
            sims = sims[sims < 0.9999]
            if len(sims) > 0:
                df.loc[idx, 'clip_sim_mean5'] = sims.mean()
    
    return df


def add_audio_change_features(df):
    """
    Add audio change detection features
    
    Args:
        df: DataFrame with audio features
    
    Returns:
        DataFrame with added audio change features
    """
    print("  Adding audio change features...")
    
    # Audio energy change score
    if 'audio_energy_rms' in df.columns:
        df['audio_change_score'] = df['audio_energy_rms'].diff(1).abs().fillna(0)
        
        # Smooth the change score
        df['audio_change_score'] = df['audio_change_score'].rolling(
            window=3, center=True, min_periods=1
        ).mean()
    
    # Silence to speech transition
    if 'audio_is_speaking' in df.columns:
        df['silence_to_speech'] = (
            (df['audio_is_speaking'].diff(1) > 0).astype(float)
        )
        df['speech_to_silence'] = (
            (df['audio_is_speaking'].diff(1) < 0).astype(float)
        )
    
    # Speaker change detection
    if 'speaker_id' in df.columns:
        df['speaker_change'] = (
            (df['speaker_id'].diff(1) != 0).astype(float)
        )
    
    # Pitch change
    if 'pitch_f0' in df.columns:
        df['pitch_change'] = df['pitch_f0'].diff(1).abs().fillna(0)
        df['pitch_change'] = df['pitch_change'].rolling(
            window=3, center=True, min_periods=1
        ).mean()
    
    return df


def add_cumulative_statistics(df):
    """
    Add cumulative statistics (if active labels exist)
    
    Args:
        df: DataFrame with optional 'active' column
    
    Returns:
        DataFrame with added cumulative statistics
    """
    print("  Adding cumulative statistics...")
    
    # Cumulative position (frame index / total frames)
    df['cumulative_position'] = np.arange(len(df)) / len(df)
    
    # If active labels exist, compute cumulative adoption rate
    if 'active' in df.columns:
        df['cumulative_adoption_rate'] = df['active'].expanding().mean()
    
    return df


def add_visual_change_features(df):
    """
    Add visual change detection features
    
    Args:
        df: DataFrame with visual features
    
    Returns:
        DataFrame with added visual change features
    """
    print("  Adding visual change features...")
    
    # Visual motion change
    if 'visual_motion' in df.columns:
        df['visual_motion_change'] = df['visual_motion'].diff(1).abs().fillna(0)
    
    # Face count change
    if 'face_count' in df.columns:
        df['face_count_change'] = df['face_count'].diff(1).abs().fillna(0)
    
    # Saliency movement
    if 'saliency_x' in df.columns and 'saliency_y' in df.columns:
        df['saliency_movement'] = np.sqrt(
            df['saliency_x'].diff(1)**2 + df['saliency_y'].diff(1)**2
        ).fillna(0)
    
    return df


def process_single_video(feature_file, output_dir):
    """
    Process a single video's feature file
    
    Args:
        feature_file: Path to input feature CSV
        output_dir: Path to output directory
    
    Returns:
        Success status and statistics
    """
    try:
        # Load features
        df = pd.read_csv(feature_file, low_memory=False)
        original_cols = len(df.columns)
        original_rows = len(df)
        
        # Ensure time column exists
        if 'time' not in df.columns:
            print(f"    Warning: No 'time' column in {feature_file.name}")
            return False, None
        
        # Sort by time
        df = df.sort_values('time').reset_index(drop=True)
        
        # 1. Add moving statistics for key columns (5, 10, 30, 60, 120 frame windows)
        key_columns = [
            'audio_energy_rms', 'visual_motion', 'face_count',
            'pitch_f0', 'spectral_centroid', 'scene_change'
        ]
        df = add_moving_statistics(df, key_columns, windows=[5, 10, 30, 60, 120])
        
        # 2. Add cut timing features
        df = add_cut_timing_features(df)
        
        # 3. Add CLIP similarity features
        df = add_clip_similarity_features(df)
        
        # 4. Add audio change features
        df = add_audio_change_features(df)
        
        # 5. Add visual change features
        df = add_visual_change_features(df)
        
        # 6. Add cumulative statistics
        df = add_cumulative_statistics(df)
        
        # Fill any remaining NaN values
        df = df.fillna(0)
        
        # Save enhanced features
        output_file = output_dir / feature_file.name
        df.to_csv(output_file, index=False)
        
        new_cols = len(df.columns)
        added_cols = new_cols - original_cols
        
        stats = {
            'video': feature_file.stem,
            'original_cols': original_cols,
            'new_cols': new_cols,
            'added_cols': added_cols,
            'rows': original_rows
        }
        
        return True, stats
        
    except Exception as e:
        print(f"    Error processing {feature_file.name}: {e}")
        return False, None


def main():
    print("=" * 60)
    print("Adding Temporal and Contextual Features")
    print("=" * 60)
    
    # Paths
    input_dir = Path('data/processed/source_features')
    output_dir = Path('data/processed/source_features_enhanced')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Find all feature files
    feature_files = list(input_dir.glob('*_features.csv'))
    
    if not feature_files:
        print(f"\n❌ No feature files found in {input_dir}")
        return
    
    print(f"\nFound {len(feature_files)} feature files")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Process each video
    print("\nProcessing videos...")
    all_stats = []
    success_count = 0
    
    for feature_file in tqdm(feature_files, desc="Processing"):
        print(f"\n{feature_file.name}")
        success, stats = process_single_video(feature_file, output_dir)
        
        if success:
            success_count += 1
            all_stats.append(stats)
            print(f"  ✓ {stats['original_cols']} → {stats['new_cols']} columns (+{stats['added_cols']})")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Successfully processed: {success_count}/{len(feature_files)} videos")
    
    if all_stats:
        avg_original = np.mean([s['original_cols'] for s in all_stats])
        avg_new = np.mean([s['new_cols'] for s in all_stats])
        avg_added = np.mean([s['added_cols'] for s in all_stats])
        
        print(f"\nAverage columns:")
        print(f"  Original: {avg_original:.0f}")
        print(f"  Enhanced: {avg_new:.0f}")
        print(f"  Added: {avg_added:.0f}")
        
        print(f"\nNew feature categories:")
        print(f"  • Moving statistics (MA5, MA10, MA30, MA60, MA120)")
        print(f"  • Moving variance (STD5, STD30, STD120)")
        print(f"  • Change rates (DIFF1, DIFF2, DIFF30)")
        print(f"  • Cut timing (time_since_prev, time_to_next, cut_duration, position, density)")
        print(f"  • CLIP similarity (prev, next, mean5)")
        print(f"  • Audio changes (change_score, silence transitions, speaker_change, pitch_change)")
        print(f"  • Visual changes (motion_change, face_count_change, saliency_movement)")
        print(f"  • Cumulative stats (position, adoption_rate)")
        
        print(f"\n✅ Enhanced features saved to: {output_dir}")
        print(f"\nNext steps:")
        print(f"  1. Update create_cut_selection_data.py to use enhanced features")
        print(f"  2. Regenerate training data with new features")
        print(f"  3. Retrain model with expanded feature set")


if __name__ == '__main__':
    main()
