"""
Combine train and val sequences into a single file for K-Fold CV (Enhanced version)

This script merges train_sequences_cut_selection_enhanced.npz and 
val_sequences_cut_selection_enhanced.npz into combined_sequences_cut_selection_enhanced.npz
"""
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("Combining Enhanced Sequences for K-Fold CV")
    logger.info("=" * 60)
    
    data_dir = Path('preprocessed_data')
    
    # Load train and val data
    logger.info("\nLoading train sequences...")
    train_data = np.load(data_dir / 'train_sequences_cut_selection_enhanced.npz', allow_pickle=True)
    
    logger.info("Loading val sequences...")
    val_data = np.load(data_dir / 'val_sequences_cut_selection_enhanced.npz', allow_pickle=True)
    
    # Combine
    logger.info("\nCombining sequences...")
    combined_audio = np.concatenate([train_data['audio'], val_data['audio']], axis=0)
    combined_visual = np.concatenate([train_data['visual'], val_data['visual']], axis=0)
    combined_temporal = np.concatenate([train_data['temporal'], val_data['temporal']], axis=0)
    combined_active = np.concatenate([train_data['active'], val_data['active']], axis=0)
    combined_video_names = np.concatenate([train_data['video_names'], val_data['video_names']])
    
    logger.info(f"  Combined audio: {combined_audio.shape}")
    logger.info(f"  Combined visual: {combined_visual.shape}")
    logger.info(f"  Combined temporal: {combined_temporal.shape}")
    logger.info(f"  Combined active: {combined_active.shape}")
    logger.info(f"  Combined video names: {len(combined_video_names)}")
    logger.info(f"  Unique videos: {len(set(combined_video_names))}")
    
    # Calculate statistics
    total_samples = combined_active.size
    active_count = np.sum(combined_active == 1)
    inactive_count = np.sum(combined_active == 0)
    
    logger.info(f"\nClass distribution:")
    logger.info(f"  Active: {active_count:,} ({active_count/total_samples*100:.2f}%)")
    logger.info(f"  Inactive: {inactive_count:,} ({inactive_count/total_samples*100:.2f}%)")
    
    # Save combined data
    output_path = data_dir / 'combined_sequences_cut_selection_enhanced.npz'
    logger.info(f"\nSaving combined data to {output_path}...")
    
    np.savez(
        output_path,
        audio=combined_audio,
        visual=combined_visual,
        temporal=combined_temporal,
        active=combined_active,
        video_names=combined_video_names
    )
    
    logger.info(f"\n✅ Combined sequences saved!")
    logger.info(f"   Total sequences: {len(combined_audio)}")
    logger.info(f"   Ready for K-Fold Cross Validation")


if __name__ == '__main__':
    main()
