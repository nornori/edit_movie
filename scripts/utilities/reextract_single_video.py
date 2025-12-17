"""
Re-extract features for a single video with telop information
"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data_preparation.extract_video_features_parallel import extract_features_worker
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test video
video_name = "bandicam 2025-03-03 22-34-57-492"
video_path = rf"D:\切り抜き\2025-3\2025-3-03\{video_name}.mp4"
output_dir = "input_features"

logger.info(f"Re-extracting features with telop for: {video_name}")
logger.info(f"Video path: {video_path}")
logger.info(f"Output dir: {output_dir}")

# Check if video exists
if not os.path.exists(video_path):
    logger.error(f"Video file not found: {video_path}")
    sys.exit(1)

# Check if XML exists
xml_path = f"editxml/{video_name}.xml"
if os.path.exists(xml_path):
    logger.info(f"✅ XML file found: {xml_path}")
else:
    logger.warning(f"⚠️ XML file not found: {xml_path}")

# Extract features
logger.info("\nStarting feature extraction...")
result = extract_features_worker(video_path, output_dir)

logger.info(f"\nResult: {result}")

if result['status'] == 'Success':
    # Check the output CSV
    output_path = os.path.join(output_dir, f"{video_name}_features.csv")
    
    df = pd.read_csv(output_path)
    
    logger.info(f"\n✅ Feature extraction successful!")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Columns ({len(df.columns)}): {df.columns.tolist()}")
    
    # Check for telop columns
    has_telop = 'telop_active' in df.columns and 'telop_text' in df.columns
    has_speech_emb = any(col.startswith('speech_emb_') for col in df.columns)
    has_telop_emb = any(col.startswith('telop_emb_') for col in df.columns)
    
    logger.info(f"\n  Telop columns: {'✅' if has_telop else '❌'}")
    logger.info(f"  Speech embeddings: {'✅' if has_speech_emb else '❌'}")
    logger.info(f"  Telop embeddings: {'✅' if has_telop_emb else '❌'}")
    
    if has_telop:
        telop_active_count = df['telop_active'].sum()
        logger.info(f"\n  Telop active frames: {telop_active_count} / {len(df)}")
        
        if telop_active_count > 0:
            logger.info(f"\n  Sample telop data:")
            sample = df[df['telop_active'] == 1][['time', 'telop_active', 'telop_text']].head(10)
            for _, row in sample.iterrows():
                logger.info(f"    {row['time']:.1f}s: {row['telop_text']}")
    
    # Count numeric features (excluding text columns)
    text_cols = ['text_word', 'telop_text', 'speaker_id']
    numeric_cols = [col for col in df.columns if col not in text_cols and col != 'time']
    logger.info(f"\n  Total numeric features: {len(numeric_cols)}")
    
else:
    logger.error(f"❌ Feature extraction failed: {result.get('message', 'Unknown error')}")
