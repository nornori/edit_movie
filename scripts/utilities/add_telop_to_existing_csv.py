"""
Add telop information to existing feature CSV files
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data_preparation.telop_extractor import TelopExtractor
from src.data_preparation.text_embedding import SimpleTextEmbedder
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_telop_to_csv(csv_path: str, xml_dir: str = 'editxml', output_path: str = None):
    """
    Add telop information to an existing feature CSV
    
    Args:
        csv_path: Path to existing feature CSV
        xml_dir: Directory containing XML files
        output_path: Output path (if None, overwrites input)
    """
    # Read existing CSV
    df = pd.read_csv(csv_path)
    
    # Get video name
    video_name = Path(csv_path).stem.replace('_features', '')
    xml_path = Path(xml_dir) / f"{video_name}.xml"
    
    # Check if already has embeddings
    has_speech_emb = any(col.startswith('speech_emb_') for col in df.columns)
    has_telop_emb = any(col.startswith('telop_emb_') for col in df.columns)
    
    if has_speech_emb and has_telop_emb:
        logger.info(f"  ⚠️ Already has text embeddings, skipping")
        return False
    
    # Extract telop information
    if xml_path.exists():
        try:
            extractor = TelopExtractor(fps=10.0)
            total_duration = df['time'].max() + 0.1
            df_telop = extractor.extract_and_convert(str(xml_path), total_duration)
            
            # Align with existing data
            min_len = min(len(df), len(df_telop))
            df = df.iloc[:min_len].copy()
            df_telop = df_telop.iloc[:min_len].copy()
            
            # Add telop columns
            df['telop_active'] = df_telop['telop_active'].values
            df['telop_text'] = df_telop['telop_text'].values
            
            # Add telop embeddings
            embedder = SimpleTextEmbedder()
            telop_embeddings = embedder.embed_series(df['telop_text'])
            for i in range(embedder.embedding_dim):
                df[f'telop_emb_{i}'] = telop_embeddings[:, i]
            
            # Add speech embeddings if text_word exists
            if 'text_word' in df.columns:
                speech_embeddings = embedder.embed_series(df['text_word'])
                for i in range(embedder.embedding_dim):
                    df[f'speech_emb_{i}'] = speech_embeddings[:, i]
            
            # Save
            if output_path is None:
                output_path = csv_path
            df.to_csv(output_path, index=False, float_format='%.6f')
            
            telop_count = df['telop_active'].sum()
            logger.info(f"  ✅ Added telop info: {telop_count} active frames")
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Failed to add telop: {e}")
            return False
    else:
        # No XML file, add empty telop columns
        df['telop_active'] = 0
        df['telop_text'] = np.nan
        
        # Add zero embeddings
        embedder = SimpleTextEmbedder()
        for i in range(embedder.embedding_dim):
            df[f'telop_emb_{i}'] = 0.0
        
        # Add speech embeddings if text_word exists
        if 'text_word' in df.columns:
            speech_embeddings = embedder.embed_series(df['text_word'])
            for i in range(embedder.embedding_dim):
                df[f'speech_emb_{i}'] = speech_embeddings[:, i]
        
        # Save
        if output_path is None:
            output_path = csv_path
        df.to_csv(output_path, index=False, float_format='%.6f')
        
        logger.info(f"  ⚠️ No XML, added empty telop columns")
        return True


def batch_add_telop(features_dir: str = 'input_features', xml_dir: str = 'editxml'):
    """
    Add telop information to all feature CSV files
    
    Args:
        features_dir: Directory containing feature CSV files
        xml_dir: Directory containing XML files
    """
    # Find all feature CSV files (not visual_features)
    csv_files = list(Path(features_dir).glob('*_features.csv'))
    csv_files = [f for f in csv_files if 'visual' not in f.name]
    
    logger.info(f"Found {len(csv_files)} feature CSV files")
    logger.info(f"XML directory: {xml_dir}")
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for csv_path in tqdm(csv_files, desc="Adding telop info"):
        logger.info(f"\nProcessing: {csv_path.name}")
        
        try:
            result = add_telop_to_csv(str(csv_path), xml_dir)
            if result:
                success_count += 1
            else:
                skip_count += 1
        except Exception as e:
            logger.error(f"  ❌ Error: {e}")
            error_count += 1
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Batch processing complete!")
    logger.info(f"  Success: {success_count}")
    logger.info(f"  Skipped: {skip_count}")
    logger.info(f"  Errors: {error_count}")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Add telop information to existing feature CSVs")
    parser.add_argument('--features_dir', type=str, default='input_features',
                       help='Directory containing feature CSV files')
    parser.add_argument('--xml_dir', type=str, default='editxml',
                       help='Directory containing XML files')
    parser.add_argument('--single', type=str, default=None,
                       help='Process single CSV file')
    
    args = parser.parse_args()
    
    if args.single:
        logger.info(f"Processing single file: {args.single}")
        add_telop_to_csv(args.single, args.xml_dir)
    else:
        logger.info("Processing all feature CSV files...")
        batch_add_telop(args.features_dir, args.xml_dir)
