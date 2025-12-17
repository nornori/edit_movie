"""
Batch process Premiere Pro XML files to extract track sequences

This script processes all XML files in the editxml directory and generates:
- NPZ files with track sequences (for model training)
- CSV files with detailed track information (for inspection)
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.data_preparation.premiere_xml_parser import PremiereXMLParser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def batch_process_xml_files(
    input_dir: str = 'editxml',
    output_dir: str = 'preprocessed_data/xml_tracks',
    max_tracks: int = 20,
    fps: float = 10.0,
    output_format: str = 'both'
):
    """
    Batch process all XML files in input directory
    
    Args:
        input_dir: Directory containing XML files
        output_dir: Directory to save output files
        max_tracks: Maximum number of tracks to extract
        fps: Sampling rate (frames per second)
        output_format: Output format ('npz', 'csv', or 'both')
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all XML files
    xml_files = list(input_path.glob('*.xml'))
    
    if not xml_files:
        logger.error(f"No XML files found in {input_dir}")
        return
    
    logger.info(f"Found {len(xml_files)} XML files in {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Settings: max_tracks={max_tracks}, fps={fps}, format={output_format}")
    logger.info("="*70)
    
    # Statistics
    successful = 0
    failed = 0
    failed_files = []
    
    # Process each XML file
    for xml_file in tqdm(xml_files, desc="Processing XML files"):
        try:
            # Get video ID from filename
            video_id = xml_file.stem
            
            # Skip if already processed (optional - comment out to reprocess)
            npz_output = output_path / f'{video_id}_tracks.npz'
            if npz_output.exists() and output_format in ['npz', 'both']:
                logger.info(f"Skipping {video_id} (already processed)")
                successful += 1
                continue
            
            # Create parser
            parser = PremiereXMLParser(max_tracks=max_tracks, fps=fps)
            
            # Parse XML
            clips, total_duration = parser.parse_premiere_xml(str(xml_file))
            
            # Convert to sequence
            sequence = parser.clips_to_track_sequence(clips, total_duration)
            
            # Save outputs
            if output_format in ['npz', 'both']:
                npz_path = output_path / f'{video_id}_tracks.npz'
                parser.save_sequence(sequence, str(npz_path), video_id)
            
            if output_format in ['csv', 'both']:
                csv_path = output_path / f'{video_id}_tracks.csv'
                parser.save_csv(sequence, str(csv_path), video_id, clips)
            
            successful += 1
            logger.info(f"✓ Processed: {video_id} ({total_duration:.2f}s, {len(clips)} clips)")
            
        except Exception as e:
            failed += 1
            failed_files.append((xml_file.name, str(e)))
            logger.error(f"✗ Failed: {xml_file.name} - {e}")
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("Batch Processing Summary")
    logger.info("="*70)
    logger.info(f"Total files: {len(xml_files)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    
    if failed_files:
        logger.info("\nFailed files:")
        for filename, error in failed_files:
            logger.info(f"  - {filename}: {error}")
    
    logger.info("="*70)
    logger.info(f"\nOutput saved to: {output_dir}")
    
    return successful, failed


def main():
    parser = argparse.ArgumentParser(description='Batch process Premiere Pro XML files')
    parser.add_argument('--input_dir', type=str, default='editxml',
                       help='Directory containing XML files')
    parser.add_argument('--output_dir', type=str, default='preprocessed_data/xml_tracks',
                       help='Output directory')
    parser.add_argument('--max_tracks', type=int, default=20,
                       help='Maximum number of tracks')
    parser.add_argument('--fps', type=float, default=10.0,
                       help='Sampling rate (frames per second)')
    parser.add_argument('--format', type=str, choices=['npz', 'csv', 'both'], default='both',
                       help='Output format')
    parser.add_argument('--reprocess', action='store_true',
                       help='Reprocess files even if they already exist')
    
    args = parser.parse_args()
    
    logger.info("\n" + "="*70)
    logger.info("Batch XML Processing")
    logger.info("="*70 + "\n")
    
    batch_process_xml_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_tracks=args.max_tracks,
        fps=args.fps,
        output_format=args.format
    )


if __name__ == "__main__":
    main()
