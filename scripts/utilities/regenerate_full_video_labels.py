"""
全動画の完全版ラベルを再生成

元動画全体の長さに対してラベルを生成し、カット判断を学習できるようにする
"""
import sys
sys.path.insert(0, '.')

from src.data_preparation.full_video_xml_parser import FullVideoXMLParser
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    # ディレクトリ設定
    xml_dir = Path('data/raw/editxml')
    output_dir = Path('data/processed/output_labels_full')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 統計情報を保存
    stats_file = output_dir / 'generation_stats.json'
    
    # XMLファイルを取得
    xml_files = sorted(xml_dir.glob('*.xml'))
    logger.info(f"Found {len(xml_files)} XML files")
    
    # パーサーを初期化
    parser = FullVideoXMLParser(max_tracks=20, fps=10.0)
    
    # 統計情報
    stats = {
        'total_videos': len(xml_files),
        'successful': 0,
        'failed': 0,
        'videos': []
    }
    
    # 各XMLファイルを処理
    for idx, xml_file in enumerate(xml_files, 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing {idx}/{len(xml_files)}: {xml_file.name}")
        logger.info(f"{'='*70}")
        
        try:
            # XMLをパース
            df, metadata = parser.parse_premiere_xml_full(str(xml_file))
            
            # CSVとして保存
            output_file = output_dir / f"{xml_file.stem}_tracks.csv"
            df.to_csv(output_file, index=False)
            
            logger.info(f"✅ Saved: {output_file}")
            logger.info(f"   Shape: {df.shape}")
            
            # 統計情報を記録
            video_stats = {
                'name': xml_file.stem,
                'source_path': metadata['source_video_path'],
                'duration': metadata['source_duration'],
                'timesteps': metadata['num_timesteps'],
                'used_frames': metadata['used_frames'],
                'cut_frames': metadata['cut_frames'],
                'cut_ratio': metadata['cut_ratio']
            }
            stats['videos'].append(video_stats)
            stats['successful'] += 1
            
        except Exception as e:
            logger.error(f"❌ Failed to process {xml_file.name}: {e}")
            stats['failed'] += 1
            stats['videos'].append({
                'name': xml_file.stem,
                'error': str(e)
            })
    
    # 統計情報を保存
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n{'='*70}")
    logger.info("Generation Complete!")
    logger.info(f"{'='*70}")
    logger.info(f"Total: {stats['total_videos']}")
    logger.info(f"Successful: {stats['successful']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Stats saved to: {stats_file}")
    
    # 全体統計を計算
    if stats['successful'] > 0:
        total_duration = sum(v['duration'] for v in stats['videos'] if 'duration' in v)
        total_timesteps = sum(v['timesteps'] for v in stats['videos'] if 'timesteps' in v)
        total_used = sum(v['used_frames'] for v in stats['videos'] if 'used_frames' in v)
        total_cut = sum(v['cut_frames'] for v in stats['videos'] if 'cut_frames' in v)
        avg_cut_ratio = total_cut / (total_used + total_cut) if (total_used + total_cut) > 0 else 0
        
        logger.info(f"\nOverall Statistics:")
        logger.info(f"  Total duration: {total_duration/60:.1f} minutes")
        logger.info(f"  Total timesteps: {total_timesteps:,}")
        logger.info(f"  Used frames: {total_used:,} ({100*(1-avg_cut_ratio):.1f}%)")
        logger.info(f"  Cut frames: {total_cut:,} ({100*avg_cut_ratio:.1f}%)")


if __name__ == '__main__':
    main()
