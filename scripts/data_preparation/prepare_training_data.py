"""
トレーニングデータ準備スクリプト

XMLファイルから動画パスを抽出し、特徴量とラベルを生成します。
"""
import os
import glob
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote
from tqdm import tqdm
import logging
import sys

# Pythonパスを設定（プロジェクトルートを追加）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data_preparation.extract_video_features_parallel import extract_features_worker
from src.data_preparation.premiere_xml_parser import PremiereXMLParser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_video_path_from_xml(xml_path: str) -> str:
    """XMLファイルから動画ファイルのパスを抽出"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # pathurlを探す
        pathurl_elem = root.find('.//pathurl')
        if pathurl_elem is not None and pathurl_elem.text:
            # file://localhost/ を削除してデコード
            path = pathurl_elem.text.replace('file://localhost/', '')
            path = unquote(path)
            
            # Windowsパスに変換
            path = path.replace('%3a', ':').replace('%3A', ':')
            path = path.replace('/', '\\')
            
            return path
    except Exception as e:
        logger.warning(f"XMLパース失敗 {xml_path}: {e}")
    
    return None


def main():
    logger.info("="*70)
    logger.info("トレーニングデータ準備")
    logger.info("="*70)
    
    # ディレクトリ設定
    xml_dir = "data/raw/editxml"
    features_output_dir = "data/processed/input_features"
    labels_output_dir = "data/processed/output_labels"
    
    os.makedirs(features_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)
    
    # XMLファイルを取得
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
    logger.info(f"\nXMLファイル数: {len(xml_files)}")
    
    # 動画パスを抽出
    video_to_xml = {}
    for xml_file in xml_files:
        video_path = extract_video_path_from_xml(xml_file)
        if video_path and os.path.exists(video_path):
            video_to_xml[video_path] = xml_file
    
    logger.info(f"有効な動画ファイル数: {len(video_to_xml)}")
    
    if not video_to_xml:
        logger.error("動画ファイルが見つかりませんでした")
        return
    
    # ステップ1: 特徴量抽出
    logger.info("\n" + "="*70)
    logger.info("ステップ1: 動画から特徴量を抽出")
    logger.info("="*70)
    
    features_success = 0
    features_failed = 0
    
    for video_path in tqdm(list(video_to_xml.keys()), desc="特徴量抽出"):
        video_stem = Path(video_path).stem
        output_path = os.path.join(features_output_dir, f"{video_stem}_features.csv")
        
        # 既に存在する場合はスキップ
        if os.path.exists(output_path):
            logger.info(f"スキップ（既存）: {video_stem}")
            features_success += 1
            continue
        
        try:
            result = extract_features_worker(video_path, features_output_dir)
            
            if result['status'] == 'Success':
                features_success += 1
                logger.info(f"✅ {video_stem}: {result['timesteps']}タイムステップ")
            else:
                features_failed += 1
                logger.error(f"❌ {video_stem}: {result.get('message', '不明なエラー')}")
        
        except Exception as e:
            features_failed += 1
            logger.error(f"❌ {video_stem}: {e}")
    
    logger.info(f"\n特徴量抽出完了: 成功={features_success}, 失敗={features_failed}")
    
    # ステップ2: ラベル抽出
    logger.info("\n" + "="*70)
    logger.info("ステップ2: XMLからラベルを抽出")
    logger.info("="*70)
    
    labels_success = 0
    labels_failed = 0
    
    for video_path, xml_file in tqdm(video_to_xml.items(), desc="ラベル抽出"):
        video_stem = Path(video_path).stem
        csv_output = os.path.join(labels_output_dir, f"{video_stem}_tracks.csv")
        
        # 既に存在する場合はスキップ
        if os.path.exists(csv_output):
            logger.info(f"スキップ（既存）: {video_stem}")
            labels_success += 1
            continue
        
        try:
            # XMLをパース
            parser = PremiereXMLParser(max_tracks=20, fps=10.0)
            clips, total_duration = parser.parse_premiere_xml(xml_file)
            
            # シーケンスに変換
            sequence = parser.clips_to_track_sequence(clips, total_duration)
            
            # CSVとして保存
            parser.save_csv(sequence, csv_output, video_stem, clips)
            
            labels_success += 1
            logger.info(f"✅ {video_stem}: {total_duration:.2f}秒, {len(clips)}クリップ")
        
        except Exception as e:
            labels_failed += 1
            logger.error(f"❌ {video_stem}: {e}")
    
    logger.info(f"\nラベル抽出完了: 成功={labels_success}, 失敗={labels_failed}")
    
    # サマリー
    logger.info("\n" + "="*70)
    logger.info("トレーニングデータ準備完了")
    logger.info("="*70)
    logger.info(f"処理対象動画: {len(video_to_xml)}個")
    logger.info(f"特徴量: 成功={features_success}, 失敗={features_failed}")
    logger.info(f"ラベル: 成功={labels_success}, 失敗={labels_failed}")
    logger.info(f"\n出力先:")
    logger.info(f"  特徴量: {features_output_dir}")
    logger.info(f"  ラベル: {labels_output_dir}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
