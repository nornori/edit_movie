"""
トレーニング用の特徴量を全動画から抽出

XMLファイルから動画パスを読み取り、特徴量を抽出します。
"""
import os
import glob
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote
from tqdm import tqdm
import logging

from src.data_preparation.extract_video_features_parallel import extract_features_worker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_video_path_from_xml(xml_path: str) -> str:
    """
    XMLファイルから動画ファイルのパスを抽出
    
    Args:
        xml_path: XMLファイルのパス
    
    Returns:
        動画ファイルのパス（存在しない場合はNone）
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # pathurlを探す
        pathurl_elem = root.find('.//pathurl')
        if pathurl_elem is not None and pathurl_elem.text:
            # file://localhost/ を削除してデコード
            path = pathurl_elem.text.replace('file://localhost/', '')
            path = unquote(path)
            
            # Windowsパスに変換（%3a -> :）
            path = path.replace('%3a', ':').replace('%3A', ':')
            
            # スラッシュをバックスラッシュに変換
            path = path.replace('/', '\\')
            
            return path
    except Exception as e:
        logger.warning(f"XMLパース失敗 {xml_path}: {e}")
    
    return None


def main():
    # XMLディレクトリ
    xml_dir = "data/raw/editxml"
    
    # 出力ディレクトリ
    output_dir = "data/processed/input_features"
    os.makedirs(output_dir, exist_ok=True)
    
    # XMLファイルを取得（.xmlのみ）
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
    logger.info(f"XMLファイル数: {len(xml_files)}")
    
    # 動画パスを抽出
    video_paths = []
    xml_to_video = {}
    
    for xml_file in xml_files:
        video_path = extract_video_path_from_xml(xml_file)
        if video_path:
            # 動画ファイルが存在するか確認
            if os.path.exists(video_path):
                video_paths.append(video_path)
                xml_to_video[xml_file] = video_path
            else:
                logger.warning(f"動画ファイルが見つかりません: {video_path}")
    
    logger.info(f"有効な動画ファイル数: {len(video_paths)}")
    
    if not video_paths:
        logger.error("動画ファイルが見つかりませんでした")
        return
    
    # 既に抽出済みのファイルをスキップ
    to_process = []
    for video_path in video_paths:
        video_stem = Path(video_path).stem
        output_path = os.path.join(output_dir, f"{video_stem}_features.csv")
        
        if os.path.exists(output_path):
            logger.info(f"スキップ（既に存在）: {video_stem}")
        else:
            to_process.append(video_path)
    
    logger.info(f"処理対象: {len(to_process)}個")
    
    if not to_process:
        logger.info("全ての動画が既に処理済みです")
        return
    
    # 特徴量抽出を実行
    logger.info("\n" + "="*70)
    logger.info("特徴量抽出を開始")
    logger.info("="*70)
    
    success_count = 0
    error_count = 0
    
    for video_path in tqdm(to_process, desc="特徴量抽出"):
        try:
            result = extract_features_worker(video_path, output_dir)
            
            if result['status'] == 'Success':
                success_count += 1
                logger.info(f"✅ {result['file']}: {result['timesteps']}タイムステップ, {result['features']}特徴量")
            else:
                error_count += 1
                logger.error(f"❌ {result['file']}: {result.get('message', '不明なエラー')}")
        
        except Exception as e:
            error_count += 1
            logger.error(f"❌ {Path(video_path).name}: {e}")
    
    # サマリー
    logger.info("\n" + "="*70)
    logger.info("特徴量抽出完了")
    logger.info("="*70)
    logger.info(f"成功: {success_count}個")
    logger.info(f"失敗: {error_count}個")
    logger.info(f"出力先: {output_dir}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
