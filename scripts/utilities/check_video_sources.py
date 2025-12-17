"""
XMLから抽出される動画パスを確認

編集後の動画なのか、元動画（bandicam）なのかを確認します。
"""
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_video_path_from_xml(xml_path: str) -> dict:
    """
    XMLファイルから動画ファイルのパスを抽出
    
    Returns:
        dict: 動画パス情報
    """
    result = {
        'xml_name': Path(xml_path).stem,
        'video_path': None,
        'is_bandicam': False,
        'is_output': False,
        'error': None
    }
    
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
            
            result['video_path'] = path
            
            # パスの種類を判定
            path_lower = path.lower()
            if 'bandicam' in path_lower:
                result['is_bandicam'] = True
            
            # 出力動画のパターンをチェック
            # 通常、編集後の動画は特定のフォルダに出力される
            # 例: "全財産を賭けた", "ひなーの", など日本語タイトル
            path_parts = Path(path).parts
            for part in path_parts:
                # bandicamではない日本語ファイル名は出力動画の可能性が高い
                if not 'bandicam' in part.lower() and any(ord(c) > 127 for c in part):
                    # 拡張子がmp4で、bandicamを含まない日本語ファイル名
                    if part.endswith('.mp4'):
                        result['is_output'] = True
                        break
            
    except Exception as e:
        result['error'] = str(e)
    
    return result


def main():
    logger.info("="*70)
    logger.info("XMLから抽出される動画パスの確認")
    logger.info("="*70)
    
    xml_dir = Path("data/raw/editxml")
    xml_files = sorted(xml_dir.glob("*.xml"))
    
    logger.info(f"XMLファイル数: {len(xml_files)}\n")
    
    results = []
    bandicam_count = 0
    output_count = 0
    unknown_count = 0
    error_count = 0
    
    for xml_file in xml_files:
        result = extract_video_path_from_xml(str(xml_file))
        results.append(result)
        
        if result['error']:
            error_count += 1
            logger.error(f"❌ {result['xml_name']}: {result['error']}")
        elif result['video_path']:
            if result['is_bandicam']:
                bandicam_count += 1
                logger.info(f"✅ {result['xml_name']}: BANDICAM")
                logger.info(f"   {result['video_path']}")
            elif result['is_output']:
                output_count += 1
                logger.warning(f"⚠️  {result['xml_name']}: OUTPUT VIDEO (編集後)")
                logger.warning(f"   {result['video_path']}")
            else:
                unknown_count += 1
                logger.info(f"❓ {result['xml_name']}: UNKNOWN")
                logger.info(f"   {result['video_path']}")
        else:
            error_count += 1
            logger.error(f"❌ {result['xml_name']}: パスが見つかりません")
    
    # サマリー
    logger.info("\n" + "="*70)
    logger.info("確認完了")
    logger.info("="*70)
    logger.info(f"総XMLファイル数: {len(xml_files)}")
    logger.info(f"元動画（bandicam）: {bandicam_count}")
    logger.info(f"編集後動画: {output_count}")
    logger.info(f"不明: {unknown_count}")
    logger.info(f"エラー: {error_count}")
    
    if output_count > 0:
        logger.warning("\n⚠️  警告: 編集後の動画から特徴量を抽出している可能性があります！")
        logger.warning("編集後の動画:")
        for r in results:
            if r['is_output']:
                logger.warning(f"  - {r['xml_name']}: {r['video_path']}")
    
    # 結果を保存
    output_file = Path("video_sources_check.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total': len(xml_files),
                'bandicam': bandicam_count,
                'output': output_count,
                'unknown': unknown_count,
                'error': error_count
            },
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n詳細レポート保存: {output_file}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
