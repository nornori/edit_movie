"""
Premiere Pro互換のXMLを生成

注意: 以前はOpenTimelineIO (OTIO)を使った実装がありましたが、
Premiere Pro互換性の問題により、direct_xml_generator.pyに
処理を委譲しています。

OTIOを使った実装は以下のブランチに保存されています:
- ブランチ: archive/otio-implementation
- 理由: docs/architecture/WHY_NOT_OTIO.md を参照

詳細な分析: DEAD_CODE_ANALYSIS.md
"""
import logging

logger = logging.getLogger(__name__)


def create_premiere_xml_with_otio(
    video_path: str,
    video_name: str,
    total_frames: int,
    fps: float,
    tracks_data: list,
    telops: list,
    output_path: str,
    ai_telops: list = None
) -> str:
    """
    Premiere Pro互換のXMLを生成
    
    この関数は、direct_xml_generator.pyに処理を委譲しています。
    
    Args:
        video_path: 元動画のパス
        video_name: 動画名
        total_frames: 総フレーム数（推論FPS基準）
        fps: 推論時のフレームレート（10fps）
        tracks_data: ビデオトラックデータのリスト
        telops: テロップ情報のリスト（OCR）
        output_path: 出力XMLパス
        ai_telops: AI生成テロップ情報のリスト（音声認識+感情検出）
    
    Returns:
        出力XMLファイルのパス
    
    Note:
        以前はOpenTimelineIO (OTIO)を使った実装がありましたが、
        Premiere Pro互換性の問題により、direct_xml_generator.pyに
        処理を委譲しています。
        
        OTIOを使った実装の詳細:
        - ブランチ: archive/otio-implementation
        - ドキュメント: docs/architecture/WHY_NOT_OTIO.md
        - 分析: DEAD_CODE_ANALYSIS.md
        
        主な問題点:
        1. Premiere Pro互換性の問題（要素の順序、必須属性）
        2. 音声トラックの同期問題
        3. テロップがグラフィックとして認識されない
        4. 実装の複雑さ（Gap機能、複数トラック配置）
    """
    from src.inference.direct_xml_generator import create_premiere_xml_direct
    
    return create_premiere_xml_direct(
        video_path=video_path,
        video_name=video_name,
        total_frames=total_frames,
        fps=fps,
        tracks_data=tracks_data,
        telops=telops,
        output_path=output_path,
        ai_telops=ai_telops
    )
