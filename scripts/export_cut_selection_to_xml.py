"""
カット選択モデルの推論結果をPremiere Pro XML形式で出力

direct_xml_generator.pyの実装を完全に真似して作成
"""
import argparse
import torch
import numpy as np
import pandas as pd
import logging
import cv2
from pathlib import Path
from urllib.parse import quote

from src.cut_selection.inference.inference_cut_selection import CutSelectionInference
from src.utils.feature_alignment import FeatureAligner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_features_from_video(video_path: str, fps: float = 10.0) -> tuple:
    """
    動画から特徴量を抽出
    
    Args:
        video_path: 動画ファイルのパス
        fps: フレームレート
    
    Returns:
        (audio_features, visual_features) as numpy arrays
    """
    # 一時ディレクトリに特徴量を保存
    temp_dir = Path("temp_features")
    temp_dir.mkdir(exist_ok=True)
    
    video_name = Path(video_path).stem
    output_path = temp_dir / f"{video_name}_features.csv"
    
    # 既に抽出済みの場合はスキップ
    if output_path.exists():
        logger.info(f"  既存の特徴量ファイルを使用: {output_path}")
        df_all = pd.read_csv(output_path)
    else:
        # 特徴量抽出を実行
        logger.info(f"  動画を解析中...")
        from src.data_preparation.extract_video_features_parallel import extract_features_worker

        result = extract_features_worker(video_path, str(temp_dir))
        logger.info(f"  特徴量抽出結果: {result}")

        # エラーチェック
        if result and result.get('status') == 'Error':
            raise RuntimeError(f"特徴量抽出エラー: {result.get('message')}")

        # 抽出されたCSVを読み込み
        if output_path.exists():
            df_all = pd.read_csv(output_path)
        else:
            raise FileNotFoundError(f"特徴量抽出に失敗しました: {output_path}")
    
    # 音声特徴量と映像特徴量に分割
    # create_cut_selection_data.pyと同じロジックを使用

    # Audio features (215 dimensions)
    audio_cols = [
        'audio_energy_rms', 'audio_is_speaking', 'silence_duration_ms', 'speaker_id',
        'text_is_active', 'telop_active',
        'pitch_f0', 'pitch_std', 'spectral_centroid', 'zcr'
    ]
    audio_cols += [f'speaker_emb_{i}' for i in range(192)]
    audio_cols += [f'mfcc_{i}' for i in range(13)]

    # Visual features (522 dimensions)
    visual_cols = [
        'scene_change', 'visual_motion', 'saliency_x', 'saliency_y',
        'face_count', 'face_center_x', 'face_center_y',
        'face_size', 'face_mouth_open', 'face_eyebrow_raise'
    ]
    visual_cols += [f'clip_{i}' for i in range(512)]

    # Extract existing columns only
    audio_cols_exist = [c for c in audio_cols if c in df_all.columns]
    visual_cols_exist = [c for c in visual_cols if c in df_all.columns]

    # Convert to numeric, coercing errors to NaN, then fill with 0
    # Extract only feature columns (no time column)
    audio_features = df_all[audio_cols_exist].apply(pd.to_numeric, errors='coerce').fillna(0).values
    visual_features = df_all[visual_cols_exist].apply(pd.to_numeric, errors='coerce').fillna(0).values

    logger.info(f"  音声特徴量: {audio_features.shape}")
    logger.info(f"  映像特徴量: {visual_features.shape}")

    return audio_features, visual_features


def create_premiere_xml_from_clips(
    video_path: str,
    video_name: str,
    clips: list,
    fps: float,
    output_path: str
) -> str:
    """
    クリップリストからPremiere Pro互換のXMLを直接生成
    direct_xml_generator.pyの実装を完全に真似
    
    Args:
        video_path: 元動画のパス
        video_name: 動画名
        clips: クリップのリスト [(start_time, end_time), ...]
        fps: 推論時のフレームレート（10fps）
        output_path: 出力XMLパス
    
    Returns:
        出力XMLファイルのパス
    """
    # 元動画のFPS・解像度を取得
    cap = cv2.VideoCapture(video_path)
    video_fps = 60.0  # デフォルト
    video_width = 1080  # デフォルト
    video_height = 1920  # デフォルト
    
    if cap.isOpened():
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 60.0
        
        # 解像度を取得
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if width > 0 and height > 0:
            video_width = width
            video_height = height
        
        cap.release()
    
    logger.info(f"  Video properties: {video_width}x{video_height} @ {video_fps}fps")
    
    # FPS変換比率
    fps_ratio = video_fps / fps
    
    # クリップをフレーム番号に変換
    tracks_data = []
    for idx, (start_time, end_time) in enumerate(clips):
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        # 動画FPSに変換
        start_frame_video = int(start_frame * fps_ratio)
        end_frame_video = int(end_frame * fps_ratio)
        
        tracks_data.append({
            'start_frame': start_frame_video,
            'end_frame': end_frame_video,
            'idx': idx
        })
    
    # 総フレーム数を計算（最後のクリップの終了位置）
    if tracks_data:
        total_frames_video = max(track['end_frame'] for track in tracks_data)
    else:
        total_frames_video = 0
    
    # XMLを手動で構築
    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<!DOCTYPE xmeml>')
    lines.append('<xmeml version="4">')
    lines.append('\t<sequence id="sequence-1">')
    lines.append(f'\t\t<name>{video_name}</name>')
    lines.append(f'\t\t<duration>{total_frames_video}</duration>')
    lines.append('\t\t<rate>')
    lines.append(f'\t\t\t<timebase>{int(video_fps)}</timebase>')
    ntsc = 'TRUE' if abs(video_fps - 29.97) < 0.1 or abs(video_fps - 59.94) < 0.1 else 'FALSE'
    lines.append(f'\t\t\t<ntsc>{ntsc}</ntsc>')
    lines.append('\t\t</rate>')
    lines.append('\t\t<media>')
    lines.append('\t\t\t<video>')
    lines.append('\t\t\t\t<format/>')
    
    # メイン動画トラック
    lines.append('\t\t\t\t<track>')
    
    if tracks_data:
        timeline_pos = 0  # タイムライン上の現在位置を追跡
        for idx, track_data in enumerate(tracks_data):
            start_frame_video = track_data['start_frame']
            end_frame_video = track_data['end_frame']
            
            lines.append(f'\t\t\t\t\t<clipitem frameBlend="FALSE" id="clipitem-{idx+1}">')
            
            # file参照
            if idx == 0:
                # 最初のclipitemは完全なfile定義
                lines.append('\t\t\t\t\t\t<file id="file-1">')
                
                # pathurl
                video_path_normalized = video_path.replace('\\', '/')
                encoded_path = quote(video_path_normalized, safe='/:')
                lines.append(f'\t\t\t\t\t\t\t<pathurl>file://localhost/{encoded_path}</pathurl>')
                
                # name
                lines.append(f'\t\t\t\t\t\t\t<name>{Path(video_path).name}</name>')
                
                # rate
                lines.append('\t\t\t\t\t\t\t<rate>')
                lines.append(f'\t\t\t\t\t\t\t\t<timebase>{int(video_fps)}</timebase>')
                lines.append(f'\t\t\t\t\t\t\t\t<ntsc>{ntsc}</ntsc>')
                lines.append('\t\t\t\t\t\t\t</rate>')
                
                # duration
                lines.append(f'\t\t\t\t\t\t\t<duration>{total_frames_video}</duration>')
                
                # timecode
                lines.append('\t\t\t\t\t\t\t<timecode>')
                lines.append('\t\t\t\t\t\t\t\t<rate>')
                lines.append(f'\t\t\t\t\t\t\t\t\t<timebase>{int(video_fps)}</timebase>')
                lines.append(f'\t\t\t\t\t\t\t\t\t<ntsc>{ntsc}</ntsc>')
                lines.append('\t\t\t\t\t\t\t\t</rate>')
                lines.append('\t\t\t\t\t\t\t\t<string>00:00:00:00</string>')
                lines.append('\t\t\t\t\t\t\t\t<frame>0</frame>')
                lines.append('\t\t\t\t\t\t\t\t<displayformat>NDF</displayformat>')
                lines.append('\t\t\t\t\t\t\t</timecode>')
                
                # media
                lines.append('\t\t\t\t\t\t\t<media>')
                lines.append('\t\t\t\t\t\t\t\t<video/>')
                lines.append('\t\t\t\t\t\t\t\t<audio/>')
                lines.append('\t\t\t\t\t\t\t</media>')
                
                lines.append('\t\t\t\t\t\t</file>')
            else:
                # 2番目以降は参照のみ
                lines.append('\t\t\t\t\t\t<file id="file-1"/>')
            
            # name
            lines.append(f'\t\t\t\t\t\t<name>{video_name}</name>')
            
            # rate (2回)
            for _ in range(2):
                lines.append('\t\t\t\t\t\t<rate>')
                lines.append(f'\t\t\t\t\t\t\t<timebase>{int(video_fps)}</timebase>')
                lines.append(f'\t\t\t\t\t\t\t<ntsc>{ntsc}</ntsc>')
                lines.append('\t\t\t\t\t\t</rate>')
            
            # duration, start, end, in, out
            duration = end_frame_video - start_frame_video
            lines.append(f'\t\t\t\t\t\t<duration>{duration}</duration>')
            
            # タイムライン上の位置を計算（連続して配置）
            timeline_start = timeline_pos
            timeline_end = timeline_start + duration
            
            lines.append(f'\t\t\t\t\t\t<start>{timeline_start}</start>')
            lines.append(f'\t\t\t\t\t\t<end>{timeline_end}</end>')
            lines.append(f'\t\t\t\t\t\t<in>{start_frame_video}</in>')
            lines.append(f'\t\t\t\t\t\t<out>{end_frame_video}</out>')
            
            lines.append('\t\t\t\t\t</clipitem>')
            
            # 次のクリップのために位置を更新
            timeline_pos = timeline_end
    
    lines.append('\t\t\t\t</track>')
    lines.append('\t\t\t</video>')
    
    # 音声トラック
    lines.append('\t\t\t<audio>')
    lines.append('\t\t\t\t<track>')
    
    if tracks_data:
        timeline_pos = 0
        for idx, track_data in enumerate(tracks_data):
            start_frame_video = track_data['start_frame']
            end_frame_video = track_data['end_frame']
            duration = end_frame_video - start_frame_video
            
            lines.append(f'\t\t\t\t\t<clipitem frameBlend="FALSE" id="audioclip-{idx+1}">')
            lines.append('\t\t\t\t\t\t<file id="file-1"/>')
            lines.append(f'\t\t\t\t\t\t<name>{video_name} - Audio</name>')
            
            # rate (2回)
            for _ in range(2):
                lines.append('\t\t\t\t\t\t<rate>')
                lines.append(f'\t\t\t\t\t\t\t<timebase>{int(video_fps)}</timebase>')
                lines.append(f'\t\t\t\t\t\t\t<ntsc>{ntsc}</ntsc>')
                lines.append('\t\t\t\t\t\t</rate>')
            
            lines.append(f'\t\t\t\t\t\t<duration>{duration}</duration>')
            lines.append(f'\t\t\t\t\t\t<start>{timeline_pos}</start>')
            lines.append(f'\t\t\t\t\t\t<end>{timeline_pos + duration}</end>')
            lines.append(f'\t\t\t\t\t\t<in>{start_frame_video}</in>')
            lines.append(f'\t\t\t\t\t\t<out>{end_frame_video}</out>')
            
            lines.append('\t\t\t\t\t</clipitem>')
            
            timeline_pos += duration
    
    lines.append('\t\t\t\t</track>')
    lines.append('\t\t\t</audio>')
    lines.append('\t\t</media>')
    lines.append('\t</sequence>')
    lines.append('</xmeml>')
    
    # ファイルに書き込み
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"  ✅ XML generated: {output_path}")
    
    return str(output_path)


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="カット選択モデルの推論結果をPremiere Pro XMLで出力")
    
    parser.add_argument('video_path', type=str,
                       help='入力動画ファイルのパス')
    parser.add_argument('--model', type=str, default='checkpoints_cut_selection/best_model.pth',
                       help='学習済みモデルのパス')
    parser.add_argument('--output', type=str, default=None,
                       help='出力XMLファイルのパス（デフォルト: outputs/{video_name}_cut_selection.xml）')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='使用デバイス')
    parser.add_argument('--fps', type=float, default=10.0,
                       help='フレームレート')
    
    args = parser.parse_args()
    
    # 出力パスを決定
    video_name = Path(args.video_path).stem
    if args.output is None:
        args.output = f"outputs/{video_name}_cut_selection.xml"
    
    logger.info(f"\n{'='*80}")
    logger.info(f"カット選択推論パイプライン")
    logger.info(f"{'='*80}\n")
    logger.info(f"入力動画: {args.video_path}")
    logger.info(f"モデル: {args.model}")
    logger.info(f"出力XML: {args.output}")
    
    # ステップ1: 特徴量抽出
    logger.info("\nStep 1: 特徴量抽出...")
    audio_features, visual_features = extract_features_from_video(args.video_path, args.fps)
    
    # ステップ2: モデルで予測
    logger.info("\nStep 2: カット選択予測...")
    pipeline = CutSelectionInference(
        model_path=args.model,
        device=args.device,
        fps=args.fps
    )
    
    results = pipeline.predict(
        audio_features=audio_features,
        visual_features=visual_features,
        smooth=True
    )
    
    clips = results['clips']
    logger.info(f"  検出されたクリップ数: {len(clips)}")
    
    total_duration = sum(end - start for start, end in clips)
    logger.info(f"  合計時間: {total_duration:.1f}秒")
    
    # クリップの詳細を表示
    for i, (start, end) in enumerate(clips):
        logger.info(f"    クリップ {i+1}: {start:.2f}s - {end:.2f}s ({end-start:.2f}s)")
    
    # ステップ3: XMLに変換
    logger.info("\nStep 3: Premiere Pro XMLに変換...")
    xml_path = create_premiere_xml_from_clips(
        video_path=args.video_path,
        video_name=video_name,
        clips=clips,
        fps=args.fps,
        output_path=args.output
    )
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ 完了！")
    logger.info(f"出力XML: {xml_path}")
    logger.info(f"{'='*80}\n")
    logger.info(f"Premiere Proで開いて編集を確認してください。")


if __name__ == "__main__":
    main()
