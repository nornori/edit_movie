"""
Premiere Pro互換のXMLを直接生成（OTIOを使わない）
"""
import xml.etree.ElementTree as ET
from pathlib import Path
import logging
from urllib.parse import quote

logger = logging.getLogger(__name__)


def create_premiere_xml_direct(
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
    Premiere Pro互換のXMLを直接生成
    
    Args:
        video_path: 元動画のパス
        video_name: 動画名
        total_frames: 総フレーム数（推論FPS基準）
        fps: 推論時のフレームレート（10fps）
        tracks_data: ビデオトラックデータのリスト
        telops: テロップ情報のリスト（OCR）
        output_path: 出力XMLパス
        ai_telops: AI生成テロップ情報のリスト
    
    Returns:
        出力XMLファイルのパス
    """
    if ai_telops is None:
        ai_telops = []
    
    # 元動画のFPSを取得
    import cv2
    cap = cv2.VideoCapture(video_path)
    video_fps = 60.0  # デフォルト
    if cap.isOpened():
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 60.0
        cap.release()
    
    logger.info(f"  Video FPS: {video_fps}, Inference FPS: {fps}")
    
    # FPS変換比率
    fps_ratio = video_fps / fps
    
    # 総フレーム数を変換
    total_frames_video = int(total_frames * fps_ratio)
    
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
            start_frame_video = int(track_data['start_frame'] * fps_ratio)
            end_frame_video = int(track_data['end_frame'] * fps_ratio)
            
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
    
    # テロップトラック（OCR + AI）
    # 時間が重ならないテロップを同じトラックにまとめる
    all_telops = telops + ai_telops
    if all_telops:
        logger.info(f"  Adding {len(all_telops)} telops")
        
        # テロップを開始時間でソート
        sorted_telops = sorted(all_telops, key=lambda t: t['start_frame'])
        
        # トラックのリスト（各トラックの最後のend_frameを記録）
        telop_track_lines = []  # 各トラックのXML行のリスト
        track_end_times = []  # 各トラックの最後のend_frame
        
        first_telop_file_defined = False
        
        for telop_idx, telop in enumerate(sorted_telops):
            telop_start = int(telop['start_frame'] * fps_ratio)
            telop_end = int(telop['end_frame'] * fps_ratio)
            
            # clipitemのXML行を生成
            clipitem_lines = []
            clipitem_lines.append(f'\t\t\t\t\t<clipitem frameBlend="FALSE" id="clipitem-telop-{telop_idx+1}">')
            
            # file参照
            if not first_telop_file_defined:
                # 最初のテロップは完全なfile定義
                clipitem_lines.append('\t\t\t\t\t\t<file id="file-2">')
                clipitem_lines.append('\t\t\t\t\t\t\t<name>グラフィック</name>')
                clipitem_lines.append('\t\t\t\t\t\t\t<mediaSource>GraphicAndType</mediaSource>')
                clipitem_lines.append('\t\t\t\t\t\t\t<rate>')
                clipitem_lines.append(f'\t\t\t\t\t\t\t\t<timebase>{int(video_fps)}</timebase>')
                clipitem_lines.append(f'\t\t\t\t\t\t\t\t<ntsc>{ntsc}</ntsc>')
                clipitem_lines.append('\t\t\t\t\t\t\t</rate>')
                clipitem_lines.append('\t\t\t\t\t\t\t<timecode>')
                clipitem_lines.append('\t\t\t\t\t\t\t\t<rate>')
                clipitem_lines.append(f'\t\t\t\t\t\t\t\t\t<timebase>{int(video_fps)}</timebase>')
                clipitem_lines.append(f'\t\t\t\t\t\t\t\t\t<ntsc>{ntsc}</ntsc>')
                clipitem_lines.append('\t\t\t\t\t\t\t\t</rate>')
                clipitem_lines.append('\t\t\t\t\t\t\t\t<string>00:00:00:00</string>')
                clipitem_lines.append('\t\t\t\t\t\t\t\t<frame>0</frame>')
                clipitem_lines.append('\t\t\t\t\t\t\t\t<displayformat>NDF</displayformat>')
                clipitem_lines.append('\t\t\t\t\t\t\t</timecode>')
                clipitem_lines.append('\t\t\t\t\t\t\t<media>')
                clipitem_lines.append('\t\t\t\t\t\t\t\t<video>')
                clipitem_lines.append('\t\t\t\t\t\t\t\t\t<samplecharacteristics>')
                clipitem_lines.append('\t\t\t\t\t\t\t\t\t\t<rate>')
                clipitem_lines.append(f'\t\t\t\t\t\t\t\t\t\t\t<timebase>{int(video_fps)-1}</timebase>')
                clipitem_lines.append(f'\t\t\t\t\t\t\t\t\t\t\t<ntsc>{ntsc}</ntsc>')
                clipitem_lines.append('\t\t\t\t\t\t\t\t\t\t</rate>')
                clipitem_lines.append('\t\t\t\t\t\t\t\t\t\t<width>1080</width>')
                clipitem_lines.append('\t\t\t\t\t\t\t\t\t\t<height>1920</height>')
                clipitem_lines.append('\t\t\t\t\t\t\t\t\t\t<anamorphic>FALSE</anamorphic>')
                clipitem_lines.append('\t\t\t\t\t\t\t\t\t\t<pixelaspectratio>square</pixelaspectratio>')
                clipitem_lines.append('\t\t\t\t\t\t\t\t\t\t<fielddominance>none</fielddominance>')
                clipitem_lines.append('\t\t\t\t\t\t\t\t\t</samplecharacteristics>')
                clipitem_lines.append('\t\t\t\t\t\t\t\t</video>')
                clipitem_lines.append('\t\t\t\t\t\t\t</media>')
                clipitem_lines.append('\t\t\t\t\t\t</file>')
                first_telop_file_defined = True
            else:
                # 2番目以降は参照のみ
                clipitem_lines.append('\t\t\t\t\t\t<file id="file-2"/>')
            
            # name
            clipitem_lines.append(f'\t\t\t\t\t\t<name>グラフィック</name>')
            
            # rate (2回)
            for _ in range(2):
                clipitem_lines.append('\t\t\t\t\t\t<rate>')
                clipitem_lines.append(f'\t\t\t\t\t\t\t<timebase>{int(video_fps)}</timebase>')
                clipitem_lines.append(f'\t\t\t\t\t\t\t<ntsc>{ntsc}</ntsc>')
                clipitem_lines.append('\t\t\t\t\t\t</rate>')
            
            # duration, start, end, in, out
            duration = telop_end - telop_start
            clipitem_lines.append(f'\t\t\t\t\t\t<duration>{duration}</duration>')
            clipitem_lines.append(f'\t\t\t\t\t\t<start>{telop_start}</start>')
            clipitem_lines.append(f'\t\t\t\t\t\t<end>{telop_end}</end>')
            clipitem_lines.append('\t\t\t\t\t\t<in>0</in>')
            clipitem_lines.append(f'\t\t\t\t\t\t<out>{duration}</out>')
            
            # グラフィックエフェクトを追加（テロップテキストを含む）
            # 改行を&#13;に変換
            telop_text = telop["text"].replace('\n', '&#13;').replace('\r', '&#13;')
            clipitem_lines.append('\t\t\t\t\t\t<filter>')
            clipitem_lines.append('\t\t\t\t\t\t\t<effect>')
            clipitem_lines.append(f'\t\t\t\t\t\t\t\t<name>{telop_text}</name>')
            clipitem_lines.append('\t\t\t\t\t\t\t\t<effectid>GraphicAndType</effectid>')
            clipitem_lines.append('\t\t\t\t\t\t\t\t<effectcategory>graphic</effectcategory>')
            clipitem_lines.append('\t\t\t\t\t\t\t\t<effecttype>filter</effecttype>')
            clipitem_lines.append('\t\t\t\t\t\t\t\t<mediatype>video</mediatype>')
            clipitem_lines.append('\t\t\t\t\t\t\t\t<pproBypass>false</pproBypass>')
            clipitem_lines.append('\t\t\t\t\t\t\t</effect>')
            clipitem_lines.append('\t\t\t\t\t\t</filter>')
            
            clipitem_lines.append('\t\t\t\t\t</clipitem>')
            
            # 既存のトラックで時間が重ならないものを探す
            placed = False
            for track_idx, (track_lines, last_end) in enumerate(zip(telop_track_lines, track_end_times)):
                if telop_start >= last_end:
                    # このトラックに配置可能
                    track_lines.extend(clipitem_lines)
                    track_end_times[track_idx] = telop_end
                    placed = True
                    break
            
            if not placed:
                # 新しいトラックを作成
                new_track_lines = []
                new_track_lines.extend(clipitem_lines)
                telop_track_lines.append(new_track_lines)
                track_end_times.append(telop_end)
        
        # 全てのテロップトラックをXMLに追加
        for track_lines in telop_track_lines:
            lines.append('\t\t\t\t<track>')
            lines.extend(track_lines)
            lines.append('\t\t\t\t</track>')
        
        logger.info(f"  Organized {len(all_telops)} telops into {len(telop_track_lines)} tracks")
    
    lines.append('\t\t\t</video>')
    
    # 音声トラック
    lines.append('\t\t\t<audio>')
    lines.append('\t\t\t\t<track>')
    
    if tracks_data:
        timeline_pos = 0
        for idx, track_data in enumerate(tracks_data):
            start_frame_video = int(track_data['start_frame'] * fps_ratio)
            end_frame_video = int(track_data['end_frame'] * fps_ratio)
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
