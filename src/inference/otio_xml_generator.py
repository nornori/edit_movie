"""
OpenTimelineIO を使用してPremiere Pro互換のXMLを生成
"""
import opentimelineio as otio
from pathlib import Path
import logging
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


def _post_process_telop_to_graphics(xml_path: str, telops: list, fps_ratio: float):
    """
    XMLを後処理してテロップをPremiere Pro互換のグラフィック/テキストレイヤーに変換
    
    Args:
        xml_path: XMLファイルのパス
        telops: テロップ情報のリスト
        fps_ratio: FPS変換比率
    """
    try:
        # XMLを読み込み（パーサーを使用してフォーマットを保持）
        ET.register_namespace('', 'http://www.w3.org/2001/XMLSchema')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # すべてのclipitemを検索
        for clipitem in root.iter('clipitem'):
            name_elem = clipitem.find('name')
            if name_elem is not None and name_elem.text and '[Telop]' in name_elem.text:
                # テロップクリップを見つけた
                logger.info(f"  Converting telop clip to graphic: {name_elem.text}")
                
                # fileエレメントを探す
                file_elem = clipitem.find('file')
                if file_elem is not None:
                    # ファイル参照をグラフィックに変更
                    file_elem.set('id', f"graphic_{clipitem.get('id')}")
                    
                    # pathurlを削除（グラフィックには不要）
                    pathurl = file_elem.find('pathurl')
                    if pathurl is not None:
                        file_elem.remove(pathurl)
                    
                    # nameを更新または作成
                    file_name = file_elem.find('name')
                    if file_name is None:
                        file_name = ET.Element('name')
                        file_elem.insert(0, file_name)
                    file_name.text = 'グラフィック'
                    
                    # mediaタグを削除（グラフィックには不要）
                    media_elem = file_elem.find('media')
                    if media_elem is not None:
                        file_elem.remove(media_elem)
                    
                    # 既存のmediaSourceを削除してから追加
                    existing_media_source = file_elem.find('mediaSource')
                    if existing_media_source is not None:
                        file_elem.remove(existing_media_source)
                    
                    # mediaSourceを追加（グラフィックとして認識させる）
                    # nameの直後に配置
                    name_index = list(file_elem).index(file_name)
                    media_source = ET.Element('mediaSource')
                    media_source.text = 'GraphicAndType'
                    file_elem.insert(name_index + 1, media_source)
                
                # テキストをクリップ名として設定（Premiere Proで編集可能）
                # 元のテキストを抽出
                original_text = name_elem.text.replace('[Telop] ', '')
                name_elem.text = original_text
        
        # 変更を保存（インデントを保持）
        _indent_xml(root)
        tree.write(xml_path, encoding='utf-8', xml_declaration=True)
        logger.info(f"  Post-processed {len(telops)} telop clips to graphics")
        
    except Exception as e:
        logger.warning(f"  Failed to post-process telops: {e}")
        logger.warning(f"  Telops will be imported as video clips instead of graphics")


def _post_process_telop_to_graphics_correct(xml_path: str, telops: list, video_fps: float):
    """
    XMLを後処理してテロップをグラフィックに変換（正しい形式）
    """
    try:
        # XMLを文字列として読み込んで処理
        with open(xml_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()
        
        # [Telop] マーカーを削除
        import re
        xml_content = re.sub(r'\[Telop\] ', '', xml_content)
        
        # テロップのfile要素を正しい形式に変換
        # <name/>を<name>グラフィック</name>に、<media><video/></media>を正しい形式に
        def replace_telop_file(match):
            file_content = match.group(0)
            file_id = match.group(1)
            
            # 空の<name/>を持つfile = テロップ
            if '<name/>' in file_content or '<name></name>' in file_content:
                # nameを置換
                file_content = re.sub(r'<name\s*/?>|<name></name>', '<name>グラフィック</name>', file_content)
                
                # mediaSourceを追加（nameの直後）
                if '<mediaSource>' not in file_content:
                    file_content = file_content.replace(
                        '<name>グラフィック</name>',
                        '<name>グラフィック</name>\n                                    <mediaSource>GraphicAndType</mediaSource>'
                    )
                
                # <media><video/></media>を正しい形式に置換
                timebase = str(int(video_fps))
                ntsc = 'TRUE' if abs(video_fps - 29.97) < 0.1 or abs(video_fps - 59.94) < 0.1 else 'FALSE'
                
                correct_media = f'''<media>
                                        <video>
                                            <samplecharacteristics>
                                                <rate>
                                                    <timebase>{timebase}</timebase>
                                                    <ntsc>{ntsc}</ntsc>
                                                </rate>
                                                <width>1080</width>
                                                <height>1920</height>
                                                <anamorphic>FALSE</anamorphic>
                                                <pixelaspectratio>square</pixelaspectratio>
                                                <fielddominance>none</fielddominance>
                                            </samplecharacteristics>
                                        </video>
                                    </media>'''
                
                file_content = re.sub(
                    r'<media>\s*<video\s*/>\s*</media>',
                    correct_media,
                    file_content
                )
            
            return file_content
        
        # file要素を処理
        xml_content = re.sub(
            r'<file id="(file-\d+)">(.*?)</file>',
            replace_telop_file,
            xml_content,
            flags=re.DOTALL
        )
        
        # 保存
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(xml_content)
        
        logger.info(f"  Post-processed {len(telops)} telop clips to graphics")
        
    except Exception as e:
        logger.warning(f"  Failed to post-process telops: {e}")
        logger.warning(f"  Telops will be imported as video clips instead of graphics")


def _post_process_telop_to_graphics_no_indent(xml_path: str, telops: list, fps_ratio: float):
    """
    XMLを後処理してテロップをグラフィックに変換（整形なし版）
    """
    try:
        # XMLを文字列として読み込んで処理
        with open(xml_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()
        
        # [Telop] マーカーを削除
        import re
        xml_content = re.sub(r'\[Telop\] ', '', xml_content)
        
        # <media><video/></media> を削除（テロップのfileタグ内のみ）
        # グラフィックのfile要素を特定して処理
        def replace_telop_media(match):
            file_content = match.group(0)
            # mediaSourceがある場合のみ処理
            if 'mediaSource' not in file_content and '<name/>' in file_content:
                # 空のnameを持つfileタグ = テロップ
                # <media><video/></media>を削除
                file_content = re.sub(r'<media>\s*<video/>\s*</media>', '', file_content)
                # 空の<name/>を<name>グラフィック</name>に置換
                file_content = file_content.replace('<name/>', '<name>グラフィック</name>')
                # mediaSourceを追加（nameの後）
                file_content = file_content.replace(
                    '<name>グラフィック</name>',
                    '<name>グラフィック</name>\n                                    <mediaSource>GraphicAndType</mediaSource>'
                )
            return file_content
        
        # file要素を処理
        xml_content = re.sub(
            r'<file id="file-\d+">(.*?)</file>',
            replace_telop_media,
            xml_content,
            flags=re.DOTALL
        )
        
        # 保存
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(xml_content)
        
        logger.info(f"  Post-processed {len(telops)} telop clips to graphics")
        
    except Exception as e:
        logger.warning(f"  Failed to post-process telops: {e}")
        logger.warning(f"  Telops will be imported as video clips instead of graphics")


def _fix_audio_clips_in_xml(root, tracks_data, fps_ratio, rate, video_name, media_reference):
    """
    XMLの音声クリップを強制的に修正（OTIOが正しく書き出さないため）
    csv2xml3.pyと同じアプローチ
    """
    # シーケンスレベルのmedia要素を探す（file要素内のmediaではない）
    for sequence in root.findall(".//sequence"):
        media = sequence.find("media")
        if media is None:
            continue
        
        audio = media.find("audio")
        if audio is None:
            # audioタグがない場合は作成
            audio = ET.SubElement(media, "audio")
        
        # 既存の音声トラックを全て削除
        for track in list(audio.findall("track")):
            audio.remove(track)
        
        # 新しい音声トラックを作成
        audio_track = ET.SubElement(audio, "track")
        
        if tracks_data:
            logger.info(f"  Recreating {len(tracks_data)} audio clips to match video")
            timeline_pos = 0
            for idx, track_data in enumerate(tracks_data):
                start_frame_video = int(track_data['start_frame'] * fps_ratio)
                end_frame_video = int(track_data['end_frame'] * fps_ratio)
                duration = end_frame_video - start_frame_video
                
                # clipitemを作成（duration=0も含める - csv2xml3.pyと同じ）
                clipitem = ET.SubElement(audio_track, "clipitem", {
                    "frameBlend": "FALSE",
                    "id": f"audioclip-{idx+1}"
                })
                
                # file参照（最初に配置）
                ET.SubElement(clipitem, "file", id="file-1")
                
                # name
                ET.SubElement(clipitem, "name").text = f"{video_name} - Audio"
                
                # rate（2回追加 - csv2xml3.pyと同じ）
                rate_elem1 = ET.SubElement(clipitem, "rate")
                ET.SubElement(rate_elem1, "timebase").text = str(int(rate))
                ET.SubElement(rate_elem1, "ntsc").text = "TRUE"
                
                rate_elem2 = ET.SubElement(clipitem, "rate")
                ET.SubElement(rate_elem2, "timebase").text = str(int(rate))
                ET.SubElement(rate_elem2, "ntsc").text = "TRUE"
                
                # duration
                ET.SubElement(clipitem, "duration").text = str(duration)
                
                # start/end/in/out
                ET.SubElement(clipitem, "start").text = str(timeline_pos)
                ET.SubElement(clipitem, "end").text = str(timeline_pos + duration)
                ET.SubElement(clipitem, "in").text = str(start_frame_video)
                ET.SubElement(clipitem, "out").text = str(end_frame_video)
                
                timeline_pos += duration


def _convert_telops_to_graphics_in_xml(root, telops, video_fps):
    """
    XMLのテロップをグラフィックに変換
    """
    import re
    
    # すべてのclipitemを検索
    for clipitem in root.iter('clipitem'):
        name_elem = clipitem.find('name')
        if name_elem is not None and name_elem.text and '[Telop]' in name_elem.text:
            # テロップクリップを見つけた
            logger.info(f"  Converting telop clip to graphic: {name_elem.text}")
            
            # fileエレメントを探す
            file_elem = clipitem.find('file')
            if file_elem is not None:
                # ファイル参照をグラフィックに変更
                file_elem.set('id', f"graphic_{clipitem.get('id')}")
                
                # pathurlを削除（グラフィックには不要）
                pathurl = file_elem.find('pathurl')
                if pathurl is not None:
                    file_elem.remove(pathurl)
                
                # nameを更新または作成
                file_name = file_elem.find('name')
                if file_name is None:
                    file_name = ET.Element('name')
                    file_elem.insert(0, file_name)
                file_name.text = 'グラフィック'
                
                # mediaタグを削除（グラフィックには不要）
                media_elem = file_elem.find('media')
                if media_elem is not None:
                    file_elem.remove(media_elem)
                
                # 既存のmediaSourceを削除してから追加
                existing_media_source = file_elem.find('mediaSource')
                if existing_media_source is not None:
                    file_elem.remove(existing_media_source)
                
                # mediaSourceを追加（グラフィックとして認識させる）
                # nameの直後に配置
                name_index = list(file_elem).index(file_name)
                media_source = ET.Element('mediaSource')
                media_source.text = 'GraphicAndType'
                file_elem.insert(name_index + 1, media_source)
            
            # テキストをクリップ名として設定（Premiere Proで編集可能）
            # 元のテキストを抽出
            original_text = name_elem.text.replace('[Telop] ', '')
            name_elem.text = original_text


def _post_process_telop_to_graphics_simple(xml_path: str):
    """
    XMLを後処理してテロップをグラフィックに変換（Premiere Pro互換）
    bandicam_2025-06-02_final_test.xmlと同じ構造にする
    """
    try:
        import re
        
        # XMLを文字列として読み込み
        with open(xml_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()
        
        # [Telop]を含むclipitemを探して処理
        telop_counter = 0
        first_telop_file_id = None
        
        def process_telop_clipitem(match):
            nonlocal telop_counter, first_telop_file_id
            telop_counter += 1
            
            clipitem_block = match.group(0)
            telop_text = match.group(1).strip()  # [Telop]の後のテキスト
            
            # clipitem内のnameから[Telop]マーカーを削除（テキスト内容を保持）
            clipitem_block = clipitem_block.replace(f'<name>[Telop] {telop_text}</name>', f'<name>{telop_text}</name>')
            
            # file要素を探して置き換え
            def replace_file(file_match):
                nonlocal first_telop_file_id
                file_id = file_match.group(1)
                
                # 最初のテロップの場合は完全なfile要素を生成
                if first_telop_file_id is None:
                    first_telop_file_id = file_id
                    
                    # Premiere Pro互換のグラフィックfile要素を生成
                    new_file = f'''<file id="{file_id}">
                                    <name>グラフィック</name>
                                    <mediaSource>GraphicAndType</mediaSource>
                                    <rate>
                                        <timebase>60</timebase>
                                        <ntsc>TRUE</ntsc>
                                    </rate>
                                    <timecode>
                                        <rate>
                                            <timebase>60</timebase>
                                            <ntsc>TRUE</ntsc>
                                        </rate>
                                        <string>00:00:00:00</string>
                                        <frame>0</frame>
                                        <displayformat>NDF</displayformat>
                                    </timecode>
                                    <media>
                                        <video>
                                            <samplecharacteristics>
                                                <rate>
                                                    <timebase>59</timebase>
                                                    <ntsc>TRUE</ntsc>
                                                </rate>
                                                <width>1080</width>
                                                <height>1920</height>
                                                <anamorphic>FALSE</anamorphic>
                                                <pixelaspectratio>square</pixelaspectratio>
                                                <fielddominance>none</fielddominance>
                                            </samplecharacteristics>
                                        </video>
                                    </media>
                                </file>'''
                    return new_file
                else:
                    # 2番目以降のテロップは最初のfile要素を参照
                    return f'<file id="{first_telop_file_id}"/>'
            
            # clipitem内のfile要素を置き換え（空のfileタグと完全なfileタグ両方に対応）
            clipitem_block = re.sub(
                r'<file id="([^"]*)"(?:\s*/?>|>(.*?)</file>)',
                replace_file,
                clipitem_block,
                flags=re.DOTALL
            )
            
            return clipitem_block
        
        # [Telop]を含むclipitemを処理（改行を含むnameに対応）
        xml_content = re.sub(
            r'<clipitem[^>]*>.*?<name>\[Telop\]\s*([^<]*(?:\n[^<]*)*)</name>.*?</clipitem>',
            process_telop_clipitem,
            xml_content,
            flags=re.DOTALL
        )
        
        # 保存
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(xml_content)
        
        logger.info(f"  ✅ Converted {telop_counter} telops to Premiere Pro graphics successfully")
        
    except Exception as e:
        logger.warning(f"  ⚠️  Failed to post-process telops: {e}")
        import traceback
        logger.warning(traceback.format_exc())
        logger.warning(f"  Telops will remain as video clips")
        import traceback
        logger.warning(traceback.format_exc())


def _indent_xml(elem, level=0):
    """XMLを整形するヘルパー関数"""
    indent = "\n" + "    " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "    "
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
        for child in elem:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent


def _generate_premiere_xml_directly_with_audio(
    output_path: str,
    video_path: str,
    video_name: str,
    video_fps: float,
    total_frames_video: int,
    tracks_data: list,
    telops: list,
    fps_ratio: float
):
    """
    Premiere Pro互換のXMLを直接生成（音声トラック対応）
    """
    from urllib.parse import quote
    from xml.dom import minidom
    
    # ルート要素
    root = ET.Element('xmeml', version='5')
    
    # シーケンス
    sequence = ET.SubElement(root, 'sequence', id='sequence-1')
    ET.SubElement(sequence, 'name').text = video_name
    ET.SubElement(sequence, 'duration').text = str(total_frames_video)
    
    # レート
    rate = ET.SubElement(sequence, 'rate')
    ET.SubElement(rate, 'timebase').text = str(int(video_fps))
    ET.SubElement(rate, 'ntsc').text = 'TRUE' if abs(video_fps - 29.97) < 0.1 or abs(video_fps - 59.94) < 0.1 else 'FALSE'
    
    # メディア
    media = ET.SubElement(sequence, 'media')
    
    # ビデオ
    video = ET.SubElement(media, 'video')
    
    # フォーマット
    fmt = ET.SubElement(video, 'format')
    sc = ET.SubElement(fmt, 'samplecharacteristics')
    ET.SubElement(sc, 'width').text = '1080'
    ET.SubElement(sc, 'height').text = '1920'
    ET.SubElement(sc, 'pixelaspectratio').text = 'square'
    
    # メイン動画トラック
    track = ET.SubElement(video, 'track')
    clipitem = ET.SubElement(track, 'clipitem', id='clipitem-1')
    ET.SubElement(clipitem, 'name').text = video_name
    ET.SubElement(clipitem, 'start').text = '0'
    ET.SubElement(clipitem, 'end').text = str(total_frames_video)
    ET.SubElement(clipitem, 'in').text = '0'
    ET.SubElement(clipitem, 'out').text = str(total_frames_video)
    
    # ファイル参照（Premiere Pro形式）
    file_elem = ET.SubElement(clipitem, 'file', id='file-1')
    ET.SubElement(file_elem, 'name').text = Path(video_path).name
    
    # パスをURLエンコード（Premiere Pro形式: file://localhost/D:/...）
    video_path_normalized = video_path.replace('\\', '/')
    encoded_path = quote(video_path_normalized, safe='/:')
    ET.SubElement(file_elem, 'pathurl').text = f"file://localhost/{encoded_path}"
    
    # レート
    file_rate = ET.SubElement(file_elem, 'rate')
    ET.SubElement(file_rate, 'timebase').text = str(int(video_fps))
    ET.SubElement(file_rate, 'ntsc').text = 'TRUE' if abs(video_fps - 29.97) < 0.1 or abs(video_fps - 59.94) < 0.1 else 'FALSE'
    
    ET.SubElement(file_elem, 'duration').text = str(total_frames_video)
    
    # メディア
    file_media = ET.SubElement(file_elem, 'media')
    ET.SubElement(file_media, 'video')
    ET.SubElement(file_media, 'audio')
    
    # テロップトラック
    if telops:
        logger.info(f"  Adding {len(telops)} telop tracks as graphics")
        for telop_idx, telop in enumerate(telops):
            telop_start = int(telop['start_frame'] * fps_ratio)
            telop_end = int(telop['end_frame'] * fps_ratio)
            
            telop_track = ET.SubElement(video, 'track')
            telop_clipitem = ET.SubElement(telop_track, 'clipitem', id=f'clipitem-telop-{telop_idx}')
            ET.SubElement(telop_clipitem, 'name').text = telop['text']
            ET.SubElement(telop_clipitem, 'start').text = str(telop_start)
            ET.SubElement(telop_clipitem, 'end').text = str(telop_end)
            ET.SubElement(telop_clipitem, 'in').text = '0'
            ET.SubElement(telop_clipitem, 'out').text = str(telop_end - telop_start)
            
            # グラフィックファイル
            telop_file = ET.SubElement(telop_clipitem, 'file', id=f'graphic-{telop_idx}')
            ET.SubElement(telop_file, 'name').text = 'グラフィック'
            ET.SubElement(telop_file, 'mediaSource').text = 'GraphicAndType'
    
    # 音声トラック
    audio = ET.SubElement(media, 'audio')
    
    # 音声フォーマット
    audio_fmt = ET.SubElement(audio, 'format')
    audio_sc = ET.SubElement(audio_fmt, 'samplecharacteristics')
    ET.SubElement(audio_sc, 'depth').text = '16'
    ET.SubElement(audio_sc, 'samplerate').text = '48000'
    
    # 音声トラック（映像と同じカット位置で）
    audio_track = ET.SubElement(audio, 'track')
    
    if tracks_data:
        # 映像トラックと同じセグメントで音声をカット
        timeline_pos = 0
        for idx, track_data in enumerate(tracks_data):
            start_frame_video = int(track_data['start_frame'] * fps_ratio)
            end_frame_video = int(track_data['end_frame'] * fps_ratio)
            duration = end_frame_video - start_frame_video
            
            audio_clipitem = ET.SubElement(audio_track, 'clipitem', id=f'audioclip-{idx+1}')
            ET.SubElement(audio_clipitem, 'name').text = f'{video_name} - Audio'
            ET.SubElement(audio_clipitem, 'start').text = str(timeline_pos)
            ET.SubElement(audio_clipitem, 'end').text = str(timeline_pos + duration)
            ET.SubElement(audio_clipitem, 'in').text = str(start_frame_video)
            ET.SubElement(audio_clipitem, 'out').text = str(end_frame_video)
            
            # 同じファイルを参照
            ET.SubElement(audio_clipitem, 'file', id='file-1')
            
            timeline_pos += duration
    else:
        # トラックデータがない場合は全体を追加
        audio_clipitem = ET.SubElement(audio_track, 'clipitem', id='audioclip-1')
        ET.SubElement(audio_clipitem, 'name').text = f'{video_name} - Audio'
        ET.SubElement(audio_clipitem, 'start').text = '0'
        ET.SubElement(audio_clipitem, 'end').text = str(total_frames_video)
        ET.SubElement(audio_clipitem, 'in').text = '0'
        ET.SubElement(audio_clipitem, 'out').text = str(total_frames_video)
        
        # 同じファイルを参照
        ET.SubElement(audio_clipitem, 'file', id='file-1')
    
    # XMLを整形して保存
    _indent_xml(root)
    tree = ET.ElementTree(root)
    
    # DOCTYPE付きで保存
    xml_str = ET.tostring(root, encoding='unicode')
    xml_with_doctype = f'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n{xml_str}'
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(xml_with_doctype)


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
    Premiere Pro互換のXMLを生成（直接生成方式）
    
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
    """
    # 直接XML生成方式を使用（OTIOを使わない）
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
    # 元動画のFPSを取得
    import cv2
    cap = cv2.VideoCapture(video_path)
    video_fps = 30.0  # デフォルト
    if cap.isOpened():
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30.0
        cap.release()
    
    logger.info(f"  Video FPS: {video_fps}, Inference FPS: {fps}")
    
    # FPS変換比率
    fps_ratio = video_fps / fps
    
    # タイムラインを作成
    timeline = otio.schema.Timeline(name=video_name)
    
    # フレームレートを設定（元動画のFPSを使用）
    rate = video_fps
    
    # 総フレーム数を変換
    total_frames_video = int(total_frames * fps_ratio)
    
    # ビデオトラックを作成
    video_track = otio.schema.Track(
        name="Video 1",
        kind=otio.schema.TrackKind.Video
    )
    timeline.tracks.append(video_track)
    
    # 元動画への参照を作成
    available_range = otio.opentime.TimeRange(
        start_time=otio.opentime.RationalTime(0, rate),
        duration=otio.opentime.RationalTime(total_frames_video, rate)
    )
    
    # Windowsパスをfile:// URLに変換（Premiere Pro形式: file://localhost/D%3A/...）
    from urllib.parse import quote
    video_path_normalized = video_path.replace(chr(92), '/')  # バックスラッシュをスラッシュに
    # コロンを含む全体をエンコード（safe='/' でスラッシュのみ保護）
    encoded_path = quote(video_path_normalized, safe='/')
    video_url = f"file://localhost/{encoded_path}"
    
    media_reference = otio.schema.ExternalReference(
        target_url=video_url,
        available_range=available_range
    )
    
    # 音声トラックを作成（映像と同時に追加するため、ここで作成）
    audio_track = otio.schema.Track(
        name="Audio 1",
        kind=otio.schema.TrackKind.Audio
    )
    
    # ビデオクリップと音声クリップを同時に追加
    if tracks_data:
        for idx, track_data in enumerate(tracks_data):
            # フレーム数を変換
            start_frame_video = int(track_data['start_frame'] * fps_ratio)
            end_frame_video = int(track_data['end_frame'] * fps_ratio)
            
            # 時間範囲を作成
            source_range = otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(start_frame_video, rate),
                duration=otio.opentime.RationalTime(
                    end_frame_video - start_frame_video, 
                    rate
                )
            )
            
            # ビデオクリップを作成
            video_clip = otio.schema.Clip(
                name=video_name,
                media_reference=media_reference,
                source_range=source_range
            )
            
            # エフェクトを追加（モーション）
            # 注: OTIOのエフェクトは限定的なので、メタデータとして保存
            video_clip.metadata['premiere'] = {
                'scale': track_data['scale'] * 100,  # パーセンテージに変換
                'position_x': track_data['position_x'],
                'position_y': track_data['position_y'],
                'crop_left': track_data['crop_left'],
                'crop_right': track_data['crop_right'],
                'crop_top': track_data['crop_top'],
                'crop_bottom': track_data['crop_bottom']
            }
            
            video_track.append(video_clip)
            
            # 音声クリップを作成（同じsource_rangeを使用）
            audio_clip = otio.schema.Clip(
                name=f"{video_name} - Audio",
                media_reference=media_reference,
                source_range=source_range
            )
            audio_track.append(audio_clip)
    else:
        # トラックデータがない場合は、全体を1つのクリップとして追加
        source_range = otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(0, rate),
            duration=otio.opentime.RationalTime(total_frames_video, rate)
        )
        
        video_clip = otio.schema.Clip(
            name=video_name,
            media_reference=media_reference,
            source_range=source_range
        )
        video_track.append(video_clip)
        
        audio_clip = otio.schema.Clip(
            name=f"{video_name} - Audio",
            media_reference=media_reference,
            source_range=source_range
        )
        audio_track.append(audio_clip)
    
    # テロップトラックを追加（グラフィックとして）
    # 時間が重ならないテロップを同じトラックにまとめる
    if telops:
        logger.info(f"  Adding {len(telops)} telop clips as graphics")
        
        # テロップを開始時間でソート
        sorted_telops = sorted(telops, key=lambda t: t['start_frame'])
        
        # トラックのリスト（各トラックの最後のend_frameを記録）
        telop_tracks = []
        track_end_times = []
        
        for telop_idx, telop in enumerate(sorted_telops):
            # フレーム数を変換
            telop_start_video = int(telop['start_frame'] * fps_ratio)
            telop_end_video = int(telop['end_frame'] * fps_ratio)
            
            # テロップクリップを作成
            telop_range = otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(telop_start_video, rate),
                duration=otio.opentime.RationalTime(
                    telop_end_video - telop_start_video,
                    rate
                )
            )
            
            # テロップクリップ（グラフィックとして）
            telop_clip = otio.schema.Clip(
                name=f"[Telop] {telop['text'][:30]}",  # 名前にテロップマーカーを追加
                source_range=telop_range
            )
            
            # メタデータにテロップ情報を保存（後処理用）
            telop_clip.metadata['telop'] = {
                'text': telop['text'],
                'type': 'graphic',
                'is_text_layer': True
            }
            
            # 既存のトラックで時間が重ならないものを探す
            placed = False
            for track_idx, (track, last_end) in enumerate(zip(telop_tracks, track_end_times)):
                if telop_start_video >= last_end:
                    # このトラックに配置可能
                    # ギャップを追加（前のクリップとの間）
                    if telop_start_video > last_end:
                        gap = otio.schema.Gap(
                            source_range=otio.opentime.TimeRange(
                                start_time=otio.opentime.RationalTime(last_end, rate),
                                duration=otio.opentime.RationalTime(telop_start_video - last_end, rate)
                            )
                        )
                        track.append(gap)
                    
                    track.append(telop_clip)
                    track_end_times[track_idx] = telop_end_video
                    placed = True
                    break
            
            if not placed:
                # 新しいトラックを作成
                telop_track = otio.schema.Track(
                    name=f"Telops {len(telop_tracks) + 1}",
                    kind=otio.schema.TrackKind.Video
                )
                
                # ギャップを追加（開始位置まで）
                if telop_start_video > 0:
                    gap = otio.schema.Gap(
                        source_range=otio.opentime.TimeRange(
                            start_time=otio.opentime.RationalTime(0, rate),
                            duration=otio.opentime.RationalTime(telop_start_video, rate)
                        )
                    )
                    telop_track.append(gap)
                
                telop_track.append(telop_clip)
                telop_tracks.append(telop_track)
                track_end_times.append(telop_end_video)
        
        # 全てのテロップトラックをタイムラインに追加
        for track in telop_tracks:
            timeline.tracks.append(track)
        
        logger.info(f"  Organized {len(telops)} telops into {len(telop_tracks)} tracks")
    
    # AI字幕トラックを追加（時間が重ならないものを同じトラックにまとめる）
    if ai_telops:
        logger.info(f"  Adding {len(ai_telops)} AI telop clips")
        
        # AI字幕を開始時間でソート
        sorted_ai_telops = sorted(ai_telops, key=lambda t: t['start_frame'])
        
        # トラックのリスト（各トラックの最後のend_frameを記録）
        ai_telop_tracks = []
        ai_track_end_times = []
        
        for ai_telop_idx, ai_telop in enumerate(sorted_ai_telops):
            # フレーム数を変換
            ai_telop_start_video = int(ai_telop['start_frame'] * fps_ratio)
            ai_telop_end_video = int(ai_telop['end_frame'] * fps_ratio)
            
            # AI字幕クリップを作成
            telop_type = ai_telop.get('type', 'speech')
            emotion_type = ai_telop.get('emotion_type', '')
            
            if telop_type == 'speech':
                marker = "[AI-Speech]"
            elif telop_type == 'emotion':
                marker = f"[AI-Emotion-{emotion_type}]"
            else:
                marker = "[AI-Telop]"
            
            ai_telop_range = otio.opentime.TimeRange(
                start_time=otio.opentime.RationalTime(ai_telop_start_video, rate),
                duration=otio.opentime.RationalTime(
                    ai_telop_end_video - ai_telop_start_video,
                    rate
                )
            )
            
            # AI字幕クリップ（グラフィックとして）
            ai_telop_clip = otio.schema.Clip(
                name=f"{marker} {ai_telop['text'][:30]}",
                source_range=ai_telop_range
            )
            
            # メタデータにAI字幕情報を保存
            ai_telop_clip.metadata['telop'] = {
                'text': ai_telop['text'],
                'type': 'graphic',
                'is_text_layer': True,
                'ai_generated': True,
                'telop_type': telop_type,
                'emotion_type': emotion_type
            }
            
            # 既存のトラックで時間が重ならないものを探す
            placed = False
            for track_idx, (track, last_end) in enumerate(zip(ai_telop_tracks, ai_track_end_times)):
                if ai_telop_start_video >= last_end:
                    # このトラックに配置可能
                    # ギャップを追加（前のクリップとの間）
                    if ai_telop_start_video > last_end:
                        gap = otio.schema.Gap(
                            source_range=otio.opentime.TimeRange(
                                start_time=otio.opentime.RationalTime(last_end, rate),
                                duration=otio.opentime.RationalTime(ai_telop_start_video - last_end, rate)
                            )
                        )
                        track.append(gap)
                    
                    track.append(ai_telop_clip)
                    ai_track_end_times[track_idx] = ai_telop_end_video
                    placed = True
                    break
            
            if not placed:
                # 新しいトラックを作成
                ai_telop_track = otio.schema.Track(
                    name=f"AI-Telops {len(ai_telop_tracks) + 1}",
                    kind=otio.schema.TrackKind.Video
                )
                
                # ギャップを追加（開始位置まで）
                if ai_telop_start_video > 0:
                    gap = otio.schema.Gap(
                        source_range=otio.opentime.TimeRange(
                            start_time=otio.opentime.RationalTime(0, rate),
                            duration=otio.opentime.RationalTime(ai_telop_start_video, rate)
                        )
                    )
                    ai_telop_track.append(gap)
                
                ai_telop_track.append(ai_telop_clip)
                ai_telop_tracks.append(ai_telop_track)
                ai_track_end_times.append(ai_telop_end_video)
        
        # 全てのAI字幕トラックをタイムラインに追加
        for track in ai_telop_tracks:
            timeline.tracks.append(track)
        
        logger.info(f"  Organized {len(ai_telops)} AI telops into {len(ai_telop_tracks)} tracks")
    
    # 音声トラックを追加（既にクリップは追加済み）
    timeline.tracks.append(audio_track)
    
    # XMLとして書き出し（OTIOに忠実に従う）
    logger.info("Writing XML via OTIO...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    otio.adapters.write_to_file(timeline, str(output_path), adapter_name='fcp_xml')
    
    logger.info(f"  OTIO timeline exported to {output_path}")
    
    # 後処理: テロップをグラフィックに変換（完全なPremiere Pro互換形式）
    if telops or ai_telops:
        logger.info(f"  Post-processing: Converting telops to Premiere Pro graphics...")
        from src.inference.fix_telop_complete import fix_telops_complete
        
        # 一時ファイルとして保存
        temp_output = str(output_path).replace('.xml', '_temp.xml')
        import shutil
        shutil.move(str(output_path), temp_output)
        
        # テロップをグラフィックに変換
        fix_telops_complete(temp_output, str(output_path))
        
        # 一時ファイルを削除
        import os
        os.remove(temp_output)
    
    return str(output_path)
