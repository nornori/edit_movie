"""
Premiere Pro互換のXMLを完全に再構築するスクリプト
OTIOが生成したXMLを解析して、Premiere Proが確実に読み込める形式に再構築
"""
import xml.etree.ElementTree as ET
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def rebuild_xml_for_premiere(input_xml: str, output_xml: str):
    """
    XMLを解析してPremiere Pro互換の形式に完全再構築
    
    Args:
        input_xml: 入力XMLファイルのパス
        output_xml: 出力XMLファイルのパス
    """
    logger.info(f"  Rebuilding XML for Premiere Pro: {input_xml}")
    
    # XMLを解析
    tree = ET.parse(input_xml)
    root = tree.getroot()
    
    # sequenceを探す（projectタグの中にある場合もある）
    sequence = root.find('.//sequence')
    if sequence is None:
        raise ValueError("No sequence found in XML")
    
    # 新しいXMLを手動で構築
    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<!DOCTYPE xmeml>')
    lines.append('<xmeml version="4">')
    
    # sequenceを処理
    seq_id = sequence.get('id', 'sequence-1')
    lines.append(f'\t<sequence id="{seq_id}">')
    
    # sequence内の要素を処理
    # name
    name_elem = sequence.find('name')
    if name_elem is not None and name_elem.text:
        lines.append(f'\t\t<name>{name_elem.text}</name>')
    
    # duration
    duration_elem = sequence.find('duration')
    if duration_elem is not None and duration_elem.text:
        lines.append(f'\t\t<duration>{duration_elem.text}</duration>')
    
    # rate
    rate_elem = sequence.find('rate')
    if rate_elem is not None:
        timebase = rate_elem.find('timebase')
        ntsc = rate_elem.find('ntsc')
        lines.append('\t\t<rate>')
        if timebase is not None and timebase.text:
            lines.append(f'\t\t\t<timebase>{timebase.text}</timebase>')
        if ntsc is not None and ntsc.text:
            lines.append(f'\t\t\t<ntsc>{ntsc.text}</ntsc>')
        lines.append('\t\t</rate>')
    
    # media
    media_elem = sequence.find('media')
    if media_elem is not None:
        lines.append('\t\t<media>')
        
        # video
        video_elem = media_elem.find('video')
        if video_elem is not None:
            lines.append('\t\t\t<video>')
            
            # format
            lines.append('\t\t\t\t<format>')
            lines.append('\t\t\t\t\t<samplecharacteristics>')
            lines.append('\t\t\t\t\t\t<rate>')
            lines.append('\t\t\t\t\t\t\t<timebase>59</timebase>')
            lines.append('\t\t\t\t\t\t\t<ntsc>FALSE</ntsc>')
            lines.append('\t\t\t\t\t\t</rate>')
            lines.append('\t\t\t\t\t\t<codec>')
            lines.append('\t\t\t\t\t\t\t<name>Apple ProRes 422</name>')
            lines.append('\t\t\t\t\t\t\t<appspecificdata>')
            lines.append('\t\t\t\t\t\t\t\t<appname>Final Cut Pro</appname>')
            lines.append('\t\t\t\t\t\t\t\t<appmanufacturer>Apple Inc.</appmanufacturer>')
            lines.append('\t\t\t\t\t\t\t\t<appversion>7.0</appversion>')
            lines.append('\t\t\t\t\t\t\t\t<data>')
            lines.append('\t\t\t\t\t\t\t\t\t<qtcodec>')
            lines.append('\t\t\t\t\t\t\t\t\t\t<codecname>Apple ProRes 422</codecname>')
            lines.append('\t\t\t\t\t\t\t\t\t\t<codectypename>Apple ProRes 422</codectypename>')
            lines.append('\t\t\t\t\t\t\t\t\t\t<codectypecode>apcn</codectypecode>')
            lines.append('\t\t\t\t\t\t\t\t\t\t<codecvendorcode>appl</codecvendorcode>')
            lines.append('\t\t\t\t\t\t\t\t\t\t<spatialquality>1024</spatialquality>')
            lines.append('\t\t\t\t\t\t\t\t\t\t<temporalquality>0</temporalquality>')
            lines.append('\t\t\t\t\t\t\t\t\t\t<keyframerate>0</keyframerate>')
            lines.append('\t\t\t\t\t\t\t\t\t\t<datarate>0</datarate>')
            lines.append('\t\t\t\t\t\t\t\t\t</qtcodec>')
            lines.append('\t\t\t\t\t\t\t\t</data>')
            lines.append('\t\t\t\t\t\t\t</appspecificdata>')
            lines.append('\t\t\t\t\t\t</codec>')
            lines.append('\t\t\t\t\t\t<width>1080</width>')
            lines.append('\t\t\t\t\t\t<height>1920</height>')
            lines.append('\t\t\t\t\t\t<anamorphic>FALSE</anamorphic>')
            lines.append('\t\t\t\t\t\t<pixelaspectratio>square</pixelaspectratio>')
            lines.append('\t\t\t\t\t\t<fielddominance>none</fielddominance>')
            lines.append('\t\t\t\t\t\t<colordepth>24</colordepth>')
            lines.append('\t\t\t\t\t</samplecharacteristics>')
            lines.append('\t\t\t\t</format>')
            
            # tracks
            telop_count = 0
            file_elements = {}  # file idをキャッシュ
            
            for track in video_elem.findall('track'):
                lines.append('\t\t\t\t<track>')
                
                for clipitem in track.findall('clipitem'):
                    clip_id = clipitem.get('id', 'clipitem-1')
                    lines.append(f'\t\t\t\t\t<clipitem id="{clip_id}">')
                    
                    # file要素を先に処理してfile idを取得
                    file_elem = clipitem.find('file')
                    file_id = file_elem.get('id', 'file-1') if file_elem is not None else 'file-1'
                    
                    # masterclipid
                    lines.append(f'\t\t\t\t\t\t<masterclipid>{file_id}</masterclipid>')
                    
                    # name（テロップマーカーを削除）
                    name_elem = clipitem.find('name')
                    if name_elem is not None and name_elem.text:
                        name_text = name_elem.text
                        # マーカーを削除
                        name_text = name_text.replace('[Telop] ', '')
                        name_text = name_text.replace('[AI-Speech] ', '')
                        import re
                        name_text = re.sub(r'\[AI-Emotion-\w+\]\s*', '', name_text)
                        lines.append(f'\t\t\t\t\t\t<name>{name_text}</name>')
                    
                    # enabled
                    lines.append('\t\t\t\t\t\t<enabled>TRUE</enabled>')
                    
                    # duration
                    duration_elem = clipitem.find('duration')
                    if duration_elem is not None and duration_elem.text:
                        lines.append(f'\t\t\t\t\t\t<duration>{duration_elem.text}</duration>')
                    
                    # rate (2回)
                    rate_elem = clipitem.find('rate')
                    if rate_elem is not None:
                        timebase = rate_elem.find('timebase')
                        ntsc = rate_elem.find('ntsc')
                        for _ in range(2):
                            lines.append('\t\t\t\t\t\t<rate>')
                            if timebase is not None and timebase.text:
                                lines.append(f'\t\t\t\t\t\t\t<timebase>{timebase.text}</timebase>')
                            if ntsc is not None and ntsc.text:
                                lines.append(f'\t\t\t\t\t\t\t<ntsc>{ntsc.text}</ntsc>')
                            lines.append('\t\t\t\t\t\t</rate>')
                    
                    # start, end, in, out
                    for tag_name in ['start', 'end', 'in', 'out']:
                        elem = clipitem.find(tag_name)
                        if elem is not None and elem.text:
                            lines.append(f'\t\t\t\t\t\t<{tag_name}>{elem.text}</{tag_name}>')
                    
                    # alphatype, pixelaspectratio, anamorphic
                    lines.append('\t\t\t\t\t\t<alphatype>none</alphatype>')
                    lines.append('\t\t\t\t\t\t<pixelaspectratio>square</pixelaspectratio>')
                    lines.append('\t\t\t\t\t\t<anamorphic>FALSE</anamorphic>')
                    
                    # file要素
                    if file_elem is not None:
                        # file-2（テロップ）の場合
                        if file_id == 'file-2':
                            telop_count += 1
                            if telop_count == 1:
                                # 最初のfile-2は完全な定義
                                lines.append(f'\t\t\t\t\t\t<file id="{file_id}">')
                                lines.append('\t\t\t\t\t\t\t<name>グラフィック</name>')
                                lines.append('\t\t\t\t\t\t\t<pathurl></pathurl>')
                                lines.append('\t\t\t\t\t\t\t<rate>')
                                lines.append('\t\t\t\t\t\t\t\t<timebase>59</timebase>')
                                lines.append('\t\t\t\t\t\t\t\t<ntsc>FALSE</ntsc>')
                                lines.append('\t\t\t\t\t\t\t</rate>')
                                lines.append('\t\t\t\t\t\t\t<duration>1000</duration>')
                                lines.append('\t\t\t\t\t\t\t<timecode>')
                                lines.append('\t\t\t\t\t\t\t\t<rate>')
                                lines.append('\t\t\t\t\t\t\t\t\t<timebase>59</timebase>')
                                lines.append('\t\t\t\t\t\t\t\t\t<ntsc>FALSE</ntsc>')
                                lines.append('\t\t\t\t\t\t\t\t</rate>')
                                lines.append('\t\t\t\t\t\t\t\t<string>00:00:00:00</string>')
                                lines.append('\t\t\t\t\t\t\t\t<frame>0</frame>')
                                lines.append('\t\t\t\t\t\t\t\t<displayformat>NDF</displayformat>')
                                lines.append('\t\t\t\t\t\t\t</timecode>')
                                lines.append('\t\t\t\t\t\t\t<media>')
                                lines.append('\t\t\t\t\t\t\t\t<video>')
                                lines.append('\t\t\t\t\t\t\t\t\t<samplecharacteristics>')
                                lines.append('\t\t\t\t\t\t\t\t\t\t<rate>')
                                lines.append('\t\t\t\t\t\t\t\t\t\t\t<timebase>59</timebase>')
                                lines.append('\t\t\t\t\t\t\t\t\t\t\t<ntsc>FALSE</ntsc>')
                                lines.append('\t\t\t\t\t\t\t\t\t\t</rate>')
                                lines.append('\t\t\t\t\t\t\t\t\t\t<width>1080</width>')
                                lines.append('\t\t\t\t\t\t\t\t\t\t<height>1920</height>')
                                lines.append('\t\t\t\t\t\t\t\t\t\t<anamorphic>FALSE</anamorphic>')
                                lines.append('\t\t\t\t\t\t\t\t\t\t<pixelaspectratio>square</pixelaspectratio>')
                                lines.append('\t\t\t\t\t\t\t\t\t\t<fielddominance>none</fielddominance>')
                                lines.append('\t\t\t\t\t\t\t\t\t</samplecharacteristics>')
                                lines.append('\t\t\t\t\t\t\t\t</video>')
                                lines.append('\t\t\t\t\t\t\t</media>')
                                lines.append('\t\t\t\t\t\t</file>')
                                file_elements[file_id] = True
                            else:
                                # 2番目以降は参照のみ
                                lines.append(f'\t\t\t\t\t\t<file id="{file_id}"/>')
                        else:
                            # file-1（動画ファイル）の場合
                            if file_id not in file_elements:
                                # 最初の出現時は完全な定義
                                lines.append(f'\t\t\t\t\t\t<file id="{file_id}">')
                                
                                # name
                                file_name = file_elem.find('name')
                                if file_name is not None and file_name.text:
                                    lines.append(f'\t\t\t\t\t\t\t<name>{file_name.text}</name>')
                                
                                # pathurl
                                pathurl = file_elem.find('pathurl')
                                if pathurl is not None and pathurl.text:
                                    lines.append(f'\t\t\t\t\t\t\t<pathurl>{pathurl.text}</pathurl>')
                                
                                # rate
                                file_rate = file_elem.find('rate')
                                if file_rate is not None:
                                    timebase = file_rate.find('timebase')
                                    ntsc = file_rate.find('ntsc')
                                    lines.append('\t\t\t\t\t\t\t<rate>')
                                    if timebase is not None and timebase.text:
                                        lines.append(f'\t\t\t\t\t\t\t\t<timebase>{timebase.text}</timebase>')
                                    if ntsc is not None and ntsc.text:
                                        lines.append(f'\t\t\t\t\t\t\t\t<ntsc>{ntsc.text}</ntsc>')
                                    lines.append('\t\t\t\t\t\t\t</rate>')
                                
                                # duration
                                file_duration = file_elem.find('duration')
                                if file_duration is not None and file_duration.text:
                                    lines.append(f'\t\t\t\t\t\t\t<duration>{file_duration.text}</duration>')
                                
                                # timecode
                                timecode = file_elem.find('timecode')
                                if timecode is not None:
                                    lines.append('\t\t\t\t\t\t\t<timecode>')
                                    tc_rate = timecode.find('rate')
                                    if tc_rate is not None:
                                        timebase = tc_rate.find('timebase')
                                        ntsc = tc_rate.find('ntsc')
                                        lines.append('\t\t\t\t\t\t\t\t<rate>')
                                        if timebase is not None and timebase.text:
                                            lines.append(f'\t\t\t\t\t\t\t\t\t<timebase>{timebase.text}</timebase>')
                                        if ntsc is not None and ntsc.text:
                                            lines.append(f'\t\t\t\t\t\t\t\t\t<ntsc>{ntsc.text}</ntsc>')
                                        lines.append('\t\t\t\t\t\t\t\t</rate>')
                                    
                                    tc_string = timecode.find('string')
                                    if tc_string is not None and tc_string.text:
                                        lines.append(f'\t\t\t\t\t\t\t\t<string>{tc_string.text}</string>')
                                    
                                    tc_frame = timecode.find('frame')
                                    if tc_frame is not None and tc_frame.text:
                                        lines.append(f'\t\t\t\t\t\t\t\t<frame>{tc_frame.text}</frame>')
                                    
                                    tc_format = timecode.find('displayformat')
                                    if tc_format is not None and tc_format.text:
                                        lines.append(f'\t\t\t\t\t\t\t\t<displayformat>{tc_format.text}</displayformat>')
                                    
                                    lines.append('\t\t\t\t\t\t\t</timecode>')
                                
                                # media
                                file_media = file_elem.find('media')
                                if file_media is not None:
                                    lines.append('\t\t\t\t\t\t\t<media>')
                                    
                                    # video
                                    file_video = file_media.find('video')
                                    if file_video is not None:
                                        lines.append('\t\t\t\t\t\t\t\t<video>')
                                        file_video_sc = file_video.find('samplecharacteristics')
                                        if file_video_sc is not None:
                                            lines.append('\t\t\t\t\t\t\t\t\t<samplecharacteristics>')
                                            
                                            # rate
                                            sc_rate = file_video_sc.find('rate')
                                            if sc_rate is not None:
                                                timebase = sc_rate.find('timebase')
                                                ntsc = sc_rate.find('ntsc')
                                                lines.append('\t\t\t\t\t\t\t\t\t\t<rate>')
                                                if timebase is not None and timebase.text:
                                                    lines.append(f'\t\t\t\t\t\t\t\t\t\t\t<timebase>{timebase.text}</timebase>')
                                                if ntsc is not None and ntsc.text:
                                                    lines.append(f'\t\t\t\t\t\t\t\t\t\t\t<ntsc>{ntsc.text}</ntsc>')
                                                lines.append('\t\t\t\t\t\t\t\t\t\t</rate>')
                                            
                                            # width, height
                                            for tag_name in ['width', 'height', 'anamorphic', 'pixelaspectratio', 'fielddominance']:
                                                elem = file_video_sc.find(tag_name)
                                                if elem is not None and elem.text:
                                                    lines.append(f'\t\t\t\t\t\t\t\t\t\t<{tag_name}>{elem.text}</{tag_name}>')
                                            
                                            lines.append('\t\t\t\t\t\t\t\t\t</samplecharacteristics>')
                                        lines.append('\t\t\t\t\t\t\t\t</video>')
                                    
                                    # audio
                                    file_audio = file_media.find('audio')
                                    if file_audio is not None:
                                        lines.append('\t\t\t\t\t\t\t\t<audio>')
                                        file_audio_sc = file_audio.find('samplecharacteristics')
                                        if file_audio_sc is not None:
                                            lines.append('\t\t\t\t\t\t\t\t\t<samplecharacteristics>')
                                            
                                            depth = file_audio_sc.find('depth')
                                            if depth is not None and depth.text:
                                                lines.append(f'\t\t\t\t\t\t\t\t\t\t<depth>{depth.text}</depth>')
                                            
                                            samplerate = file_audio_sc.find('samplerate')
                                            if samplerate is not None and samplerate.text:
                                                lines.append(f'\t\t\t\t\t\t\t\t\t\t<samplerate>{samplerate.text}</samplerate>')
                                            
                                            lines.append('\t\t\t\t\t\t\t\t\t</samplecharacteristics>')
                                        
                                        channelcount = file_audio.find('channelcount')
                                        if channelcount is not None and channelcount.text:
                                            lines.append(f'\t\t\t\t\t\t\t\t\t<channelcount>{channelcount.text}</channelcount>')
                                        
                                        lines.append('\t\t\t\t\t\t\t\t</audio>')
                                    
                                    lines.append('\t\t\t\t\t\t\t</media>')
                                
                                lines.append('\t\t\t\t\t\t</file>')
                                file_elements[file_id] = True
                            else:
                                # 2番目以降は参照のみ
                                lines.append(f'\t\t\t\t\t\t<file id="{file_id}"/>')
                    
                    lines.append('\t\t\t\t\t</clipitem>')
                
                lines.append('\t\t\t\t</track>')
            
            lines.append('\t\t\t</video>')
        
        # audio
        audio_elem = media_elem.find('audio')
        if audio_elem is not None:
            lines.append('\t\t\t<audio>')
            
            # format
            lines.append('\t\t\t\t<format>')
            lines.append('\t\t\t\t\t<samplecharacteristics>')
            lines.append('\t\t\t\t\t\t<depth>16</depth>')
            lines.append('\t\t\t\t\t\t<samplerate>48000</samplerate>')
            lines.append('\t\t\t\t\t</samplecharacteristics>')
            lines.append('\t\t\t\t</format>')
            
            # tracks
            for track in audio_elem.findall('track'):
                lines.append('\t\t\t\t<track>')
                
                for clipitem in track.findall('clipitem'):
                    clip_id = clipitem.get('id', 'audioclip-1')
                    lines.append(f'\t\t\t\t\t<clipitem frameBlend="FALSE" id="{clip_id}">')
                    
                    # file参照
                    file_elem = clipitem.find('file')
                    file_id = file_elem.get('id', 'file-1') if file_elem is not None else 'file-1'
                    lines.append(f'\t\t\t\t\t\t<file id="{file_id}"/>')
                    
                    # name
                    name_elem = clipitem.find('name')
                    if name_elem is not None and name_elem.text:
                        lines.append(f'\t\t\t\t\t\t<name>{name_elem.text}</name>')
                    
                    # rate (2回)
                    rate_elem = clipitem.find('rate')
                    if rate_elem is not None:
                        timebase = rate_elem.find('timebase')
                        ntsc = rate_elem.find('ntsc')
                        for _ in range(2):
                            lines.append('\t\t\t\t\t\t<rate>')
                            if timebase is not None and timebase.text:
                                lines.append(f'\t\t\t\t\t\t\t<timebase>{timebase.text}</timebase>')
                            if ntsc is not None and ntsc.text:
                                lines.append(f'\t\t\t\t\t\t\t<ntsc>{ntsc.text}</ntsc>')
                            lines.append('\t\t\t\t\t\t</rate>')
                    
                    # duration, start, end, in, out
                    for tag_name in ['duration', 'start', 'end', 'in', 'out']:
                        elem = clipitem.find(tag_name)
                        if elem is not None and elem.text:
                            lines.append(f'\t\t\t\t\t\t<{tag_name}>{elem.text}</{tag_name}>')
                    
                    lines.append('\t\t\t\t\t</clipitem>')
                
                lines.append('\t\t\t\t</track>')
            
            lines.append('\t\t\t</audio>')
        
        lines.append('\t\t</media>')
    
    lines.append('\t</sequence>')
    lines.append('</xmeml>')
    
    # ファイルに書き込み
    output_path = Path(output_xml)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_xml, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"  ✅ Rebuilt XML saved to: {output_xml}")
    logger.info(f"     Total telops: {telop_count}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python rebuild_xml_for_premiere.py <input.xml> <output.xml>")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO)
    rebuild_xml_for_premiere(sys.argv[1], sys.argv[2])
