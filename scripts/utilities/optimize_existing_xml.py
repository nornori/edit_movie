"""
既存のXMLファイルのトラックを最適化
test_ai_telop_final_fixed.xml のトラックを減らす
"""
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

input_file = "outputs/test_ai_telop_final_fixed.xml"
output_file = "outputs/test_ai_telop_final_fixed_optimized.xml"

logger.info(f"Reading {input_file}...")
with open(input_file, 'r', encoding='utf-8') as f:
    content = f.read()

# file-2を参照しているtrackを検出（グラフィックトラック）
telop_tracks = re.findall(r'<track[^>]*>.*?<file id="file-\d+"[^>]*>.*?<mediaSource>GraphicAndType</mediaSource>.*?</track>', content, re.DOTALL)

logger.info(f"Found {len(telop_tracks)} telop tracks")

if len(telop_tracks) <= 1:
    logger.info("No optimization needed")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
else:
    # 各トラックからclipitemを抽出
    telop_clips = []
    for track in telop_tracks:
        # clipitemを抽出
        clipitems = re.findall(r'<clipitem[^>]*>.*?</clipitem>', track, re.DOTALL)
        for clipitem in clipitems:
            # start/endを抽出
            start_match = re.search(r'<start>(\d+)</start>', clipitem)
            end_match = re.search(r'<end>(\d+)</end>', clipitem)
            
            if start_match and end_match:
                telop_clips.append({
                    'xml': clipitem,
                    'start': int(start_match.group(1)),
                    'end': int(end_match.group(1))
                })
    
    logger.info(f"Extracted {len(telop_clips)} telop clips")
    
    # 開始時間でソート
    telop_clips.sort(key=lambda c: c['start'])
    
    # トラックに配置
    optimized_tracks = []
    track_end_times = []
    
    for clip in telop_clips:
        # 既存のトラックで時間が重ならないものを探す
        placed = False
        for track_idx, (track_clips, last_end) in enumerate(zip(optimized_tracks, track_end_times)):
            if clip['start'] >= last_end:
                # このトラックに配置可能
                track_clips.append(clip)
                track_end_times[track_idx] = clip['end']
                placed = True
                break
        
        if not placed:
            # 新しいトラックを作成
            optimized_tracks.append([clip])
            track_end_times.append(clip['end'])
    
    logger.info(f"Optimized to {len(optimized_tracks)} tracks")
    
    # 新しいトラックXMLを生成
    new_tracks_xml = []
    for track_idx, track_clips in enumerate(optimized_tracks):
        track_xml = '\t\t\t\t<track TL.SQTrackShy="0" TL.SQTrackExpandedHeight="25" TL.SQTrackExpanded="0" MZ.TrackTargeted="0">\n'
        for clip in track_clips:
            # インデントを調整
            clip_lines = clip['xml'].split('\n')
            adjusted_lines = []
            for line in clip_lines:
                # 既存のインデントを削除して新しいインデントを追加
                stripped = line.lstrip('\t')
                adjusted_lines.append('\t\t\t\t\t' + stripped)
            track_xml += '\n'.join(adjusted_lines) + '\n'
        track_xml += '\t\t\t\t\t<enabled>TRUE</enabled>\n'
        track_xml += '\t\t\t\t\t<locked>FALSE</locked>\n'
        track_xml += '\t\t\t\t</track>'
        new_tracks_xml.append(track_xml)
    
    # 元のテロップトラックを全て削除
    for track in telop_tracks:
        content = content.replace(track, '<<<TELOP_PLACEHOLDER>>>', 1)
    
    # プレースホルダーを新しいトラックに置き換え
    # 最初のプレースホルダーに全てのトラックを挿入
    first_placeholder = content.find('<<<TELOP_PLACEHOLDER>>>')
    if first_placeholder != -1:
        content = content.replace('<<<TELOP_PLACEHOLDER>>>', '\n'.join(new_tracks_xml), 1)
        # 残りのプレースホルダーを削除
        content = content.replace('<<<TELOP_PLACEHOLDER>>>', '')
    
    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"✅ Saved optimized XML to {output_file}")

# トラック数を確認
import xml.etree.ElementTree as ET
tree = ET.parse(output_file)
root = tree.getroot()
video_tracks = root.findall('.//video/track')
print(f"\n📊 Result:")
print(f"   Original tracks: {len(telop_tracks) + 1}")  # +1 for main video track
print(f"   Optimized tracks: {len(video_tracks)}")
print(f"   Reduction: {len(telop_tracks) + 1 - len(video_tracks)} tracks removed")
