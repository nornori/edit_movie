"""
テロップトラックを最適化して統合
時間が重ならないテロップを同じトラックにまとめる
"""
import re
import sys
import logging

logger = logging.getLogger(__name__)


def optimize_telop_tracks(input_xml: str, output_xml: str):
    """
    XMLのテロップトラックを最適化
    時間が重ならないテロップを同じトラックにまとめる
    
    Args:
        input_xml: 入力XMLファイルのパス
        output_xml: 出力XMLファイルのパス
    """
    with open(input_xml, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # file-2を参照しているtrackを検出（グラフィックトラック）
    telop_tracks = re.findall(r'<track[^>]*>.*?<file id="file-2".*?</track>', content, re.DOTALL)
    
    if len(telop_tracks) <= 1:
        # 最適化不要
        logger.info(f"  No optimization needed ({len(telop_tracks)} telop tracks)")
        with open(output_xml, 'w', encoding='utf-8') as f:
            f.write(content)
        return
    
    logger.info(f"  Found {len(telop_tracks)} telop tracks to optimize")
    
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
    
    logger.info(f"  Optimized to {len(optimized_tracks)} tracks")
    
    # 新しいトラックXMLを生成
    new_tracks_xml = []
    for track_clips in optimized_tracks:
        track_xml = '\t\t\t\t<track>\n'
        for clip in track_clips:
            # インデントを調整
            clip_xml = clip['xml'].replace('\n\t\t\t\t\t', '\n\t\t\t\t\t')
            track_xml += '\t' + clip_xml + '\n'
        track_xml += '\t\t\t\t</track>'
        new_tracks_xml.append(track_xml)
    
    # 元のテロップトラックを削除
    for track in telop_tracks:
        content = content.replace(track, '', 1)
    
    # 新しいトラックを挿入（最初のvideoトラックの後）
    video_track_end = content.find('</track>', content.find('<video>'))
    if video_track_end != -1:
        insert_pos = video_track_end + len('</track>')
        content = content[:insert_pos] + '\n' + '\n'.join(new_tracks_xml) + content[insert_pos:]
    
    # 保存
    with open(output_xml, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"  Output: {output_xml}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python optimize_telop_tracks.py <input> <output>")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO)
    optimize_telop_tracks(sys.argv[1], sys.argv[2])
