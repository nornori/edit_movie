"""
タイムライン形式で採用/不採用を視覚化
"""
import xml.etree.ElementTree as ET

xml_path = "outputs/test_full_pipeline_v2.xml"

# XMLを読み込み
tree = ET.parse(xml_path)
root = tree.getroot()

# ビデオトラックのclipitemを取得
video_track = root.find(".//video/track")
clipitems = video_track.findall("clipitem")

# 元動画のフレーム範囲を追跡
used_ranges = []
for clipitem in clipitems:
    in_frame = int(clipitem.find("in").text)
    out_frame = int(clipitem.find("out").text)
    if out_frame > in_frame:  # duration > 0のみ
        used_ranges.append((in_frame, out_frame))

used_ranges.sort()

# 元動画の総フレーム数
total_frames = int(root.find(".//sequence/duration").text)

# タイムラインを作成（1秒ごと）
print("="*80)
print("タイムライン（1秒ごと）")
print("="*80)
print("✅ = 採用  ❌ = カット")
print()

fps = 59.98
timeline_length = int(total_frames / fps) + 1

for sec in range(timeline_length):
    start_frame = int(sec * fps)
    end_frame = int((sec + 1) * fps)
    
    # この1秒間に採用されたフレームの割合を計算
    used_count = 0
    for frame in range(start_frame, end_frame):
        for used_start, used_end in used_ranges:
            if used_start <= frame < used_end:
                used_count += 1
                break
    
    ratio = used_count / (end_frame - start_frame)
    
    # 視覚化
    if ratio > 0.8:
        symbol = "✅✅✅"
    elif ratio > 0.5:
        symbol = "✅✅ "
    elif ratio > 0.2:
        symbol = "✅  "
    else:
        symbol = "❌❌❌"
    
    print(f"{sec:3d}秒: {symbol} ({ratio*100:5.1f}%)")

print()
print("="*80)
