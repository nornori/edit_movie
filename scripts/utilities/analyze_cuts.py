"""
XMLファイルを解析して、採用/不採用の部分を表示
"""
import xml.etree.ElementTree as ET

xml_path = "outputs/test_full_pipeline_v2.xml"

# XMLを読み込み
tree = ET.parse(xml_path)
root = tree.getroot()

# ビデオトラックのclipitemを取得
video_track = root.find(".//video/track")
clipitems = video_track.findall("clipitem")

print("="*80)
print("採用/不採用の分析")
print("="*80)
print(f"総クリップ数: {len(clipitems)}")
print()

# 元動画のフレーム範囲を追跡
used_ranges = []
for idx, clipitem in enumerate(clipitems, 1):
    in_frame = int(clipitem.find("in").text)
    out_frame = int(clipitem.find("out").text)
    start_frame = int(clipitem.find("start").text)
    end_frame = int(clipitem.find("end").text)
    duration = out_frame - in_frame
    
    used_ranges.append((in_frame, out_frame))
    
    # 秒に変換（59.98fps）
    in_sec = in_frame / 59.98
    out_sec = out_frame / 59.98
    duration_sec = duration / 59.98
    
    print(f"✅ クリップ {idx}: 採用")
    print(f"   元動画: {in_frame}～{out_frame}フレーム ({in_sec:.2f}～{out_sec:.2f}秒)")
    print(f"   長さ: {duration}フレーム ({duration_sec:.2f}秒)")
    print(f"   タイムライン: {start_frame}～{end_frame}フレーム")
    print()

# 不採用の部分を計算
print("="*80)
print("❌ 不採用（カットされた部分）")
print("="*80)

# 元動画の総フレーム数を取得
total_frames = int(root.find(".//sequence/duration").text)
print(f"元動画の総フレーム数: {total_frames} ({total_frames/59.98:.2f}秒)")
print()

# 使用されていない範囲を計算
used_ranges.sort()
cut_ranges = []

# 最初のクリップの前
if used_ranges[0][0] > 0:
    cut_ranges.append((0, used_ranges[0][0]))

# クリップ間のギャップ
for i in range(len(used_ranges) - 1):
    gap_start = used_ranges[i][1]
    gap_end = used_ranges[i + 1][0]
    if gap_end > gap_start:
        cut_ranges.append((gap_start, gap_end))

# 最後のクリップの後
if used_ranges[-1][1] < total_frames:
    cut_ranges.append((used_ranges[-1][1], total_frames))

if cut_ranges:
    for idx, (start, end) in enumerate(cut_ranges, 1):
        duration = end - start
        start_sec = start / 59.98
        end_sec = end / 59.98
        duration_sec = duration / 59.98
        
        print(f"カット {idx}:")
        print(f"   元動画: {start}～{end}フレーム ({start_sec:.2f}～{end_sec:.2f}秒)")
        print(f"   長さ: {duration}フレーム ({duration_sec:.2f}秒)")
        print()
else:
    print("カットされた部分はありません（全て採用）")

# 統計
total_used = sum(end - start for start, end in used_ranges)
total_cut = sum(end - start for start, end in cut_ranges)

print("="*80)
print("統計")
print("="*80)
print(f"採用: {total_used}フレーム ({total_used/59.98:.2f}秒) - {total_used/total_frames*100:.1f}%")
print(f"不採用: {total_cut}フレーム ({total_cut/59.98:.2f}秒) - {total_cut/total_frames*100:.1f}%")
print(f"合計: {total_frames}フレーム ({total_frames/59.98:.2f}秒)")
