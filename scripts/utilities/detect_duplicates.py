"""
重複しているフレーム範囲を検出
"""
import xml.etree.ElementTree as ET

xml_path = "outputs/test_no_duplicates.xml"

# XMLを読み込み
tree = ET.parse(xml_path)
root = tree.getroot()

# ビデオトラックのclipitemを取得
video_track = root.find(".//video/track")
clipitems = video_track.findall("clipitem")

print("="*80)
print("重複フレーム検出")
print("="*80)

# 各クリップの使用範囲を記録
clips = []
for idx, clipitem in enumerate(clipitems, 1):
    in_frame = int(clipitem.find("in").text)
    out_frame = int(clipitem.find("out").text)
    if out_frame > in_frame:  # duration > 0のみ
        clips.append({
            'id': idx,
            'in': in_frame,
            'out': out_frame
        })

# 重複を検出
overlaps = []
for i in range(len(clips)):
    for j in range(i + 1, len(clips)):
        clip1 = clips[i]
        clip2 = clips[j]
        
        # 重複範囲を計算
        overlap_start = max(clip1['in'], clip2['in'])
        overlap_end = min(clip1['out'], clip2['out'])
        
        if overlap_start < overlap_end:
            overlaps.append({
                'clip1': clip1['id'],
                'clip2': clip2['id'],
                'range1': (clip1['in'], clip1['out']),
                'range2': (clip2['in'], clip2['out']),
                'overlap': (overlap_start, overlap_end),
                'overlap_frames': overlap_end - overlap_start
            })

if overlaps:
    print(f"⚠️  {len(overlaps)}個の重複が見つかりました\n")
    
    for idx, overlap in enumerate(overlaps[:20], 1):  # 最初の20個だけ表示
        print(f"重複 {idx}:")
        print(f"  クリップ {overlap['clip1']}: {overlap['range1'][0]}～{overlap['range1'][1]}フレーム")
        print(f"  クリップ {overlap['clip2']}: {overlap['range2'][0]}～{overlap['range2'][1]}フレーム")
        print(f"  重複範囲: {overlap['overlap'][0]}～{overlap['overlap'][1]}フレーム ({overlap['overlap_frames']}フレーム)")
        print()
    
    if len(overlaps) > 20:
        print(f"... 他 {len(overlaps) - 20}個の重複")
    
    # 統計
    total_overlap_frames = sum(o['overlap_frames'] for o in overlaps)
    print(f"合計重複フレーム数: {total_overlap_frames}")
else:
    print("✅ 重複は見つかりませんでした")
