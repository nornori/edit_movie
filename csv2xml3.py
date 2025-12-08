import pandas as pd
import opentimelineio as otio
import os
import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom
import cv2

# ==========================================
# 設定: 縦長シーケンス (1080 x 1920)
# ==========================================
SEQ_WIDTH = 1080
SEQ_HEIGHT = 1920

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = 30.0
    width = 1920
    height = 1080
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    return fps if fps > 0 else 30.0, width, height

def create_vertical_robust_otio(csv_path, video_path, output_xml_path):
    fps, src_w, src_h = get_video_info(video_path)
    print(f"🎬 Source: {src_w}x{src_h} @ {fps}fps")
    
    df = pd.read_csv(csv_path)
    
    # 1. OTIOで骨組み作成 (ここは任せる)
    timeline = otio.schema.Timeline(name="AI_Vertical_Final")
    track_v1 = otio.schema.Track(name="V1_Game", kind=otio.schema.TrackKind.Video)
    track_v2 = otio.schema.Track(name="V2_Face", kind=otio.schema.TrackKind.Video)
    track_a1 = otio.schema.Track(name="A1_Main", kind=otio.schema.TrackKind.Audio)
    timeline.tracks.extend([track_v1, track_v2, track_a1])

    abs_video_path = os.path.abspath(video_path)
    media_reference = otio.schema.ExternalReference(
        target_url=abs_video_path,
        available_range=otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(0, fps),
            duration=otio.opentime.RationalTime(1000000, fps)
        )
    )

    if len(df) > 1:
        interval = df.iloc[1]['time'] - df.iloc[0]['time']
    else:
        interval = 0.1

    face_effects_map = {} 
    clips_data = []
    current = None

    # データ結合処理
    for index, row in df.iterrows():
        time = row['time']
        is_keep = (row.get('decision') == 'KEEP') or (row.get('ai_cut_score', 0) > 0.5)
        
        scale = row.get('ai_scale', 100.0)
        pos_x = row.get('ai_pos_x', 0.0)
        pos_y = row.get('ai_pos_y', 0.0)
        has_face = (row.get('face_count', 0) > 0)
        face_x = row.get('face_center_x', 0.5)
        face_y = row.get('face_center_y', 0.5)
        face_size = row.get('face_size', 0.2)

        if is_keep:
            if current is None:
                current = {'start': time, 'end': time, 'scales': [scale], 'xs': [pos_x], 'ys': [pos_y], 'has_face': [has_face], 'fx': [face_x], 'fy': [face_y], 'fs': [face_size]}
            else:
                current['end'] = time
                current['scales'].append(scale)
                current['xs'].append(pos_x)
                current['ys'].append(pos_y)
                current['has_face'].append(has_face)
                current['fx'].append(face_x)
                current['fy'].append(face_y)
                current['fs'].append(face_size)
        else:
            if current:
                clips_data.append(current)
                current = None
    if current: clips_data.append(current)

    # クリップ生成
    for i, c in enumerate(clips_data):
        duration_sec = (c['end'] - c['start']) + interval
        start_frame = int(round(c['start'] * fps))
        duration_frame = int(round(duration_sec * fps))
        src_range = otio.opentime.TimeRange(start_time=otio.opentime.RationalTime(start_frame, fps), duration=otio.opentime.RationalTime(duration_frame, fps))

        # V1 Game
        clip_name_v1 = f"Game_{i}"
        clip_v1 = otio.schema.Clip(name=clip_name_v1, media_reference=media_reference, source_range=src_range)
        track_v1.append(clip_v1)
        
        # A1 Audio
        clip_a1 = otio.schema.Clip(name=f"Audio_{i}", media_reference=media_reference, source_range=src_range)
        track_a1.append(clip_a1)

        # V2 Face
        if sum(c['has_face']) / len(c['has_face']) > 0.5:
            clip_name_v2 = f"Face_{i}"
            clip_v2 = otio.schema.Clip(name=clip_name_v2, media_reference=media_reference, source_range=src_range)
            track_v2.append(clip_v2)

            avg_fx = sum(c['fx']) / len(c['fx'])
            avg_fy = sum(c['fy']) / len(c['fy'])
            avg_fs = sum(c['fs']) / len(c['fs'])
            margin = avg_fs * 0.8
            
            face_effects_map[clip_name_v2] = {
                'crop': (max(0, (avg_fx-margin)*100), max(0, (1.0-(avg_fx+margin))*100), max(0, (avg_fy-margin)*100), max(0, (1.0-(avg_fy+margin))*100)),
                'scale': 150, 'pos_x': 0.0, 'pos_y': 0.6 
            }

    # 4. OTIO書き出し (文字列として取得)
    print("Generating base XML via OTIO...")
    raw_xml = otio.adapters.write_to_string(timeline, adapter_name='fcp_xml')

    # ==============================================================================
    # 5. XML強制整形 (ここが修正のキモ)
    # ==============================================================================
    print(f"Force-Injecting Sequence Settings: {SEQ_WIDTH}x{SEQ_HEIGHT}")
    root = ET.fromstring(raw_xml)

    # 1. すべての sequence タグを探す
    for seq in root.findall(".//sequence"):
        # mediaタグを探す、なければ作る
        media = seq.find("media")
        if media is None: media = ET.SubElement(seq, "media")
        
        # videoタグを探す、なければ作る
        video = media.find("video")
        if video is None: video = ET.SubElement(media, "video")
        
        # formatタグを探す、なければ作る
        fmt = video.find("format")
        if fmt is None: fmt = ET.SubElement(video, "format")
        
        # samplecharacteristicsタグを探す、なければ作る
        sc = fmt.find("samplecharacteristics")
        if sc is None: sc = ET.SubElement(fmt, "samplecharacteristics")
        
        # ★ ここで中身を一旦全部消して、正しい設定を書き込む (一番確実)
        for child in list(sc):
            sc.remove(child)
            
        # 縦長設定を注入
        ET.SubElement(sc, "width").text = str(SEQ_WIDTH)
        ET.SubElement(sc, "height").text = str(SEQ_HEIGHT)
        ET.SubElement(sc, "pixelaspectratio").text = "square"
        
        # レート設定も入れる (入れないとPremiereがエラーになることがあるため)
        rate = ET.SubElement(sc, "rate")
        ET.SubElement(rate, "timebase").text = str(int(fps))
        ET.SubElement(rate, "ntsc").text = "TRUE"

    # 2. エフェクト注入 (これまで通り)
    clip_items = root.findall(".//clipitem")
    for item in clip_items:
        name_node = item.find("name")
        if name_node is None: continue
        name = name_node.text
        
        if name.startswith("Game_"):
            try:
                clip_idx = int(name.split('_')[1])
                data = clips_data[clip_idx]
                s = sum(data['scales']) / len(data['scales'])
                x = sum(data['xs']) / len(data['xs'])
                y = sum(data['ys']) / len(data['ys'])
                add_motion_xml(item, s, x, y)
            except: pass
            
        elif name in face_effects_map:
            eff = face_effects_map[name]
            cl, cr, ct, cb = eff['crop']
            add_crop_xml(item, cl, cr, ct, cb)
            add_motion_xml(item, eff['scale'], eff['pos_x'], eff['pos_y'])

    # 保存
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    with open(output_xml_path, "w", encoding="utf-8") as f:
        f.write(xml_str)
    print(f"✅ DONE! Saved to: {output_xml_path}")

def add_crop_xml(node, left, right, top, bottom):
    filt = ET.SubElement(node, 'filter')
    eff = ET.SubElement(filt, 'effect')
    ET.SubElement(eff, 'name').text = "Crop"
    ET.SubElement(eff, 'effectid').text = "Crop"
    params = {'left': left, 'right': right, 'top': top, 'bottom': bottom}
    for key, val in params.items():
        p = ET.SubElement(eff, 'parameter')
        ET.SubElement(p, 'parameterid').text = key
        ET.SubElement(p, 'name').text = key.capitalize()
        ET.SubElement(p, 'valuemin').text = "0"
        ET.SubElement(p, 'valuemax').text = "100"
        ET.SubElement(p, 'value').text = str(int(val))

def add_motion_xml(node, scale, x, y):
    filt = ET.SubElement(node, 'filter')
    eff = ET.SubElement(filt, 'effect')
    ET.SubElement(eff, 'name').text = "Basic Motion"
    ET.SubElement(eff, 'effectid').text = "basic"
    ps = ET.SubElement(eff, 'parameter')
    ET.SubElement(ps, 'parameterid').text = "scale"
    ET.SubElement(ps, 'name').text = "Scale"
    ET.SubElement(ps, 'value').text = str(int(scale))
    pc = ET.SubElement(eff, 'parameter')
    ET.SubElement(pc, 'parameterid').text = "center"
    ET.SubElement(pc, 'name').text = "Center"
    val = ET.SubElement(pc, 'value')
    ET.SubElement(val, 'horiz').text = str(x)
    ET.SubElement(val, 'vert').text = str(y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("video_path")
    parser.add_argument("--output", default="vertical_robust.xml")
    args = parser.parse_args()
    
    create_vertical_robust_otio(args.csv_path, args.video_path, args.output)