import pandas as pd
import argparse
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import cv2

def get_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = 30.0
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
    return fps if fps > 0 else 30.0

def create_dynamic_xml(csv_path, video_path, output_xml_path):
    fps = get_fps(video_path)
    df = pd.read_csv(csv_path)
    
    # 時間間隔の推定
    interval = df.iloc[1]['time'] - df.iloc[0]['time'] if len(df) > 1 else 0.1

    # ==========================================
    # 1. すべての「要素」を抽出してリスト化する
    # ==========================================
    # ここでは「ベース動画」「画像要素」「テキスト要素」を全部フラットなリストにします
    
    elements = [] # {'start': 0.0, 'end': 1.0, 'type': 'video', 'content': ...}
    
    current_video = None
    current_broll = None
    current_text = None
    
    for index, row in df.iterrows():
        time = row['time']
        
        # --- ベース動画 (V1固定) ---
        is_keep = (row.get('decision') == 'KEEP') or (row.get('ai_cut_score', 0) > 0.5)
        scale = row.get('ai_scale', 100.0)
        pos_x = row.get('ai_pos_x', 0.0)
        pos_y = row.get('ai_pos_y', 0.0)
        
        if is_keep:
            if current_video is None:
                current_video = {'start': time, 'end': time, 'type': 'main_video', 
                                 'scales': [scale], 'xs': [pos_x], 'ys': [pos_y]}
            else:
                current_video['end'] = time
                current_video['scales'].append(scale)
                current_video['xs'].append(pos_x)
                current_video['ys'].append(pos_y)
        else:
            if current_video:
                elements.append(current_video)
                current_video = None

        # --- AI要素 (B-Roll) ---
        # 閾値を超えたら「要素」として認識
        if row.get('ai_broll', 0) > 0.6:
            if current_broll is None:
                current_broll = {'start': time, 'end': time, 'type': 'broll', 'content': 'placeholder'}
            else:
                current_broll['end'] = time
        else:
            if current_broll:
                elements.append(current_broll)
                current_broll = None

        # --- AI要素 (Text) ---
        # テキストがあり、かつAIが字幕必要と判断した場合
        text_content = str(row.get('text', ''))
        has_text = (text_content != "nan" and len(text_content) > 0)
        
        if has_text and row.get('ai_graphic', 0) > 0.5:
            if current_text is None:
                current_text = {'start': time, 'end': time, 'type': 'text', 'texts': [text_content]}
            else:
                current_text['end'] = time
                current_text['texts'].append(text_content)
        else:
            if current_text:
                elements.append(current_text)
                current_text = None

    # ループ後の残り処理
    if current_video: elements.append(current_video)
    if current_broll: elements.append(current_broll)
    if current_text: elements.append(current_text)

    # ==========================================
    # 2. 動的トラック割り当て (パッキングアルゴリズム)
    # ==========================================
    # V1はメイン動画専用として予約。V2以降を動的に使う。
    tracks = {1: []} # {track_id: [end_time_of_last_clip]}
    
    # 処理しやすいように開始時間順にソート
    elements.sort(key=lambda x: x['start'])
    
    print(f"Assigning tracks for {len(elements)} elements...")
    
    for el in elements:
        duration = (el['end'] - el['start']) + interval
        el['duration_sec'] = duration
        
        # ベース動画は絶対にV1
        if el['type'] == 'main_video':
            el['track_id'] = 1
            tracks[1].append(el['end']) # V1の末尾時間を更新
            continue
            
        # それ以外（画像やテキスト）は「空いている一番下のトラック」を探す
        assigned_track = -1
        
        # V2から順にチェック (最大V10まで探索)
        for t_id in range(2, 10):
            if t_id not in tracks:
                tracks[t_id] = []
                
            # このトラックの既存クリップと被っていないかチェック
            is_overlap = False
            for existing_end_time in tracks[t_id]:
                # 簡易判定: 既存の最後のクリップの終了時間よりも、今回の開始時間が後ならOK
                # ※厳密には全てのクリップと比較すべきですが、今回は簡易的にリスト管理
                pass
            
            # もっと厳密な衝突判定
            # このトラックに既に配置されている全要素を取得して比較するのは重いので
            # 「トラックごとの専有時間リスト」を持つのが正解
            
            # 今回はシンプルに: tracks[t_id] には (start, end) のタプルを入れる
            can_put_here = True
            for placed_start, placed_end in tracks[t_id]:
                # 被り判定: (StartA < EndB) and (EndA > StartB)
                if (el['start'] < placed_end) and (el['end'] > placed_start):
                    can_put_here = False
                    break
            
            if can_put_here:
                assigned_track = t_id
                tracks[t_id].append((el['start'], el['end']))
                break
        
        el['track_id'] = assigned_track

    # ==========================================
    # 3. XML生成
    # ==========================================
    root = ET.Element('xmeml', version="4")
    seq = ET.SubElement(root, 'sequence')
    ET.SubElement(seq, 'name').text = "AI_Dynamic_Layout"
    
    rate = ET.SubElement(seq, 'rate')
    ET.SubElement(rate, 'timebase').text = str(int(fps))
    ET.SubElement(rate, 'ntsc').text = "TRUE"
    
    media = ET.SubElement(seq, 'media')
    video = ET.SubElement(media, 'video')
    
    fmt = ET.SubElement(video, 'format')
    sc = ET.SubElement(fmt, 'samplecharacteristics')
    ET.SubElement(sc, 'width').text = "1920"
    ET.SubElement(sc, 'height').text = "1080"

    # 必要なトラック数分だけ作成
    max_track = max(tracks.keys())
    xml_tracks = {}
    for t_id in range(1, max_track + 1):
        xml_tracks[t_id] = ET.SubElement(video, 'track')

    # ファイル参照ノード
    file_node = ET.Element('file', id="file-1")
    ET.SubElement(file_node, 'name').text = os.path.basename(video_path)
    ET.SubElement(file_node, 'pathurl').text = "file://localhost/" + os.path.abspath(video_path).replace("\\", "/")
    ET.SubElement(file_node, 'rate').append(rate)

    # クリップ配置
    for i, el in enumerate(elements):
        t_id = el['track_id']
        if t_id == -1: continue # 配置場所がなかった（エラー）
        
        start_f = int(el['start'] * fps)
        end_f = int((el['end'] + interval) * fps)
        duration_f = end_f - start_f
        
        # タイムライン上の位置（ベース動画は詰める、それ以外は絶対時間）
        # ※ここが重要: マルチトラックXMLの場合、通常は「絶対時間配置」か「シーケンシャル配置」か選ぶ必要がある
        # V1（ベース）はカット編集なので時間が詰まる。V2以降は元のタイムコードに同期させる必要がある。
        
        # 簡易実装: V1は「詰める」。V2以降は「V1の現在の時間」に合わせて配置する必要があるが、
        # AIの出力が「元動画の絶対時間」なので、カット編集後の時間へのマッピング計算が必要。
        
        # ★ マッピング計算 ★
        # 元動画の時間(original_time) -> 編集後のタイムライン時間(timeline_time) への変換マップを作る必要があります
        # しかし今回は複雑になるため、V1に合わせて「V2以降も同じように詰める」ロジックにします
        # (AIがカットした部分にあるテロップは、当然消えるべきなので、これで合います)
        
        # 修正: 全要素に対して「累積デュレーション」を計算するのはV1だけ。
        # V2以降は「V1のどのクリップに対応するか」で位置が決まる。
        pass 

    # ★再設計: シンプルなループで書き出す
    # V1（メイン）を基準にタイムライン時間を進める
    
    timeline_cursor = 0
    # V1要素だけを先に抽出して、タイムライン時間を確定させる
    v1_elements = [e for e in elements if e['type'] == 'main_video']
    v1_elements.sort(key=lambda x: x['start'])
    
    # マッピングテーブル: { 元動画の開始時間: タイムラインの開始時間 }
    time_map = {}
    
    for v1_el in v1_elements:
        dur = int((v1_el['end'] - v1_el['start'] + interval) * fps)
        time_map[v1_el['start']] = timeline_cursor
        timeline_cursor += dur

    # 全要素の書き出し
    for i, el in enumerate(elements):
        t_id = el['track_id']
        track_node = xml_tracks[t_id]
        
        # この要素が「どのV1クリップに属しているか」を探して、タイムライン時間を計算
        # 近似的な開始時間を持つV1クリップを探す
        closest_start = min(time_map.keys(), key=lambda x: abs(x - el['start']))
        
        # もし乖離が大きすぎる（カットされた部分のテロップなど）場合はスキップ
        if abs(closest_start - el['start']) > 1.0:
            continue
            
        timeline_start = time_map[closest_start]
        # V1とのズレ（オフセット）を加算
        offset = int((el['start'] - closest_start) * fps)
        timeline_in = timeline_start + offset
        
        src_in = int(el['start'] * fps)
        src_out = int((el['end'] + interval) * fps)
        duration = src_out - src_in
        timeline_out = timeline_in + duration
        
        clipitem = ET.SubElement(track_node, 'clipitem', id=f"clip-{i}")
        ET.SubElement(clipitem, 'name').text = f"{el['type']}"
        ET.SubElement(clipitem, 'duration').text = str(duration)
        ET.SubElement(clipitem, 'start').text = str(timeline_in)
        ET.SubElement(clipitem, 'end').text = str(timeline_out)
        
        # タイプ別処理
        if el['type'] == 'main_video':
            ET.SubElement(clipitem, 'in').text = str(src_in)
            ET.SubElement(clipitem, 'out').text = str(src_out)
            clipitem.append(file_node)
            
            # エフェクト (平均)
            avg_s = sum(el['scales']) / len(el['scales'])
            avg_x = sum(el['xs']) / len(el['xs'])
            avg_y = sum(el['ys']) / len(el['ys'])
            add_motion_effect(clipitem, avg_s, avg_x, avg_y)
            
        elif el['type'] == 'text':
            # 最頻出テキスト
            txts = el['texts']
            final_txt = max(set(txts), key=txts.count) if txts else "Text"
            ET.SubElement(clipitem, 'name').text = final_txt
            
            gen = ET.SubElement(clipitem, 'generator')
            ET.SubElement(gen, 'effectid').text = "Text"
            ET.SubElement(gen, 'name').text = "Text Generator"
            p = ET.SubElement(gen, 'parameter')
            ET.SubElement(p, 'parameterid').text = "str"
            ET.SubElement(p, 'value').text = final_txt
            
        elif el['type'] == 'broll':
            # プレースホルダー
            ET.SubElement(clipitem, 'name').text = "Image_PlaceHolder"
            ET.SubElement(clipitem, 'file', id="slug")

    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    with open(output_xml_path, "w", encoding="utf-8") as f:
        f.write(xml_str)
    print(f"✅ Created Dynamic XML: {output_xml_path}")

def add_motion_effect(node, s, x, y):
    filt = ET.SubElement(node, 'filter')
    eff = ET.SubElement(filt, 'effect')
    ET.SubElement(eff, 'name').text = "Basic Motion"
    ET.SubElement(eff, 'effectid').text = "basic"
    
    ps = ET.SubElement(eff, 'parameter')
    ET.SubElement(ps, 'parameterid').text = "scale"
    ET.SubElement(ps, 'value').text = str(int(s))
    
    pc = ET.SubElement(eff, 'parameter')
    ET.SubElement(pc, 'parameterid').text = "center"
    val = ET.SubElement(pc, 'value')
    ET.SubElement(val, 'horiz').text = str(x)
    ET.SubElement(val, 'vert').text = str(y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("video_path")
    parser.add_argument("--output", default="dynamic_timeline.xml")
    args = parser.parse_args()
    create_dynamic_xml(args.csv_path, args.video_path, args.output)