#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
学習データ量産ツール (Structure Edition v5 - Ultimate Attributes)
XMLを直接解析して、Premiere特有の「スケール」「位置」「回転」「不透明度」「音量」「クロップ」
などの詳細パラメータを抽出し、学習しやすい形式でJSONに追加保存する完全版。
"""

import argparse
import os
import glob
import json
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
import opentimelineio as otio
from tqdm import tqdm

# ==========================================
# 設定
# ==========================================
# 出力先のデフォルト（環境に合わせて書き換えてください）
DEFAULT_OUTPUT_DIR = r"C:\Users\yushi\Documents\プログラム\xmlai\edit_triaining"

VIDEO_EXTS = {".mp4", ".mov", ".mxf", ".mkv", ".avi", ".webm"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".psd", ".bmp", ".tiff", ".gif", ".ai"}
FPS_BASE = 59.94

# ==========================================
# ユーティリティ
# ==========================================
def clean_xml_path(raw_url: str) -> str:
    if not raw_url: return ""
    decoded_path = urllib.parse.unquote(raw_url)
    if decoded_path.startswith("file://localhost/"):
        decoded_path = decoded_path.replace("file://localhost/", "")
    elif decoded_path.startswith("file://"):
        decoded_path = decoded_path.replace("file://", "")
    elif decoded_path.startswith("file:"):
        decoded_path = decoded_path.replace("file:", "")
    
    if os.name == 'nt':
        if decoded_path.startswith("/") and len(decoded_path) > 2 and decoded_path[2] == ":":
            decoded_path = decoded_path.lstrip("/")
        decoded_path = decoded_path.replace("/", "\\")
    return decoded_path

# ==========================================
# 詳細パラメータ抽出ロジック (Core Logic)
# ==========================================
def extract_detailed_attributes(clipitem_node):
    """
    クリップのXMLノードから、エフェクトやモーション情報を辞書として抽出する
    """
    # デフォルト値
    attributes = {
        # Basic Motion
        "scale": 100.0,
        "pos_x": 0.5,
        "pos_y": 0.5,
        "rotation": 0.0,
        # Opacity
        "opacity": 100.0,
        # Audio
        "audio_level": 0.0, # dBなど（Premiereの内部値）
        # Crop
        "crop_left": 0.0,
        "crop_top": 0.0,
        "crop_right": 0.0,
        "crop_bottom": 0.0
    }

    # <filter> (エフェクト) を全走査
    for filter_node in clipitem_node.findall("filter"):
        effect_node = filter_node.find("effect")
        if effect_node is None: continue
        
        effect_id_node = effect_node.find("effectid")
        if effect_id_node is None: continue
        effect_id = effect_id_node.text.lower()

        # パラメータ走査
        for param in effect_node.findall("parameter"):
            pid_node = param.find("parameterid")
            if pid_node is None: continue
            pid = pid_node.text.lower()
            
            # 値の取得 (数値変換できるものだけ)
            val_node = param.find("value")
            val = None
            if val_node is not None:
                try: val = float(val_node.text)
                except: pass

            # --- 1. Basic Motion (スケール, 位置, 回転) ---
            if "basic" in effect_id:
                if pid == "scale" and val is not None: attributes["scale"] = val
                if pid == "rotation" and val is not None: attributes["rotation"] = val
                if pid == "center" and val_node is not None:
                    horiz = val_node.find("horiz")
                    vert = val_node.find("vert")
                    if horiz is not None:
                        try: attributes["pos_x"] = float(horiz.text)
                        except: pass
                    if vert is not None:
                        try: attributes["pos_y"] = float(vert.text)
                        except: pass
            
            # --- 2. Opacity (不透明度) ---
            elif "opacity" in effect_id:
                if pid == "opacity" and val is not None: attributes["opacity"] = val

            # --- 3. Audio Levels (音量) ---
            elif "audiolevels" in effect_id:
                if pid == "level" and val is not None:
                    attributes["audio_level"] = val 

            # --- 4. Crop (クロップ) ---
            # エフェクトIDに "crop" が含まれていればクロップエフェクトとみなす
            # (またはBasic Motion内にパラメータがある場合も考慮してパラメータ名で判定)
            if "crop" in effect_id or "crop" in pid:
                if val is not None:
                    if "left" in pid: attributes["crop_left"] = val
                    elif "top" in pid: attributes["crop_top"] = val
                    elif "right" in pid: attributes["crop_right"] = val
                    elif "bottom" in pid: attributes["crop_bottom"] = val

    return attributes

def extract_clip_details_from_xml(xml_file_path):
    """
    XML全体を走査し、全クリップの詳細情報をリスト化する
    """
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
    except:
        return []

    clip_details = []

    # シーケンス内の全トラック・全クリップを走査
    for sequence in root.iter("sequence"):
        media = sequence.find("media")
        if media is None: continue
        
        # Video と Audio 両方走査する
        for media_type in ["video", "audio"]:
            media_node = media.find(media_type)
            if media_node is None: continue

            for track_idx, track in enumerate(media_node.findall("track"), start=1):
                for clipitem in track.findall("clipitem"):
                    # 基本情報
                    name_node = clipitem.find("name")
                    name = name_node.text if name_node is not None else "Unknown"
                    
                    # 時間情報 (フレーム)
                    start_node = clipitem.find("start")
                    end_node = clipitem.find("end")
                    if start_node is None or end_node is None: continue
                    
                    start_frame = int(start_node.text)
                    end_frame = int(end_node.text)
                    if start_frame < 0: start_frame = 0
                    
                    # 秒換算
                    t_start = round(start_frame / FPS_BASE, 4)
                    t_end = round(end_frame / FPS_BASE, 4)

                    # ★詳細属性の抽出
                    attrs = extract_detailed_attributes(clipitem)

                    # データ結合
                    data = {
                        "track_kind": media_type, # video or audio
                        "track_index": track_idx,
                        "name": name,
                        "timeline_start": t_start,
                        "timeline_end": t_end,
                    }
                    data.update(attrs) # scale, opacity等を結合

                    clip_details.append(data)

    return clip_details

# ==========================================
# パス探索
# ==========================================
def find_all_media_paths(xml_file_path: str):
    xml_base_dir = os.path.dirname(os.path.abspath(xml_file_path))
    found_videos = []
    found_images = []
    
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
    except:
        return [], []

    seen_paths = set()
    for path_node in root.iter("pathurl"):
        raw_url = path_node.text
        if not raw_url: continue
        file_path = clean_xml_path(raw_url)
        
        if file_path in seen_paths: continue
        seen_paths.add(file_path)

        ext = os.path.splitext(file_path)[1].lower()
        valid_path = None
        
        if os.path.exists(file_path):
            valid_path = file_path
        else:
            local_path = os.path.join(xml_base_dir, os.path.basename(file_path))
            if os.path.exists(local_path):
                valid_path = local_path
        
        if valid_path:
            if ext in VIDEO_EXTS:
                found_videos.append(valid_path)
            elif ext in IMAGE_EXTS:
                found_images.append(valid_path)

    return found_videos, found_images

# ==========================================
# OTIOシリアライズ
# ==========================================
def serialize_timeline(timeline: otio.schema.Timeline):
    serialized_string = otio.adapters.write_to_string(timeline, adapter_name="otio_json")
    return json.loads(serialized_string)

# ==========================================
# 個別処理
# ==========================================
def process_structure(xml_path):
    video_paths, image_paths = find_all_media_paths(xml_path)
    if not video_paths: return None
    
    try:
        timeline = otio.adapters.read_from_file(xml_path)
    except: return None

    # ★詳細データ抽出
    details = extract_clip_details_from_xml(xml_path)

    dataset_entry = {
        "meta": {
            "xml_path": os.path.abspath(xml_path),
            "video_paths": [os.path.abspath(p) for p in video_paths],
            "image_paths": [os.path.abspath(p) for p in image_paths],
            "original_filename": os.path.basename(xml_path)
        },
        
        # ★ここが今回の目玉データ（学習用）
        "clip_details": details,
        
        # 念のためOTIO生データ
        "timeline_data": serialize_timeline(timeline)
    }
    
    return dataset_entry

# ==========================================
# メイン
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", help="Target folder containing XMLs")
    parser.add_argument("-o", "--output_dir", default=DEFAULT_OUTPUT_DIR, help="Output Directory")
    args = parser.parse_args()
    
    target_files = glob.glob(os.path.join(args.input_folder, "**", "*.xml"), recursive=True)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f"Found {len(target_files)} XML files. Processing with Ultimate Attributes...")
    
    success_count = 0
    for xml_path in tqdm(target_files):
        try:
            result = process_structure(xml_path)
            if result:
                success_count += 1
                filename = f"{success_count:02d}.json"
                save_path = os.path.join(args.output_dir, filename)
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
        except Exception as e:
            # print(e) # デバッグ時はコメントアウト解除
            pass

    print(f"\nDone! Saved {success_count} json files to: {args.output_dir}")

if __name__ == "__main__":
    main()