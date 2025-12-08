#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
学習データ量産ツール (Structure Edition v4 - Video & Image Support)
動画パス(video_paths)と画像パス(image_paths)を分けてmetaに保存する完全版。
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
DEFAULT_OUTPUT_DIR = r"C:\Users\yushi\Documents\プログラム\xmlai\edit_triaining"

VIDEO_EXTS = {".mp4", ".mov", ".mxf", ".mkv", ".avi", ".webm"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".psd", ".bmp", ".tiff", ".gif", ".ai"} # 画像拡張子を追加

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

def find_all_media_paths(xml_file_path: str):
    """
    XML内のすべてのメディアパスを「動画」と「画像」に分けてリストで返す
    Returns: (video_paths_list, image_paths_list)
    """
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
        
        # 実在チェックロジック
        valid_path = None
        if os.path.exists(file_path):
            valid_path = file_path
        else:
            filename = os.path.basename(file_path)
            local_path = os.path.join(xml_base_dir, filename)
            if os.path.exists(local_path):
                valid_path = local_path
        
        # 分類して追加
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
    # 1. メディアパスの特定 (動画と画像を両方取得)
    video_paths, image_paths = find_all_media_paths(xml_path)
    
    if not video_paths:
        # 動画がない場合は学習に使えないためスキップ
        return None
    
    # 2. XML読み込み
    try:
        timeline = otio.adapters.read_from_file(xml_path)
    except Exception:
        return None

    # 3. データセット構築
    dataset_entry = {
        "meta": {
            "xml_path": os.path.abspath(xml_path),
            "video_paths": [os.path.abspath(p) for p in video_paths], # 動画リスト
            "image_paths": [os.path.abspath(p) for p in image_paths], # 画像リスト(New!)
            "original_filename": os.path.basename(xml_path)
        },
        "timeline_data": serialize_timeline(timeline)
    }
    
    return dataset_entry

# ==========================================
# メイン処理
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", help="Target folder containing XMLs")
    parser.add_argument("-o", "--output_dir", default=DEFAULT_OUTPUT_DIR, help="Output Directory")
    args = parser.parse_args()
    
    target_files = glob.glob(os.path.join(args.input_folder, "**", "*.xml"), recursive=True)
    if not target_files:
        print("XML file not found.")
        return

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f"Found {len(target_files)} XML files. Processing...")
    
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
                    
        except Exception:
            pass

    print(f"\nDone! Saved {success_count} json files to:")
    print(f"  -> {args.output_dir}")

if __name__ == "__main__":
    main()