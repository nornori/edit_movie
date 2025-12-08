#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
学習データ補完ツール (Decoder Only)
既存のJSONデータを読み込み、XML内に隠された「エッセンシャルグラフィックスのテキスト」のみを
バイナリデコードで発掘して追記する軽量・高速スクリプト。
"""

import argparse
import os
import glob
import json
import base64
import zlib
import re
import xml.etree.ElementTree as ET
from tqdm import tqdm

# ==========================================
# 設定
# ==========================================
FPS_BASE = 59.94

# ==========================================
# XML Decoder (隠しテキスト発掘ロジック)
# ==========================================
def extract_hidden_text_from_binary(binary_data):
    """バイナリデータから意味のある文字列パターンを探す"""
    found_texts = []
    
    # ノイズ除去フィルタ
    def is_valid_text(s):
        if len(s) < 2: return False
        if s.startswith("xml") or s.startswith("Adobe") or "uuid" in s: return False
        if re.match(r'^[a-zA-Z0-9\s\.,_]+$', s): # 英数字だけなら4文字以上
            return len(s) > 3
        return True

    # UTF-8 でトライ
    try:
        text = binary_data.decode('utf-8', errors='ignore')
        # 日本語・英語・記号を含む文字列を抽出
        candidates = re.findall(r'[\w\u3040-\u30FF\u4E00-\u9FFF\uFF66-\uFF9F\?!\(\)\[\]]+', text)
        for c in candidates:
            if is_valid_text(c): found_texts.append(c)
    except: pass
    
    # UTF-16 でトライ (Premiereは内部UTF-16が多い)
    try:
        text = binary_data.decode('utf-16', errors='ignore')
        candidates = re.findall(r'[\w\u3040-\u30FF\u4E00-\u9FFF\uFF66-\uFF9F\?!\(\)\[\]]+', text)
        for c in candidates:
            if is_valid_text(c): found_texts.append(c)
    except: pass

    return list(set(found_texts))

def hunt_text_in_clipitem(clipitem_node):
    """クリップ内の全パラメータからBase64隠しテキストを探す"""
    extracted_texts = []

    # A. Filter/Effect内のパラメータを走査
    for param in clipitem_node.findall(".//parameter"):
        val_node = param.find("value")
        if val_node is None or val_node.text is None: continue
        
        raw_text = val_node.text.strip()
        
        # Base64っぽいもの (空白が含まれず、ある程度長い)
        if len(raw_text) > 40 and " " not in raw_text:
            try:
                # 1. Base64デコード
                decoded_bytes = base64.b64decode(raw_text)
                
                # 2. Zlib解凍トライ (失敗したらそのまま使う)
                try:
                    data = zlib.decompress(decoded_bytes)
                except:
                    data = decoded_bytes
                
                # 3. テキスト抽出
                texts = extract_hidden_text_from_binary(data)
                extracted_texts.extend(texts)
            except: pass

    # B. SourceDataタグ (Legacy Titleなどはここにある場合も)
    for source_data in clipitem_node.findall(".//sourcedata"):
        if source_data.text:
            try:
                decoded = base64.b64decode(source_data.text)
                texts = extract_hidden_text_from_binary(decoded)
                extracted_texts.extend(texts)
            except: pass

    return list(set(extracted_texts))

# ==========================================
# メイン処理
# ==========================================
def process_single_json(json_path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # XMLパスの取得
        xml_path = data.get("meta", {}).get("xml_path", "")
        if not xml_path or not os.path.exists(xml_path):
            return False

        # クリップリストの取得
        clips = data.get("ground_truth_clips", [])
        if not clips: clips = data.get("clip_details", [])
        if not clips: return False

        changed = False

        # XMLをパース
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # XML構造を走査して、JSON内のクリップとマッチングさせる
            for seq in root.iter("sequence"):
                media = seq.find("media")
                if not media: continue
                
                for mtype in ["video", "audio"]:
                    mnode = media.find(mtype)
                    if not mnode: continue
                    
                    for trk_idx, track in enumerate(mnode.findall("track"), start=1):
                        for clipitem in track.findall("clipitem"):
                            # 開始時間で特定
                            start_node = clipitem.find("start")
                            if start_node is None: continue
                            
                            start_frame = int(start_node.text)
                            if start_frame < 0: start_frame = 0
                            t_start = round(start_frame / FPS_BASE, 4)
                            
                            # ★隠しテキスト発掘★
                            hidden_texts = hunt_text_in_clipitem(clipitem)
                            
                            if hidden_texts:
                                # JSON側で該当するクリップを探す
                                for d_clip in clips:
                                    # トラック種別、番号、時間が一致
                                    if (d_clip.get("track_kind") == mtype and 
                                        d_clip.get("track_index") == trk_idx and 
                                        abs(d_clip.get("timeline_start", -1) - t_start) < 0.05):
                                        
                                        # すでに同じデータが入っていれば更新しない
                                        if d_clip.get("decoded_text") != hidden_texts:
                                            d_clip["decoded_text"] = hidden_texts
                                            changed = True
                                        break
        except Exception:
            return False

        # 変更があれば保存
        if changed:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=None, ensure_ascii=False)
            return True
        else:
            return False

    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder containing existing JSON files")
    args = parser.parse_args()
    
    # JSONファイル一覧取得
    json_files = glob.glob(os.path.join(args.folder, "*.json"))
    print(f"Found {len(json_files)} JSON files. Scanning for hidden text...")

    count = 0
    # プログレスバー付きで実行
    for jf in tqdm(json_files):
        if process_single_json(jf):
            count += 1
            
    print(f"\nDone! Updated {count} files with decoded text data.")

if __name__ == "__main__":
    main()