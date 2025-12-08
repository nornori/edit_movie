#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
動画編集AI 学習・分析ツール (Ver.6 - Raw XML Search Fix)
OTIOの解析に頼らず、XMLを直接スキャンして動画パスを確実に発見するバージョン。
"""

import argparse
import subprocess
import tempfile
import os
import sys
import urllib.parse
import xml.etree.ElementTree as ET  # 追加: 直接XMLを読む
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import librosa
import opentimelineio as otio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ==========================================
# 0. ユーティリティ (パス解析)
# ==========================================

VIDEO_EXTS = {".mp4", ".mov", ".mxf", ".mkv", ".avi", ".webm"}

def clean_xml_path(raw_url: str) -> str:
    """XML内のURL(file://...)をOSのパスに変換する (xml2test2.pyと同等のロジック)"""
    if not raw_url: return ""
    
    decoded_path = urllib.parse.unquote(raw_url)
    
    if decoded_path.startswith("file://localhost/"):
        decoded_path = decoded_path.replace("file://localhost/", "")
    elif decoded_path.startswith("file://"):
        decoded_path = decoded_path.replace("file://", "")
    elif decoded_path.startswith("file:"):
        decoded_path = decoded_path.replace("file:", "")
        
    if os.name == 'nt':
        # Windows: /D:/path -> D:/path
        if decoded_path.startswith("/") and len(decoded_path) > 2 and decoded_path[2] == ":":
            decoded_path = decoded_path.lstrip("/")
        decoded_path = decoded_path.replace("/", "\\")
        
    return decoded_path

def find_video_path_raw_xml(xml_file_path: str, xml_base_dir: str) -> str:
    """
    OTIOを使わず、XMLテキストを直接走査して動画パスを見つける。
    これが最も確実な方法。
    """
    print("[INFO] XML(テキスト)から元動画を検索中...")
    
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
    except Exception as e:
        print(f"  [ERROR] XML Parsing Failed: {e}")
        return None

    # すべての <pathurl> タグを検索
    # xmemlの構造上、すべてのファイル参照はここにある
    candidates = []
    
    for path_node in root.iter("pathurl"):
        raw_url = path_node.text
        if not raw_url: continue
        
        file_path = clean_xml_path(raw_url)
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in VIDEO_EXTS:
            if file_path not in candidates:
                candidates.append(file_path)
                print(f"  [候補発見] {file_path}")
                
                # A. 絶対パスチェック
                if os.path.exists(file_path):
                    print(f"    -> [OK] 見つかりました (絶対パス)")
                    return file_path
                
                # B. 同階層チェック
                filename = os.path.basename(file_path)
                local_path = os.path.join(xml_base_dir, filename)
                if os.path.exists(local_path):
                    print(f"    -> [OK] 見つかりました (XML同階層): {local_path}")
                    return local_path
                
                print("    -> [NG] 存在しません")

    print("\n[ERROR] 有効な動画ファイルが見つかりませんでした。")
    if not candidates:
        print("  XML内に <pathurl> タグで指定された動画ファイル(.mp4等)が一つもありません。")
    return None

# ==========================================
# 1. 特徴量抽出 (動画全体を時系列で取得)
# ==========================================

def extract_continuous_features(video_path: str, window_sec: float = 0.5):
    print(f"[INFO] 動画解析開始: {video_path}")
    
    # --- Audio RMS ---
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "temp.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "22050", "-vn", str(wav_path)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if not os.path.exists(wav_path):
            raise FileNotFoundError("音声抽出に失敗しました。ffmpegがインストールされているか確認してください。")

        y, sr = librosa.load(str(wav_path), sr=None)
        hop_len = int(sr * window_sec)
        if len(y) < hop_len: hop_len = len(y)
            
        rms = librosa.feature.rms(y=y, frame_length=hop_len, hop_length=hop_len)[0]
        times_audio = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_len)

    # --- Video Motion ---
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    motion_avgs = []
    step_frames = int(fps * window_sec)
    if step_frames < 1: step_frames = 1
    
    prev_gray = None
    
    print("[INFO] 映像モーション解析中...")
    for i in range(0, total_frames, step_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 90))
        
        val = 0.0
        if prev_gray is not None:
            val = float(np.mean(cv2.absdiff(gray, prev_gray)))
        
        motion_avgs.append(val)
        prev_gray = gray
        
    cap.release()

    min_len = min(len(rms), len(motion_avgs))
    
    data = []
    for i in range(min_len):
        data.append({
            "time_sec": float(times_audio[i]),
            "audio_vol": float(rms[i]),
            "video_motion": float(motion_avgs[i])
        })
        
    return pd.DataFrame(data)

# ==========================================
# 2. 正解ラベル付け (XMLからUsed判定)
# ==========================================

def get_used_ranges_from_timeline(timeline: otio.schema.Timeline, fps_base: float = 59.94) -> list:
    """タイムラインオブジェクトから採用区間リストを返す"""
    used_ranges = []
    
    if timeline.tracks:
        for track in timeline.tracks:
            if track.kind != otio.schema.TrackKind.Video: continue
            for item in track:
                if not isinstance(item, otio.schema.Clip): continue
                clip = item
                
                # 動画ファイルかどうか判定 (OTIO経由でmedia_referenceが見えない場合への対策)
                # 今回は単純に「Videoトラックに置かれた長さ」を採用とみなす
                if clip.source_range:
                    start_frame = clip.source_range.start_time.value
                    dur_frame = clip.source_range.duration.value
                    
                    s_start = start_frame / fps_base
                    s_end = (start_frame + dur_frame) / fps_base
                    used_ranges.append((s_start, s_end))
                
    return used_ranges

def label_data(df: pd.DataFrame, used_ranges: list) -> pd.DataFrame:
    labels = []
    for t in df["time_sec"]:
        is_used = 0
        for start, end in used_ranges:
            if start <= t < end:
                is_used = 1
                break
        labels.append(is_used)
    
    df["is_used"] = labels
    return df

# ==========================================
# 3. 学習と分析
# ==========================================

def train_and_analyze(df: pd.DataFrame):
    if df['is_used'].nunique() < 2:
        print("[ERROR] データに「採用(1)」または「不採用(0)」のどちらかが欠けています。")
        print(f"現在のラベル分布:\n{df['is_used'].value_counts()}")
        return None

    X = df[["audio_vol", "video_motion"]]
    y = df["is_used"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    score = clf.score(X_test, y_test)
    print("\n" + "="*30)
    print(f"AI予測精度 (Accuracy): {score:.2%}")
    print("="*30)
    
    importances = clf.feature_importances_
    print("\n[編集判断の重要度分析]")
    print(f"  - 音量 (Audio): {importances[0]:.4f}")
    print(f"  - 動き (Motion): {importances[1]:.4f}")
    
    if importances[0] > importances[1]:
        print(">> 結論: この編集は「音量（喋り）」を基準にカットしています。")
    else:
        print(">> 結論: この編集は「映像の動き」を基準にカットしています。")

    print("\n[採用/不採用エリアの平均値]")
    stats = df.groupby("is_used")[["audio_vol", "video_motion"]].mean()
    print(stats)
    
    return clf

# ==========================================
# Main
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("xml", help="Input Premiere XML (.xml)")
    args = parser.parse_args()
    
    xml_path = args.xml
    xml_base_dir = os.path.dirname(os.path.abspath(xml_path))
    
    if not os.path.exists(xml_path):
        print(f"Error: XMLファイルが見つかりません: {xml_path}")
        sys.exit(1)

    # 1. パス検索 (Raw XML)
    video_path = find_video_path_raw_xml(xml_path, xml_base_dir)
    
    if not video_path:
        sys.exit(1)
        
    print(f"[SUCCESS] 解析対象の動画を特定しました: {video_path}")

    # 2. XML読み込み (OTIO for Logic)
    try:
        timeline = otio.adapters.read_from_file(xml_path)
    except Exception as e:
        print(f"Error reading XML with OTIO: {e}")
        sys.exit(1)

    df = extract_continuous_features(video_path)
    ranges = get_used_ranges_from_timeline(timeline)
    df = label_data(df, ranges)
    
    print(f"\nデータセット生成完了: {len(df)} サンプル")
    print(f"動画全体の採用率: {df['is_used'].mean():.2%}")
    
    model = train_and_analyze(df)
    
    out_csv = "training_result.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n詳細データは {out_csv} に保存されました。")
    