#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
動画編集AI 学習実行ツール (v2)
- 入力: 7batch...で作ったJSONフォルダ (edit_triaining)
- 処理: 動画を解析して「音量」「動き」を抽出 + 編集点と突き合わせ
- 出力: 学習済みモデル (.pkl) + 分析レポート
"""

import argparse
import os
import glob
import json
import subprocess
import tempfile
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import librosa
import joblib  # モデル保存用
import opentimelineio as otio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

# ==========================================
# 設定
# ==========================================
WINDOW_SEC = 0.5  # 解析する窓の大きさ（秒）
FPS_BASE = 59.94  # 基準FPS

# ==========================================
# 特徴量抽出 (動画の実体解析)
# ==========================================

def extract_features(video_path: str):
    """動画から Audio RMS と Video Motion を抽出"""
    if not os.path.exists(video_path):
        return None

    # --- Audio RMS ---
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "temp.wav"
        # 高速化のためffmpegで音声のみ抽出
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "22050", "-vn", str(wav_path)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if not os.path.exists(wav_path): return None

        y, sr = librosa.load(str(wav_path), sr=None)
        hop_len = int(sr * WINDOW_SEC)
        if len(y) < hop_len: hop_len = len(y)
        
        rms = librosa.feature.rms(y=y, frame_length=hop_len, hop_length=hop_len)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_len)

    # --- Video Motion ---
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    motion_avgs = []
    
    # 処理軽量化のためリサイズ＆間引き
    step_frames = int(fps * WINDOW_SEC)
    if step_frames < 1: step_frames = 1
    
    prev_gray = None
    
    # フレームループ
    for i in range(0, total_frames, step_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 90)) # 画素数を減らして高速化
        
        val = 0.0
        if prev_gray is not None:
            # 前フレームとの差分平均を「動き量」とする
            val = float(np.mean(cv2.absdiff(gray, prev_gray)))
        
        motion_avgs.append(val)
        prev_gray = gray
        
    cap.release()

    # 長さを合わせる
    min_len = min(len(rms), len(motion_avgs))
    
    # DataFrame化
    df = pd.DataFrame({
        "time": times[:min_len],
        "audio_vol": rms[:min_len],
        "video_motion": motion_avgs[:min_len]
    })
    
    return df

# ==========================================
# 正解ラベル抽出 (JSON -> OTIO解析)
# ==========================================

def get_used_ranges_from_json(json_data):
    """保存されたOTIO JSONデータから採用区間リストを取得"""
    # OTIOオブジェクトに復元
    json_str = json.dumps(json_data["timeline_data"])
    timeline = otio.adapters.read_from_string(json_str, "otio_json")
    
    used_ranges = []
    if timeline.tracks:
        for track in timeline.tracks:
            if track.kind != otio.schema.TrackKind.Video: continue
            for item in track:
                if not isinstance(item, otio.schema.Clip): continue
                
                # Source Rangeがあれば採用とみなす
                if item.source_range:
                    s = item.source_range.start_time.value / FPS_BASE
                    e = s + (item.source_range.duration.value / FPS_BASE)
                    used_ranges.append((s, e))
    return used_ranges

# ==========================================
# データセット作成メイン
# ==========================================

def process_dataset(json_folder):
    json_files = glob.glob(os.path.join(json_folder, "*.json"))
    if not json_files:
        print("JSON file not found.")
        return None

    all_dataframes = []

    print(f"Loading {len(json_files)} datasets...")
    
    for jf in tqdm(json_files):
        with open(jf, "r", encoding="utf-8") as f:
            data_entry = json.load(f)
        
        meta = data_entry["meta"]
        video_paths = meta.get("video_paths", [])
        
        if not video_paths: continue
        
        # 簡易化のため、リストの最初の動画をメイン素材として解析する
        # (マルチカムの場合は工夫が必要だが、まずはメイン1本で学習)
        target_video = video_paths[0]
        
        if not os.path.exists(target_video):
            # print(f"Video missing: {target_video}")
            continue
            
        # 1. 特徴量抽出
        df = extract_features(target_video)
        if df is None or len(df) == 0: continue
        
        # 2. ラベル付け
        ranges = get_used_ranges_from_json(data_entry)
        
        # タイムスタンプごとに判定
        labels = []
        for t in df["time"]:
            is_used = 0
            for start, end in ranges:
                if start <= t < end:
                    is_used = 1
                    break
            labels.append(is_used)
        
        df["is_used"] = labels
        all_dataframes.append(df)

    if not all_dataframes:
        print("No valid training data generated.")
        return None

    # 全データを結合
    full_df = pd.concat(all_dataframes, ignore_index=True)
    return full_df

# ==========================================
# 学習と保存
# ==========================================

def train_and_save(df, output_model="editor_model.pkl"):
    print("\n--- Training Start ---")
    print(f"Total Samples: {len(df)}")
    print(f"Used Rate: {df['is_used'].mean():.2%}")
    
    X = df[["audio_vol", "video_motion"]]
    y = df["is_used"]
    
    # 分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 学習 (RandomForest)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # 評価
    score = clf.score(X_test, y_test)
    print(f"Model Accuracy: {score:.2%}")
    
    # 重要度表示
    imps = clf.feature_importances_
    print(f"Importance - Audio: {imps[0]:.3f}, Motion: {imps[1]:.3f}")
    
    # 保存
    joblib.dump(clf, output_model)
    print(f"Model saved to: {output_model}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder with training JSONs (edit_triaining)")
    parser.add_argument("-o", "--model", default="editor_model.pkl", help="Output model path")
    args = parser.parse_args()
    
    # 1. データセット作成
    df = process_dataset(args.folder)
    
    if df is not None:
        # (オプション) データCSV保存
        df.to_csv("combined_training_data.csv", index=False)
        print("Dataset CSV saved to combined_training_data.csv")
        
        # 2. 学習
        train_and_save(df, args.model)

if __name__ == "__main__":
    main()