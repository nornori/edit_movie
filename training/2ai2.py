#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
動画編集AI 学習実行ツール (Ver.4 - GPU & Parallel Edition)
- 動画解析: CPU全コアを使った並列処理 (Parallel Processing)
- AI学習: GPUを使った XGBoost (要: pip install xgboost)
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
import joblib
import opentimelineio as otio
from xgboost import XGBClassifier # GPU対応の強力なモデル
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from joblib import Parallel, delayed

# ==========================================
# 設定
# ==========================================
WINDOW_SEC = 0.5
FPS_BASE = 59.94
AUDIO_SAMPLE_RATE = 16000 # 16kHzに下げて高速化 (音量判定には十分)
N_JOBS = -1 # -1ならPCの限界まで並列処理 (重すぎる場合は 4 などに指定)

# ==========================================
# 特徴量抽出 (並列処理用に最適化)
# ==========================================

def extract_features_fast(video_path: str):
    """動画から Audio RMS と Video Motion を抽出 (軽量版)"""
    if not os.path.exists(video_path): return None

    # --- Audio RMS ---
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "temp.wav"
        # -ar 16000 で軽量化, ログ抑制
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path, "-ac", "1", 
            "-ar", str(AUDIO_SAMPLE_RATE), "-vn", "-loglevel", "error", str(wav_path)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if not os.path.exists(wav_path): return None

        # 指定レートで読み込み
        y, sr = librosa.load(str(wav_path), sr=AUDIO_SAMPLE_RATE)
        hop_len = int(sr * WINDOW_SEC)
        if len(y) < hop_len: hop_len = len(y)
        
        rms = librosa.feature.rms(y=y, frame_length=hop_len, hop_length=hop_len)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_len)

    # --- Video Motion ---
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    motion_avgs = []
    step_frames = int(fps * WINDOW_SEC)
    if step_frames < 1: step_frames = 1
    
    prev_gray = None
    
    # ループ回数を減らして高速化
    for i in range(0, total_frames, step_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: break
        
        # 解像度を極小にして差分計算を爆速にする (80x45)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (80, 45)) 
        
        val = 0.0
        if prev_gray is not None:
            val = float(np.mean(cv2.absdiff(gray, prev_gray)))
        
        motion_avgs.append(val)
        prev_gray = gray
        
    cap.release()

    # データ長合わせ
    min_len = min(len(rms), len(motion_avgs))
    
    df = pd.DataFrame({
        "time": times[:min_len],
        "audio_vol": rms[:min_len],
        "video_motion": motion_avgs[:min_len]
    })
    
    return df

# ==========================================
# 正解ラベル抽出
# ==========================================

def get_used_ranges_from_json(json_data):
    try:
        json_str = json.dumps(json_data["timeline_data"])
        timeline = otio.adapters.read_from_string(json_str, "otio_json")
        
        used_ranges = []
        if timeline.tracks:
            for track in timeline.tracks:
                if track.kind != otio.schema.TrackKind.Video: continue
                for item in track:
                    if not isinstance(item, otio.schema.Clip): continue
                    if item.source_range:
                        s = item.source_range.start_time.value / FPS_BASE
                        e = s + (item.source_range.duration.value / FPS_BASE)
                        used_ranges.append((s, e))
        return used_ranges
    except:
        return []

# ==========================================
# 並列ワーカー関数
# ==========================================

def process_single_json(json_file):
    """1つのJSONと動画を処理してDataFrameを返す"""
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data_entry = json.load(f)
        
        meta = data_entry["meta"]
        
        # 動画パス取得 (v3/v4対応)
        video_paths = meta.get("video_paths", [])
        if not video_paths and "video_path" in meta:
            video_paths = [meta["video_path"]]
            
        if not video_paths: return None
        
        target_video = video_paths[0]
        if not os.path.exists(target_video): return None
            
        # 特徴量抽出 (ここが重いので並列化が効く)
        df = extract_features_fast(target_video)
        if df is None or len(df) == 0: return None
        
        # ラベル付け
        ranges = get_used_ranges_from_json(data_entry)
        labels = []
        for t in df["time"]:
            is_used = 0
            for start, end in ranges:
                if start <= t < end:
                    is_used = 1
                    break
            labels.append(is_used)
        
        df["is_used"] = labels
        return df
        
    except Exception:
        return None

# ==========================================
# メイン
# ==========================================

def train_and_save(df, output_model="editor_model_xgb.pkl"):
    print("\n--- Training Start (XGBoost GPU) ---")
    print(f"Total Samples: {len(df)}")
    print(f"Used Rate: {df['is_used'].mean():.2%}")
    
    X = df[["audio_vol", "video_motion"]]
    y = df["is_used"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # GPUを使用する設定 (device="cuda")
    # GPUがない環境でエラーになる場合は device="cpu" に自動で落ちるか、エラーになります
    try:
        clf = XGBClassifier(
            n_estimators=200, 
            learning_rate=0.1,
            device="cuda",  # GPU使用
            tree_method="hist", 
            random_state=42
        )
        clf.fit(X_train, y_train)
        print("[INFO] GPU学習に成功しました。")
    except Exception as e:
        print(f"[WARN] GPU学習に失敗 ({e})。CPUで再試行します。")
        clf = XGBClassifier(n_estimators=200, n_jobs=-1, random_state=42)
        clf.fit(X_train, y_train)
    
    score = clf.score(X_test, y_test)
    print(f"Model Accuracy: {score:.2%}")
    
    # 特徴量重要度
    imps = clf.feature_importances_
    print(f"Importance - Audio: {imps[0]:.3f}, Motion: {imps[1]:.3f}")
    
    joblib.dump(clf, output_model)
    print(f"Model saved to: {output_model}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder with training JSONs")
    parser.add_argument("-o", "--model", default="editor_model_xgb.pkl", help="Output model path")
    args = parser.parse_args()
    
    json_files = glob.glob(os.path.join(args.folder, "*.json"))
    if not json_files:
        print("JSON file not found.")
        return

    print(f"Found {len(json_files)} datasets.")
    print(f"Starting parallel processing (Using all CPU cores)...")
    
    # 並列処理実行
    results = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(process_single_json)(jf) for jf in json_files
    )
    
    # 結合
    valid_dfs = [res for res in results if res is not None]
    if not valid_dfs:
        print("No valid training data generated.")
        return

    print("Merging data...")
    full_df = pd.concat(valid_dfs, ignore_index=True)
    
    # CSV保存 (確認用)
    full_df.to_csv("combined_training_data.csv", index=False)
    
    # 学習
    train_and_save(full_df, args.model)

if __name__ == "__main__":
    main()