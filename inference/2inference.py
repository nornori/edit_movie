#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
動画編集AI 推論・JSON生成ツール (Ver.1)
training_result.csv の学習結果をもとに、動画から「切り抜きJSON」を自動生成する。
"""

import argparse
import subprocess
import tempfile
import os
import sys
import json
from pathlib import Path
from itertools import groupby

import numpy as np
import pandas as pd
import cv2
import librosa
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. 特徴量抽出 (1ai.py と同じロジック)
# ==========================================

def extract_features_for_inference(video_path: str, window_sec: float = 0.5):
    print(f"[INFO] 推論用解析開始: {video_path}")
    
    # --- Audio ---
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "temp.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "22050", "-vn", str(wav_path)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        y, sr = librosa.load(str(wav_path), sr=None)
        hop_len = int(sr * window_sec)
        if len(y) < hop_len: hop_len = len(y)
        rms = librosa.feature.rms(y=y, frame_length=hop_len, hop_length=hop_len)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_len)

    # --- Video ---
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    motion_avgs = []
    step_frames = int(fps * window_sec)
    if step_frames < 1: step_frames = 1
    prev_gray = None
    
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
            "time_sec": float(times[i]),
            "audio_vol": float(rms[i]),
            "video_motion": float(motion_avgs[i])
        })
    return pd.DataFrame(data)

# ==========================================
# 2. 推論とスムージング
# ==========================================

def train_model_from_csv(csv_path: str):
    """保存されたCSVからモデルを再学習して復元する"""
    print(f"[INFO] 学習データをロード中: {csv_path}")
    df = pd.read_csv(csv_path)
    X = df[["audio_vol", "video_motion"]]
    y = df["is_used"]
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    return clf

def smooth_predictions(predictions, window_size=3):
    """
    ガタガタな予測（0,1,0,1,1...）を滑らかにする簡易フィルタ
    例: [0, 1, 0] -> [0, 0, 0] (ノイズ除去)
    """
    # 移動平均的な処理で「孤立した1」や「孤立した0」を埋める
    smoothed = predictions.copy()
    # 簡易的に: 前後が1なら自分も1にする（穴埋め）
    for i in range(1, len(predictions) - 1):
        if predictions[i-1] == 1 and predictions[i+1] == 1:
            smoothed[i] = 1
    return smoothed

# ==========================================
# 3. JSON生成
# ==========================================

def generate_pipeline_json(video_path: str, df: pd.DataFrame, predictions: list, output_json: str):
    """推論結果(0/1の配列)を、start/endのクリップリストに変換してJSON保存"""
    
    clips = []
    # タイムライン上の現在のヘッド位置 (0秒からスタートして積んでいく)
    current_timeline_head = 0.0
    
    # 0.5秒ごとの判定を「連続した区間」にまとめる
    # df['time_sec'] は各区間の開始時刻
    
    # itertools.groupby で連続する 0 と 1 をまとめる
    # data structure: [(is_used, [row, row, ...]), ...]
    
    step_sec = df["time_sec"][1] - df["time_sec"][0] if len(df) > 1 else 0.5
    
    current_idx = 0
    for is_used, group in groupby(predictions):
        count = len(list(group))
        duration = count * step_sec
        
        # 採用区間 (1) の場合のみクリップを作成
        if is_used == 1:
            # 元動画の開始位置
            source_start = df["time_sec"][current_idx]
            
            clip = {
                "track": 1,
                "name": f"AutoCut_{current_idx}",
                "file": os.path.abspath(video_path).replace("\\", "/"),
                
                # タイムライン上の配置場所 (前のクリップの直後)
                "start": round(current_timeline_head, 4),
                "end": round(current_timeline_head + duration, 4),
                
                # 素材のどこを使うか (ここが重要)
                "source_start": round(source_start, 4)
            }
            clips.append(clip)
            
            # ヘッドを進める
            current_timeline_head += duration
            
        current_idx += count

    # JSON全体構造
    final_data = {
        "timeline_name": "AI_Auto_Edit",
        "fps": 59.94,
        "clips": clips
    }
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)
        
    print(f"\n[SUCCESS] 切り抜きJSON生成完了: {output_json}")
    print(f"  - 生成クリップ数: {len(clips)}")
    print(f"  - 編集後トータル尺: {current_timeline_head:.2f}秒")

# ==========================================
# Main
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="New Video file (.mp4) to edit")
    parser.add_argument("--csv", default="training_result.csv", help="Learned data csv")
    parser.add_argument("-o", "--output", default="ai_edit.json", help="Output JSON path")
    args = parser.parse_args()
    
    if not os.path.exists(args.csv):
        print(f"Error: 学習データ {args.csv} がありません。先に 1ai.py を実行してください。")
        sys.exit(1)
        
    # 1. モデル復元
    model = train_model_from_csv(args.csv)
    
    # 2. 新しい動画を解析
    df = extract_features_for_inference(args.video)
    
    # 3. AIによる判定
    X = df[["audio_vol", "video_motion"]]
    raw_preds = model.predict(X)
    
    # 4. ノイズ除去（スムージング）
    final_preds = smooth_predictions(raw_preds)
    
    # 5. JSON生成
    generate_pipeline_json(args.video, df, final_preds, args.output)