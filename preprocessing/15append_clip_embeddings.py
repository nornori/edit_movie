#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CLIP Embeddings 追記ツール
既存のJSON (0.1秒特徴量を含む) に、1.0秒間隔のCLIPベクトルを計算し追記する。

【修正点】
- N_JOBSを高い値に設定し、CPU(動画デコード)の並列処理を最大化。
- initialize_clip()を calculate_clip_embeddings 内に移動し、プロセスごとの独立したモデルロードとする。
- joblibに maxtasksperchild=1 を設定し、タスク完了ごとにGPUメモリを解放して安定性と効率を向上させる。
"""

import argparse
import os
import glob
import json
import numpy as np
import cv2
from PIL import Image
import torch
import clip # CLIPモデル用
from tqdm import tqdm
from joblib import Parallel, delayed
import sys

# ==========================================
# 設定
# ==========================================
CLIP_SAMPLE_SEC = 1.0   # CLIP Embeddingsの解像度
CLIP_MODEL_NAME = "ViT-B/32"
# N_JOBSをCPUコア数に近い高い値に設定。GPUメモリに応じて調整してください。
N_JOBS = 8 

# ==========================================
# ユーティリティ
# ==========================================
def initialize_clip():
    """CLIPモデルのロード（GPU使用）"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 各子プロセスでロードするため、進捗ログは冗長にならないよう注意
    # print(f"[INFO] Loading CLIP Model ({CLIP_MODEL_NAME}) on {device}...")
    try:
        model, preprocess = clip.load(CLIP_MODEL_NAME, device=device)
        return model, preprocess, device
    except Exception as e:
        # 子プロセスでのエラーを明確化
        print(f"[ERROR in Process] CLIPモデルのロードに失敗しました: {e}", file=sys.stderr)
        return None, None, None

# ==========================================
# メイン処理 (CLIP計算)
# ==========================================
def calculate_clip_embeddings(json_path):
    """
    JSONファイルにCLIP Embeddingsを計算し、追記する。
    この関数内でモデルをロードし、各プロセスが独立してGPUリソースを使用する。
    """
    
    # 修正点2: モデルをプロセス内でロードする
    model, preprocess, device = initialize_clip()
    if model is None:
        return {"file": os.path.basename(json_path), "status": "Error", "message": "CLIP model initialization failed in worker process."}

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 1. メタデータから動画パスを取得
        meta = data.get("meta", {})
        video_path = meta.get("video_path")
        
        if not video_path or not os.path.exists(video_path):
            return {"file": os.path.basename(json_path), "status": "Error", "message": "Video file not found or path is missing."}

        # 2. 動画処理
        cap = cv2.VideoCapture(video_path)
        
        # OpenCV/FFmpegのエラーチェックを強化
        if not cap.isOpened():
             return {"file": os.path.basename(json_path), "status": "Error", "message": "Failed to open video file (OpenCV error). Check file integrity."}

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        clip_embeddings = {}
        step_frames = int(fps * CLIP_SAMPLE_SEC)
        if step_frames < 1: step_frames = 1
        
        for i in range(0, total_frames, step_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret: 
                # デコードエラーやファイル終端に達した場合
                # print(f"Warning: Failed to read frame {i} from {os.path.basename(video_path)}")
                break
            
            # CLIP処理
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_input = preprocess(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                embedding = model.encode_image(image_input)
                
            time_sec = round(i / fps, 1)
            clip_embeddings[str(time_sec)] = embedding.cpu().numpy().flatten().tolist()
            
        cap.release()
        
        if not clip_embeddings:
            return {"file": os.path.basename(json_path), "status": "Error", "message": "No frames were successfully processed. Video might be corrupted or very short."}

        # 3. JSONに追記 (上書き)
        data["clip_embeddings_1s"] = clip_embeddings
        
        with open(json_path, "w", encoding="utf-8") as f:
            # indent=None でファイルをコンパクトに保つ
            json.dump(data, f, indent=None, ensure_ascii=False)
            
        return {"file": os.path.basename(json_path), "status": "Success", "count": len(clip_embeddings)}

    except Exception as e:
        return {"file": os.path.basename(json_path), "status": "Error", "message": str(e)}

# ==========================================
# メイン
# ==========================================
# ==========================================
# メイン
# ==========================================
# ==========================================
# メイン
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder containing existing JSON files")
    args = parser.parse_args()
    
    # フォルダ内の全てのJSONファイルをリストアップ
    json_files = glob.glob(os.path.join(args.folder, "*.json"))
    if not json_files:
        print("Error: JSON files not found in the specified folder.")
        sys.exit(1)

    print(f"[INFO] Initializing CLIP (Test Load) on {'cuda' if torch.cuda.is_available() else 'cpu'}...")
    
    print(f"\nFound {len(json_files)} files. Starting CLIP Embeddings calculation with N_JOBS={N_JOBS}...")
    
    # 【最終修正】backend="multiprocessing" を追加し、maxtasksperchild=1 を有効にする
    results = Parallel(
        n_jobs=N_JOBS, 
        verbose=1,
        # 修正: multiprocessing バックエンドを指定し、maxtasksperchild 引数を有効化
        backend="multiprocessing", 
        maxtasksperchild=1 # タスク完了ごとにプロセスを再起動し、GPUメモリを確実に解放
    )(
        delayed(calculate_clip_embeddings)(jf) for jf in json_files
    )
    
    success_count = sum(1 for r in results if r and r["status"] == "Success")
    error_results = [r for r in results if r and r["status"] == "Error"]
    error_count = len(error_results)

    print("\n" + "="*40)
    print("CLIP Embeddings 追記完了")
    print(f"  - 成功ファイル数: {success_count}")
    print(f"  - 失敗ファイル数: {error_count}")
    print(f"  - 完了フォルダ: {os.path.abspath(args.folder)}")
    
    if error_count > 0:
        print("\n--- 失敗したファイル詳細 ---")
        for i, r in enumerate(error_results[:5]):
            print(f"[{i+1}] {r['file']}: {r['message']}")
        if error_count > 5:
             print(f"(他 {error_count - 5} 件)")

    print("="*40)

if __name__ == "__main__":
    main()