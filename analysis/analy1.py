#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
学習データ量産ツール (Parallel Speed Edition v2 - Verbose Logging)
CPU並列処理とGPU順次処理の進捗を、ターミナルに詳細に表示して可視化するバージョン。
"""

import argparse
import os
import glob
import json
import urllib.parse
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
import uuid
import sys

import numpy as np
import cv2
import librosa
import torch
import whisper
import mediapipe as mp
import opentimelineio as otio
from tqdm import tqdm
from joblib import Parallel, delayed

# ==========================================
# 設定
# ==========================================
DEFAULT_OUTPUT_DIR = r"C:\Users\yushi\Documents\プログラム\xmlai\night_run_data_parallel"
TEMP_JSON_DIR = r"C:\Users\yushi\Documents\プログラム\xmlai\temp_cpu_results"
WINDOW_SEC = 0.1
WHISPER_MODEL = "small"
VIDEO_EXTS = {".mp4", ".mov", ".mxf", ".mkv", ".avi", ".webm"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".psd", ".bmp", ".tiff", ".gif", ".ai"}
FPS_BASE = 59.94
N_JOBS = -1 

# ==========================================
# ログ出力関数 (並列処理でも崩れにくいように)
# ==========================================
def log_progress(pid, action, filename):
    # PID(プロセスID)を表示することで、複数のコアが動いていることを可視化
    print(f"[CPU:{pid}] {action}: {filename}")

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

def find_video_path_raw_xml(xml_file_path: str) -> str:
    xml_base_dir = os.path.dirname(os.path.abspath(xml_file_path))
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
    except: return None
    for path_node in root.iter("pathurl"):
        raw_url = path_node.text
        if not raw_url: continue
        file_path = clean_xml_path(raw_url)
        if os.path.splitext(file_path)[1].lower() in VIDEO_EXTS:
            if os.path.exists(file_path): return file_path
            local_path = os.path.join(xml_base_dir, os.path.basename(file_path))
            if os.path.exists(local_path): return local_path
    return None

def find_all_image_paths(xml_file_path: str) -> list:
    xml_base_dir = os.path.dirname(os.path.abspath(xml_file_path))
    images = set()
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
    except: return []
    for path_node in root.iter("pathurl"):
        raw_url = path_node.text
        if not raw_url: continue
        file_path = clean_xml_path(raw_url)
        if os.path.splitext(file_path)[1].lower() in IMAGE_EXTS:
            valid_path = None
            if os.path.exists(file_path): valid_path = file_path
            else:
                local = os.path.join(xml_base_dir, os.path.basename(file_path))
                if os.path.exists(local): valid_path = local
            if valid_path: images.add(valid_path)
    return list(images)

# ==========================================
# XML詳細解析
# ==========================================
def extract_detailed_attributes(clipitem_node):
    attributes = {
        "scale": 100.0, "pos_x": 0.5, "pos_y": 0.5, "rotation": 0.0,
        "opacity": 100.0, "audio_level": 0.0,
        "crop_left": 0.0, "crop_top": 0.0, "crop_right": 0.0, "crop_bottom": 0.0,
        "speed_percent": 100.0, "is_time_remapped": False
    }
    rate_node = clipitem_node.find("rate")
    file_node = clipitem_node.find("file")
    if rate_node is not None and file_node is not None:
        file_rate_node = file_node.find("rate")
        if file_rate_node is not None:
            try:
                clip_tb = float(rate_node.find("timebase").text)
                file_tb = float(file_rate_node.find("timebase").text)
                if file_tb > 0: attributes["speed_percent"] = (clip_tb / file_tb) * 100.0
            except: pass

    for filter_node in clipitem_node.findall("filter"):
        effect_node = filter_node.find("effect")
        if effect_node is None: continue
        effect_id_node = effect_node.find("effectid")
        if effect_id_node is None: continue
        effect_id = effect_id_node.text.lower()
        if "timeremap" in effect_id:
            attributes["is_time_remapped"] = True
            continue
        for param in effect_node.findall("parameter"):
            pid_node = param.find("parameterid")
            if pid_node is None: continue
            pid = pid_node.text.lower()
            val_node = param.find("value")
            val = None
            if val_node is not None:
                try: val = float(val_node.text)
                except: pass
            
            if "basic" in effect_id:
                if pid == "scale" and val is not None: attributes["scale"] = val
                if pid == "rotation" and val is not None: attributes["rotation"] = val
                if pid == "center" and val_node is not None:
                    horiz = val_node.find("horiz"); vert = val_node.find("vert")
                    if horiz is not None: attributes["pos_x"] = float(horiz.text)
                    if vert is not None: attributes["pos_y"] = float(vert.text)
            elif "opacity" in effect_id and pid == "opacity" and val is not None:
                attributes["opacity"] = val
            elif "audiolevels" in effect_id and pid == "level" and val is not None:
                attributes["audio_level"] = val
            elif "crop" in effect_id or "crop" in pid:
                if val is not None:
                    if "left" in pid: attributes["crop_left"] = val
                    elif "top" in pid: attributes["crop_top"] = val
                    elif "right" in pid: attributes["crop_right"] = val
                    elif "bottom" in pid: attributes["crop_bottom"] = val
    return attributes

def get_ground_truth_structure(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except: return []
    clip_details = []
    for sequence in root.iter("sequence"):
        media = sequence.find("media")
        if media is None: continue
        for media_type in ["video", "audio"]:
            media_node = media.find(media_type)
            if media_node is None: continue
            for track_idx, track in enumerate(media_node.findall("track"), start=1):
                for clipitem in track.findall("clipitem"):
                    name_node = clipitem.find("name")
                    name = name_node.text if name_node is not None else "Unknown"
                    start_node = clipitem.find("start")
                    end_node = clipitem.find("end")
                    if start_node is None or end_node is None: continue
                    start_frame = int(start_node.text)
                    end_frame = int(end_node.text)
                    if start_frame < 0: start_frame = 0
                    t_start = round(start_frame / FPS_BASE, 4)
                    t_end = round(end_frame / FPS_BASE, 4)
                    attrs = extract_detailed_attributes(clipitem)
                    data = {
                        "track_kind": media_type, "track_index": track_idx, "name": name,
                        "timeline_start": t_start, "timeline_end": t_end,
                    }
                    data.update(attrs)
                    clip_details.append(data)
    return clip_details

def analyze_image_asset(image_path: str):
    try:
        n = np.fromfile(image_path, np.uint8)
        img = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)
        if img is None: return None
        h, w = img.shape[:2]
        channels = img.shape[2] if len(img.shape) > 2 else 1
        has_alpha = (channels == 4)
        mean_color = cv2.mean(img)
        return {
            "filename": os.path.basename(image_path), "width": w, "height": h,
            "aspect_ratio": float(w)/h if h>0 else 0, "has_alpha": has_alpha,
            "mean_blue": mean_color[0], "mean_green": mean_color[1], "mean_red": mean_color[2],
            "mean_alpha": mean_color[3] if has_alpha else 255.0
        }
    except: return None

# ==========================================
# フェーズ1: CPU並列処理用ワーカー
# ==========================================
def process_cpu_tasks(xml_path, save_dir):
    pid = os.getpid() # 現在のプロセスID
    xml_name = os.path.basename(xml_path)
    
    try:
        video_path = find_video_path_raw_xml(xml_path)
        if not video_path: 
            return None

        log_progress(pid, "START", xml_name)

        # --- A. XML解析 ---
        clip_details = get_ground_truth_structure(xml_path)
        image_paths = find_all_image_paths(xml_path)
        image_assets_data = []
        if image_paths:
            for img_p in image_paths:
                data = analyze_image_asset(img_p)
                if data: image_assets_data.append(data)

        # --- B. 音声特徴量 (Librosa) ---
        log_progress(pid, "AUDIO", xml_name)
        temp_wav_name = f"temp_{uuid.uuid4()}.wav"
        wav_path = Path(tempfile.gettempdir()) / temp_wav_name
        
        subprocess.run(["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "16000", "-vn", "-loglevel", "error", str(wav_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if not os.path.exists(wav_path): return None
        
        y, sr = librosa.load(str(wav_path), sr=16000)
        hop_len = int(sr * WINDOW_SEC)
        
        rms = librosa.feature.rms(y=y, frame_length=hop_len*2, hop_length=hop_len)[0]
        zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=hop_len*2, hop_length=hop_len)[0]
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=hop_len*2, hop_length=hop_len)[0]
        melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=20, hop_length=hop_len)
        melspec_db = librosa.power_to_db(melspec, ref=np.max).T
        times_audio = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_len)
        
        os.remove(wav_path)

        # --- C. 映像解析 (Motion & Face) ---
        log_progress(pid, "VIDEO", xml_name)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        motion_avgs = []
        face_stats = []
        step_frames = int(fps * WINDOW_SEC)
        if step_frames < 1: step_frames = 1
        prev_gray = None

        mp_face = mp.solutions.face_detection
        with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detector:
            for i in range(0, total_frames, step_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret: break
                
                # Face
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detector.process(frame_rgb)
                f_count = 0; f_max_ratio = 0.0; f_cx = 0.5; f_cy = 0.5
                if results.detections:
                    f_count = len(results.detections)
                    for d in results.detections:
                        b = d.location_data.relative_bounding_box
                        area = b.width * b.height
                        if area > f_max_ratio:
                            f_max_ratio = area
                            f_cx = b.xmin + b.width/2; f_cy = b.ymin + b.height/2
                face_stats.append({"cnt": f_count, "ratio": float(f_max_ratio), "cx": float(f_cx), "cy": float(f_cy)})

                # Motion
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (64, 36))
                val = 0.0
                if prev_gray is not None: val = float(np.mean(cv2.absdiff(gray, prev_gray)))
                motion_avgs.append(val)
                prev_gray = gray
        cap.release()

        # データ統合
        min_len = min(len(rms), len(motion_avgs), len(face_stats))
        time_series_base = []
        
        for i in range(min_len):
            row = {
                "time": float(times_audio[i]),
                "audio_vol": float(rms[i]),
                "audio_zcr": float(zcr[i]),
                "audio_centroid": float(centroid[i]),
                "video_motion": float(motion_avgs[i]),
                "face_count": face_stats[i]["cnt"],
                "face_ratio": face_stats[i]["ratio"],
                "face_cx": face_stats[i]["cx"],
                "face_cy": face_stats[i]["cy"],
                "audio_mel": melspec_db[i].tolist() if i < len(melspec_db) else []
            }
            time_series_base.append(row)

        temp_data = {
            "meta": {"xml_path": os.path.abspath(xml_path), "video_path": os.path.abspath(video_path)},
            "ground_truth_clips": clip_details,
            "image_assets_analysis": image_assets_data,
            "time_series_features": time_series_base
        }
        
        save_path = os.path.join(save_dir, f"{xml_name}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(temp_data, f, indent=None, ensure_ascii=False)
        
        log_progress(pid, "DONE ", xml_name)
        return save_path

    except Exception as e:
        log_progress(pid, f"ERROR({e})", xml_name)
        return None

# ==========================================
# フェーズ2: GPU文字起こし処理
# ==========================================
def process_whisper_task(json_path, model, output_dir):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        video_path = data["meta"]["video_path"]
        xml_name = os.path.basename(json_path).replace(".json", "")
        # print(f"  [GPU Whisper] {xml_name}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "temp_w.wav"
            subprocess.run(["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "16000", "-vn", "-loglevel", "error", str(wav_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            if not os.path.exists(wav_path): return

            transcription = model.transcribe(str(wav_path), language="ja")
            segments = transcription["segments"]
        
        time_series = data["time_series_features"]
        for row in time_series:
            t = row["time"]
            txt = ""
            for s in segments:
                if s["start"] <= t <= s["end"]:
                    txt = s["text"]
                    break
            row["text"] = txt
            
        final_path = os.path.join(output_dir, os.path.basename(json_path))
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=None, ensure_ascii=False)
            
        os.remove(json_path)
        
    except Exception as e:
        print(f"  [Whisper Error] {json_path}: {e}")

# ==========================================
# メイン
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Target folder")
    parser.add_argument("-o", "--output_dir", default=DEFAULT_OUTPUT_DIR, help="Output Directory")
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    if not os.path.exists(TEMP_JSON_DIR): os.makedirs(TEMP_JSON_DIR)
    
    target_files = glob.glob(os.path.join(args.folder, "**", "*.xml"), recursive=True)
    print(f"Found {len(target_files)} XML files.")
    
    # --- Phase 1: CPU Parallel Processing ---
    print("\n=== Phase 1: CPU Analysis (Motion/Face/Audio/XML) ===")
    print("This will utilize multiple CPU cores. Watch the logs below:")
    
    # joblibのverboseは下げて、独自のlog_progressを見やすくする
    temp_files = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(process_cpu_tasks)(xml, TEMP_JSON_DIR) for xml in target_files
    )
    
    valid_temp_files = [f for f in temp_files if f is not None]
    print(f"\nPhase 1 Done. Generated {len(valid_temp_files)} intermediate files.")
    
    # --- Phase 2: GPU Whisper Processing ---
    print("\n=== Phase 2: GPU Analysis (Whisper Transcription) ===")
    
    print(f"[INFO] Loading Whisper ({WHISPER_MODEL})...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(WHISPER_MODEL, device=device)
    print(f"[INFO] Model Loaded on {device}.")
    
    for json_path in tqdm(valid_temp_files, desc="Whisper Progress"):
        process_whisper_task(json_path, model, args.output_dir)

    print(f"\nAll Done! Final data saved to: {args.output_dir}")
    try: os.rmdir(TEMP_JSON_DIR)
    except: pass

if __name__ == "__main__":
    main()