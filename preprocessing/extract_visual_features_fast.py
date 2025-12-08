import os
import glob
import math
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import Dict, Any

# ==============================================================================
#  ⚡ 高速化設定エリア ⚡
# ==============================================================================
INPUT_JSON_DIR = "./night_run_data_parallel"
OUTPUT_FEATURE_DIR = "./input_features"
JSON_FILE_EXT = ".xml.json"
VIDEO_FILE_EXT = ".mp4"
OUTPUT_FILE_SUFFIX = "_visual_features.csv"

TIME_STEP = 0.1             # 基本サンプリング (0.1秒)
CLIP_STEP = 1.0             # CLIP解析間隔 (1.0秒)
ANALYSIS_WIDTH = 640        # ★高速化: 解析用画像の横幅 (推奨: 640 or 480)
USE_GPU = torch.cuda.is_available()
# ==============================================================================

# --- GPU最適化 ---
if USE_GPU:
    print(f"✅ GPU Mode: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True # 高速化フラグ
else:
    print("⚠️ CPU Mode (CLIP may be slow)")

# --- モデル初期化 ---
print("Loading CLIP model...")
# 【修正】use_safetensors=True を追加してセキュリティエラーを回避
try:
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
except Exception as e:
    print(f"Model loading failed with safetensors: {e}")
    print("Retrying with standard load...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

if USE_GPU:
    clip_model = clip_model.to("cuda")

# MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# --- ヘルパー関数 ---
def load_xml_json(json_path: str) -> Dict[str, Any]:
    import json
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_base_video_path_and_name(data: Dict[str, Any]) -> tuple[str, str, str]:
    video_path_raw = data.get("meta", {}).get("video_path", "")
    if not video_path_raw: return "", "", ""
    video_path_normalized = video_path_raw.replace("\\", "/")
    base_name = os.path.basename(video_path_normalized)
    video_stem = base_name.replace(VIDEO_FILE_EXT, "")
    if "." not in base_name: base_name += VIDEO_FILE_EXT
    return video_path_normalized, base_name, video_stem

# --- 特徴量計算ロジック ---

def calculate_mouth_openness(landmarks, h, w):
    """口の開き (MAR)"""
    upper = np.array([landmarks[13].x * w, landmarks[13].y * h])
    lower = np.array([landmarks[14].x * w, landmarks[14].y * h])
    left = np.array([landmarks[61].x * w, landmarks[61].y * h])
    right = np.array([landmarks[291].x * w, landmarks[291].y * h])
    return np.linalg.norm(upper - lower) / (np.linalg.norm(left - right) + 1e-6)

def calculate_eyebrow_raise(landmarks, h, w):
    """眉の上がり具合"""
    left_eye = np.array([landmarks[159].x * w, landmarks[159].y * h])
    left_brow = np.array([landmarks[65].x * w, landmarks[65].y * h])
    face_h = abs(landmarks[152].y * h - landmarks[10].y * h)
    return np.linalg.norm(left_eye - left_brow) / (face_h + 1e-6)

def calculate_scene_change(prev_hist, curr_frame):
    """シーン転換スコア"""
    curr_hsv = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)
    curr_hist = cv2.calcHist([curr_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(curr_hist, curr_hist, 0, 1, cv2.NORM_MINMAX)
    
    if prev_hist is None: return 0.0, curr_hist
    
    similarity = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
    return max(0.0, 1.0 - similarity), curr_hist

def calculate_motion_and_saliency(prev_gray, curr_frame):
    """動きと注目点"""
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    if prev_gray is None: return 0.0, np.nan, np.nan, curr_gray
    
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    
    non_zero = cv2.countNonZero(thresh)
    h, w = diff.shape
    motion = non_zero / (h * w)
    
    M = cv2.moments(thresh)
    if M["m00"] > 0:
        cX = (M["m10"] / M["m00"]) / w
        cY = (M["m01"] / M["m00"]) / h
    else:
        cX, cY = np.nan, np.nan
        
    return motion, cX, cY, curr_gray

# --- メイン抽出関数 ---

def extract_visual_features_fast(video_path: str) -> pd.DataFrame:
    if not os.path.exists(video_path):
        print(f"  [ERROR] 動画不在: {video_path}")
        return pd.DataFrame()

    print(f"  -> 解析開始 (Resize={ANALYSIS_WIDTH}px): {os.path.basename(video_path)}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return pd.DataFrame()

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_step = int(fps * TIME_STEP)
    if frame_step < 1: frame_step = 1

    records = []
    current_frame_idx = 0
    
    # 状態保持
    prev_gray = None
    prev_hist = None
    last_clip_emb = [0.0] * 512
    last_clip_time = -999.0

    while True:
        ret, raw_frame = cap.read()
        if not ret: break
        
        if current_frame_idx % frame_step == 0:
            timestamp = current_frame_idx / fps
            
            # ★ ここでリサイズして高速化
            h_raw, w_raw = raw_frame.shape[:2]
            scale = ANALYSIS_WIDTH / w_raw
            h_new = int(h_raw * scale)
            frame = cv2.resize(raw_frame, (ANALYSIS_WIDTH, h_new))
            
            # 1. シーン転換
            scene_score, prev_hist = calculate_scene_change(prev_hist, frame)
            
            # 2. 動き & 注目点
            motion, sal_x, sal_y, prev_gray = calculate_motion_and_saliency(prev_gray, frame)
            
            # 3. 顔・表情 (MediaPipe)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            
            face_data = {
                'face_count': 0, 'face_center_x': np.nan, 'face_center_y': np.nan,
                'face_size': 0.0, 'face_mouth_open': 0.0, 'face_eyebrow_raise': 0.0
            }
            
            if results.multi_face_landmarks:
                face_data['face_count'] = len(results.multi_face_landmarks)
                lm = results.multi_face_landmarks[0].landmark
                
                # 座標計算
                xs = [l.x for l in lm]
                ys = [l.y for l in lm]
                face_data['face_center_x'] = sum(xs) / len(xs)
                face_data['face_center_y'] = sum(ys) / len(ys)
                face_data['face_size'] = (max(xs) - min(xs)) * (max(ys) - min(ys))
                face_data['face_mouth_open'] = calculate_mouth_openness(lm, h_new, ANALYSIS_WIDTH)
                face_data['face_eyebrow_raise'] = calculate_eyebrow_raise(lm, h_new, ANALYSIS_WIDTH)

            # 4. CLIP (1秒ごと, GPU使用)
            if (timestamp - last_clip_time) >= CLIP_STEP:
                pil_img = Image.fromarray(frame_rgb)
                inputs = clip_processor(images=pil_img, return_tensors="pt")
                if USE_GPU: inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
                with torch.no_grad():
                    img_feats = clip_model.get_image_features(**inputs)
                last_clip_emb = img_feats.cpu().numpy().flatten().tolist()
                last_clip_time = timestamp

            # 記録
            row = {
                'time': round(timestamp, 3),
                'scene_change': scene_score,
                'visual_motion': motion,
                'saliency_x': sal_x,
                'saliency_y': sal_y,
                **face_data
            }
            # CLIP次元 (col_0..511)
            row.update({f'clip_{i}': v for i, v in enumerate(last_clip_emb)})
            records.append(row)

        current_frame_idx += 1

    cap.release()
    return pd.DataFrame(records)

def batch_process_fast():
    os.makedirs(OUTPUT_FEATURE_DIR, exist_ok=True)
    json_paths = glob.glob(os.path.join(INPUT_JSON_DIR, f"*{JSON_FILE_EXT}"))
    print(f"🚀 高速映像特徴量抽出を開始 ({len(json_paths)} 件)")

    for idx, json_path in enumerate(json_paths):
        print(f"--- ({idx+1}/{len(json_paths)}) {os.path.basename(json_path)}")
        data = load_xml_json(json_path)
        video_full_path, _, video_stem = get_base_video_path_and_name(data)
        
        if not video_full_path: continue
        
        output_path = os.path.join(OUTPUT_FEATURE_DIR, f"{video_stem}{OUTPUT_FILE_SUFFIX}")
        
        df = extract_visual_features_fast(video_full_path)
        
        if not df.empty:
            df.to_csv(output_path, index=False, float_format='%.6f')
            print(f"  [SUCCESS] 保存完了 ({len(df)}行): {os.path.basename(output_path)}")

if __name__ == "__main__":
    batch_process_fast()