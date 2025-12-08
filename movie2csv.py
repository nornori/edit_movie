import cv2
import pandas as pd
import numpy as np
import librosa
import torch
import whisper
import mediapipe as mp
from moviepy.editor import VideoFileClip
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import argparse
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 設定
# ==========================================
INTERVAL_SEC = 0.1  # 学習データに合わせて0.1秒刻みに変更（細かいほうが精度が出ます）
USE_CLIP = True
USE_WHISPER = True

# ==========================================
# モデルロード
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = None
clip_processor = None
whisper_model = None
face_mesh = None

def load_models():
    global clip_model, clip_processor, whisper_model, face_mesh
    print(f"Using device: {device}")

    if USE_CLIP:
        print("Loading CLIP model...")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    if USE_WHISPER:
        print("Loading Whisper model...")
        whisper_model = whisper.load_model("small", device=device) # 精度重視でsmall推奨

    print("Loading MediaPipe FaceMesh...")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

# ==========================================
# 音声解析
# ==========================================
def analyze_audio(video_path, duration, interval):
    print("Extracting Audio Features...")
    try:
        temp_audio = "temp_audio.wav"
        video = VideoFileClip(video_path)
        if video.audio is None:
            return None
        
        video.audio.write_audiofile(temp_audio, logger=None)
        y, sr = librosa.load(temp_audio, sr=None)
        os.remove(temp_audio)
        
        # RMS (音量)
        hop_length = int(sr * interval)
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
        
        # 長さ合わせ
        target_len = int(duration / interval) + 1
        rms = np.pad(rms, (0, max(0, target_len - len(rms))))[:target_len]
        
        # 沈黙期間 & 発話フラグ
        threshold = 0.01
        is_speaking = (rms > threshold).astype(float)
        
        # 沈黙継続時間 (ミリ秒)
        silence_dur = np.zeros_like(rms)
        count = 0
        for i in range(len(rms)):
            if rms[i] < threshold:
                count += interval * 1000
            else:
                count = 0
            silence_dur[i] = count
            
        return rms, is_speaking, silence_dur
        
    except Exception as e:
        print(f"Audio Error: {e}")
        return None

# ==========================================
# 顔・視覚解析 (MediaPipe)
# ==========================================
def get_face_features(image):
    h, w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    
    features = {
        'face_center_x': 0.5, 'face_center_y': 0.5,
        'face_count': 0, 'face_size': 0.0,
        'face_eyebrow_raise': 0.0, 'face_mouth_open': 0.0
    }
    
    if results.multi_face_landmarks:
        features['face_count'] = len(results.multi_face_landmarks)
        landmarks = results.multi_face_landmarks[0].landmark
        
        # 座標 (鼻の頭)
        features['face_center_x'] = landmarks[1].x
        features['face_center_y'] = landmarks[1].y
        
        # 顔サイズ (バウンディングボックスの対角線)
        xs = [l.x for l in landmarks]
        ys = [l.y for l in landmarks]
        features['face_size'] = np.sqrt((max(xs)-min(xs))**2 + (max(ys)-min(ys))**2)
        
        # 口の開き (上唇: 13, 下唇: 14)
        lip_dist = np.linalg.norm(np.array([landmarks[13].x, landmarks[13].y]) - 
                                  np.array([landmarks[14].x, landmarks[14].y]))
        features['face_mouth_open'] = lip_dist * 100 # 簡易スケーリング
        
        # 眉の上がり (眉: 105, 目: 33 の距離など)
        eyebrow_dist = landmarks[105].y - landmarks[33].y
        features['face_eyebrow_raise'] = abs(eyebrow_dist) * 100
        
    return features

# ==========================================
# サリエンシー (注目度マップ - 簡易版)
# ==========================================
def get_saliency(image):
    # OpenCVのSaliency APIがない場合があるため、簡易的に「中心からの明るさ重み」で代用
    # もし opencv-contrib-python が入っていれば SpectralResidual が使えます
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 簡易実装: 輝度重心を中心からのズレとして計算
    moments = cv2.moments(gray)
    if moments["m00"] != 0:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        return cx / image.shape[1], cy / image.shape[0]
    return 0.5, 0.5

# ==========================================
# メイン処理
# ==========================================
def extract_features(video_path, output_csv):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Video not found.")
        return

    load_models()
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    print(f"Video Info: {duration:.2f}s, {fps}fps")

    # 音声 & 字幕 (一括)
    audio_data = analyze_audio(video_path, duration, INTERVAL_SEC)
    whisper_res = whisper_model.transcribe(video_path) if USE_WHISPER else {'segments': []}
    
    data_rows = []
    frame_interval = int(fps * INTERVAL_SEC)
    current_frame = 0
    step = 0
    prev_gray = None
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if current_frame % frame_interval == 0:
            time_sec = current_frame / fps
            
            # --- 1. 基本画像処理 ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Motion (前フレームとの差分)
            motion = 0.0
            scene_change = 0.0
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                motion = np.mean(diff) / 255.0
                if motion > 0.3: scene_change = 1.0 # 簡易シーンチェンジ判定
            prev_gray = gray
            
            # --- 2. 顔認識 & サリエンシー ---
            face_feats = get_face_features(frame)
            sal_x, sal_y = get_saliency(frame)
            
            # --- 3. CLIP ---
            clip_embeds = [0.0] * 512
            if USE_CLIP:
                img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                with torch.no_grad():
                    inputs = clip_processor(images=img_pil, return_tensors="pt").to(device)
                    clip_embeds = clip_model.get_image_features(**inputs).cpu().numpy().flatten().tolist()

            # --- 4. 音声 & テキスト ---
            rms = audio_data[0][step] if audio_data and step < len(audio_data[0]) else 0
            is_speaking = audio_data[1][step] if audio_data and step < len(audio_data[1]) else 0
            silence_ms = audio_data[2][step] if audio_data and step < len(audio_data[2]) else 0
            
            text_word = ""
            for seg in whisper_res['segments']:
                if seg['start'] <= time_sec <= seg['end']:
                    text_word = seg['text']
                    break
            
            # データ格納 (カラム順序は厳密でなくてもDataFrame作成時にキーで合わさりますが、要素は網羅する)
            row = {
                'time': round(time_sec, 2),
                'audio_energy_rms': rms,
                'audio_is_speaking': is_speaking,
                'silence_duration_ms': silence_ms,
                'speaker_id': 0, # 話者識別は簡易的に0
                'text_is_active': 1.0 if text_word else 0.0,
                'text_word': text_word,
                # Visual
                'face_center_x': face_feats['face_center_x'],
                'face_center_y': face_feats['face_center_y'],
                'face_count': face_feats['face_count'],
                'face_eyebrow_raise': face_feats['face_eyebrow_raise'],
                'face_mouth_open': face_feats['face_mouth_open'],
                'face_size': face_feats['face_size'],
                'saliency_x': sal_x,
                'saliency_y': sal_y,
                'scene_change': scene_change,
                'visual_motion': motion,
            }
            # CLIP展開
            for i, v in enumerate(clip_embeds):
                row[f'clip_{i}'] = v
                
            data_rows.append(row)
            if step % 10 == 0: print(f"Processed {time_sec:.1f}s...")
            step += 1
            
        current_frame += 1

    cap.release()
    df = pd.DataFrame(data_rows)
    df.to_csv(output_csv, index=False)
    print(f"Done! Saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_video")
    args = parser.parse_args()
    
    out_name = os.path.splitext(os.path.basename(args.input_video))[0] + "_features.csv"
    extract_features(args.input_video, out_name)