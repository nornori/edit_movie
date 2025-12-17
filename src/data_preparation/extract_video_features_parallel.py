"""
Video Feature Extraction Script (並列処理版)

動画から音声・視覚特徴量を並列処理で高速抽出します。

【特徴】
- 並列処理対応（joblib使用）
- GPU/CPUの自動切り替え
- メモリ効率化（maxtasksperchild=1）
- プログレスバー表示
- エラーハンドリング強化

【抽出される特徴量】
- 音声: 7次元 (rms, speaking, silence, speaker_id, text_active, text_word)
- 視覚: 522次元 (10スカラー + 512 CLIP)
- 合計: 529次元
"""
import os
import argparse
import glob
import math
import numpy as np
import pandas as pd
import cv2
import librosa
import whisper
import torch
import mediapipe as mp
from PIL import Image
from pydub import AudioSegment
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import List, Dict, Any
import warnings
import sys
warnings.filterwarnings('ignore')

# ==========================================
# 設定
# ==========================================
TIME_STEP = 0.1              # 基本サンプリング (0.1秒 = 10 FPS)
CLIP_STEP = 1.0              # CLIP解析間隔 (1.0秒)
ANALYSIS_WIDTH = 640         # 解析用画像の横幅
WHISPER_MODEL_SIZE = "small" # Whisperモデル
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
N_JOBS = 4                   # 並列処理数（CPUコア数に応じて調整）

USE_GPU = torch.cuda.is_available()

print("="*70)
print("Video Feature Extraction Script (Parallel)")
print("="*70)
print(f"Device: {'GPU' if USE_GPU else 'CPU'}")
if USE_GPU:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Parallel Jobs: {N_JOBS}")
print(f"Sampling Rate: {1.0/TIME_STEP:.1f} FPS")
print(f"CLIP Interval: {CLIP_STEP}s")
print("="*70 + "\n")

# ==========================================
# 特徴量抽出関数（プロセスごとに実行）
# ==========================================
def extract_features_worker(video_path: str, output_dir: str):
    """
    1つの動画から特徴量を抽出（ワーカー関数）
    各プロセスで独立してモデルをロード
    """
    try:
        # プロセス内でモデルをロード
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Whisper
        whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
        
        # CLIP
        try:
            clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, use_safetensors=True)
            clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        except:
            clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
            clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        
        if device == "cuda":
            clip_model = clip_model.to("cuda")
        
        # MediaPipe (skip on Windows due to path issues)
        face_mesh = None
        try:
            mp_face_mesh = mp.solutions.face_mesh
            face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
        except Exception as e:
            print(f"WARNING: MediaPipe FaceMesh initialization failed: {e}")
            print("   Continuing without face features...")
        
        # 音声特徴量を抽出（テロップ情報を含む）
        df_audio = _extract_audio_features(video_path, whisper_model, xml_dir='editxml')
        
        # 視覚特徴量を抽出
        df_visual = _extract_visual_features(
            video_path, clip_model, clip_processor, face_mesh, device
        )
        
        # 統合
        min_len = min(len(df_audio), len(df_visual))
        df_audio = df_audio.iloc[:min_len].reset_index(drop=True)
        df_visual = df_visual.iloc[:min_len].reset_index(drop=True)
        df_visual = df_visual.drop(columns=['time'])
        df_final = pd.concat([df_audio, df_visual], axis=1)
        
        # 保存
        video_stem = Path(video_path).stem
        output_path = os.path.join(output_dir, f"{video_stem}_features.csv")
        df_final.to_csv(output_path, index=False, float_format='%.6f')
        
        return {
            "file": Path(video_path).name,
            "status": "Success",
            "timesteps": len(df_final),
            "features": len(df_final.columns)
        }
        
    except Exception as e:
        return {
            "file": Path(video_path).name,
            "status": "Error",
            "message": str(e)
        }

def _extract_audio_features(video_path: str, whisper_model, xml_dir: str = 'editxml') -> pd.DataFrame:
    """音声特徴量を抽出（テロップ情報を含む）"""
    temp_wav = "temp_audio.wav"
    
    try:
        # 音声を抽出
        audio = AudioSegment.from_file(video_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(temp_wav, format="wav")
        
        # Librosaで読み込み
        y, sr = librosa.load(temp_wav, sr=16000)
        total_duration = librosa.get_duration(y=y, sr=sr)
        
        # RMS Energy
        frame_length = int(TIME_STEP * sr)
        hop_length = frame_length
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # VAD
        vad_threshold = 0.01
        is_speaking = (rms > vad_threshold).astype(int)
        
        # 無音時間
        silence_duration_ms = []
        current_silence = 0
        for speak_flag in is_speaking:
            if speak_flag == 0:
                current_silence += int(TIME_STEP * 1000)
            else:
                current_silence = 0
            silence_duration_ms.append(current_silence)
        
        # Whisper文字起こし
        whisper_results = _get_whisper_features(temp_wav, whisper_model)
        df_text = _align_text_features(whisper_results, total_duration)
        
        # テロップ情報を抽出
        df_telop = _extract_telop_features(video_path, total_duration, xml_dir)
        
        # DataFrame統合
        min_len = min(len(rms), len(df_text), len(df_telop))
        
        df_audio = pd.DataFrame({
            'time': df_text['time'][:min_len],
            'audio_energy_rms': rms[:min_len],
            'audio_is_speaking': is_speaking[:min_len],
            'silence_duration_ms': silence_duration_ms[:min_len],
            'speaker_id': np.nan,
            'text_is_active': df_text['text_is_active'][:min_len],
            'text_word': df_text['text_word'][:min_len],
            'telop_active': df_telop['telop_active'][:min_len],
            'telop_text': df_telop['telop_text'][:min_len]
        })
        
        return df_audio
        
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

def _get_whisper_features(audio_path: str, whisper_model) -> List[Dict]:
    """Whisperで文字起こし"""
    try:
        result = whisper_model.transcribe(audio_path, word_timestamps=True)
        word_list = []
        for segment in result.get('segments', []):
            for word_info in segment.get('words', []):
                word_list.append({
                    "word": word_info['word'],
                    "start": word_info['start'],
                    "end": word_info['end']
                })
        return word_list
    except:
        return []

def _align_text_features(whisper_results: List[Dict], total_duration: float) -> pd.DataFrame:
    """Whisper結果を時系列データに変換"""
    num_steps = int(math.ceil(total_duration / TIME_STEP))
    time_points = [round(i * TIME_STEP, 6) for i in range(num_steps + 1)]
    
    text_records = []
    for t in time_points:
        current_word = np.nan
        is_active = 0
        
        for w in whisper_results:
            if w['start'] <= t < w['end']:
                current_word = w['word'].strip()
                is_active = 1
                break
        
        text_records.append({
            'time': t,
            'text_is_active': is_active,
            'text_word': current_word
        })
    
    return pd.DataFrame(text_records)

def _extract_telop_features(video_path: str, total_duration: float, xml_dir: str = 'editxml') -> pd.DataFrame:
    """XMLからテロップ情報を抽出して時系列データに変換"""
    from src.data_preparation.telop_extractor import TelopExtractor
    from pathlib import Path
    
    # 対応するXMLファイルを探す
    video_stem = Path(video_path).stem
    xml_path = Path(xml_dir) / f"{video_stem}.xml"
    
    if not xml_path.exists():
        # XMLが見つからない場合は空のDataFrameを返す
        num_steps = int(np.ceil(total_duration / TIME_STEP))
        time_points = [round(i * TIME_STEP, 6) for i in range(num_steps + 1)]
        return pd.DataFrame({
            'time': time_points,
            'telop_active': 0,
            'telop_text': np.nan
        })
    
    try:
        # テロップを抽出
        extractor = TelopExtractor(fps=1.0/TIME_STEP)
        df_telop = extractor.extract_and_convert(str(xml_path), total_duration)
        return df_telop
    except Exception as e:
        # エラーが発生した場合は空のDataFrameを返す
        num_steps = int(np.ceil(total_duration / TIME_STEP))
        time_points = [round(i * TIME_STEP, 6) for i in range(num_steps + 1)]
        return pd.DataFrame({
            'time': time_points,
            'telop_active': 0,
            'telop_text': np.nan
        })

def _extract_visual_features(video_path: str, clip_model, clip_processor, face_mesh, device) -> pd.DataFrame:
    """視覚特徴量を抽出"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_step = int(fps * TIME_STEP)
    if frame_step < 1:
        frame_step = 1
    
    records = []
    current_frame_idx = 0
    
    prev_gray = None
    prev_hist = None
    last_clip_emb = [0.0] * 512
    last_clip_time = -999.0
    
    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break
        
        if current_frame_idx % frame_step == 0:
            timestamp = current_frame_idx / fps
            
            # リサイズ
            h_raw, w_raw = raw_frame.shape[:2]
            scale = ANALYSIS_WIDTH / w_raw
            h_new = int(h_raw * scale)
            frame = cv2.resize(raw_frame, (ANALYSIS_WIDTH, h_new))
            
            # シーン転換
            scene_score, prev_hist = _calculate_scene_change(prev_hist, frame)
            
            # 動き & 注目点
            motion, sal_x, sal_y, prev_gray = _calculate_motion_and_saliency(prev_gray, frame)
            
            # 顔
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_data = _extract_face_features(frame_rgb, face_mesh, h_new, ANALYSIS_WIDTH)
            
            # CLIP
            if (timestamp - last_clip_time) >= CLIP_STEP:
                last_clip_emb = _extract_clip_features(frame_rgb, clip_model, clip_processor, device)
                last_clip_time = timestamp
            
            # レコード
            row = {
                'time': round(timestamp, 3),
                'scene_change': scene_score,
                'visual_motion': motion,
                'saliency_x': sal_x,
                'saliency_y': sal_y,
                **face_data
            }
            row.update({f'clip_{i}': v for i, v in enumerate(last_clip_emb)})
            records.append(row)
        
        current_frame_idx += 1
    
    cap.release()
    return pd.DataFrame(records)

def _calculate_scene_change(prev_hist, curr_frame):
    """シーン転換スコア"""
    curr_hsv = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)
    curr_hist = cv2.calcHist([curr_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(curr_hist, curr_hist, 0, 1, cv2.NORM_MINMAX)
    
    if prev_hist is None:
        return 0.0, curr_hist
    
    similarity = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
    return max(0.0, 1.0 - similarity), curr_hist

def _calculate_motion_and_saliency(prev_gray, curr_frame):
    """動きと注目点"""
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    if prev_gray is None:
        return 0.0, np.nan, np.nan, curr_gray
    
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

def _extract_face_features(frame_rgb, face_mesh, h, w):
    """顔特徴量"""
    face_data = {
        'face_count': 0,
        'face_center_x': np.nan,
        'face_center_y': np.nan,
        'face_size': 0.0,
        'face_mouth_open': 0.0,
        'face_eyebrow_raise': 0.0
    }
    
    # If face_mesh is None, return default values
    if face_mesh is None:
        return face_data
    
    results = face_mesh.process(frame_rgb)
    
    if results.multi_face_landmarks:
        face_data['face_count'] = len(results.multi_face_landmarks)
        lm = results.multi_face_landmarks[0].landmark
        
        xs = [l.x for l in lm]
        ys = [l.y for l in lm]
        face_data['face_center_x'] = sum(xs) / len(xs)
        face_data['face_center_y'] = sum(ys) / len(ys)
        face_data['face_size'] = (max(xs) - min(xs)) * (max(ys) - min(ys))
        
        # 口の開き
        upper = np.array([lm[13].x * w, lm[13].y * h])
        lower = np.array([lm[14].x * w, lm[14].y * h])
        left = np.array([lm[61].x * w, lm[61].y * h])
        right = np.array([lm[291].x * w, lm[291].y * h])
        face_data['face_mouth_open'] = np.linalg.norm(upper - lower) / (np.linalg.norm(left - right) + 1e-6)
        
        # 眉の上がり
        left_eye = np.array([lm[159].x * w, lm[159].y * h])
        left_brow = np.array([lm[65].x * w, lm[65].y * h])
        face_h = abs(lm[152].y * h - lm[10].y * h)
        face_data['face_eyebrow_raise'] = np.linalg.norm(left_eye - left_brow) / (face_h + 1e-6)
    
    return face_data

def _extract_clip_features(frame_rgb, clip_model, clip_processor, device):
    """CLIP embeddings"""
    pil_img = Image.fromarray(frame_rgb)
    inputs = clip_processor(images=pil_img, return_tensors="pt")
    
    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    with torch.no_grad():
        img_feats = clip_model.get_image_features(**inputs)
    
    return img_feats.cpu().numpy().flatten().tolist()

# ==========================================
# メイン処理
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Extract features from videos (parallel)")
    parser.add_argument("input_dir", type=str, help="Directory containing video files")
    parser.add_argument("--output-dir", type=str, default="./input_features", help="Output directory")
    parser.add_argument("--n-jobs", type=int, default=N_JOBS, help="Number of parallel jobs")
    parser.add_argument("--extensions", type=str, nargs="+", default=[".mp4", ".mov", ".avi", ".mkv"],
                        help="Video file extensions")
    
    args = parser.parse_args()
    
    # 動画ファイルを検索
    video_files = []
    for ext in args.extensions:
        pattern = os.path.join(args.input_dir, f"**/*{ext}")
        video_files.extend(glob.glob(pattern, recursive=True))
    
    if not video_files:
        print(f"No video files found in {args.input_dir}")
        return
    
    # 出力ディレクトリを作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\nFound {len(video_files)} video files")
    print(f"Starting parallel extraction with {args.n_jobs} jobs...\n")
    
    # 並列処理
    results = Parallel(
        n_jobs=args.n_jobs,
        verbose=1,
        backend="multiprocessing",
        maxtasksperchild=1  # メモリ解放
    )(
        delayed(extract_features_worker)(vf, args.output_dir) for vf in video_files
    )
    
    # サマリー
    success_count = sum(1 for r in results if r and r["status"] == "Success")
    error_results = [r for r in results if r and r["status"] == "Error"]
    
    print("\n" + "="*70)
    print("Batch Processing Complete!")
    print("="*70)
    print(f"Total files: {len(video_files)}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {len(error_results)}")
    
    if error_results:
        print("\nFailed files:")
        for r in error_results[:5]:
            print(f"  - {r['file']}: {r['message']}")
        if len(error_results) > 5:
            print(f"  (and {len(error_results) - 5} more...)")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
