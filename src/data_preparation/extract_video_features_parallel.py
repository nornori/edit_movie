"""
Video Feature Extraction Script (並列処理版)

動画から音声・視覚特徴量を並列処理で高速抽出します。

【特徴】
- 並列処理対応（joblib使用）
- GPU/CPUの自動切り替え
- メモリ効率化（maxtasksperchild=1）
- プログレスバー表示
- エラーハンドリング強化
- 話者識別機能（pyannote.audio）
- 感情表現検出（ピッチ、MFCC、スペクトル特徴）

【抽出される特徴量】
- 音声: 215次元
  - 基本: 4次元 (rms, speaking, silence, speaker_id)
  - 話者埋め込み: 192次元 (speaker embedding)
  - 感情表現: 16次元 (pitch, pitch_std, spectral_centroid, zcr, mfcc×13)
  - テキスト: 3次元 (text_active, telop_active, + 文字列)
- 視覚: 522次元 (10スカラー + 512 CLIP)
- 合計: 737次元 (215 + 522)
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
from typing import List, Dict, Any, Optional, Tuple
import warnings
import sys
warnings.filterwarnings('ignore')

# Speaker Diarization
try:
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("WARNING: pyannote.audio not installed. Speaker identification will be disabled.")
    print("   To enable: pip install pyannote.audio")
    print("   You may also need a Hugging Face token for some models.")

# ==========================================
# 設定
# ==========================================
TIME_STEP = 0.1              # 基本サンプリング (0.1秒 = 10 FPS)
CLIP_STEP = 1.0              # CLIP解析間隔 (1.0秒)
ANALYSIS_WIDTH = 640         # 解析用画像の横幅
WHISPER_MODEL_SIZE = "small" # Whisperモデル
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
N_JOBS = 4                   # 並列処理数（CPUコア数に応じて調整）

# Speaker Identification
SPEAKER_EMBEDDING_MODEL = "speechbrain/spkrec-ecapa-voxceleb"  # 192次元の話者埋め込み
SPEAKER_EMBEDDING_DIM = 192  # 埋め込みの次元数
ENABLE_SPEAKER_ID = True     # 話者識別を有効化

# Emotion-related Audio Features
ENABLE_EMOTION_FEATURES = True  # 感情表現検出用の音響特徴を有効化
MFCC_DIM = 13  # MFCC係数の次元数

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
print(f"Speaker ID: {'Enabled' if ENABLE_SPEAKER_ID and PYANNOTE_AVAILABLE else 'Disabled'}")
if ENABLE_SPEAKER_ID and PYANNOTE_AVAILABLE:
    print(f"  Model: {SPEAKER_EMBEDDING_MODEL}")
    print(f"  Embedding Dim: {SPEAKER_EMBEDDING_DIM}")
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
        
        # Speaker Embedding Model
        speaker_model = None
        if ENABLE_SPEAKER_ID and PYANNOTE_AVAILABLE:
            try:
                # Windowsでシンボリックリンクの問題を回避
                os.environ['SPEECHBRAIN_CACHE_STRATEGY'] = 'copy'
                
                speaker_model = PretrainedSpeakerEmbedding(
                    SPEAKER_EMBEDDING_MODEL,
                    device=torch.device(device)
                )
                print(f"  OK: Speaker embedding model loaded: {SPEAKER_EMBEDDING_MODEL}")
            except Exception as e:
                print(f"  WARNING: Failed to load speaker model: {e}")
                print(f"     Speaker embeddings will be set to zeros")
                speaker_model = None
        
        # MediaPipe FaceMesh
        # Note: MediaPipe may fail to initialize due to:
        # - Non-ASCII characters in path (Japanese, Chinese, etc.) - MOST COMMON ISSUE
        # - Missing dependencies (protobuf, opencv-contrib-python)
        # - GPU/CUDA compatibility issues
        # If initialization fails, face features will be set to default values (zeros)
        face_mesh = None
        face_mesh_available = False
        try:
            mp_face_mesh = mp.solutions.face_mesh
            face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            face_mesh_available = True
            print(f"  OK: MediaPipe FaceMesh initialized successfully")
        except Exception as e:
            print(f"  WARNING: MediaPipe FaceMesh initialization failed: {e}")
            print(f"     Face features will be set to default values (zeros)")
            print(f"     This won't affect other features (audio, visual, CLIP)")
            print(f"     Common causes:")
            print(f"       - Non-ASCII characters in project path (日本語など)")
            print(f"       - Missing dependencies: pip install mediapipe opencv-contrib-python")
            print(f"     To fix path issue: Move project to ASCII-only path (e.g., C:\\projects\\xmlai)")
            face_mesh_available = False
        
        # 音声特徴量を抽出（テロップ情報と話者識別を含む）
        df_audio = _extract_audio_features(
            video_path, 
            whisper_model, 
            speaker_model=speaker_model,
            xml_dir='editxml'
        )
        
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

def _extract_audio_features(
    video_path: str, 
    whisper_model, 
    speaker_model=None,
    xml_dir: str = 'editxml'
) -> pd.DataFrame:
    """音声特徴量を抽出（テロップ情報と話者識別を含む）"""
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
        
        # 感情表現検出用の音響特徴量
        emotion_features = None
        if ENABLE_EMOTION_FEATURES:
            emotion_features = _extract_emotion_features(y, sr, frame_length, hop_length)
        
        # Whisper文字起こし
        whisper_results = _get_whisper_features(temp_wav, whisper_model)
        df_text = _align_text_features(whisper_results, total_duration)
        
        # テロップ情報を抽出
        df_telop = _extract_telop_features(video_path, total_duration, xml_dir)
        
        # 話者識別と埋め込み抽出
        df_speaker = _extract_speaker_features(
            temp_wav, y, sr, total_duration, is_speaking, speaker_model
        )
        
        # DataFrame統合
        min_len = min(len(rms), len(df_text), len(df_telop), len(df_speaker))
        if emotion_features is not None:
            min_len = min(min_len, len(emotion_features['pitch']))
        
        # 基本的な音声特徴量
        df_audio = pd.DataFrame({
            'time': df_text['time'][:min_len],
            'audio_energy_rms': rms[:min_len],
            'audio_is_speaking': is_speaking[:min_len],
            'silence_duration_ms': silence_duration_ms[:min_len],
            'speaker_id': df_speaker['speaker_id'][:min_len],
            'text_is_active': df_text['text_is_active'][:min_len],
            'text_word': df_text['text_word'][:min_len],
            'telop_active': df_telop['telop_active'][:min_len],
            'telop_text': df_telop['telop_text'][:min_len]
        })
        
        # 感情表現特徴量を追加（16次元）
        if emotion_features is not None:
            df_audio['pitch_f0'] = emotion_features['pitch'][:min_len]
            df_audio['pitch_std'] = emotion_features['pitch_std'][:min_len]
            df_audio['spectral_centroid'] = emotion_features['spectral_centroid'][:min_len]
            df_audio['zcr'] = emotion_features['zcr'][:min_len]
            
            # MFCC（13次元）
            for i in range(MFCC_DIM):
                df_audio[f'mfcc_{i}'] = emotion_features['mfcc'][i][:min_len]
        
        # 話者埋め込みを追加（192次元）
        for i in range(SPEAKER_EMBEDDING_DIM):
            df_audio[f'speaker_emb_{i}'] = df_speaker[f'speaker_emb_{i}'][:min_len]
        
        return df_audio
        
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

def _extract_emotion_features(y: np.ndarray, sr: int, frame_length: int, hop_length: int) -> Dict[str, np.ndarray]:
    """
    感情表現検出用の音響特徴量を抽出
    
    Args:
        y: 音声波形データ
        sr: サンプリングレート
        frame_length: フレーム長
        hop_length: ホップ長
    
    Returns:
        感情関連特徴量の辞書（16次元）
        - pitch_f0: 基本周波数（ピッチ）
        - pitch_std: ピッチの標準偏差（抑揚）
        - spectral_centroid: スペクトル重心（音色の明るさ）
        - zcr: ゼロ交差率
        - mfcc: メル周波数ケプストラム係数（13次元）
    """
    try:
        # 基本周波数（F0/ピッチ）を抽出
        # pyin: 確率的YINアルゴリズム（ノイズに強い）
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'),  # 65 Hz（低い男性の声）
            fmax=librosa.note_to_hz('C7'),  # 2093 Hz（高い女性の声）
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length
        )
        
        # NaNを0で埋める（無音区間）
        f0 = np.nan_to_num(f0, nan=0.0)
        
        # ピッチの変動（標準偏差）を計算
        # ウィンドウサイズ: 1秒（10フレーム @ 10 FPS）
        window_size = 10
        pitch_std = []
        for i in range(len(f0)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(f0), i + window_size // 2 + 1)
            window = f0[start_idx:end_idx]
            # 無音区間（f0=0）を除外して標準偏差を計算
            non_zero = window[window > 0]
            if len(non_zero) > 0:
                pitch_std.append(np.std(non_zero))
            else:
                pitch_std.append(0.0)
        pitch_std = np.array(pitch_std)
        
        # スペクトル重心（音色の明るさ）
        spectral_centroid = librosa.feature.spectral_centroid(
            y=y, 
            sr=sr, 
            hop_length=hop_length
        )[0]
        
        # ゼロ交差率（音の粗さ）
        zcr = librosa.feature.zero_crossing_rate(
            y, 
            frame_length=frame_length, 
            hop_length=hop_length
        )[0]
        
        # MFCC（メル周波数ケプストラム係数）
        # 音色の詳細な特徴（13次元）
        mfcc = librosa.feature.mfcc(
            y=y, 
            sr=sr, 
            n_mfcc=MFCC_DIM,
            hop_length=hop_length
        )
        
        return {
            'pitch': f0,
            'pitch_std': pitch_std,
            'spectral_centroid': spectral_centroid,
            'zcr': zcr,
            'mfcc': mfcc
        }
        
    except Exception as e:
        print(f"    WARNING: Failed to extract emotion features: {e}")
        # エラー時はゼロで埋める
        num_frames = int(np.ceil(len(y) / hop_length))
        return {
            'pitch': np.zeros(num_frames),
            'pitch_std': np.zeros(num_frames),
            'spectral_centroid': np.zeros(num_frames),
            'zcr': np.zeros(num_frames),
            'mfcc': np.zeros((MFCC_DIM, num_frames))
        }


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

def _extract_speaker_features(
    audio_path: str,
    y: np.ndarray,
    sr: int,
    total_duration: float,
    is_speaking: np.ndarray,
    speaker_model
) -> pd.DataFrame:
    """
    話者識別と埋め込みベクトルを抽出
    
    Args:
        audio_path: 音声ファイルのパス
        y: 音声波形データ
        sr: サンプリングレート
        total_duration: 総時間（秒）
        is_speaking: 発話フラグの配列
        speaker_model: 話者埋め込みモデル
    
    Returns:
        話者ID（0-based）と埋め込みベクトル（256次元）を含むDataFrame
    """
    num_steps = int(np.ceil(total_duration / TIME_STEP))
    time_points = [round(i * TIME_STEP, 6) for i in range(num_steps + 1)]
    
    # デフォルト値（話者識別が無効の場合）
    default_speaker_id = -1  # -1 = 話者不明
    default_embedding = np.zeros(SPEAKER_EMBEDDING_DIM)
    
    # 話者モデルが利用できない場合
    if speaker_model is None or not ENABLE_SPEAKER_ID:
        records = []
        for t in time_points:
            record = {'speaker_id': default_speaker_id}
            for i in range(SPEAKER_EMBEDDING_DIM):
                record[f'speaker_emb_{i}'] = default_embedding[i]
            records.append(record)
        return pd.DataFrame(records)
    
    try:
        # 発話区間を検出（0.5秒以上の連続発話）
        speech_segments = _detect_speech_segments(is_speaking, min_duration=0.5)
        
        if len(speech_segments) == 0:
            # 発話がない場合
            records = []
            for t in time_points:
                record = {'speaker_id': default_speaker_id}
                for i in range(SPEAKER_EMBEDDING_DIM):
                    record[f'speaker_emb_{i}'] = default_embedding[i]
                records.append(record)
            return pd.DataFrame(records)
        
        # 各発話区間から話者埋め込みを抽出
        segment_embeddings = []
        for start_time, end_time in speech_segments:
            # 音声セグメントを切り出し
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment_audio = y[start_sample:end_sample]
            
            # 最小長チェック（0.5秒以上）
            if len(segment_audio) < sr * 0.5:
                continue
            
            # 話者埋め込みを抽出
            try:
                # PyTorchテンソルに変換（16kHzのモノラル音声）
                # 形状: (batch_size, num_channels, num_samples) = (1, 1, length)
                segment_tensor = torch.from_numpy(segment_audio).float().unsqueeze(0).unsqueeze(0)
                
                # 埋め込みを抽出
                with torch.no_grad():
                    # PretrainedSpeakerEmbeddingは(batch, channel, samples)の形状を期待
                    result = speaker_model(segment_tensor)
                    
                    # 結果の型を確認
                    if isinstance(result, tuple):
                        # タプルの場合は最初の要素を取得
                        embedding = result[0]
                    else:
                        embedding = result
                    
                    # 結果を numpy配列に変換
                    if torch.is_tensor(embedding):
                        embedding = embedding.cpu().numpy().flatten()
                    elif isinstance(embedding, np.ndarray):
                        embedding = embedding.flatten()
                    else:
                        # その他の型の場合はスキップ
                        print(f"    WARNING: Unexpected embedding type for segment {start_time:.2f}-{end_time:.2f}s: {type(embedding)}")
                        continue
                
                segment_embeddings.append({
                    'start': start_time,
                    'end': end_time,
                    'embedding': embedding
                })
            except Exception as e:
                print(f"    WARNING: Failed to extract embedding for segment {start_time:.2f}-{end_time:.2f}s: {e}")
                continue
        
        if len(segment_embeddings) == 0:
            # 埋め込み抽出に失敗した場合
            records = []
            for t in time_points:
                record = {'speaker_id': default_speaker_id}
                for i in range(SPEAKER_EMBEDDING_DIM):
                    record[f'speaker_emb_{i}'] = default_embedding[i]
                records.append(record)
            return pd.DataFrame(records)
        
        # 話者クラスタリング（簡易版：コサイン類似度ベース）
        # 閾値: 0.5-0.6 = 緩い（少ない話者数）、0.7-0.8 = 厳しい（多い話者数）
        speaker_clusters = _cluster_speakers(segment_embeddings, threshold=0.55)
        
        # 各タイムステップに話者IDと埋め込みを割り当て
        records = []
        for t in time_points:
            speaker_id = default_speaker_id
            embedding = default_embedding.copy()
            
            # このタイムステップが含まれる発話区間を探す
            for seg in segment_embeddings:
                if seg['start'] <= t < seg['end']:
                    # 話者IDを取得
                    speaker_id = speaker_clusters.get(id(seg), 0)
                    embedding = seg['embedding']
                    break
            
            record = {'speaker_id': speaker_id}
            for i in range(SPEAKER_EMBEDDING_DIM):
                record[f'speaker_emb_{i}'] = embedding[i]
            records.append(record)
        
        return pd.DataFrame(records)
        
    except Exception as e:
        print(f"    WARNING: Speaker feature extraction failed: {e}")
        # エラー時はデフォルト値を返す
        records = []
        for t in time_points:
            record = {'speaker_id': default_speaker_id}
            for i in range(SPEAKER_EMBEDDING_DIM):
                record[f'speaker_emb_{i}'] = default_embedding[i]
            records.append(record)
        return pd.DataFrame(records)


def _detect_speech_segments(
    is_speaking: np.ndarray, 
    min_duration: float = 0.5
) -> List[Tuple[float, float]]:
    """
    発話区間を検出
    
    Args:
        is_speaking: 発話フラグの配列
        min_duration: 最小継続時間（秒）
    
    Returns:
        (start_time, end_time)のリスト
    """
    segments = []
    in_speech = False
    start_idx = 0
    
    for i, speaking in enumerate(is_speaking):
        if speaking and not in_speech:
            # 発話開始
            start_idx = i
            in_speech = True
        elif not speaking and in_speech:
            # 発話終了
            start_time = start_idx * TIME_STEP
            end_time = i * TIME_STEP
            duration = end_time - start_time
            
            if duration >= min_duration:
                segments.append((start_time, end_time))
            
            in_speech = False
    
    # 最後まで発話が続いていた場合
    if in_speech:
        start_time = start_idx * TIME_STEP
        end_time = len(is_speaking) * TIME_STEP
        duration = end_time - start_time
        
        if duration >= min_duration:
            segments.append((start_time, end_time))
    
    return segments


def _cluster_speakers(
    segment_embeddings: List[Dict], 
    threshold: float = 0.55
) -> Dict[int, int]:
    """
    話者埋め込みをクラスタリング（改善版：階層的クラスタリング）
    
    Args:
        segment_embeddings: 埋め込みのリスト
        threshold: コサイン類似度の閾値（0.5-0.6推奨）
    
    Returns:
        segment_id -> speaker_id のマッピング
    """
    if len(segment_embeddings) == 0:
        return {}
    
    # 話者の代表埋め込み（各話者の平均埋め込み）
    speaker_representatives = []  # [(speaker_id, mean_embedding, [segment_ids])]
    speaker_mapping = {}
    next_speaker_id = 0
    
    for seg in segment_embeddings:
        seg_id = id(seg)
        seg_emb = seg['embedding']
        
        # 既存の話者と比較
        best_match_id = None
        best_similarity = -1.0
        
        for speaker_id, rep_emb, seg_ids in speaker_representatives:
            # コサイン類似度を計算
            similarity = np.dot(seg_emb, rep_emb) / (
                np.linalg.norm(seg_emb) * np.linalg.norm(rep_emb) + 1e-8
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = speaker_id
        
        # 閾値以上の類似度があれば同じ話者、なければ新しい話者
        if best_similarity >= threshold and best_match_id is not None:
            # 既存の話者に追加
            speaker_mapping[seg_id] = best_match_id
            
            # 代表埋め込みを更新（移動平均）
            for i, (spk_id, rep_emb, seg_ids) in enumerate(speaker_representatives):
                if spk_id == best_match_id:
                    seg_ids.append(seg_id)
                    # 新しい平均を計算
                    n = len(seg_ids)
                    speaker_representatives[i] = (
                        spk_id,
                        (rep_emb * (n - 1) + seg_emb) / n,
                        seg_ids
                    )
                    break
        else:
            # 新しい話者を作成
            speaker_mapping[seg_id] = next_speaker_id
            speaker_representatives.append((next_speaker_id, seg_emb, [seg_id]))
            next_speaker_id += 1
    
    return speaker_mapping


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
    """視覚特徴量を抽出（メモリ効率化版）"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_step = int(fps * TIME_STEP)
    if frame_step < 1:
        frame_step = 1
    
    # メモリ効率化: チャンク処理用の設定
    CHUNK_SIZE = 1000  # 1000フレームごとにメモリを解放
    records = []
    temp_csv_path = video_path + ".temp_visual.csv"
    is_first_chunk = True
    column_order = None  # 列の順序を保存
    
    current_frame_idx = 0
    prev_gray = None
    prev_hist = None
    
    # CLIP特徴量の補間用
    clip_embeddings = []  # (timestamp, embedding)のリスト
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
            
            # CLIP（1秒ごとに抽出）
            if (timestamp - last_clip_time) >= CLIP_STEP:
                clip_emb = _extract_clip_features(frame_rgb, clip_model, clip_processor, device)
                clip_embeddings.append((timestamp, clip_emb))
                last_clip_time = timestamp
            
            # CLIP特徴量を線形補間
            if len(clip_embeddings) == 0:
                # まだCLIP特徴量がない場合はゼロ
                interpolated_clip = [0.0] * 512
            elif len(clip_embeddings) == 1:
                # 1つしかない場合はそれを使用
                interpolated_clip = clip_embeddings[0][1]
            else:
                # 2つ以上ある場合は線形補間
                interpolated_clip = _interpolate_clip_features(timestamp, clip_embeddings)
            
            # レコード
            row = {
                'time': round(timestamp, 3),
                'scene_change': scene_score,
                'visual_motion': motion,
                'saliency_x': sal_x,
                'saliency_y': sal_y,
                **face_data
            }
            row.update({f'clip_{i}': v for i, v in enumerate(interpolated_clip)})
            records.append(row)
            
            # チャンクサイズに達したらCSVに書き出してメモリを解放
            if len(records) >= CHUNK_SIZE:
                df_chunk = pd.DataFrame(records)
                # 最初のチャンクで列の順序を保存
                if is_first_chunk:
                    column_order = df_chunk.columns.tolist()
                else:
                    # 2番目以降は列の順序を揃える
                    df_chunk = df_chunk[column_order]
                df_chunk.to_csv(temp_csv_path, mode='a', header=is_first_chunk, index=False, float_format='%.6f')
                is_first_chunk = False
                records = []  # メモリ解放
        
        current_frame_idx += 1
    
    cap.release()
    
    # 残りのレコードを書き出し
    if records:
        df_chunk = pd.DataFrame(records)
        if not is_first_chunk:
            # 列の順序を揃える
            df_chunk = df_chunk[column_order]
        df_chunk.to_csv(temp_csv_path, mode='a', header=is_first_chunk, index=False, float_format='%.6f')
    
    # 一時ファイルから読み込んで返す
    df_final = pd.read_csv(temp_csv_path)
    
    # 一時ファイルを削除
    try:
        os.remove(temp_csv_path)
    except:
        pass
    
    return df_final

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
    """
    顔特徴量を抽出
    
    Args:
        frame_rgb: RGB画像
        face_mesh: MediaPipe FaceMeshオブジェクト（Noneの場合はデフォルト値を返す）
        h: 画像の高さ
        w: 画像の幅
    
    Returns:
        顔特徴量の辞書
    """
    face_data = {
        'face_count': 0,
        'face_center_x': np.nan,
        'face_center_y': np.nan,
        'face_size': 0.0,
        'face_mouth_open': 0.0,
        'face_eyebrow_raise': 0.0
    }
    
    # MediaPipeが利用できない場合はデフォルト値を返す
    if face_mesh is None:
        return face_data
    
    try:
        results = face_mesh.process(frame_rgb)
    except Exception as e:
        # 処理中にエラーが発生した場合もデフォルト値を返す
        return face_data
    
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


def _interpolate_clip_features(timestamp: float, clip_embeddings: list) -> list:
    """
    CLIP特徴量を線形補間
    
    Args:
        timestamp: 現在のタイムスタンプ
        clip_embeddings: (timestamp, embedding)のリスト
    
    Returns:
        補間されたCLIP特徴量
    """
    # 現在のタイムスタンプの前後のCLIP特徴量を見つける
    prev_time, prev_emb = clip_embeddings[0]
    next_time, next_emb = clip_embeddings[-1]
    
    for i in range(len(clip_embeddings) - 1):
        t1, emb1 = clip_embeddings[i]
        t2, emb2 = clip_embeddings[i + 1]
        
        if t1 <= timestamp <= t2:
            prev_time, prev_emb = t1, emb1
            next_time, next_emb = t2, emb2
            break
    
    # タイムスタンプが範囲外の場合
    if timestamp < prev_time:
        return prev_emb
    if timestamp > next_time:
        return next_emb
    
    # 線形補間
    if next_time == prev_time:
        return prev_emb
    
    alpha = (timestamp - prev_time) / (next_time - prev_time)
    
    # 各次元を線形補間
    interpolated = []
    for v1, v2 in zip(prev_emb, next_emb):
        interpolated.append(v1 * (1 - alpha) + v2 * alpha)
    
    return interpolated

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
