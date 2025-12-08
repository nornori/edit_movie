import os
import glob
import math
import pandas as pd
import numpy as np
import json
import librosa
import soundfile as sf
import whisper  # pip install openai-whisper
from pydub import AudioSegment # pip install pydub
from typing import Dict, List, Any

# ==============================================================================
#  設定エリア (環境に合わせて変更してください)
# ==============================================================================
INPUT_JSON_DIR = "./night_run_data_parallel"    # 編集結果JSONファイルがあるフォルダ
OUTPUT_FEATURE_DIR = "./input_features"         # 特徴量データ（CSV形式）の出力先
JSON_FILE_EXT = ".xml.json"
VIDEO_FILE_EXT = ".mp4"
OUTPUT_FILE_SUFFIX = "_features.csv"
TIME_STEP = 0.1                                 # サンプリング刻み幅（秒）

# Whisperモデルの設定 ("tiny", "base", "small", "medium", "large")
# 精度と速度のバランスが良い "small" を推奨
WHISPER_MODEL_SIZE = "small"
# ==============================================================================

print(f"[{WHISPER_MODEL_SIZE}] モデルをロード中... (これには少し時間がかかります)")
model = whisper.load_model(WHISPER_MODEL_SIZE)
print("モデルロード完了。バッチ処理を開始します。")

# --- ヘルパー関数 ---

def load_xml_json(json_path: str) -> Dict[str, Any]:
    """*.xml.jsonファイルを読み込む"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_base_video_path_and_name(data: Dict[str, Any]) -> tuple[str, str, str]:
    """JSONメタデータから、元の動画の絶対パスとファイル名を特定して返す"""
    video_path_raw = data.get("meta", {}).get("video_path", "")
    if not video_path_raw:
        return "", "", ""
    
    # OSのパス区切り文字を正規化
    video_path_normalized = video_path_raw.replace("\\", "/")
    
    base_name = os.path.basename(video_path_normalized)
    video_stem = base_name.replace(VIDEO_FILE_EXT, "")
    
    # ファイル名に拡張子がない場合の補完
    if "." not in base_name:
        base_name += VIDEO_FILE_EXT
        
    return video_path_normalized, base_name, video_stem

# --- 【本番用】文字起こし処理 ---

def get_whisper_features(audio_path: str) -> List[Dict[str, Any]]:
    """
    OpenAI Whisperを使って音声ファイルから単語ごとのタイムスタンプを取得する
    """
    try:
        # Whisper実行 (word_timestamps=True で単語ごとの時間を取得)
        result = model.transcribe(audio_path, word_timestamps=True)
        
        word_list = []
        # セグメント -> 単語 へと分解してリスト化
        for segment in result.get('segments', []):
            for word_info in segment.get('words', []):
                word_list.append({
                    "word": word_info['word'],
                    "start": word_info['start'],
                    "end": word_info['end']
                })
        return word_list

    except Exception as e:
        print(f"  [Whisper Error] 文字起こしに失敗: {e}")
        return []

def align_text_features(whisper_results: List[Dict[str, Any]], total_duration: float, time_step: float) -> pd.DataFrame:
    """
    Whisperの結果（疎なデータ）を、0.1秒刻みの密な時系列データにマッピングする
    """
    num_steps = int(math.ceil(total_duration / time_step))
    time_points = [round(i * time_step, 6) for i in range(num_steps + 1)]
    
    text_records = []
    
    for t in time_points:
        current_word = np.nan
        is_active = 0
        
        # 現在の時刻 t が、どの単語の期間に含まれるかチェック
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

# --- 【本番用】音声特徴量抽出 & 統合関数 ---

def extract_features_implementation(video_full_path: str, time_step: float) -> pd.DataFrame:
    """
    動画ファイルから音声・テキスト特徴量を抽出し、統合DataFrameを返す
    """
    if not os.path.exists(video_full_path):
        print(f"  [ERROR] 動画ファイルが見つかりません: {video_full_path}")
        return pd.DataFrame()

    temp_wav_path = "temp_process_audio.wav"
    
    try:
        print(f"  -> 音声抽出中: {os.path.basename(video_full_path)} ...")
        
        # 1. PyDubで動画から音声を抽出し、WAV (16kHz, mono) に変換
        audio = AudioSegment.from_file(video_full_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(temp_wav_path, format="wav")
        
        # 2. Librosaで音声データをロード (RMS計算用)
        y, sr = librosa.load(temp_wav_path, sr=16000)
        total_duration = librosa.get_duration(y=y, sr=sr)
        
        # 3. RMS (音量エネルギー) の計算
        # 0.1秒ごとのフレーム長を設定
        frame_length = int(time_step * sr)
        hop_length = frame_length 
        
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # 4. 簡易VAD (発話判定)
        # RMSがしきい値(0.01)を超えたら「発話中」とみなす簡易ロジック
        vad_threshold = 0.01
        is_speaking = (rms > vad_threshold).astype(int)
        
        # 5. 沈黙時間の計算 (連続する沈黙の長さを累積)
        silence_duration_ms = []
        current_silence = 0
        for speak_flag in is_speaking:
            if speak_flag == 0:
                current_silence += int(time_step * 1000)
            else:
                current_silence = 0
            silence_duration_ms.append(current_silence)
        
        # 6. Whisperによる文字起こし実行
        print(f"  -> Whisper文字起こし実行中...")
        whisper_results = get_whisper_features(temp_wav_path)
        
        # 7. テキストデータを時間軸にマッピング
        df_text = align_text_features(whisper_results, total_duration, time_step)
        
        # 8. 音声データをDataFrame化
        # rmsとis_speakingの長さが時間軸と微妙にずれる場合があるため調整
        min_len = min(len(rms), len(df_text))
        
        df_audio = pd.DataFrame({
            'time': df_text['time'][:min_len], # テキスト側の時間を基準にする
            'audio_energy_rms': rms[:min_len],
            'audio_is_speaking': is_speaking[:min_len],
            'silence_duration_ms': silence_duration_ms[:min_len],
            # 話者識別は高度なため今回はプレースホルダ (NaN) とします
            'speaker_id': np.nan 
        })
        
        # 9. 結合 (Audio + Text)
        # indexではなくtime列でmergeする場合もありますが、ここでは行数が揃っている前提でconcat
        df_final = pd.concat([df_audio, df_text[['text_is_active', 'text_word']].iloc[:min_len]], axis=1)
        
        return df_final

    except Exception as e:
        print(f"  [CRITICAL ERROR] 解析中にエラー発生: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
        
    finally:
        # 一時ファイルの削除
        if os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
            except:
                pass

# --- バッチ処理実行関数 ---

def batch_extract_all_features():
    """
    JSONファイルパスから動画を特定し、全特徴量を生成するメインループ
    """
    os.makedirs(OUTPUT_FEATURE_DIR, exist_ok=True)
    
    # JSONファイルを検索
    json_paths = glob.glob(os.path.join(INPUT_JSON_DIR, f"*{JSON_FILE_EXT}"))

    if not json_paths:
        print(f"[ERROR] JSONフォルダ '{INPUT_JSON_DIR}' にファイルが見つかりません。")
        return

    print(f"==============================================================")
    print(f"  🚀 特徴量抽出バッチ処理開始 ({len(json_paths)} 件) ")
    print(f"==============================================================")

    for idx, json_path in enumerate(json_paths):
        print(f"--- ({idx+1}/{len(json_paths)}) JSON解析: {os.path.basename(json_path)}")
        
        # JSONロード & パス特定
        data = load_xml_json(json_path)
        video_full_path, _, video_stem = get_base_video_path_and_name(data)
        
        # 出力パス
        output_path = os.path.join(OUTPUT_FEATURE_DIR, f"{video_stem}{OUTPUT_FILE_SUFFIX}")
        
        # 既に存在する場合はスキップするロジックを入れることも可能
        # if os.path.exists(output_path):
        #     print("  -> 既に存在するためスキップします。")
        #     continue

        if not video_full_path:
            print("  [SKIPPED] 動画パスがJSONに含まれていません。")
            continue

        # ★★★ 特徴量抽出の実行 ★★★
        df_features = extract_features_implementation(video_full_path, TIME_STEP)
        
        if not df_features.empty:
            # CSV保存
            df_features.to_csv(output_path, index=False, float_format='%.6f')
            print(f"  [SUCCESS] 保存完了 ({len(df_features)}行): {os.path.basename(output_path)}")
        else:
            print("  [FAILED] 特徴量の生成に失敗しました。")

    print("\n[完了] 全てのバッチ処理が終了しました。")


if __name__ == "__main__":
    batch_extract_all_features()