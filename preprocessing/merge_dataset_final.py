import os
import glob
import json
import pandas as pd
import numpy as np

# ==============================================================================
#  設定エリア (フォルダ名を確認してください！)
# ==============================================================================
# 正解ラベル (JSON) があるフォルダ
# ※ あなたの環境に合わせて名前を確認してください (output_labels か output.labels か)
LABEL_DIR = "./output_labels" 

# 特徴量 (CSV) があるフォルダ
FEATURE_DIR = "./input_features"

# 最終データの出力先
OUTPUT_DIR = "./final_dataset"

# ファイルの識別子
LABEL_SUFFIX = "_labels.json"
AUDIO_SUFFIX = "_features.csv"
VISUAL_SUFFIX = "_visual_features.csv"
# ==============================================================================

def load_label_json_as_df(json_path):
    """ラベルJSONを読み込み、DataFrameに変換する"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        
        # 必要な列だけを抽出・リネーム
        target_cols = {
            'time': 'time',
            'is_used': 'target_is_used',
            'main_scale': 'target_scale',
            'main_pos_x': 'target_pos_x',
            'main_pos_y': 'target_pos_y',
            'is_graphic_active': 'target_graphic',
            'is_broll_active': 'target_broll'
        }
        
        available_cols = {k: v for k, v in target_cols.items() if k in df.columns}
        df = df[list(available_cols.keys())].rename(columns=available_cols)
        
        # 時間合わせ (小数点1桁)
        df['time'] = df['time'].round(1)
        
        # 重複時間の削除 (念のため)
        df = df.drop_duplicates(subset=['time'])
        
        return df
        
    except Exception as e:
        print(f"  [Error] ラベル読み込み失敗: {e}")
        return None

def merge_datasets():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # フォルダ内の全CSVを取得
    all_files = glob.glob(os.path.join(FEATURE_DIR, "*.csv"))
    
    # 音声特徴量ファイルだけを厳密に抽出 (映像特徴量を除外)
    audio_files = [f for f in all_files if f.endswith(AUDIO_SUFFIX) and not f.endswith(VISUAL_SUFFIX)]
    
    print(f"🚀 データセット統合を開始します (対象: {len(audio_files)} 件)")
    print(f"   参照ラベルフォルダ: {LABEL_DIR}")
    
    success_count = 0
    
    for audio_path in audio_files:
        # ファイル名から動画ID (stem) を特定
        base_name = os.path.basename(audio_path)
        video_stem = base_name.replace(AUDIO_SUFFIX, "")
        
        # パスの構築
        visual_path = os.path.join(FEATURE_DIR, f"{video_stem}{VISUAL_SUFFIX}")
        label_path = os.path.join(LABEL_DIR, f"{video_stem}{LABEL_SUFFIX}")
        output_csv_path = os.path.join(OUTPUT_DIR, f"{video_stem}_dataset.csv")
        
        # ラベルファイルの存在チェック
        if not os.path.exists(label_path):
            # ラベルがない場合はスキップ (学習できないため)
            print(f"--- Skip: {video_stem}")
            print(f"    ❌ 正解ラベルなし: {label_path}")
            continue

        print(f"\n--- 処理中: {video_stem} ---")

        # 1. 音声・テキスト特徴量
        df_audio = pd.read_csv(audio_path)
        df_audio['time'] = df_audio['time'].round(1)
        
        # 2. 映像特徴量
        if os.path.exists(visual_path):
            df_visual = pd.read_csv(visual_path)
            df_visual['time'] = df_visual['time'].round(1)
            # 重複列 (time以外) を避けてマージ
            cols_to_use = df_visual.columns.difference(df_audio.columns).tolist()
            cols_to_use.append('time')
            df_features = pd.merge(df_audio, df_visual[cols_to_use], on='time', how='outer')
        else:
            print(f"    ⚠️ 映像特徴量なし (音声のみ使用)")
            df_features = df_audio

        # 3. 正解ラベル
        df_label = load_label_json_as_df(label_path)
        if df_label is None: continue

        # 4. 最終結合 (Inner Join: 正解がある時間だけ残す)
        df_final = pd.merge(df_features, df_label, on='time', how='inner')
        
        # 欠損埋め
        df_final = df_final.fillna(0)
        
        # 保存
        if len(df_final) > 0:
            df_final.to_csv(output_csv_path, index=False)
            print(f"    ✅ 統合成功! データサイズ: {df_final.shape}")
            success_count += 1
        else:
            print(f"    ⚠️ 統合結果が0行でした (時間のズーム不一致など)")

    print(f"\n🎉 全工程終了: {success_count} / {len(audio_files)} 件のデータセットを作成しました。")
    print(f"保存先: {OUTPUT_DIR}")

if __name__ == "__main__":
    merge_datasets()