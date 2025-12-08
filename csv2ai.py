import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import os
import argparse
from sklearn.feature_extraction.text import HashingVectorizer

# ==========================================
# 設定 (学習時と同じ値にする必要があります)
# ==========================================
SEQUENCE_LENGTH = 50  # 過去5秒分を見る
TEXT_VECTOR_DIM = 32  # テキストの次元数

# ==========================================
# モデル定義 (学習時と同じ構造)
# ==========================================
class EditorAI(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super(EditorAI, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        
        # 出力層
        self.head_is_used = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.head_scale = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_pos = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 2))
        self.head_triggers = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 2), nn.Sigmoid())

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :] # 最後のステップの出力を採用
        return self.head_is_used(last_step), self.head_scale(last_step), self.head_pos(last_step), self.head_triggers(last_step)

# ==========================================
# 推論実行関数
# ==========================================
def run_inference(csv_path, model_path, scaler_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. データの読み込み
    if not os.path.exists(csv_path):
        print(f"Error: CSVファイルが見つかりません -> {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    print(f"Loaded features: {len(df)} frames")

    # 2. 前処理 (学習時と全く同じ処理を行う)
    # -----------------------------------------------------
    # 数値特徴量 (target_... と time, text_word 以外)
    feature_cols = [c for c in df.columns if not c.startswith('target_') and c not in ['time', 'text_word']]
    X_numeric = df[feature_cols].values
    X_numeric = np.nan_to_num(X_numeric, nan=0.0)
    
    # テキスト特徴量 (HashingVectorizer)
    text_vectorizer = HashingVectorizer(n_features=TEXT_VECTOR_DIM, alternate_sign=False)
    text_data = df['text_word'].fillna("").astype(str).tolist()
    X_text = text_vectorizer.transform(text_data).toarray()
    
    # 結合
    X_raw = np.hstack([X_numeric, X_text])
    
    # スケーリング (学習時に作ったscaler.pklを使う)
    if not os.path.exists(scaler_path):
        print(f"Error: {scaler_path} が見つかりません。学習を実行したフォルダにありますか？")
        return
        
    scaler = joblib.load(scaler_path)
    
    # 次元のチェック
    if X_raw.shape[1] != scaler.n_features_in_:
        print(f"Error: データの列数が合いません。")
        print(f"  学習時の列数: {scaler.n_features_in_}")
        print(f"  今回の列数: {X_raw.shape[1]}")
        print("  ヒント: 学習データと今回の推論用データの作成ロジック(列構成)を一致させてください。")
        return

    X = scaler.transform(X_raw)

    # 3. モデルの準備
    input_dim = X.shape[1]
    model = EditorAI(input_dim).to(device)
    
    if not os.path.exists(model_path):
        print(f"Error: モデルファイルが見つかりません -> {model_path}")
        return

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 4. 推論ループ (スライディングウィンドウ)
    results = []
    
    # 先頭のパディング (最初の5秒間も推論できるようにゼロ埋めデータなどを追加)
    # 学習時、シーケンスの先頭はどう扱ったかに依りますが、ここでは先頭行を複製してパディングします
    padding = np.tile(X[0], (SEQUENCE_LENGTH - 1, 1))
    X_padded = np.vstack([padding, X])
    
    print("AI is editing video... (Inference)")
    
    # バッチ処理で高速化も可能ですが、わかりやすく1つずつ処理します
    with torch.no_grad():
        for i in range(len(X)):
            # 過去50フレーム分を取得 (t-49 ~ t)
            seq = X_padded[i : i + SEQUENCE_LENGTH]
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            
            # AI予測
            p_used, p_scale, p_pos, p_trig = model(seq_tensor)
            
            results.append({
                'time': df.iloc[i]['time'],
                'text': df.iloc[i]['text_word'],
                
                # AIの判断結果
                'ai_cut_score': p_used.item(),        # 採用確率 (0.0~1.0)
                'ai_scale': p_scale.item(),           # ズーム倍率
                'ai_pos_x': p_pos[0, 0].item(),       # 位置X
                'ai_pos_y': p_pos[0, 1].item(),       # 位置Y
                'ai_graphic': p_trig[0, 0].item(),    # テロップ確率
                'ai_broll': p_trig[0, 1].item(),      # B-roll確率
            })
            
            if i % 500 == 0:
                print(f"  Processed {i}/{len(X)} frames...")

    # 5. CSV出力
    out_df = pd.DataFrame(results)
    
    # 最終判断列を追加 (閾値 0.5)
    out_df['decision'] = out_df['ai_cut_score'].apply(lambda x: 'KEEP' if x > 0.5 else 'CUT')
    
    out_df.to_csv(output_path, index=False)
    print(f"\n✅ Done! Inference result saved to: {output_path}")
    print("Preivew:")
    print(out_df[['time', 'decision', 'ai_scale', 'text']].head(10))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Video Editor Inference")
    parser.add_argument("input_csv", type=str, help="Input features CSV file")
    parser.add_argument("--model", type=str, default="editor_ai_model.pth", help="Path to trained model (.pth)")
    parser.add_argument("--scaler", type=str, default="scaler.pkl", help="Path to scaler (.pkl)")
    parser.add_argument("--output", type=str, default="final_timeline.csv", help="Output CSV file path")
    
    args = parser.parse_args()
    
    run_inference(args.input_csv, args.model, args.scaler, args.output)