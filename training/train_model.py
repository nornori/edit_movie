import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import HashingVectorizer
import joblib

# ==============================================================================
#  設定エリア
# ==============================================================================
DATASET_DIR = "./final_dataset"     # 学習データがあるフォルダ
MODEL_SAVE_PATH = "editor_ai_model.pth" # モデルの保存名
SCALER_SAVE_PATH = "scaler.pkl"     # データ変換器の保存名

SEQUENCE_LENGTH = 50  # 過去5秒分 (0.1秒 x 50) を見る
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

# テキスト特徴量の圧縮次元数 (言葉をいくつの数値で表現するか)
# 32次元あれば、日常会話の単語の違いをある程度表現できます
TEXT_VECTOR_DIM = 32

# 正解ラベル列
TARGET_COLS = [
    'target_is_used',       # カット
    'target_scale',         # ズーム
    'target_pos_x',         # 配置X
    'target_pos_y',         # 配置Y
    'target_graphic',       # テロップ
    'target_broll'          # B-roll
]
# ==============================================================================

class VideoEditorDataset(Dataset):
    def __init__(self, csv_path, scaler=None, text_vectorizer=None, sequence_length=50, is_train=True):
        self.seq_len = sequence_length
        
        # CSV読み込み
        df = pd.read_csv(csv_path)
        
        # 1. 正解データ (Y) の抽出
        Y_list = []
        for col in TARGET_COLS:
            if col in df.columns:
                Y_list.append(df[col].values)
            else:
                Y_list.append(np.zeros(len(df)))
        Y = np.column_stack(Y_list)
        self.Y = np.nan_to_num(Y, nan=0.0)

        # 2. 特徴量 (X) の抽出
        # ターゲット列と時間列を除外
        feature_cols = [c for c in df.columns if not c.startswith('target_') and c != 'time']
        
        # --- A. 数値データの処理 ---
        # 数値型の列だけを取得
        numeric_df = df[feature_cols].select_dtypes(include=[np.number])
        X_numeric = numeric_df.values
        X_numeric = np.nan_to_num(X_numeric, nan=0.0)
        
        # --- B. テキストデータの処理 (ここが修正点！) ---
        # 'text_word' 列がある場合、それをベクトル化する
        if 'text_word' in df.columns and text_vectorizer is not None:
            # NaNを空文字に変換
            text_data = df['text_word'].fillna("").astype(str).tolist()
            # ハッシュ化 (文字列 -> 固定長の数値ベクトル)
            # transform は疎行列を返すので toarray() で dense に変換
            X_text = text_vectorizer.transform(text_data).toarray()
        else:
            # テキスト列がない場合はゼロ埋め
            X_text = np.zeros((len(df), TEXT_VECTOR_DIM))

        # --- C. 数値とテキストを結合 ---
        # 横に結合 (数値特徴量 + テキスト特徴量)
        self.X_raw = np.hstack([X_numeric, X_text])
        
        # --- D. スケーリング (標準化) ---
        if is_train and scaler is not None:
            self.X = scaler.fit_transform(self.X_raw)
        elif scaler is not None:
            self.X = scaler.transform(self.X_raw)
        else:
            self.X = self.X_raw
            
        self.feature_dim = self.X.shape[1]
        self.target_dim = self.Y.shape[1]
        
    def __len__(self):
        return max(0, len(self.X) - self.seq_len)
    
    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.seq_len]
        y_label = self.Y[idx + self.seq_len]
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_label, dtype=torch.float32)

# --- LSTMモデル定義 (変更なし) ---
class EditorAI(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super(EditorAI, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        
        self.head_is_used = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.head_scale = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.head_pos = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 2))
        self.head_triggers = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 2), nn.Sigmoid())

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        return self.head_is_used(last_step), self.head_scale(last_step), self.head_pos(last_step), self.head_triggers(last_step)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    csv_files = glob.glob(os.path.join(DATASET_DIR, "*.csv"))
    if not csv_files:
        print("[Error] データセットが見つかりません。")
        return

    print(f"Found {len(csv_files)} datasets. preparing...")
    
    # ツール初期化
    scaler = StandardScaler()
    # HashingVectorizer: どんな単語でも固定次元(32次元)のベクトルに変換するすごいやつ
    text_vectorizer = HashingVectorizer(n_features=TEXT_VECTOR_DIM, alternate_sign=False)
    
    datasets = []
    
    # 最初のデータセットで次元決定 & fit
    try:
        first_ds = VideoEditorDataset(csv_files[0], scaler=scaler, text_vectorizer=text_vectorizer, sequence_length=SEQUENCE_LENGTH, is_train=True)
        datasets.append(first_ds)
        input_dim = first_ds.feature_dim
        print(f"Input feature dimension: {input_dim} (Numeric + Text Embedding)")
    except Exception as e:
        print(f"Error loading first dataset: {e}")
        import traceback
        traceback.print_exc()
        return

    # 残りのデータを追加
    for f in csv_files[1:]:
        try:
            ds = VideoEditorDataset(f, scaler=scaler, text_vectorizer=text_vectorizer, sequence_length=SEQUENCE_LENGTH, is_train=False)
            if ds.feature_dim == input_dim and len(ds) > 0:
                datasets.append(ds)
        except Exception as e:
            print(f"Skipping {os.path.basename(f)}: {e}")

    if not datasets:
        print("有効なデータがありません。")
        return

    full_dataset = ConcatDataset(datasets)
    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    print(f"Total samples: {len(full_dataset)}")
    
    model = EditorAI(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 損失関数
    criterion_cls = nn.BCELoss()
    criterion_reg = nn.MSELoss()

    print("\n--- Start Training ---")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            y_used = y[:, 0].unsqueeze(1)
            y_scale = y[:, 1].unsqueeze(1)
            y_pos = y[:, 2:4]
            y_trig = y[:, 4:6]
            
            p_used, p_scale, p_pos, p_trig = model(x)
            
            loss = criterion_cls(p_used, y_used) + criterion_reg(p_scale, y_scale) + criterion_reg(p_pos, y_pos) + criterion_cls(p_trig, y_trig)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 100 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Step {i} | Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(train_loader):.4f}")
        
        # モデルとスケーラーを保存
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        joblib.dump(scaler, SCALER_SAVE_PATH)

    print(f"\n🎉 Training Finished! Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()