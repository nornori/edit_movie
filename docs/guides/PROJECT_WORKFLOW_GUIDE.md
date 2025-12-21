# 動画編集AI - プロジェクト全体の流れ（カット選択版）

## 🎯 プロジェクトの目的
動画から自動的に**最適なカット位置**を予測し、Premiere Pro用のXMLを生成する

**現在の開発フォーカス**: カット選択（Cut Selection）に特化
- ✅ カット選択モデル: 高精度で動作中（F1スコア: 0.5630）
- ⚠️ グラフィック配置・テロップ生成: 精度が低いため今後の課題

---

## 📊 全体フロー図

```
┌─────────────────────────────────────────────────────────────────┐
│                    1. データ準備フェーズ                          │
└─────────────────────────────────────────────────────────────────┘
                                  ↓
    ┌──────────────────────────────────────────────────────┐
    │ 1-1. 編集済み動画 + Premiere Pro XMLを用意           │
    │      (data/raw/editxml/)                             │
    └──────────────────────────────────────────────────────┘
                                  ↓
    ┌──────────────────────────────────────────────────────┐
    │ 1-2. 動画から特徴量を抽出                            │
    │      python extract_video_features_parallel.py       │
    │      → data/processed/source_features/*.csv          │
    │      (音声 + 映像特徴量)                             │
    └──────────────────────────────────────────────────────┘
                                  ↓
    ┌──────────────────────────────────────────────────────┐
    │ 1-3. XMLからアクティブラベルを抽出                   │
    │      python scripts/extract_active_labels.py         │
    │      → data/processed/active_labels/*.csv            │
    │      (採用/不採用の判定)                             │
    └──────────────────────────────────────────────────────┘
                                  ↓
    ┌──────────────────────────────────────────────────────┐
    │ 1-4. カット選択用データを作成                        │
    │      python scripts/create_cut_selection_data.py     │
    │      → preprocessed_data/train_sequences.npz         │
    │      → preprocessed_data/val_sequences.npz           │
    │      (動画単位で分割、データリーク防止)               │
    └──────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    2. 学習フェーズ                                │
└─────────────────────────────────────────────────────────────────┘
                                  ↓
    ┌──────────────────────────────────────────────────────┐
    │ 2-1. カット選択モデルの学習                          │
    │      train_cut_selection.bat                         │
    │      → checkpoints_cut_selection/best_model.pth      │
    │      (Transformer + Gated Fusion + Focal Loss)       │
    └──────────────────────────────────────────────────────┘
                                  ↓
    ┌──────────────────────────────────────────────────────┐
    │ 2-2. 学習状況をリアルタイム確認                      │
    │      ブラウザで view_training.html を開く            │
    │      (2秒ごとに自動更新される6つのグラフ)            │
    └──────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    3. 推論フェーズ                                │
└─────────────────────────────────────────────────────────────────┘
                                  ↓
    ┌──────────────────────────────────────────────────────┐
    │ 3-1. 新しい動画から特徴量を抽出                       │
    │      (inference_pipeline.py内で自動実行)             │
    └──────────────────────────────────────────────────────┘
                                  ↓
    ┌──────────────────────────────────────────────────────┐
    │ 3-2. モデルでカット位置を予測                        │
    │      run_inference.bat "video.mp4"                   │
    │      → outputs/inference_results/result.xml          │
    │      (クリップフィルタリング、ギャップ結合)           │
    └──────────────────────────────────────────────────────┘
                                  ↓
    ┌──────────────────────────────────────────────────────┐
    │ 3-3. Premiere Proで開く                              │
    │      result.xmlをインポート                          │
    └──────────────────────────────────────────────────────┘
```

---

## 📝 詳細な手順

### フェーズ1: データ準備

#### ステップ1-1: 編集済み動画とXMLを用意
```
data/raw/editxml/
├── video1.mp4
├── video1.xml  (Premiere ProからエクスポートしたXML)
├── video2.mp4
└── video2.xml
```

#### ステップ1-2: 動画から特徴量を抽出
```bash
python -m src.data_preparation.extract_video_features_parallel
```

**出力**: `data/processed/source_features/video1_features.csv`

**抽出される特徴量**:
- **音声特徴量（215次元）**:
  - 基本音声: RMS energy, 発話検出, 無音区間, speaker_id
  - 話者埋め込み: 192次元（pyannote.audio）
  - 感情表現: pitch, spectral_centroid, MFCC等
  - テキスト・テロップフラグ
  
- **映像特徴量（522次元）**:
  - 基本映像: シーン変化, モーション, 顕著性マップ
  - 顔特徴量: 位置, サイズ, 表情（MediaPipe）
  - CLIP特徴量: 512次元の視覚的意味表現

#### ステップ1-3: XMLからアクティブラベルを抽出
```bash
python scripts/extract_active_labels.py
```

**出力**: `data/processed/active_labels/video1_active.csv`

**抽出される情報**:
- time: タイムスタンプ
- active: 採用（1）/ 不採用（0）

#### ステップ1-4: カット選択用データを作成
```bash
python scripts/create_cut_selection_data.py
```

**処理内容**:
1. 特徴量とアクティブラベルを時間ベースでマージ
2. シーケンス分割（長さ1000フレーム、オーバーラップ500）
3. **動画単位で学習/検証に分割**（データリーク防止）
   - 68本の動画 → 学習54本、検証14本
   - 同じ動画のシーケンスは必ず同じセットに配置
4. 特徴量の正規化（StandardScaler）

**出力**:
- `preprocessed_data/train_sequences_cut_selection.npz` (210シーケンス)
- `preprocessed_data/val_sequences_cut_selection.npz` (91シーケンス)
- `preprocessed_data/audio_scaler_cut_selection.pkl`
- `preprocessed_data/visual_scaler_cut_selection.pkl`

---

### フェーズ2: 学習

#### ステップ2-1: カット選択モデルの学習
```bash
train_cut_selection.bat
```

**モデルアーキテクチャ**:
- **入力**: 音声特徴量（215次元）+ 映像特徴量（522次元）
- **エンコーダ**: Transformer Encoder（6層、8ヘッド、256次元）
- **融合**: Gated Fusion（動的な重み付け）
- **出力**: Active判定（2クラス分類）

**学習設定**:
- エポック数: 50
- バッチサイズ: 16
- 学習率: 0.0001
- 損失関数: Focal Loss（alpha=0.70、gamma=2.0）
  - 採用見逃し（False Negative）に2.3倍のペナルティ
- Total Variation Loss: 0.05（時間的な滑らかさ）
- Early Stopping: 20エポック

**学習時間**: 約1-2時間（GPU使用時）

#### ステップ2-2: 学習状況をリアルタイム確認

ブラウザで `checkpoints_cut_selection/view_training.html` を開く

**可視化される情報**（2秒ごとに自動更新）:
1. 損失関数（Train/Val Loss）
2. 損失の内訳（CE Loss vs TV Loss）
3. 分類性能（Accuracy & F1 Score）
4. Precision, Recall, Specificity
5. 最適閾値の推移
6. 予測の採用/不採用割合

**保存されるファイル**:
- `best_model.pth`: 最良モデル（F1スコア最大）
- `inference_params.yaml`: 推論パラメータ（最適閾値等）
- `training_progress.png`: リアルタイムグラフ
- `training_final.png`: 最終グラフ（高解像度）
- `training_history.csv`: 学習履歴

---

### フェーズ3: 推論

#### ステップ3-1: 新しい動画でカット位置を予測
```bash
run_inference.bat "path\to\your_video.mp4"
```

**処理フロー**:
1. 動画から特徴量を抽出
2. 特徴量を正規化
3. モデルで予測
4. **クリップフィルタリング**:
   - 最適閾値でActive判定
   - 最小継続時間: 3.0秒
   - ギャップ結合: 2.0秒以内
   - 優先順位付け: Active確率順
   - 合計時間制限: 目標90秒、最大150秒
5. Premiere Pro XML生成

**出力**: `outputs/inference_results/result.xml`

#### ステップ3-2: Premiere Proで開く

生成されたXMLをPremiere Proで開くと、自動的にカット編集されたタイムラインが表示されます。

---

## 🔧 設定ファイル

### `configs/config_cut_selection.yaml`
カット選択モデルの学習設定

```yaml
# モデル設定
d_model: 256
nhead: 8
num_encoder_layers: 6
dropout: 0.15

# 学習設定
batch_size: 16
num_epochs: 50
learning_rate: 0.0001
weight_decay: 0.0001

# 損失関数
use_focal_loss: true
focal_alpha: 0.70  # 採用見逃しに2.3倍のペナルティ
focal_gamma: 2.0
tv_weight: 0.05    # 時間的な滑らかさ
```

### `checkpoints_cut_selection/inference_params.yaml`
推論パラメータ（学習時に自動生成）

```yaml
confidence_threshold: -0.200  # 最適閾値
target_duration: 90.0         # 目標合計時間（秒）
max_duration: 150.0           # 最大合計時間（秒）
```

---

## 📊 データフロー詳細

### 特徴量の次元数
- **音声**: 215次元
  - 基本音声: 4次元
  - 話者埋め込み: 192次元
  - 感情表現: 16次元
  - テキスト・テロップ: 3次元

- **映像**: 522次元
  - 基本映像: 10次元
  - CLIP: 512次元

### シーケンスデータ
- **シーケンス長**: 1000フレーム（約100秒 @ 10FPS）
- **オーバーラップ**: 500フレーム
- **学習データ**: 210シーケンス（54動画）
- **検証データ**: 91シーケンス（14動画）

### モデル出力
- **Active判定**: (batch, seq_len, 2)
  - クラス0: 不採用
  - クラス1: 採用

---

## 💡 ヒント

### カット数を調整したい
`checkpoints_cut_selection/inference_params.yaml` を編集：
- `confidence_threshold` を下げる → カット数が増える
- `confidence_threshold` を上げる → カット数が減る

### 学習データを増やしたい
1. 新しい動画とXMLを `data/raw/editxml/` に追加
2. ステップ1-2から再実行

### 学習が進まない場合
- データ数を確認（最低50本以上推奨）
- GPU使用を確認（`nvidia-smi`）
- 設定を確認（`configs/config_cut_selection.yaml`）

---

## 🎯 性能指標

### 学習結果
- **Best F1スコア**: 0.5630（Epoch 33）
- **最適閾値**: -0.200
- **学習時間**: 約1-2時間（50エポック、GPU使用時）

### 推論結果
- **推論時間**: 5~10分/動画（特徴量抽出含む）
- **カット数**: 約8〜12個のクリップ
- **出力動画長**: 約2分（90秒〜150秒）

---

## 📖 関連ドキュメント

- [README](../../README.md) - プロジェクト概要
- [QUICK_START](../QUICK_START.md) - クイックスタートガイド
- [PROJECT_SPECIFICATION](../PROJECT_SPECIFICATION.md) - 詳細仕様

---

**最終更新**: 2025-12-22
**バージョン**: 2.0.0（カット選択特化版）
