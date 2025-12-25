# 動画編集AI - 自動カット選択システム

動画から自動的に**最適なカット位置**を予測し、Premiere Pro用のXMLを生成するAIシステムです。

**想定用途**: 10分程度の動画を約2分（90秒〜150秒）のハイライト動画に自動編集

## 📑 目次

- [現在の開発フォーカス](#-現在の開発フォーカス)
- [機能](#-機能)
- [プロジェクト構造](#-プロジェクト構造)
- [クイックスタート](#-クイックスタート)
  - [必要な環境](#必要な環境)
  - [インストール](#インストール)
  - [新しい動画を自動編集](#新しい動画を自動編集)
- [**教師データの作り方（最重要）**](#-教師データの作り方最重要)
- [ドキュメント](#-ドキュメント)
- [開発](#-開発)
- [性能](#-性能)
- [既知の問題点・改善点](#-既知の問題点改善点)
- [トラブルシューティング](#-トラブルシューティング)
- [貢献](#-貢献)
- [ライセンス](#-ライセンス)

## 🎯 現在の開発フォーカス

**本プロジェクトは現在、カット選択（Cut Selection）に特化して開発中です。**

- ✅ **カット選択モデル**: 動作中（平均F1スコア: 42.30%、Recall: 76.10%）
  - K-Fold Cross Validation（5-Fold）で検証済み
  - 真の汎化性能を測定（データリークなし）
  - 最良モデル: Fold 1（F1: 49.42%）
- ⚠️ **グラフィック配置・テロップ生成**: 精度が低いため今後の課題
  - 現在のマルチモーダルモデル（音声・映像・トラック統合）は、グラフィック配置やテロップ生成の精度が実用レベルに達していません
  - カット選択に集中することで、より高品質な自動編集を実現します
  - グラフィック・テロップ機能は将来的に改善予定です

## 🎯 機能

### 現在実装済み（カット選択）
- **自動カット検出**: AIが最適なカット位置を予測
  - 平均F1スコア: 42.30%（K-Fold CV）
  - Recall: 76.10%（採用すべきカットを見逃さない）
  - 最良モデル: 49.42% F1（Fold 1）
- **音声同期カット**: 映像と音声を同じ位置で自動カット
- **クリップフィルタリング**: 短すぎるクリップの除外、ギャップ結合、優先順位付け
- **Premiere Pro連携**: 生成されたXMLをそのままPremiere Proで開ける
- **リアルタイム学習可視化**: 6つのグラフで学習状況を監視
- **K-Fold Cross Validation**: 5-Foldで真の汎化性能を測定

### 将来的に実装予定の機能（精度改善後）
- **グラフィック配置の自動化**: キャラクター立ち絵の配置・スケール・位置調整
  - 現在のモデルでは精度が低く実用レベルに達していません
  - カット選択の精度向上を優先し、その後に取り組みます
- **AI字幕生成**: 音声認識（Whisper）と感情検出による自動字幕生成
- **テロップ自動配置**: OCRで検出したテロップのXML出力
  - Base64エンコード形式の解析が必要
- **動的な解像度対応**: 入力動画に応じた自動シーケンス設定

## 📁 プロジェクト構造

```
xmlai/
├── src/                          # ソースコード
│   ├── data_preparation/         # データ準備
│   ├── model/                    # モデル定義
│   ├── training/                 # 学習
│   ├── inference/                # 推論
│   └── utils/                    # ユーティリティ
├── scripts/                      # 補助スクリプト
├── tests/                        # テストコード
├── configs/                      # 設定ファイル
├── docs/                         # ドキュメント
├── data/                         # データ（.gitignoreで除外）
├── checkpoints/                  # 学習済みモデル（.gitignoreで除外）
├── preprocessed_data/            # 前処理済みデータ（.gitignoreで除外）
├── outputs/                      # 出力ファイル
├── archive/                      # アーカイブ（.gitignoreで除外）
└── backups/                      # バックアップ（.gitignoreで除外）
```

## 🚀 クイックスタート

### 必要な環境

#### ハードウェア要件
- **GPU**: NVIDIA GPU（CUDA対応）**必須**
  - **VRAM**: 最低8GB、推奨12GB以上
  - 使用モデル: CLIP (512次元埋め込み)、Whisper (音声認識)、MediaPipe (顔検出)
- **RAM**: 16GB以上推奨
- **ストレージ**: 10GB以上の空き容量（モデルキャッシュ + 特徴量ファイル）

#### ソフトウェア要件
- **OS**: Windows 10/11（バッチファイルを使用）
  - Mac/Linuxの場合は、Pythonコマンドを直接実行してください
- **Python**: 3.8以上（3.11推奨）
- **CUDA**: 11.8以上（PyTorchのCUDAバージョンに依存）
- **ffmpeg**: **必須**（パスを通す必要あり）
  - pydub、librosaが内部で使用
  - インストール: [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
  - 確認コマンド: `ffmpeg -version`
- **Adobe Premiere Pro**: 2020以降（XML読み込み用）
  - 生成されるXMLはPremiere Pro 2020以降で動作確認済み

#### プロジェクトパスの制約
- **⚠️ 重要**: プロジェクトパスに日本語などの非ASCII文字を含めないでください
  - MediaPipeが正常に動作しない可能性があります
  - 推奨: `C:\projects\xmlai` のようなASCII文字のみのパス

### インストール
```bash
# 1. リポジトリをクローン
git clone <repository_url>
cd xmlai

# 2. 依存パッケージをインストール
pip install -r requirements.txt

# 3. ffmpegがインストールされているか確認
ffmpeg -version
```

**注意**: `requirements.txt`には動作確認済みのバージョンがコメントで記載されています。最小バージョン要件を満たしていれば、より新しいバージョンでも動作する可能性があります。

### 新しい動画を自動編集

**方法1: バッチファイルを使う（推奨）**
```bash
run_inference.bat "path\to\your_video.mp4"
```

**方法2: 手動で実行**
```bash
# 推論実行
python -m src.inference.inference_pipeline "your_video.mp4" outputs/inference_results/output.xml

# Premiere Proで output.xml を開く
```

詳しくは [QUICK_START.md](docs/QUICK_START.md) を参照してください。

## 🎓 教師データの作り方（最重要）

**このシステムは「あなたの編集スタイル」を学習します。** 学習データの品質が推論結果に直結します。

### ステップ1: Premiere Proで編集したXMLを用意

1. **Premiere Proで通常通り編集**
   - 元動画（10分程度）から、採用したいシーンをカット
   - タイムライン上に2分程度のハイライト動画を作成
   - **重要**: 元動画ファイルと同じファイル名を使用してください

2. **XMLとしてエクスポート**
   - `ファイル` → `書き出し` → `Final Cut Pro XMLとして書き出し`
   - 保存先: `editxml/` フォルダ
   - ファイル名: 元動画と同じ名前（例: `video1.xml`）

3. **複数の動画で繰り返す**
   - **最低30本以上**の編集済みXMLを用意することを推奨
   - 多様な編集スタイル（緩急、長さ、カット位置）を含めると精度向上

### ステップ2: 元動画を配置

```
videos/
├── video1.mp4
├── video2.mp4
└── video3.mp4

editxml/
├── video1.xml  ← Premiere Proで編集したXML
├── video2.xml
└── video3.xml
```

### ステップ3: 特徴量抽出

```bash
# 並列処理で高速抽出（推奨）
python -m src.data_preparation.extract_video_features_parallel \
    --video_dir videos \
    --output_dir data/processed/source_features \
    --n_jobs 4
```

**処理時間の目安:**
- 10分の動画1本: 約5-10分（GPU使用時）
- 30本の動画: 約2.5-5時間

**抽出される特徴量:**
- **音声（215次元）**:
  - RMS Energy（音量）
  - VAD（話している/沈黙の検出）
  - Speaker Embedding（192次元、話者識別）
  - MFCC（13次元、音響特徴）
  - ピッチ、スペクトル重心など
- **映像（522次元）**:
  - CLIP埋め込み（512次元、画像の意味理解）
  - MediaPipe顔検出（10次元、顔の位置・表情）
  - シーン変化、動き検出
- **テキスト**:
  - Whisperによる音声認識（`text_is_active`として使用）

### ステップ4: ラベル抽出

```bash
# Premiere Pro XMLから「採用/不採用」ラベルを自動抽出
python -m src.data_preparation.extract_active_labels \
    --xml_dir editxml \
    --feature_dir data/processed/source_features \
    --output_dir data/processed/active_labels
```

このスクリプトが行うこと:
- XMLファイルから採用されたクリップの時間範囲を取得
- 特徴量ファイルの各フレームに対して `Active(1)` / `Inactive(0)` ラベルを付与
- 出力: `data/processed/active_labels/video1_active.csv`

### ステップ5: 学習用データセット作成

```bash
# 特徴量とラベルを結合し、学習用シーケンスに分割
python scripts/create_cut_selection_data.py
```

生成されるファイル:
- `preprocessed_data/train_sequences_cut_selection.npz`
- `preprocessed_data/val_sequences_cut_selection.npz`

**シーケンス分割の仕組み:**
- 各動画を1000フレーム（100秒 @ 10fps）のシーケンスに分割
- オーバーラップ500フレームで重複させる
- 同じ動画のシーケンスは必ず同じsplit（train/val）に配置（データリーク防止）

### ステップ6: 学習実行

```bash
# 学習開始（可視化付き）
train_cut_selection.bat

# 学習状況をブラウザで確認
# checkpoints_cut_selection/view_training.html を開く
```

### 教師データの品質を上げるコツ

1. **多様性を確保**
   - 異なる話者、異なるトピック、異なる長さの動画を含める
   - 「盛り上がるシーン」「静かなシーン」の両方を含める

2. **編集の一貫性**
   - 「採用する基準」を明確にする（例: 笑い声がある、重要な発言がある）
   - 一貫性のない編集はモデルを混乱させる

3. **データ量**
   - 最低30本、理想は50-100本以上
   - 少ないデータでは過学習のリスク

4. **ラベルの確認**
   - `data/processed/active_labels/`のCSVファイルを確認
   - Active率が極端（5%未満、95%以上）な動画は要確認

## 📚 ドキュメント

### 基本ガイド
- [クイックスタート](docs/QUICK_START.md)
- [K-Fold Cross Validation](docs/K_FOLD_CROSS_VALIDATION.md)
- [プロジェクト全体の流れ](docs/guides/PROJECT_WORKFLOW_GUIDE.md)
- [必要なファイル一覧](docs/guides/REQUIRED_FILES_BY_PHASE.md)
- [音声カット & テロップ変換](docs/summaries/AUDIO_CUT_AND_TELOP_GRAPHICS_SUMMARY.md)

## 🔧 開発

### データ準備
```bash
# カット選択用データの作成
python scripts/create_cut_selection_data.py
```

### 学習

**カット選択モデルのトレーニング:**
```bash
# 1. データ準備（動画単位で分割）
python scripts/create_cut_selection_data.py

# 2. トレーニング実行（可視化付き）
train_cut_selection.bat

# 3. 学習状況の確認
# ブラウザで checkpoints_cut_selection/view_training.html を開く
# 2秒ごとに自動更新されるグラフで学習の様子をリアルタイム確認
```

**K-Fold Cross Validation（より信頼性の高い評価）:**
```bash
# 1. データ準備（train + valを結合）
python scripts/create_combined_data_for_kfold.py

# 2. K-Fold学習実行（5分割）
train_cut_selection_kfold.bat

# 3. 結果の確認
# checkpoints_cut_selection_kfold/kfold_comparison.png - 全Foldの比較
# checkpoints_cut_selection_kfold/kfold_summary.csv - 統計サマリー
```

### テスト
```bash
# カット選択モデルのテストは今後実装予定
```

## 📊 性能

### カット選択モデル（Cut Selection Model）

#### 最終性能（2025-12-26検証済み）

**K-Fold Cross Validation（5-Fold）結果:**

| 指標 | 平均値 | 標準偏差 | 最良（Fold 1） |
|------|--------|----------|----------------|
| **F1 Score** | **42.30%** | ±5.75% | **49.42%** |
| **Accuracy** | 50.24% | ±14.92% | 73.63% |
| **Precision** | 29.83% | ±5.80% | 36.94% |
| **Recall** | **76.10%** | ±5.19% | 74.65% |

**評価方法:**
- 各Foldで完全に未見のデータで評価
- データリークなし（動画単位でFold分割）
- 真の汎化性能を測定

**推奨モデル:** Fold 1（F1: 49.42%、最も安定した性能）

#### データセット
- **学習データ**: 67動画、289シーケンス
  - K-Fold Cross Validation: 5分割（GroupKFoldでデータリーク防止）
  - 同じ動画のシーケンスは必ず同じFoldに配置
  - シーケンス長: 1000フレーム、オーバーラップ: 500フレーム
- **採用率**: 全体23.12%
- **特徴量**: 784次元（音声235 + 映像543 + 時系列6）
- **想定入力**: 10分程度の動画
- **出力**: 約2分（90秒〜150秒）のハイライト動画

#### 処理時間（実測値）

**学習フェーズ:**
- **特徴量抽出**: 5-10分/動画（10分の動画、GPU: RTX 3060 Ti使用）
  - 並列処理（n_jobs=4）で高速化可能
- **学習時間**: 50エポック × 5 Folds = 約2-3時間（GPU使用時）
  - 1エポック: 約1-2分
  - Early Stopping: 平均7.4エポックで収束

**推論フェーズ:**
- **特徴量抽出**: 5-10分/動画（10分の動画）
  - ボトルネック: Whisper（音声認識）、CLIP（画像埋め込み）
- **モデル推論**: 5-30秒/動画
- **XML生成**: <1秒
- **合計**: 約5-10分/動画

**VRAM使用量:**
- **学習時**: 約6-8GB（バッチサイズ16）
- **推論時**: 約4-6GB（特徴量抽出時）

#### モデルアーキテクチャ

```yaml
Transformer Encoder:
  - d_model: 256
  - attention_heads: 8
  - encoder_layers: 6
  - feedforward_dim: 1024
  - dropout: 0.15

Loss Function:
  - Focal Loss (alpha=0.5, gamma=2.0)
  - TV Regularization (weight=0.02)
  - Adoption Penalty (weight=10.0)

Training:
  - Optimizer: AdamW
  - Learning Rate: 0.0001
  - Batch Size: 16
  - Mixed Precision: Enabled
  - Random Seed: 42（再現性確保）
```

#### K-Fold Cross Validation結果

**全Fold比較:**

![K-Fold Comparison](checkpoints_cut_selection_kfold_enhanced/kfold_comparison.png)

**リアルタイム進捗:**

![Realtime Progress](checkpoints_cut_selection_kfold_enhanced/kfold_realtime_progress.png)

詳細な結果分析は [最終結果レポート](docs/FINAL_RESULTS.md) を参照してください。

## ⚠️ 既知の問題点・改善点

### 現在の問題点

#### 1. テロップ関連
- **テロップがBase64エンコードで特徴量に含められていない**
  - Premiere ProのBase64エンコード形式のため、テロップの内容や位置情報を学習に活用できていない
  - OCRで検出したテロップ情報が学習データに反映されていない
- **テロップのXML出力未対応**
  - 学習したテロップ情報をXMLに出力する機能が実装されていない
  - 現在はテロップ生成を無効化して対応（`configs/config_telop_generation.yaml`）

#### 2. 編集の自由度
- **単一トラック配置**
  - 現在は1つのトラックに全クリップが時系列順に配置される
  - 複数トラックへの分散配置は未実装（編集の自由度が低い）

#### 3. フレーム単位の回帰予測のジッター（将来の課題）
- **ScaleやPosition（x, y）の予測が不安定**
  - **注意**: この問題は現在のカット選択モデルには関係ありません（カット選択は2値分類のみ）
  - 将来的にグラフィック配置機能を実装する際の課題として記載
  - フレームごとに独立して予測するため、値が微妙に震える（ジッター）
  - 生成された動画で画像がガクガク震える現象が発生
  - **提案**:
    - 移動平均フィルタ（Moving Average）の適用
    - サビツキー・ゴーレイ・フィルタなどで数値を滑らかにする
    - キーフレーム補間を考慮した予測方法の検討
    - LSTMやGRUなど時系列を考慮したモデルの使用

#### 4. XMLパースの複雑さ
- **premiere_xml_parser.pyの制限**
  - 標準的なXML構造のみを想定
  - **対応できない構造**:
    - ネストされたシーケンス（Nested Sequence）
    - マルチカムクリップ
    - 複雑なエフェクトチェーン
  - **問題点**:
    - ネストされた構造内のクリップが無視される
    - 時間計算が正しく行えない可能性
  - **提案**:
    - 再帰的にネストを掘るロジックの実装
    - より堅牢なXMLパーサーの採用（例: OpenTimelineIO）

#### 5. シーケンス設定の未対応
- **Premiere Proのシーケンス設定が反映されない**
  - 解像度、フレームレート、アスペクト比などの設定が固定
  - 縦長動画（1080x1920）に対応しているが、他の解像度は未検証
  - **問題点**:
    - 異なる解像度の動画で正しく動作しない可能性
    - フレームレートの不一致による音ズレの可能性
  - **提案**:
    - 入力動画のメタデータから自動的にシーケンス設定を生成
    - 設定ファイルでシーケンス設定をカスタマイズ可能に

### 改善予定（優先度別）

#### 高優先度（残り）
- [ ] **テロップデコード**: Base64エンコードされたテロップをデコードして特徴量に含める
- [ ] **テロップのXML出力**: 学習したテロップ情報をXMLに出力する機能を実装
- [ ] **Asset ID管理の改善**: 特徴量ベースのマッチングまたは役割ベースのID管理
- [ ] **トラック配置改善**: 複数トラックに分散配置して編集しやすいXMLを生成

#### 中優先度（残り）
- [ ] **学習データの品質向上**: より多様な編集スタイルのデータを追加
- [ ] **XMLパーサーの強化**: ネストされたシーケンスやマルチカムクリップへの対応
- [ ] **特徴量抽出の高速化**: 並列処理の最適化
- [ ] **モデルの軽量化**: 推論速度の向上

#### 低優先度
- [ ] **ユニットテストの拡充**: カバレッジ向上
- [ ] **ドキュメントの充実化**: チュートリアルやFAQの追加
- [ ] **UIの追加**: GUIベースの設定・実行ツール

### 技術的負債

#### 未解決（機能面）
- **Base64形式の解析処理未実装**: Premiere ProのBase64エンコード形式の解析・デコード処理が必要
- **XMLパーサーの制限**: ネストされたシーケンスやマルチカムクリップに未対応
- **Asset ID管理の問題**: ファイル名ベースのID割り当てで汎用性がない
- **単一トラック配置の制限**: 複数トラック対応への改修が必要

#### 未解決（コード品質）
- **特徴量次元数の不一致**: コメントと実装が合致していない可能性（軽微）

### 設定ファイルによるカスタマイズ

推論パラメータは`configs/config_inference.yaml`で設定可能です：

```yaml
# クリップフィルタリング
clip_filtering:
  active_threshold: 0.29      # Active判定の閾値（学習時に自動最適化）
  min_clip_duration: 3.0      # 最小クリップ継続時間（秒）
  max_gap_duration: 2.0       # ギャップ結合の最大長（秒）
  target_duration: 90.0       # 目標合計時間（秒）
  max_duration: 150.0         # 最大合計時間（秒）

# 予測値の平滑化
smoothing:
  enabled: true               # 平滑化の有効/無効
  method: 'savgol'           # 手法: moving_average, savgol, ema
  window_size: 5             # ウィンドウサイズ
```

学習時の重み付けは`configs/config_multimodal_experiment.yaml`で設定：

```yaml
# クラス不均衡の自動調整
auto_balance_weights: true   # 自動的に最適な重みを計算

# Loss重み（auto_balance_weights=falseの場合に使用）
active_weight: 1.0
asset_weight: 1.0
scale_weight: 1.0
position_weight: 1.0
```

**想定される処理フロー**:
1. 10分（600秒）の動画を入力
2. モデルが重要なシーンを予測（Active確率）
3. ギャップ結合で短い不採用区間を埋める
4. 3秒未満のクリップを除外
5. 予測値を平滑化（ジッター軽減）
6. スコア（確信度）順に並べて上位を選択
7. 合計90秒（最大150秒）のハイライト動画を生成

## 🔧 トラブルシューティング

### よくある問題

#### 1. MediaPipe初期化エラー
```
MediaPipe FaceMesh initialization failed
```

**原因**: プロジェクトパスに日本語などの非ASCII文字が含まれている

**解決策**:
```bash
# プロジェクトを ASCII のみのパスに移動
# 例: D:\切り抜き\xmlai → C:\projects\xmlai
```

#### 2. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```

**原因**: VRAM不足

**解決策**:
```bash
# 1. バッチサイズを減らす（configs/config_cut_selection.yaml）
batch_size: 16  # デフォルト: 32

# 2. 並列処理数を減らす
python -m src.data_preparation.extract_video_features_parallel --n_jobs 2
```

#### 3. ffmpeg not found
```
FileNotFoundError: [WinError 2] The system cannot find the file specified
```

**原因**: ffmpegがインストールされていない、またはパスが通っていない

**解決策**:
```bash
# 1. ffmpegをインストール
# https://ffmpeg.org/download.html からダウンロード

# 2. 環境変数PATHに追加
# システム環境変数 → Path → 編集 → ffmpegのbinフォルダを追加

# 3. 確認
ffmpeg -version
```

#### 4. 推論で0個のクリップが検出される

**原因1**: 学習データに推論対象の動画が含まれており、Active率が極端に低い

**確認方法**:
```bash
# 学習データに含まれているか確認
python -c "
import numpy as np
data = np.load('preprocessed_data/train_sequences_cut_selection.npz', allow_pickle=True)
print(set(data['video_names']))
"
```

**原因2**: モデルが過学習している

**解決策**:
- より多くの動画（最低30本以上）で再学習
- データの多様性を確保
- K-Fold Cross Validationで評価

#### 5. 学習データと検証データが重複している

**確認方法**:
```bash
python -c "
import numpy as np
train = np.load('preprocessed_data/train_sequences_cut_selection.npz', allow_pickle=True)
val = np.load('preprocessed_data/val_sequences_cut_selection.npz', allow_pickle=True)
overlap = set(train['video_names']) & set(val['video_names'])
print(f'重複動画数: {len(overlap)}')
if overlap:
    print('重複動画:', overlap)
"
```

**解決策**: `scripts/create_cut_selection_data.py`を修正して再実行

#### 6. 長い動画でPositional Encoding エラー

```
RuntimeError: The size of tensor a (15820) must match the size of tensor b (5000)
```

**原因**: 5000フレーム（約8分 @ 10fps）を超える動画

**解決策**: 最新版の`src/cut_selection/inference_cut_selection.py`を使用（自動チャンク処理対応済み）

### デバッグモード

詳細なログを出力するには：

```bash
# ログレベルをDEBUGに設定
export LOG_LEVEL=DEBUG  # Linux/Mac
set LOG_LEVEL=DEBUG     # Windows

# 推論実行
python -m scripts.export_cut_selection_to_xml video.mp4
```

## 🤝 貢献

プルリクエストを歓迎します！特に以下の分野での貢献を募集しています：
- 学習データの提供
- パフォーマンス最適化
- ドキュメントの改善
- バグ修正

## 📝 ライセンス

MIT License


