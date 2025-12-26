# 動画編集AI - 自動カット選択システム

動画から自動的に**最適なカット位置**を予測し、Premiere Pro用のXMLを生成するAIシステムです。

**想定用途**: 10分程度の動画を約2分（90秒〜150秒）のハイライト動画に自動編集

![カット選択の概要](docs/whatcut.png)

---

## 🎯 Purpose

このプロジェクトは、動画編集において時間と労力を要する作業的な工程を自動化することで、編集者が本来注力すべき構成や演出といった創造的な部分に、より多くの時間と思考を割けるようにすることを目的としています。

## ⚠️ Notes on Automatic Editing and Ethics

自動で生成されたカットは、必ずしも制作者の意図や文脈を正確に反映するとは限りません。誤解を招く編集や倫理的に問題のある表現を防ぐためにも、本プロジェクトの出力はあくまで補助的なものとして扱い、最終的な編集判断は人が行うことを想定しています。

本プロジェクトで使用しているモデルは、動画内容の意味や倫理的妥当性を理解するものではなく、映像・音声・テキストといった特徴量に基づいてカット候補を予測します。

## 📌 Intended Use

本ツールは、配信アーカイブやエンタメ系動画など、ハイライト作成を目的とした編集作業を主な対象としています。発言の正確性や文脈の保持が特に重要となるインタビュー、ニュース、記録映像などで使用する場合は、十分な確認と注意が必要です。

---

## ⚡ 最短ルート（初めての方向け）

### 📥 インストール（5分）

```bash
# 1. リポジトリをクローン
git clone <repository_url>
cd xmlai

# 2. 依存パッケージをインストール
pip install -r requirements.txt

# 3. ffmpegがインストールされているか確認
ffmpeg -version
```

### 🎬 推論: 動画→XML（3～5分/動画）

**学習済みモデルを使って、新しい動画を自動編集:**

```bash
# ワンコマンド実行（推奨）
python scripts/video_to_xml.py "動画ファイルのパス"

# 目標秒数を指定（デフォルト: 180秒）
python scripts/video_to_xml.py "動画パス" --target 60   # 60秒目標（30～80秒）
python scripts/video_to_xml.py "動画パス" --target 120  # 120秒目標（60～140秒）

# 出力先を指定
python scripts/video_to_xml.py "動画パス" --output "custom.xml"
```

**出力**: `outputs/{動画名}_output.xml` をPremiere Proで開く → ハイライト動画が完成！

**処理の流れ**:
1. 動画から特徴量を自動抽出（3-5分）
2. Full Video Modelで重要シーンを予測（数秒）
3. 目標秒数に合わせて最適閾値を自動探索
4. クリップを結合・フィルタリング（隙間1秒未満→結合、3秒未満→除外）
5. Premiere Pro用XMLを生成

**現在の性能**: 
- **Full Video Model**: 推論テスト成功、目標範囲内に収まる最適閾値を自動探索
- **予測精度**: F1=52.90%、Recall=80.65%（学習時）
- **制約満足**: 目標180秒に対して188.8秒（範囲: 90～200秒）
- **クリップ処理**: 隙間1秒未満を結合、3秒未満を除外

**詳細**: [推論ガイド](docs/INFERENCE_GUIDE.md)

---

### 🎓 学習: あなたの編集スタイルを学習（初回のみ、数時間）

**1. 教師データを準備（30本以上推奨）**

```
videos/              # 元動画（10分程度）
├── video1.mp4
├── video2.mp4
└── video3.mp4

data/raw/editxml/    # Premiere Proで編集したXML
├── video1.xml       # ← Premiere Proで「書き出し」→「Final Cut Pro XML」
├── video2.xml
└── video3.xml
```

**2. 特徴量抽出（5-10分/動画）**

```bash
# 音声・映像・テキストの特徴量を自動抽出
python -m src.data_preparation.extract_video_features_parallel ^
    --video_dir videos ^
    --output_dir data/processed/source_features ^
    --n_jobs 4
```

**3. ラベル抽出（数秒）**

```bash
# XMLから「採用/不採用」ラベルを自動抽出
python -m src.data_preparation.extract_active_labels ^
    --xml_dir data/raw/editxml ^
    --feature_dir data/processed/source_features ^
    --output_dir data/processed/active_labels
```

**4. 時系列特徴量追加（数分）**

```bash
# 移動平均、変化率、CLIP類似度などを追加
python scripts\add_temporal_features.py
```

**5. Full Video用データセット作成（数分）**

```bash
# Full Video学習用にデータを準備
python scripts\create_cut_selection_data_enhanced_fullvideo.py
```

**6. 学習実行（1-2時間、GPU推奨）**

```bash
# Full Video学習
batch\train_fullvideo.bat

# リアルタイム可視化
# ブラウザで checkpoints_cut_selection_fullvideo/view_training.html を開く
```

**学習の特徴**:
- 1動画=1サンプル（per-video最適化）
- 動画ごとに90-200秒制約を満たす最適閾値を学習
- Early Stopping: 性能が向上しなくなったら自動停止
- Mixed Precision: GPU VRAMを効率的に使用

**出力**: 
- `checkpoints_cut_selection_fullvideo/best_model.pth` （学習済みモデル）
- `training_history.csv` （学習履歴）
- `training_progress.png` （学習グラフ）
- `view_training.html` （リアルタイム可視化）

---

### 📊 現在の性能（2025-12-26検証済み）

#### Full Video Model ✅ 推奨

**学習性能** (Epoch 9):
- F1スコア: 52.90%
- Recall: 80.65%（採用すべきカットの80%を検出）
- Precision: 38.94%
- Accuracy: 62.89%

**推論テスト結果**（bandicam 2025-05-11 19-25-14-768.mp4）:
- 動画長: 1000.1秒（約16.7分）
- **最適閾値**: 0.8952（動画ごとに自動最適化）
- **予測時間**: 181.9秒（目標180秒に完璧に一致、誤差+1.9秒）
- **採用率**: 18.2%（1,819 / 10,001フレーム）
- **抽出クリップ数**: 10個（合計138.3秒）
- **XML生成**: 成功（Premiere Pro用）

**制約満足度**:
- ✅ 90秒以上200秒以下の制約を満たす
- ✅ 目標180秒（3分）にほぼ完璧に一致
- ✅ per-video最適化（動画ごとに最適閾値を探索）

**詳細**: [推論テスト結果レポート](docs/INFERENCE_TEST_RESULTS.md)

#### 旧K-Foldモデル（改善中）

シーケンス分割の問題により改善中です。詳細は [K-Fold結果レポート](docs/K_FOLD_FINAL_RESULTS.md) を参照してください。

---

### 💡 より詳しく知りたい方へ

- **教師データの作り方**: [教師データの作り方（最重要）](#-教師データの作り方最重要)
- **詳細な使い方**: [クイックスタート](#-クイックスタート)
- **性能の詳細**: [性能](#-性能)
- **トラブルシューティング**: [トラブルシューティング](#-トラブルシューティング)

---

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

- ✅ **カット選択モデル**: Full Video Model推奨（推論テスト成功）
  - 90-200秒制約を満たす最適閾値を自動探索
  - 目標180秒にほぼ完璧に一致（+1.9秒）
  - Premiere Pro用XML生成成功
  - 旧K-Foldモデルは改善中（シーケンス分割の問題）
- ⚠️ **グラフィック配置・テロップ生成**: 精度が低いため今後の課題
  - 現在のマルチモーダルモデル（音声・映像・トラック統合）は、グラフィック配置やテロップ生成の精度が実用レベルに達していません
  - カット選択に集中することで、より高品質な自動編集を実現します
  - グラフィック・テロップ機能は将来的に改善予定です

## 🎯 機能

### 現在実装済み（カット選択）
- **自動カット検出**: AIが最適なカット位置を予測
  - Full Video Model: 推論テスト成功
  - 90-200秒制約満足
  - per-video最適化
- **音声同期カット**: 映像と音声を同じ位置で自動カット
- **クリップフィルタリング**: 短すぎるクリップの除外、ギャップ結合、優先順位付け
- **Premiere Pro連携**: 生成されたXMLをそのままPremiere Proで開ける
- **リアルタイム学習可視化**: 6つのグラフで学習状況を監視

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

**ワンコマンド実行（推奨）**
```bash
# 動画から特徴量抽出→推論→XML生成まで自動実行
python scripts/video_to_xml.py "path\to\your_video.mp4"

# 目標秒数を指定（デフォルト: 180秒）
python scripts/video_to_xml.py "path\to\your_video.mp4" --target 60

# 既存の特徴量を使用する場合
python scripts/generate_xml_from_inference.py "path\to\your_video.mp4"
```

**出力**: `outputs/{動画名}_output.xml` をPremiere Proで開く

詳しくは [推論ガイド](docs/INFERENCE_GUIDE.md) と [QUICK_START.md](docs/QUICK_START.md) を参照してください。

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
- 10分の動画1本: 約3-5分（GPU使用時）
- 30本の動画: 約1.5-2.5時間（4並列処理）

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
- [最終結果レポート](docs/FINAL_RESULTS.md)
- [推論テスト結果](docs/INFERENCE_TEST_RESULTS.md)
- [プロジェクト全体の流れ](docs/guides/PROJECT_WORKFLOW_GUIDE_GUIDE.md)
- [必要なファイル一覧](docs/guides/REQUIRED_FILES_BY_PHASE.md)

## 🔧 開発

### データ準備

**Full Video用データセット作成**:
```bash
# 1. 特徴量抽出（並列処理）
python -m src.data_preparation.extract_video_features_parallel \
    --video_dir videos \
    --output_dir data/processed/source_features \
    --n_jobs 4

# 2. ラベル抽出
python -m src.data_preparation.extract_active_labels \
    --xml_dir data/raw/editxml \
    --feature_dir data/processed/source_features \
    --output_dir data/processed/active_labels

# 3. 時系列特徴量追加
python scripts/add_temporal_features.py

# 4. Full Video用データセット作成
python scripts/create_cut_selection_data_enhanced_fullvideo.py
```

### 学習

**Full Video Model**:
```bash
# 1. データ準備（上記参照）

# 2. トレーニング実行
batch/train_fullvideo.bat

# 3. 学習状況の確認
# ブラウザで checkpoints_cut_selection_fullvideo/view_training.html を開く
```

**学習パラメータ**:
- バッチサイズ: 1（1動画=1サンプル）
- 最大エポック: 500
- Early Stopping: 100エポック
- 学習率: 0.0001
- オプティマイザ: AdamW
- 損失関数: Focal Loss + TV Regularization + Adoption Penalty

### テスト

**推論テスト**:
```bash
# Full Video推論テスト
python tests/test_inference_fullvideo.py "video_name"
```

**XML生成**:
```bash
# ワンコマンド実行（特徴量抽出→推論→XML生成）
python scripts/video_to_xml.py "path/to/video.mp4"

# 目標秒数を指定（デフォルト: 180秒）
python scripts/video_to_xml.py "path/to/video.mp4" --target 60

# 既存の特徴量を使用する場合
python scripts/generate_xml_from_inference.py "path/to/video.mp4"
```

**出力**:
- `outputs/video_name_output.xml` - Premiere Pro用XML
- 自動的に目標秒数に合わせた最適閾値を探索
- 範囲: 目標÷2 ～ 目標+20秒
- クリップ結合: 隙間1秒未満を結合
- クリップフィルタ: 3秒未満を除外

## 📊 性能

### Full Video Model（推奨）

#### 学習性能

**最良モデル**: Epoch 9

| 指標 | 値 |
|------|-----|
| F1スコア | 52.90% |
| Recall | 80.65% |
| Precision | 38.94% |
| Accuracy | 62.89% |

**特徴**:
- 採用すべきカットの80%以上を検出（高Recall）
- 1動画=1サンプルのper-video最適化
- 動画ごとに90-200秒制約を満たす閾値を自動学習

#### 推論性能

**推論テスト結果**（bandicam 2025-05-11 19-25-14-768.mp4）:
- 動画長: 1000.1秒（約16.7分）
- **予測時間**: 181.9秒（目標180秒に完璧に一致）
- **採用率**: 18.2%（1,819 / 10,001フレーム）
- **抽出クリップ数**: 10個（合計138.3秒）
- **XML生成**: 成功（Premiere Pro用）

**制約満足度**:
- ✅ 90秒以上200秒以下の制約を満たす
- ✅ 目標180秒（3分）にほぼ完璧に一致
- ✅ per-video最適化（動画ごとに最適閾値を探索）

**詳細**: [推論テスト結果レポート](docs/INFERENCE_TEST_RESULTS.md)

### 旧K-Foldモデル（改善中）

シーケンス分割の問題により改善中です。詳細は以下を参照：
- [K-Fold最終結果](docs/K_FOLD_FINAL_RESULTS.md)
- [K-Fold詳細レポート](docs/KFOLD_TRAINING_REPORT.md)

#### データセット
- **学習データ**: 67動画
  - 1動画=1サンプル（per-video最適化）
  - 動画ごとに90-200秒制約を学習
- **採用率**: 全体23.12%
- **特徴量**: 784次元（音声235 + 映像543 + 時系列6）
- **想定入力**: 10分程度の動画
- **出力**: 約2分（90秒〜200秒）のハイライト動画

#### 処理時間（実測値）

**学習フェーズ:**
- **特徴量抽出**: 3-5分/動画（10分の動画、GPU: RTX 3060 Ti使用）
  - 並列処理（n_jobs=4）で高速化可能
- **学習時間**: 約1-2時間（GPU使用時）
  - 1エポック: 約1-2分
  - Early Stopping: 性能が向上しなくなったら自動停止
  - 最良モデル: Epoch 9で達成

**推論フェーズ:**
- **特徴量抽出**: 3-5分/動画（10分の動画）
  - ボトルネック: Whisper（音声認識）、CLIP（画像埋め込み）
- **モデル推論**: 5-30秒/動画
- **XML生成**: <1秒
- **合計**: 約3-5分/動画

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
  - Batch Size: 1 (per-video)
  - Mixed Precision: Enabled
  - Random Seed: 42（再現性確保）
```

**特徴**:
- 音声・映像・時系列の3つのモダリティを融合
- Transformerで長期的な依存関係を学習
- per-video最適化で動画ごとに最適な閾値を学習
- 90-200秒制約を満たすように自動調整

詳細な結果分析は [最終結果レポート](docs/FINAL_RESULTS.md) と [推論テスト結果](docs/INFERENCE_TEST_RESULTS.md) を参照してください。

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


