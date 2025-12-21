# 動画編集AI - プロジェクト仕様書

## プロジェクト概要

**プロジェクト名**: 動画編集AI - 自動カット選択システム

**目的**: 動画から自動的に**最適なカット位置**を予測し、Premiere Pro用のXMLを生成するAIシステム

**想定用途**: 10分程度の動画を約2分（90秒〜150秒）のハイライト動画に自動編集

**現在の開発フォーカス**:
- ✅ **カット選択モデル**: 高精度で動作中（F1スコア: 0.5630）
- ⚠️ **グラフィック配置・テロップ生成**: 精度が低いため今後の課題
  - 現在のマルチモーダルモデルは、グラフィック配置やテロップ生成の精度が実用レベルに達していません
  - カット選択に集中することで、より高品質な自動編集を実現します
  - グラフィック・テロップ機能は将来的に改善予定です

**技術スタック**:
- Python 3.8+
- PyTorch (深層学習フレームワーク)
- OpenCV (動画処理)
- Transformers (CLIP特徴量抽出)
- MediaPipe (顔検出)

---

## システムアーキテクチャ

### 全体フロー

```
1. データ準備フェーズ
   ├─ Premiere Pro XMLパース → 編集履歴抽出
   ├─ 動画特徴量抽出 → 音声・映像・テロップ特徴量
   └─ データ前処理 → 学習用シーケンス生成

2. 学習フェーズ
   ├─ マルチモーダルデータセット構築
   ├─ Multi-Track Transformerモデル学習
   └─ チェックポイント保存

3. 推論フェーズ
   ├─ 新規動画の特徴量抽出
   ├─ モデルで編集パラメータ予測
   ├─ クリップフィルタリング（最小継続時間、ギャップ結合）
   └─ Premiere Pro XML生成
```

### モデルアーキテクチャ

**Multi-Track Transformer**:
- 入力: 音声特徴量（17次元）+ 映像特徴量（522次元）+ トラック特徴量（240次元）
- エンコーダ: Transformer Encoder（6層、8ヘッド）
- デコーダ: 各トラックごとに独立した予測ヘッド
- 出力: 
  - Active判定（2クラス分類）
  - Scale（回帰）
  - Position X/Y（回帰）
  - Asset ID（10クラス分類）

---

## ディレクトリ構造と各ファイルの役割


### ルートディレクトリ

#### 実行用バッチファイル
- **`run_data_preparation.bat`**: データ準備の一括実行スクリプト
  - Premiere Pro XMLパース
  - 動画特徴量抽出（並列処理）
  - データ前処理とシーケンス生成
  
- **`run_training.bat`**: 学習の実行スクリプト
  - マルチモーダルモデルの学習
  - チェックポイント自動保存
  
- **`run_inference.bat`**: 推論の実行スクリプト
  - 新規動画の自動編集
  - Premiere Pro XML生成

#### 設定・ドキュメント
- **`requirements.txt`**: Python依存パッケージリスト
  - PyTorch, OpenCV, transformers, OpenTimelineIO等
  
- **`README.md`**: プロジェクト概要、使い方、既知の問題点
  
- **`LICENSE`**: MITライセンス
  
- **`.gitignore`**: Git除外設定
  - データファイル、モデルチェックポイント、一時ファイル等

---

### `src/` - ソースコード

#### `src/data_preparation/` - データ準備

**`premiere_xml_parser.py`** (約800行)
- **役割**: Premiere Pro XMLファイルをパースして編集履歴を抽出
- **主要機能**:
  - XMLから各クリップの開始/終了フレーム、スケール、位置、Asset IDを抽出
  - トラック情報を時系列データに変換
  - テロップ情報の抽出（Base64エンコード対応）
- **出力**: CSV形式の編集履歴（`data/processed/output_labels_full/`）
- **制限事項**: 
  - ネストされたシーケンス未対応
  - マルチカムクリップ未対応

**`extract_video_features_parallel.py`** (約600行)
- **役割**: 動画から音声・映像特徴量を並列抽出
- **主要機能**:
  - 音声特徴量: RMS energy, 発話検出, 無音区間
  - 映像特徴量: シーン変化, モーション, 顕著性マップ, 顔検出
  - CLIP特徴量: 512次元の視覚的意味表現
  - MediaPipe顔特徴量: 顔の位置、サイズ、表情
- **出力**: CSV形式の特徴量（`temp_features/`）
- **パフォーマンス**: 並列処理で複数動画を同時処理
- **問題点**: 
  - メモリリーク（長時間動画でOOMの可能性）
  - CLIP特徴量の補間問題（1秒間同じ値を使い回し）

**`text_embedding.py`** (約150行)
- **役割**: 日本語テキストを数値埋め込みに変換
- **主要機能**:
  - 文字種別カウント（ひらがな、カタカナ、漢字、英数字）
  - 文長、感嘆符・疑問符の有無
  - 6次元の簡易埋め込み
- **用途**: 音声認識テキストとテロップテキストの特徴量化

**`telop_extractor.py`** (約200行)
- **役割**: Premiere Pro XMLからテロップ情報を抽出
- **主要機能**:
  - グラフィッククリップの検出
  - テロップの開始/終了時間抽出
  - Base64エンコードされたテキストのデコード（未実装）
- **問題点**: Base64デコード未対応

**`data_preprocessing.py`** (約400行)
- **役割**: 特徴量と編集履歴をアライメントして学習用データを生成
- **主要機能**:
  - 特徴量と編集履歴の時間同期
  - シーケンス分割（最大5000フレーム）
  - 学習/検証データの分割
- **出力**: NPZ形式のシーケンスデータ（`preprocessed_data/`）


#### `src/model/` - モデル定義

**`model.py`** (約500行)
- **役割**: Multi-Track Transformerモデルの定義
- **アーキテクチャ**:
  - Transformer Encoder（6層、8ヘッド、512次元）
  - 位置エンコーディング（最大5000フレーム）
  - マルチヘッド出力（Active, Scale, Position, Asset ID）
- **入力形状**:
  - Track特徴量: (batch, seq_len, 240)
  - Audio特徴量: (batch, seq_len, 17) ※マルチモーダル時
  - Visual特徴量: (batch, seq_len, 522) ※マルチモーダル時
- **出力形状**:
  - Active: (batch, seq_len, num_tracks, 2)
  - Scale: (batch, seq_len, num_tracks, 1)
  - Position: (batch, seq_len, num_tracks, 2)
  - Asset ID: (batch, seq_len, num_tracks, 10)

**`multimodal_modules.py`** (約300行)
- **役割**: マルチモーダル統合モジュール
- **主要コンポーネント**:
  - `ModalityEmbedding`: 各モダリティを共通次元に埋め込み
  - `ModalityFusion`: 複数モダリティの融合（Concatenation, Attention, Gated）
  - `CrossModalAttention`: モダリティ間のアテンション
- **融合戦略**:
  - Concatenation: 単純な結合
  - Attention: アテンション重み付き融合
  - Gated: ゲート機構による動的融合

**`loss.py`** (約250行)
- **役割**: 損失関数の定義
- **損失の種類**:
  - Active Loss: Binary Cross Entropy（重み付き）
  - Scale Loss: MSE（Activeなトラックのみ）
  - Position Loss: MSE（Activeなトラックのみ）
  - Asset ID Loss: Cross Entropy（Activeなトラックのみ）
- **重み設定**:
  - active_weight: 2.0（Active判定を重視）
  - scale_weight: 1.0
  - position_weight: 1.0
  - asset_id_weight: 0.5
- **問題点**: active_countによる除算で学習が不安定になる可能性

**`model_persistence.py`** (約150行)
- **役割**: モデルの保存・読み込み
- **主要機能**:
  - チェックポイント保存（モデル、オプティマイザ、設定）
  - モデル読み込み（後方互換性対応）
  - 設定ファイル（JSON）の管理

---

#### `src/training/` - 学習

**`train.py`** (約400行)
- **役割**: 学習のメインスクリプト
- **主要機能**:
  - YAMLファイルから設定読み込み
  - データセット・モデル・オプティマイザの初期化
  - 学習ループの実行
  - チェックポイント自動保存
  - TensorBoard対応（オプション）
- **使用例**:
  ```bash
  python -m src.training.train --config configs/config_multimodal_experiment.yaml
  ```

**`multimodal_dataset.py`** (約350行)
- **役割**: マルチモーダルデータセット
- **主要機能**:
  - NPZファイルからシーケンスデータ読み込み
  - 音声・映像・トラック特徴量のアライメント
  - パディングマスク生成
  - モダリティマスク生成（欠損データ対応）
- **データ拡張**: なし（将来実装予定）

**`multimodal_preprocessing.py`** (約400行)
- **役割**: マルチモーダル特徴量の前処理
- **主要クラス**:
  - `AudioFeaturePreprocessor`: 音声特徴量の正規化
  - `VisualFeaturePreprocessor`: 映像特徴量の正規化
- **正規化手法**:
  - StandardScaler（平均0、分散1）
  - CLIP特徴量のL2正規化
  - 顔特徴量の条件付き正規化
- **保存形式**: Pickle（`.pkl`）

**`dataset.py`** (約200行)
- **役割**: 旧バージョンのデータセット（トラックのみ）
- **状態**: 後方互換性のために残存

**`training.py`** (約300行)
- **役割**: 旧バージョンの学習ループ
- **状態**: 後方互換性のために残存


#### `src/inference/` - 推論

**`inference_pipeline.py`** (約1500行)
- **役割**: 推論パイプラインのメインスクリプト
- **主要機能**:
  1. 動画から特徴量抽出
  2. 特徴量の前処理とアライメント
  3. モデルで編集パラメータ予測
  4. クリップフィルタリング
     - 最小継続時間: 3.0秒
     - ギャップ結合: 2.0秒以内
     - 優先順位付け: Active確率順
     - 合計時間制限: 目標90秒、最大150秒
  5. Premiere Pro XML生成
  6. AI字幕生成（オプション）
     - Whisper音声認識
     - 感情検出（未実装）
- **使用例**:
  ```bash
  python -m src.inference.inference_pipeline video.mp4 --output output.xml
  ```
- **問題点**:
  - Active閾値0.29がハードコード
  - クリップフィルタリングのパラメータがハードコード

**`direct_xml_generator.py`** (約400行)
- **役割**: Premiere Pro互換XMLの直接生成
- **主要機能**:
  - 動画クリップの配置（単一トラック）
  - 音声クリップの同期配置
  - テロップトラックの生成（複数トラック対応）
  - ファイルパスのURL エンコード
- **XML構造**:
  - シーケンス設定（FPS、解像度）
  - ビデオトラック（メイン動画）
  - テロップトラック（グラフィック）
  - オーディオトラック（音声同期）
- **問題点**:
  - 解像度が1080x1920に固定
  - 単一トラック配置のみ

**`otio_xml_generator.py`** (約950行)
- **役割**: OpenTimelineIOを使用したXML生成
- **状態**: 現在は`direct_xml_generator.py`に委譲
- **問題点**:
  - デッドコード（642行目以降が実行されない）
  - 正規表現によるXML操作（アンチパターン）
  - 複数の未使用関数が残存
- **注意**: このファイルは大幅なリファクタリングが必要

**`premiere_telop_encoder.py`** (約200行)
- **役割**: Premiere ProのMOGRT形式テロップエンコーダー
- **主要機能**:
  - テロップテキストのBase64エンコード
  - MOGRT形式のバイナリデータ生成
- **状態**: 実験的実装（未使用）

**その他の修正スクリプト**:
- `fix_telop_complete.py`: テロップをグラフィックに変換
- `fix_telop_simple.py`: 簡易テロップ修正
- `fix_xml_for_premiere.py`: Premiere Pro互換性修正
- `optimize_telop_tracks.py`: テロップトラック最適化
- `remove_telops.py`: テロップ削除

---

#### `src/utils/` - ユーティリティ

**`feature_alignment.py`** (約400行)
- **役割**: 特徴量のアライメントと補間
- **主要機能**:
  - 時間ベースのアライメント
  - 線形補間（欠損値補完）
  - モダリティマスク生成
  - カバレッジ統計計算
- **アライメント戦略**:
  - 許容誤差: 0.05秒
  - 最近傍探索
  - 前方補間（forward fill）
- **問題点**:
  - CLIP特徴量のL2正規化（重複処理の可能性）
  - 特徴量次元数のコメント不一致

**`config_loader.py`** (約150行)
- **役割**: AI字幕生成の設定読み込み
- **主要機能**:
  - YAMLファイルから設定読み込み
  - デフォルト設定の提供
  - 音声認識・感情検出の有効/無効切り替え
- **設定項目**:
  - Whisperモデルサイズ（tiny, small, medium, large）
  - 言語設定（ja, en等）
  - セグメント長の制限
  - キャッシュ設定

**`sequence_processing.py`** (約200行)
- **役割**: シーケンス分割とパディング
- **主要機能**:
  - 長いシーケンスの分割（最大5000フレーム）
  - パディング処理
  - バッチ生成

---

### `configs/` - 設定ファイル

**`config_multimodal_experiment.yaml`**
- **役割**: マルチモーダルモデルの学習設定
- **主要設定**:
  - モデル: Multi-Track Transformer
  - 入力次元: audio=17, visual=522, track=240
  - エポック数: 100
  - バッチサイズ: 4
  - 学習率: 0.0001
  - 損失重み: active=2.0, scale=1.0, position=1.0, asset_id=0.5

**`config_telop_generation.yaml`**
- **役割**: AI字幕生成の設定
- **主要設定**:
  - 音声認識: 有効/無効
  - Whisperモデル: small
  - 言語: ja
  - 感情検出: 有効/無効（未実装）

**`config_telop_disabled.yaml`**
- **役割**: テロップ生成を無効化する設定


---

### `tests/` - テストコード

#### `tests/unit/` - ユニットテスト

**`test_model.py`** (約200行)
- **テスト対象**: `src/model/model.py`
- **テスト内容**:
  - モデルの初期化
  - 順伝播の動作確認
  - 出力形状の検証

**`test_multimodal_model.py`** (約250行)
- **テスト対象**: マルチモーダルモデル
- **テスト内容**:
  - マルチモーダル入力の処理
  - モダリティマスクの動作
  - 欠損データの処理

**`test_multimodal_preprocessing.py`** (約200行)
- **テスト対象**: `src/training/multimodal_preprocessing.py`
- **テスト内容**:
  - 正規化の動作確認
  - 保存・読み込みの検証

**`test_feature_alignment.py`** (約150行)
- **テスト対象**: `src/utils/feature_alignment.py`
- **テスト内容**:
  - アライメントの精度
  - 補間の動作確認

**`test_loss_compatibility.py`** (約100行)
- **テスト対象**: `src/model/loss.py`
- **テスト内容**:
  - 損失計算の正確性
  - 後方互換性

#### `tests/integration/` - 統合テスト

**`test_inference_pipeline.py`** (約300行)
- **テスト対象**: 推論パイプライン全体
- **テスト内容**:
  - エンドツーエンドの動作確認
  - XML生成の検証

**`test_telop_integration.py`** (約200行)
- **テスト対象**: テロップ統合機能
- **テスト内容**:
  - テロップ抽出
  - XML出力の検証

---

### `scripts/` - 補助スクリプト

#### `scripts/batch_processing/` - バッチ処理

**`batch_extract_features.py`**
- **役割**: 複数動画の特徴量を一括抽出
- **使用例**: 学習データの準備

**`batch_process_xml.py`**
- **役割**: 複数XMLファイルの一括処理
- **使用例**: データセット構築

#### `scripts/data_preparation/` - データ準備

**`create_training_sequences.py`**
- **役割**: 学習用シーケンスの生成
- **主要機能**:
  - 特徴量と編集履歴のマージ
  - シーケンス分割
  - NPZ形式で保存

**`prepare_training_data.py`**
- **役割**: データ準備の統合スクリプト
- **主要機能**:
  - XMLパース
  - 特徴量抽出
  - シーケンス生成

#### `scripts/utilities/` - ユーティリティ

**データ検証系**:
- `verify_data_integrity.py`: データ整合性チェック
- `validate_features.py`: 特徴量の妥当性検証
- `check_nan_in_features.py`: NaN値の検出
- `verify_sequences.py`: シーケンスデータの検証

**分析系**:
- `analyze_cuts.py`: カット分析
- `analyze_telop_structure.py`: テロップ構造分析
- `visualize_timeline.py`: タイムライン可視化

**修正系**:
- `regenerate_full_video_labels.py`: ラベル再生成
- `remove_invalid_telops.py`: 無効なテロップ削除
- `optimize_existing_xml.py`: XML最適化

---

### `docs/` - ドキュメント

#### `docs/guides/` - ガイド

**`PROJECT_WORKFLOW_GUIDE.md`**
- **内容**: プロジェクト全体のワークフロー
- **対象**: 開発者、新規参加者

**`REQUIRED_FILES_BY_PHASE.md`**
- **内容**: 各フェーズで必要なファイル一覧
- **対象**: 開発者

**`VIDEO_FEATURE_EXTRACTION_GUIDE.md`**
- **内容**: 動画特徴量抽出の詳細
- **対象**: データサイエンティスト

**`AI_TELOP_CUSTOMIZATION_GUIDE.md`**
- **内容**: AI字幕生成のカスタマイズ方法
- **対象**: ユーザー

#### `docs/summaries/` - サマリー

**`MULTIMODAL_FINAL_SUMMARY.md`**
- **内容**: マルチモーダル実装の最終サマリー
- **対象**: プロジェクトマネージャー

**`TRAINING_RESULTS.md`**
- **内容**: 学習結果の詳細
- **対象**: データサイエンティスト

**`AUDIO_CUT_AND_TELOP_GRAPHICS_SUMMARY.md`**
- **内容**: 音声カットとテロップ変換のサマリー
- **対象**: 開発者

---

### データディレクトリ（.gitignore対象）

#### `data/` - 学習データ

**`data/raw/editxml/`**
- **内容**: Premiere Pro XMLファイル（編集済み動画）
- **形式**: XML
- **サイズ**: 106本の動画
- **注意**: 個人データのため.gitignoreで除外

**`data/processed/output_labels_full/`**
- **内容**: パース済み編集履歴
- **形式**: CSV
- **カラム**: time, track_id, active, scale, position_x, position_y, asset_id

#### `preprocessed_data/` - 前処理済みデータ

**`train_sequences.npz`**
- **内容**: 学習用シーケンス（239シーケンス）
- **形式**: NumPy圧縮配列
- **サイズ**: 約500MB

**`val_sequences.npz`**
- **内容**: 検証用シーケンス（60シーケンス）
- **形式**: NumPy圧縮配列
- **サイズ**: 約100MB

#### `checkpoints/` - モデルチェックポイント

**`best_model.pth`**
- **内容**: 最良モデル（検証損失最小）
- **サイズ**: 約200MB
- **エポック**: 100

**`checkpoint_epoch_*.pth`**
- **内容**: 各エポックのチェックポイント
- **保存間隔**: 5エポックごと

#### `temp_features/` - 一時特徴量

**`{video_name}_features.csv`**
- **内容**: 抽出済み特徴量
- **形式**: CSV
- **カラム**: 音声特徴量（17列）+ 映像特徴量（522列）

**`{video_name}_speech.json`**
- **内容**: Whisper音声認識結果
- **形式**: JSON
- **キャッシュ**: 再実行時に使用


---

## データフロー詳細

### 1. データ準備フェーズ

```
入力: Premiere Pro XML + 動画ファイル
  ↓
[premiere_xml_parser.py]
  - XMLから編集履歴を抽出
  - トラック情報を時系列データに変換
  ↓
出力: data/processed/output_labels_full/{video_name}.csv
  - カラム: time, track_id, active, scale, position_x, position_y, asset_id
  
入力: 動画ファイル
  ↓
[extract_video_features_parallel.py]
  - 音声特徴量抽出（RMS, 発話検出, 無音区間）
  - 映像特徴量抽出（シーン変化, モーション, 顕著性）
  - CLIP特徴量抽出（512次元）
  - MediaPipe顔特徴量抽出
  - テキスト埋め込み（音声認識, テロップ）
  ↓
出力: temp_features/{video_name}_features.csv
  - カラム: time, audio_features(17), visual_features(522)
  
入力: 編集履歴CSV + 特徴量CSV
  ↓
[data_preprocessing.py]
  - 時間ベースでアライメント
  - シーケンス分割（最大5000フレーム）
  - 学習/検証データ分割（80/20）
  ↓
出力: preprocessed_data/train_sequences.npz, val_sequences.npz
  - 含まれるデータ:
    - audio_features: (num_sequences, seq_len, 17)
    - visual_features: (num_sequences, seq_len, 522)
    - track_features: (num_sequences, seq_len, 240)
    - labels: (num_sequences, seq_len, num_tracks, 5)
```

### 2. 学習フェーズ

```
入力: train_sequences.npz, val_sequences.npz
  ↓
[multimodal_dataset.py]
  - シーケンスデータ読み込み
  - パディングマスク生成
  - モダリティマスク生成
  ↓
[multimodal_preprocessing.py]
  - 音声特徴量の正規化（StandardScaler）
  - 映像特徴量の正規化（StandardScaler + L2 for CLIP）
  - 前処理器の保存（.pkl）
  ↓
[train.py]
  - モデル初期化（Multi-Track Transformer）
  - 学習ループ
    - 順伝播
    - 損失計算（Active, Scale, Position, Asset ID）
    - 逆伝播
    - パラメータ更新
  - チェックポイント保存
  ↓
出力: checkpoints/best_model.pth
  - モデルパラメータ
  - オプティマイザ状態
  - 設定情報
```

### 3. 推論フェーズ

```
入力: 新規動画ファイル
  ↓
[inference_pipeline.py]
  ↓
ステップ1: 特徴量抽出
  - extract_video_features_parallel.pyと同じ処理
  - 出力: temp_features/{video_name}_features.csv
  ↓
ステップ2: 前処理とアライメント
  - 音声・映像特徴量の正規化
  - 時間ベースでアライメント
  - モダリティマスク生成
  ↓
ステップ3: モデル予測
  - 学習済みモデル読み込み
  - 順伝播
  - 出力: predictions
    - active: (seq_len, num_tracks, 2)
    - scale: (seq_len, num_tracks, 1)
    - position: (seq_len, num_tracks, 2)
    - asset_id: (seq_len, num_tracks, 10)
  ↓
ステップ4: クリップフィルタリング
  - Active閾値0.29でフィルタリング
  - 最小継続時間3.0秒でフィルタリング
  - ギャップ結合（2.0秒以内）
  - 優先順位付け（Active確率順）
  - 合計時間制限（目標90秒、最大150秒）
  ↓
ステップ5: XML生成
  [direct_xml_generator.py]
  - ビデオクリップ配置
  - 音声クリップ同期配置
  - テロップトラック生成（オプション）
  ↓
出力: Premiere Pro互換XML
  - シーケンス設定（FPS, 解像度）
  - ビデオトラック
  - オーディオトラック
  - テロップトラック（複数）
```

---

## 特徴量の詳細

### 音声特徴量（17次元）

1. **基本音声特徴量（5次元）**
   - `audio_energy_rms`: RMSエネルギー（音量）
   - `audio_is_speaking`: 発話検出（0/1）
   - `silence_duration_ms`: 無音区間の長さ（ミリ秒）
   - `text_is_active`: テキストがアクティブか（0/1）
   - `telop_active`: テロップがアクティブか（0/1）

2. **音声認識テキスト埋め込み（6次元）**
   - `speech_emb_0~5`: 文字種別カウント、文長、感嘆符・疑問符

3. **テロップテキスト埋め込み（6次元）**
   - `telop_emb_0~5`: 文字種別カウント、文長、感嘆符・疑問符

### 映像特徴量（522次元）

1. **基本映像特徴量（11次元）**
   - `scene_change`: シーン変化検出（0/1）
   - `visual_motion`: モーション量（0-1）
   - `saliency_x`, `saliency_y`: 顕著性マップの中心座標
   - `face_count`: 検出された顔の数
   - `face_center_x`, `face_center_y`: 顔の中心座標
   - `face_size`: 顔のサイズ
   - `face_mouth_open`: 口の開き具合（0-1）
   - `face_eyebrow_raise`: 眉の上がり具合（0-1）

2. **CLIP特徴量（512次元）**
   - `clip_0~511`: CLIP ViT-B/32の視覚的意味表現

### トラック特徴量（240次元）

- 20トラック × 12次元/トラック
- 各トラックの特徴量:
  - `active`: アクティブか（0/1）
  - `scale`: スケール（0-2）
  - `position_x`, `position_y`: 位置（-1~1）
  - `asset_id`: 素材ID（0-9）
  - その他: 時間情報、トラックID等

---

## モデルの詳細

### Multi-Track Transformer

**入力**:
- Audio: (batch, seq_len, 17)
- Visual: (batch, seq_len, 522)
- Track: (batch, seq_len, 240)
- Padding Mask: (batch, seq_len)
- Modality Mask: (batch, seq_len, 3)

**アーキテクチャ**:
```
1. Modality Embedding
   - Audio → 512次元
   - Visual → 512次元
   - Track → 512次元

2. Modality Fusion
   - 融合戦略: Concatenation / Attention / Gated
   - 出力: (batch, seq_len, 512)

3. Positional Encoding
   - 正弦波ベースの位置エンコーディング
   - 最大長: 5000フレーム

4. Transformer Encoder
   - 層数: 6
   - ヘッド数: 8
   - 隠れ層次元: 512
   - FFN次元: 2048
   - Dropout: 0.1

5. Multi-Head Output
   - Active Head: (batch, seq_len, num_tracks, 2)
   - Scale Head: (batch, seq_len, num_tracks, 1)
   - Position Head: (batch, seq_len, num_tracks, 2)
   - Asset ID Head: (batch, seq_len, num_tracks, 10)
```

**パラメータ数**: 約50M

**学習時間**: 100エポック（約2-3時間、GPU使用時）

**推論時間**: 5~10分/動画（特徴量抽出含む）


---

## 既知の問題点と制限事項

### 緊急（コードの整合性）

1. **`otio_xml_generator.py`のデッドコード**
   - 956行のうち642行目以降が実行されない
   - 正規表現によるXML操作（アンチパターン）
   - 複数の未使用関数が残存
   - **影響**: コードの可読性低下、メンテナンス困難
   - **対策**: リファクタリングまたは削除

2. **解像度のハードコード**
   - `direct_xml_generator.py`で1080x1920に固定
   - 横長動画（1920x1080）で正しく動作しない
   - **影響**: アスペクト比の問題
   - **対策**: 入力動画から解像度を自動取得

3. **Active閾値のハードコード**
   - `inference_pipeline.py`で0.29に固定
   - モデルの学習具合によって最適値が変動
   - **影響**: 実験やチューニングが困難
   - **対策**: コマンドライン引数または設定ファイルで指定可能に

### 高優先度（パフォーマンス・安定性）

4. **特徴量抽出のメモリリーク**
   - `extract_video_features.py`で全フレームをメモリに保持
   - 長時間動画でOOMクラッシュの可能性
   - **影響**: 数十分以上の動画で処理不可
   - **対策**: Chunk処理によるメモリ解放

5. **Loss関数の不安定性**
   - `loss.py`でactive_countによる除算
   - Activeなトラックが少ない場合に不安定
   - **影響**: 学習の収束が遅い
   - **対策**: バッチ全体で統一的な平均化

6. **テロップのBase64デコード未対応**
   - Premiere ProのBase64エンコード形式を解析できない
   - テロップの内容や位置情報を学習に活用できない
   - **影響**: テロップ予測の精度低下
   - **対策**: Base64デコード処理の実装

7. **テロップのXML出力未対応**
   - 学習したテロップ情報をXMLに出力できない
   - 現在はテロップ生成を無効化して対応
   - **影響**: テロップ自動生成機能が使えない
   - **対策**: テロップXML出力機能の実装

### 中優先度（機能拡張）

8. **モデルの確信度の低さ**
   - Active閾値0.29が必要なほど確信度が低い
   - 学習データは299シーケンスと十分だが、不均衡の可能性
   - **原因の可能性**:
     - 学習データの不均衡（Activeなフレームが少ない）
     - Loss関数の`active_weight`調整が不十分
     - 動画を分割したことによる文脈情報の損失
   - **影響**: 不要なクリップも採用されてしまう
   - **対策**: Loss関数の重み付け調整、データ拡張

9. **フレーム単位の回帰予測のジッター**
   - ScaleやPositionの予測が不安定
   - 生成された動画で画像がガクガク震える
   - **現在の応急処置**: クリップ全体の平均値を使用
   - **問題点**: 動き（ズームイン等）が消えてしまう
   - **対策**: 移動平均フィルタ、サビツキー・ゴーレイ・フィルタ

10. **XMLパーサーの制限**
    - ネストされたシーケンス未対応
    - マルチカムクリップ未対応
    - 複雑なエフェクトチェーン未対応
    - **影響**: 一部のPremiere Proプロジェクトで動作しない
    - **対策**: 再帰的なネスト処理の実装

11. **Asset ID管理の問題**
    - ファイル名ベースのID割り当て（0-9）
    - 学習時と推論時で素材が異なる場合に意味をなさない
    - **影響**: 新しい動画素材に対応できない
    - **対策**: 特徴量ベースのマッチング、役割ベースのID管理

12. **単一トラック配置の制限**
    - 現在は1つのトラックに全クリップが配置される
    - 複数トラックへの分散配置は未実装
    - **影響**: 編集の自由度が低い
    - **対策**: 複数トラック対応への改修

### 低優先度（将来的な改善）

13. **CLIP特徴量の補間問題**
    - 1秒間同じ値を使い回している
    - 細かいシーン変化を取り逃がす
    - **対策**: 抽出頻度の向上

14. **例外処理の不足**
    - MediaPipe初期化失敗時の適切なハンドリングがない
    - **対策**: エラーハンドリングの強化

15. **特徴量次元数の不一致**
    - コメントと実装が合致していない可能性
    - **対策**: コードレビューと修正

---

## 性能指標

### 学習データ
- **動画数**: 106本
- **シーケンス数**: 299（学習239 + 検証60）
- **総フレーム数**: 約150万フレーム
- **動画の長さ**: 平均10分

### モデル性能
- **学習時間**: 100エポック（約2-3時間、GPU使用時）
- **推論時間**: 5~10分/動画（特徴量抽出含む）
- **カット数**: 約8〜12個のクリップ（最小3秒、ギャップ結合後）
- **出力動画長**: 約2分（90秒〜150秒）

### 損失値（100エポック後）
- **Total Loss**: 約0.5
- **Active Loss**: 約0.3
- **Scale Loss**: 約0.1
- **Position Loss**: 約0.05
- **Asset ID Loss**: 約0.05

### 精度指標
- **Active判定精度**: 約75%（閾値0.29）
- **Scale予測誤差**: 約0.1（RMSE）
- **Position予測誤差**: 約0.05（RMSE）
- **Asset ID精度**: 約60%

**注意**: これらの指標は学習データに対するものであり、新規動画での性能は異なる可能性があります。

---

## 使用例

### 1. データ準備

```bash
# バッチファイルで一括実行（推奨）
run_data_preparation.bat

# または手動で実行
python -m src.data_preparation.premiere_xml_parser
python -m src.data_preparation.extract_video_features_parallel
python -m src.data_preparation.data_preprocessing
```

### 2. 学習

```bash
# バッチファイルで実行（推奨）
run_training.bat

# または手動で実行
python -m src.training.train --config configs/config_multimodal_experiment.yaml
```

### 3. 推論

```bash
# バッチファイルで実行（推奨）
run_inference.bat "path\to\your_video.mp4"

# または手動で実行
python -m src.inference.inference_pipeline "your_video.mp4" --output output.xml

# AI字幕生成を有効化
python -m src.inference.inference_pipeline "your_video.mp4" --output output.xml --telop_config configs/config_telop_generation.yaml

# AI字幕生成を無効化
python -m src.inference.inference_pipeline "your_video.mp4" --output output.xml --no-speech --no-emotion
```

### 4. テスト

```bash
# ユニットテスト
pytest tests/unit/

# 統合テスト
pytest tests/integration/

# 特定のテスト
pytest tests/unit/test_model.py
```

---

## 開発環境

### 必要な環境
- **OS**: Windows（バッチファイルを使用）
  - Mac/Linuxの場合は、Pythonコマンドを直接実行
- **Python**: 3.8以上
- **GPU**: CUDA対応GPU（推奨）
  - CPU でも動作するが、学習・推論が遅い
- **Premiere Pro**: XML読み込み用

### 依存パッケージ
- PyTorch >= 1.10.0
- OpenCV >= 4.5.0
- transformers >= 4.20.0
- OpenTimelineIO >= 0.14.0
- MediaPipe >= 0.8.0
- Whisper (オプション)
- その他: numpy, pandas, scipy, pyyaml等

### インストール

```bash
# 仮想環境の作成（推奨）
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

# 依存パッケージのインストール
pip install -r requirements.txt
```

---

## 今後の開発計画

### 短期（1-2ヶ月）
1. デッドコードの削除とリファクタリング
2. 解像度の動的取得
3. Active閾値の設定可能化
4. メモリリーク対策
5. Loss関数の安定化

### 中期（3-6ヶ月）
1. テロップのBase64デコード実装
2. テロップのXML出力実装
3. 予測値の平滑化
4. XMLパーサーの強化
5. 複数トラック対応

### 長期（6ヶ月以上）
1. Asset ID管理の改善
2. モデルの軽量化
3. 推論速度の向上
4. GUIツールの開発
5. クラウド対応

---

## ライセンス

MIT License

---

## 貢献

プルリクエストを歓迎します！特に以下の分野での貢献を募集しています：
- 学習データの提供
- パフォーマンス最適化
- ドキュメントの改善
- バグ修正

---

## 連絡先

プロジェクトに関する質問や提案は、GitHubのIssueでお願いします。

---

**最終更新**: 2025-12-19
**バージョン**: 1.0.0
