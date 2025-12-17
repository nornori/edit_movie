# ワークスペース整理計画

## 📊 現状分析

### 問題点
- ルートディレクトリに200個以上のファイルが散乱
- テスト用XMLファイルが40個以上
- 古いスクリプトと新しいスクリプトが混在
- ドキュメントファイルが整理されていない

---

## 🎯 整理後の理想的な構造

```
xmlai/
├── 📁 src/                          # メインのソースコード
│   ├── data_preparation/            # データ準備用スクリプト
│   ├── model/                       # モデル関連
│   ├── training/                    # 学習用
│   ├── inference/                   # 推論用
│   └── utils/                       # ユーティリティ
│
├── 📁 tests/                        # テストコード
│   ├── unit/                        # ユニットテスト
│   └── integration/                 # 統合テスト
│
├── 📁 configs/                      # 設定ファイル
│   ├── config_multimodal.yaml
│   └── config.yaml
│
├── 📁 docs/                         # ドキュメント
│   ├── guides/                      # ガイド
│   └── summaries/                   # サマリー
│
├── 📁 data/                         # データディレクトリ
│   ├── raw/                         # 生データ
│   │   └── editxml/                 # 編集済み動画とXML
│   ├── processed/                   # 処理済みデータ
│   │   ├── input_features/          # 特徴量
│   │   ├── output_labels/           # ラベル
│   │   └── master_training_data.csv
│   └── temp/                        # 一時ファイル
│       └── temp_features/
│
├── 📁 models/                       # 学習済みモデル
│   ├── checkpoints_50epochs/
│   └── checkpoints/
│
├── 📁 outputs/                      # 出力ファイル
│   ├── inference_results/           # 推論結果
│   └── test_outputs/                # テスト出力
│
├── 📁 scripts/                      # 補助スクリプト
│   ├── batch_processing/            # バッチ処理
│   └── utilities/                   # ユーティリティ
│
├── 📁 archive/                      # アーカイブ（古いファイル）
│   ├── old_scripts/
│   ├── old_xmls/
│   └── old_tests/
│
├── README.md                        # プロジェクト説明
├── requirements.txt                 # 依存ライブラリ
└── .gitignore                       # Git除外設定
```

---

## 🔧 整理手順

### ステップ1: 新しいディレクトリ構造を作成
```bash
mkdir -p src/data_preparation
mkdir -p src/model
mkdir -p src/training
mkdir -p src/inference
mkdir -p src/utils
mkdir -p tests/unit
mkdir -p tests/integration
mkdir -p configs
mkdir -p docs/guides
mkdir -p docs/summaries
mkdir -p data/raw/editxml
mkdir -p data/processed/input_features
mkdir -p data/processed/output_labels
mkdir -p data/temp/temp_features
mkdir -p models
mkdir -p outputs/inference_results
mkdir -p outputs/test_outputs
mkdir -p scripts/batch_processing
mkdir -p scripts/utilities
mkdir -p archive/old_scripts
mkdir -p archive/old_xmls
mkdir -p archive/old_tests
```

### ステップ2: ファイルを移動

#### 📁 src/data_preparation/ に移動
- `premiere_xml_parser.py`
- `extract_video_features_parallel.py`
- `extract_video_features.py`
- `telop_extractor.py`
- `text_embedding.py`
- `data_preprocessing.py`
- `xml2csv.py`
- `movie2csv.py`
- `fcpxml_to_tracks.py`

#### 📁 src/model/ に移動
- `model.py`
- `multimodal_modules.py`
- `model_persistence.py`
- `loss.py`

#### 📁 src/training/ に移動
- `training.py`
- `train.py`
- `multimodal_dataset.py`
- `dataset.py`
- `multimodal_preprocessing.py`

#### 📁 src/inference/ に移動
- `inference_pipeline.py`
- `otio_xml_generator.py`
- `fix_telop_simple.py`

#### 📁 src/utils/ に移動
- `feature_alignment.py`
- `sequence_processing.py`

#### 📁 tests/unit/ に移動
- `test_model.py`
- `test_dataset.py`
- `test_loss_compatibility.py`
- `test_feature_alignment.py`
- `test_preprocessing.py`
- `test_sequence_processing.py`
- `test_multimodal_dataset.py`
- `test_multimodal_model.py`
- `test_multimodal_modules.py`
- `test_multimodal_preprocessing.py`
- `test_model_properties.py`

#### 📁 tests/integration/ に移動
- `test_inference_pipeline.py`
- `test_batch_processing.py`
- `test_backward_compatibility.py`
- `test_extract_with_telop.py`
- `test_telop_integration.py`
- `test_text_embedding_integration.py`
- `test_telop_csv_generation.py`
- `test_real_inference.py`
- `test_training_logging.py`

#### 📁 configs/ に移動
- `config_multimodal.yaml`
- `config_multimodal_experiment.yaml`
- `config.yaml`

#### 📁 docs/guides/ に移動
- `PROJECT_WORKFLOW_GUIDE.md`
- `REQUIRED_FILES_BY_PHASE.md`
- `FCPXML_EXTRACTION_GUIDE.md`
- `VIDEO_FEATURE_EXTRACTION_GUIDE.md`

#### 📁 docs/summaries/ に移動
- `AUDIO_CUT_AND_TELOP_GRAPHICS_SUMMARY.md`
- `INFERENCE_PIPELINE_SUMMARY.md`
- `MULTIMODAL_FINAL_SUMMARY.md`
- `MULTIMODAL_IMPLEMENTATION_SUMMARY.md`
- `MULTIMODAL_TRAINING_RESULTS.md`
- `MULTIMODAL_VALIDATION_SUMMARY.md`
- `PREMIERE_XML_EXTRACTION_SUMMARY.md`
- `PREMIERE_XML_PARSER_UPDATE.md`
- `TELOP_INTEGRATION_SUMMARY.md`
- `TEXT_EMBEDDING_SUMMARY.md`
- `TRAINING_50EPOCHS_RESULTS.md`
- `TRAINING_RESULTS.md`
- `FINAL_PROGRESS.md`
- `PROGRESS.md`
- `PROJECT_COMPLETE.md`

#### 📁 scripts/batch_processing/ に移動
- `batch_extract_features.py`
- `batch_process_xml.py`
- `batch_xml2csv_keyframes.py`
- `batch_test_fcpxml.bat`

#### 📁 scripts/utilities/ に移動
- `check_all_files.py`
- `check_mediapipe.py`
- `check_model_weights.py`
- `check_nan_in_features.py`
- `check_telop_in_xml.py`
- `check_telop_premiere.py`
- `check_text.py`
- `validate_features.py`
- `validate_features_quick.py`
- `verify_csv_quality.py`
- `verify_sequences.py`
- `verify_text_content.py`
- `reextract_single_video.py`
- `add_telop_to_existing_csv.py`

#### 📁 archive/old_scripts/ に移動（古いバージョン）
- `csv2xml.py`
- `csv2xml2.py`
- `csv2xml3.py`
- `csv2ai.py`
- `debug_inference.py`
- `debug_nan_in_training.py`
- `fix_telop_graphics.py`
- `fix_xml_format.py`
- `generate_audio_cut_xml.py`
- `generate_working_xml.py`

#### 📁 archive/old_xmls/ に移動（テスト用XML）
すべての `bandicam_*.xml` と `inference_*.xml` ファイル:
- `bandicam 2025-06-02 00-03-33-780_*.xml` (全バージョン)
- `bandicam_2025-06-02_*.xml` (全バージョン)
- `inference_*.xml` (全バージョン)
- `test_*.xml` (全バージョン)
- `premiere_auto.xml`

**例外**: 以下は残す
- `bandicam_2025-06-02_COMPLETE.xml` → `outputs/inference_results/` に移動（成功例として）

#### 📁 data/raw/editxml/ に移動
- 既存の `editxml/` フォルダの内容

#### 📁 data/processed/ に移動
- `input_features/` フォルダ → `data/processed/input_features/`
- `output_labels/` フォルダ → `data/processed/output_labels/`
- `master_training_data.csv` → `data/processed/`
- `preprocessed_data/` フォルダ → `data/processed/preprocessed_data/`

#### 📁 data/temp/ に移動
- `temp_features/` フォルダ → `data/temp/temp_features/`

#### 📁 models/ に移動
- `checkpoints/` フォルダ → `models/checkpoints/`
- `checkpoints_50epochs/` フォルダ → `models/checkpoints_50epochs/`
- `test_checkpoints/` フォルダ → `models/test_checkpoints/`

#### 📁 archive/ に移動（その他の古いファイル）
- `analysis/` フォルダ
- `backup/` フォルダ
- `edit_triaining/` フォルダ
- `final_dataset/` フォルダ
- `inference/` フォルダ
- `input_jsons/` フォルダ
- `night_run_data_parallel/` フォルダ
- `premiere_test_extended/` フォルダ
- `premiere_test_output/` フォルダ
- `preprocessing/` フォルダ
- `test_features/` フォルダ
- `training/` フォルダ
- `_archive/` フォルダ（既存のアーカイブ）

#### 📁 outputs/test_outputs/ に移動
- `test_features.csv`
- `test_audio_prep.pkl`
- `test_visual_prep.pkl`
- `test_model.pth`
- `test_model.json`
- `feature_validation_report.txt`
- `batch_processing.log`
- `premiere_error_log.txt`
- `final_timeline.csv`

#### その他のファイル
- `scaler.pkl` → `models/`
- `editor_ai_model.pth` → `models/`
- `bandicam 2025-12-07 21-59-57-374_features.csv` → `data/temp/`

---

## ✅ 整理後の確認事項

### ルートディレクトリに残すファイル
- `README.md` - プロジェクト説明（新規作成）
- `requirements.txt` - 依存ライブラリ（新規作成）
- `.gitignore` - Git除外設定
- `WORKSPACE_CLEANUP_PLAN.md` - この整理計画

### 削除しても良いファイル
- `__pycache__/` - Pythonキャッシュ（自動生成される）
- `.pytest_cache/` - pytestキャッシュ（自動生成される）
- `.hypothesis/` - Hypothesisキャッシュ（自動生成される）

---

## 🚀 整理後の使い方

### データ準備
```bash
python src/data_preparation/premiere_xml_parser.py
python src/data_preparation/extract_video_features_parallel.py
python src/data_preparation/data_preprocessing.py
```

### 学習
```bash
python src/training/training.py --config configs/config_multimodal.yaml
```

### 推論
```bash
python src/inference/inference_pipeline.py "video.mp4" \
    --model models/checkpoints_50epochs/best_model.pth \
    --output outputs/inference_results/temp.xml

python src/inference/fix_telop_simple.py \
    outputs/inference_results/temp.xml \
    outputs/inference_results/final.xml
```

---

## 📝 注意事項

### インポートパスの修正が必要
整理後、各スクリプトのインポートパスを修正する必要があります：

**修正前**:
```python
from model import create_model
from multimodal_modules import MultimodalEncoder
```

**修正後**:
```python
from src.model.model import create_model
from src.model.multimodal_modules import MultimodalEncoder
```

または、`src/`を Pythonパスに追加：
```bash
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
```

### `__init__.py` の追加
各ディレクトリに `__init__.py` を追加してPythonパッケージとして認識させる：
```bash
touch src/__init__.py
touch src/data_preparation/__init__.py
touch src/model/__init__.py
touch src/training/__init__.py
touch src/inference/__init__.py
touch src/utils/__init__.py
```

---

## 🎯 整理の優先順位

### 優先度: 高（すぐに実行）
1. テスト用XMLファイルを `archive/old_xmls/` に移動（40個以上）
2. ドキュメントを `docs/` に移動（15個以上）
3. 古いスクリプトを `archive/old_scripts/` に移動

### 優先度: 中（時間があれば）
4. データディレクトリを整理
5. モデルディレクトリを整理
6. テストコードを整理

### 優先度: 低（余裕があれば）
7. インポートパスの修正
8. `__init__.py` の追加
9. `README.md` と `requirements.txt` の作成
