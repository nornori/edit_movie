# 動画編集AI - 自動編集パラメータ予測システム

動画から自動的に編集パラメータ（カット位置、ズーム、クロップ、テロップ）を予測し、Premiere Pro用のXMLを生成するAIシステムです。

## 🎯 機能

- **マルチモーダル学習**: 音声・映像・トラックの3つのモダリティを統合
- **自動カット検出**: AIが最適なカット位置を予測
- **音声同期カット**: 映像と音声を同じ位置で自動カット
- **テロップ自動配置**: OCRでテロップを検出し、Premiere Pro互換のグラフィックとして出力
- **🆕 AI字幕生成**: 音声認識と感情検出で自動的に字幕を生成
  - 音声認識（Whisper）による自動文字起こし
  - 感情検出（笑い、驚き、悲しみ）による感情字幕
  - カスタマイズ可能なテキストパターン
- **Premiere Pro連携**: 生成されたXMLをそのままPremiere Proで開ける

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
- Python 3.8+
- CUDA対応GPU（推奨）
- Premiere Pro（XML読み込み用）

### インストール
```bash
pip install -r requirements.txt
```

### 新しい動画を自動編集（超簡単！）

**方法1: バッチファイルを使う（推奨）**
```bash
run_inference.bat "D:\videos\my_video.mp4"
```

**方法2: 手動で実行**
```bash
# 推論実行
python -m src.inference.inference_pipeline "your_video.mp4" outputs/inference_results/output.xml

# Premiere Proで output.xml を開く
```

詳しくは [QUICK_START.md](QUICK_START.md) を参照してください。

### 🆕 AI字幕生成を使う

```bash
# 1. 依存関係をインストール
pip install openai-whisper librosa soundfile pydub

# 2. AI字幕生成を有効にして推論
python -m src.inference.inference_pipeline "your_video.mp4" outputs/inference_results/output.xml

# 3. Premiere Proで output.xml を開く
```

**カスタマイズ**:
- 感情検出の感度調整: `configs/config_telop_generation.yaml`を編集
- テキストパターン変更: 笑い（"www"）、驚き（"！"）、悲しみ（"..."）
- 詳細は [AI字幕カスタマイズガイド](docs/guides/AI_TELOP_CUSTOMIZATION_GUIDE.md) を参照

**コマンドラインオプション**:
```bash
# カスタム設定ファイルを使用
python -m src.inference.inference_pipeline video.mp4 output.xml --telop_config my_config.yaml

# 音声認識のみ（感情検出なし）
python -m src.inference.inference_pipeline video.mp4 output.xml --no-emotion

# 感情検出のみ（音声認識なし）
python -m src.inference.inference_pipeline video.mp4 output.xml --no-speech
```

## 📚 ドキュメント

### 基本ガイド
- [プロジェクト全体の流れ](docs/guides/PROJECT_WORKFLOW_GUIDE.md)
- [必要なファイル一覧](docs/guides/REQUIRED_FILES_BY_PHASE.md)
- [音声カット & テロップ変換](docs/summaries/AUDIO_CUT_AND_TELOP_GRAPHICS_SUMMARY.md)

### 🆕 AI字幕生成ガイド
- [AI字幕カスタマイズガイド](docs/guides/AI_TELOP_CUSTOMIZATION_GUIDE.md) - 詳細な設定方法
- [AI字幕クイックリファレンス](docs/guides/AI_TELOP_QUICK_REFERENCE.md) - 簡単な設定例

## 🔧 開発

### データ準備
```bash
# バッチファイルで一括実行（推奨）
run_data_preparation.bat

# または手動で実行
python -m src.data_preparation.premiere_xml_parser
python -m src.data_preparation.extract_video_features_parallel
python -m src.data_preparation.data_preprocessing
```

### 学習
```bash
# バッチファイルで実行（推奨）
run_training.bat

# または手動で実行
python -m src.training.train --config configs/config_multimodal_experiment.yaml
```

### テスト
```bash
# ユニットテスト
pytest tests/unit/

# 統合テスト
pytest tests/integration/
```

## 📊 性能

- **学習データ**: 約10本の編集済み動画
- **学習時間**: 50エポック（約2-3時間、GPU使用時）
- **推論時間**: 約30秒/動画（特徴量抽出含む）
- **カット数**: 約500個/動画（閾値0.29の場合）

## ⚠️ 既知の問題点・改善点

### 現在の問題点
- **テロップがBase64エンコードで特徴量に含められていない**
  - Premiere ProのBase64エンコード形式のため、テロップの内容や位置情報を学習に活用できていない
- **テロップのXML出力未対応**
  - 学習したテロップ情報をXMLに出力する機能が実装されていない
- **1つのトラックに複数動画が並んで編集しづらい**
  - 単一トラックに全クリップが配置されるため、編集の自由度が低い
- **0.1秒ごとの過剰なカットで音声・動画が飛び飛び**
  - カット閾値の調整不足により、短すぎるクリップが生成され視聴体験が悪化

### 改善予定（優先度別）

#### 高優先度
- [ ] **テロップデコード**: Base64エンコードされたテロップをデコードして特徴量に含める
- [ ] **テロップのXML出力**: 学習したテロップ情報をXMLに出力する機能を実装
- [ ] **カット閾値最適化**: 短すぎるクリップ（<0.5秒）を自動的にマージする機能
- [ ] **トラック配置改善**: 複数トラックに分散配置して編集しやすいXMLを生成

#### 中優先度
- [ ] 学習データ拡充（目標: 100本以上）
- [ ] 特徴量抽出の高速化
- [ ] モデルの軽量化

#### 低優先度
- [ ] ユニットテストの拡充
- [ ] ドキュメントの充実化

### 技術的負債
- **Base64形式の解析処理未実装**: Premiere ProのBase64エンコード形式の解析・デコード処理が必要
- **カット閾値のハードコーディング**: 閾値（0.29）が固定されており、調整機能が不足
- **単一トラック配置の制限**: 複数トラック対応への改修が必要
- **異常カット検出機能の不足**: 0.1秒以下の異常なカットを検出・修正する機能が必要

## 🤝 貢献

プルリクエストを歓迎します！特に以下の分野での貢献を募集しています：
- 学習データの提供
- パフォーマンス最適化
- ドキュメントの改善
- バグ修正

## 📝 ライセンス

MIT License


