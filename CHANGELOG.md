# Changelog

## [2025-12-22] - カット選択モデルの実装と動画単位データ分割

### 🎯 主な変更
- カット選択専用モデル（Cut Selection Model）の実装
- 動画単位でのデータ分割（データリーク防止）
- プロジェクトをカット選択に特化
- リアルタイム学習可視化機能の追加

### ✅ 新機能

#### カット選択モデル
- Transformer + Gated Fusionアーキテクチャ
- Focal Loss（alpha=0.70）で採用見逃しに重いペナルティ
- Best F1スコア: 0.5630
- 最適閾値: -0.200（学習時に自動計算）

#### データ準備の改善
- 動画単位でのデータ分割を実装
  - 68本の動画を学習54本、検証14本に分割
  - データリーク防止（同じ動画のシーケンスは同じセットに配置）
  - より厳密な汎化性能の評価が可能

#### 学習可視化
- リアルタイムグラフ表示（6つのグラフ）
  - 損失関数（Train/Val Loss）
  - 損失の内訳（CE Loss vs TV Loss）
  - 分類性能（Accuracy & F1 Score）
  - Precision, Recall, Specificity
  - 最適閾値の推移
  - 予測の採用/不採用割合
- ブラウザで自動更新（2秒ごと）

### 📁 新規ファイル
- `src/cut_selection/` - カット選択モデル全体
- `configs/config_cut_selection.yaml` - モデル設定
- `scripts/create_cut_selection_data.py` - データ準備
- `train_cut_selection.bat` - 学習スクリプト
- `checkpoints_cut_selection/view_training.html` - 可視化ページ

### 📝 ドキュメント更新
- README.md: カット選択に特化した説明に更新
- docs/QUICK_START.md: カット選択モデル用に更新
- docs/PROJECT_SPECIFICATION.md: プロジェクト概要を更新
- グラフィック・テロップは精度が低く今後の課題として明記

### 🗑️ コード整理
- ルートディレクトリの大幅整理
  - デバッグスクリプト → archive/debug_scripts/
  - テストスクリプト → archive/test_scripts/
  - 古いドキュメント → archive/old_docs/
  - 古いバッチファイル → archive/old_batch_files/
- .kiroフォルダをGit管理から除外

### 📊 性能指標
- **学習データ**: 94本の動画から218,693フレーム
- **採用率**: 全体26.93%（学習28.19%、検証12.14%）
- **学習時間**: 50エポック（約1-2時間、GPU使用時）
- **Best F1スコア**: 0.5630（Epoch 33）

---

## [2025-12-17] - ワークスペース整理とパス統一

### 🎯 主な変更
- ワークスペース全体を整理し、プロジェクト構造を明確化
- すべてのインポートパスを`from src.`形式の絶対インポートに統一
- バッチファイルを正しいエントリーポイントに修正

### ✅ 検証済み
- インポートパステスト: 20/20成功
- 機能テスト: 8/8成功
- トレーニングパイプライン: 完全動作確認
- 推論パイプライン: 完全動作確認

### 📁 ディレクトリ構造の変更
- ルートディレクトリから30個以上のファイルを整理
- `archive/`ディレクトリに古いファイルを移動
- `scripts/`ディレクトリに補助スクリプトを整理
- `src/`ディレクトリ内のコードは変更なし（インポートパスのみ修正）

### 🔧 修正内容

#### インポートパス修正
- `src/model/model_persistence.py`: 相対インポートを絶対インポートに修正

#### バッチファイル修正
- `run_training.bat`: 正しいエントリーポイント(`src.training.train`)に修正
- デフォルト設定ファイルを`config_multimodal_experiment.yaml`に変更

#### .gitignore更新
- `checkpoints/`を追加（学習済みモデルを除外）
- `backups/`を追加（バックアップを除外）
- `preprocessed_data/`を追加（前処理済みデータを除外）
- `temp_features/`を追加（一時ファイルを除外）
- `test_*.py`を追加（テスト用一時ファイルを除外）
- `.kiro/`を追加（Kiro IDE設定を除外）

### 📊 システム状態
- **モデル**: MultimodalTransformer (5,212,694パラメータ)
- **ベストエポック**: 59
- **トレーニングデータ**: 239動画 (80.3%マルチモーダル)
- **バリデーションデータ**: 60動画 (83.3%マルチモーダル)

### 🚀 動作確認済みコマンド
```bash
# データ準備（カット選択）
python scripts/create_cut_selection_data.py

# トレーニング（カット選択）
train_cut_selection.bat

# 推論
run_inference.bat "video.mp4"
```

### 📝 ドキュメント更新
- README.md: プロジェクト構造とコマンド例を更新
- VERIFICATION_REPORT.md: 完全な検証レポートを作成
- FINAL_VERIFICATION_SUMMARY.md: 最終確認サマリーを作成

### 🗑️ 削除されたファイル
- 一時的な検証・分析用ファイル（BATCH_FILES_RECOMMENDATION.md等）
- 古いスクリプトとテストファイル（archive/に移動済み）

---

## 今後の予定
- [ ] K-Fold Cross Validation実装
- [ ] グラフィック配置モデルの精度改善
- [ ] テロップ生成モデルの精度改善
- [ ] 追加のユニットテスト作成
- [ ] パフォーマンス最適化
