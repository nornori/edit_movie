# Changelog

## [2025-12-25] - アンサンブル学習による大幅な性能向上 🎉

### 🎯 達成した成果

**F1スコアの劇的な改善:**
```
個別モデル平均: 41.23% F1
        ↓
アンサンブル: 60.80% F1 ✨

改善: +19.57ポイント (+47.47%)
目標達成: ✅ 55% F1 → 60.80% F1 (目標を5.80ポイント上回る)
```

### 📊 詳細な結果

#### アンサンブル性能
- **F1 Score**: 41.23% → **60.80%** (+19.57pt)
- **Accuracy**: 49.83% → **78.69%** (+28.86pt)
- **Precision**: 28.70% → **52.90%** (+24.20pt)
- **Recall**: 74.31% → **71.45%** (-2.86pt)
- **Specificity**: - → **80.87%**

#### 各Foldの個別性能
| Fold | Best Epoch | F1 Score | Accuracy | Precision | Recall |
|------|-----------|----------|----------|-----------|--------|
| 1 | 4 | **49.52%** | 73.48% | 36.88% | 75.34% |
| 2 | 1 | 41.22% | 36.44% | 27.85% | 79.24% |
| 3 | 2 | 40.69% | 43.11% | 28.52% | 71.00% |
| 4 | 19 | 40.43% | 47.18% | 27.68% | 74.95% |
| 5 | 32 | 34.27% | 48.92% | 22.58% | 71.03% |
| **平均** | **11.6±12.1** | **41.23±4.86%** | **49.83±12.58%** | **28.70±4.61%** | **74.31±3.08%** |

#### アンサンブル戦略の比較
- **Soft Voting**: **60.80%** F1 (最良)
- Hard Voting: 60.80% F1
- Weighted Voting: 60.24% F1

### ✅ 新機能

#### 1. 時系列特徴量の追加（83個）
- 移動統計量: MA5, MA10, MA30, MA60, MA120, STD5, STD30, STD120
- 変化率: DIFF1, DIFF2, DIFF30
- カットタイミング: time_since_prev, time_to_next, cut_duration, position_in_video, cut_density_10s
- CLIP類似度: clip_sim_prev, clip_sim_next, clip_sim_mean5
- 音声変化: audio_change_score, silence_to_speech, speech_to_silence, speaker_change, pitch_change
- 映像変化: visual_motion_change, face_count_change, saliency_movement
- 累積統計: cumulative_position, cumulative_adoption_rate

#### 2. アンサンブル学習の実装
- 5つのK-Foldモデルを組み合わせ
- 3つの投票戦略（Soft, Hard, Weighted）
- 最適閾値の自動探索（Recall制約付き）

#### 3. V2モデルの設計（データ拡張 + 深いネットワーク）
- 8層エンコーダー（V1は6層）
- 16個のAttentionヘッド（V1は8個）
- データ拡張: ノイズ追加、時間シフト、スケーリング、時間ワーピング
- 改善されたFusion: 残差接続付き

### 📁 新規ファイル

#### スクリプト
- `scripts/add_temporal_features.py` - 時系列特徴量追加
- `scripts/create_cut_selection_data_enhanced.py` - 拡張データ作成
- `scripts/combine_sequences_enhanced.py` - K-Fold用データ結合

#### モデル
- `src/cut_selection/cut_model_enhanced.py` - 拡張モデル（V1）
- `src/cut_selection/cut_model_enhanced_v2.py` - 改善モデル（V2）
- `src/cut_selection/cut_dataset_enhanced.py` - 拡張データセット（V1）
- `src/cut_selection/cut_dataset_enhanced_v2.py` - 拡張データセット（V2、データ拡張付き）
- `src/cut_selection/time_series_augmentation.py` - 時系列データ拡張
- `src/cut_selection/ensemble_predictor.py` - アンサンブル予測器
- `src/cut_selection/evaluate_ensemble.py` - アンサンブル評価

#### トレーニング
- `src/cut_selection/train_cut_selection_kfold_enhanced.py` - V1トレーニング
- `src/cut_selection/train_cut_selection_kfold_enhanced_v2.py` - V2トレーニング

#### 設定
- `configs/config_cut_selection_kfold_enhanced.yaml` - V1設定
- `configs/config_cut_selection_kfold_enhanced_v2.yaml` - V2設定

#### バッチファイル（batch/フォルダに整理）
- `batch/train_cut_selection_enhanced.bat` - V1トレーニング
- `batch/train_cut_selection_enhanced_v2.bat` - V2トレーニング
- `batch/evaluate_ensemble.bat` - アンサンブル評価
- `batch/run_inference.bat` - 推論実行

#### ドキュメント
- `docs/ENSEMBLE_RESULTS.md` - アンサンブル結果詳細
- `FEATURE_ENHANCEMENT_README.md` - 特徴量拡張ガイド

### 📊 出力ファイル
- `checkpoints_cut_selection_kfold_enhanced/` - V1モデルチェックポイント
  - `fold_X_best_model.pth` - 各Foldの最良モデル
  - `kfold_summary.csv` - K-Fold統計
  - `kfold_comparison.png` - 比較グラフ
  - `ensemble_comparison.csv` - アンサンブル比較
  - `ensemble_comparison.png` - アンサンブルグラフ
  - `view_training.html` - リアルタイムビューアー

### 🗑️ フォルダ整理
- `archive/old_experiments/` - 古い実験スクリプト
- `archive/old_logs/` - 古い実験ログ
- `archive/old_batch_files/` - 古いバッチファイル
- `archive/old_checkpoints/` - 古いチェックポイント
- `batch/` - 現在使用中のバッチファイル（新規作成）

### 🚀 成功の要因
1. **時系列特徴量の追加** ⭐⭐⭐⭐⭐
2. **アンサンブル学習** ⭐⭐⭐⭐⭐
3. **Focal Loss** ⭐⭐⭐⭐
4. **K-Fold Cross Validation** ⭐⭐⭐⭐
5. **最適閾値の自動探索** ⭐⭐⭐

### 🎯 次のステップ
- V2モデルのトレーニング（期待値: 65%+ F1）
- V2モデルでのアンサンブル評価
- 本番環境への展開

---

## [2025-12-22] - K-Fold Cross Validation完了と最終最適化

### 🎯 主な変更
- K-Fold Cross Validation（5分割）の実装と完了
- データリーク防止（GroupKFold）の実装
- 損失関数の最適化（採用率ペナルティの削除）
- メトリクス計算の修正（全メトリクスで最適閾値を使用）
- 可視化の改善（CE Loss vs TV Loss グラフの修正）

### ✅ 最終性能（K-Fold Cross Validation）
- **Mean F1 Score**: 0.4427 ± 0.0451
- **Mean Recall**: 0.7230 ± 0.1418（採用の72%を検出）✅
- **Mean Precision**: 0.3310 ± 0.0552（予測の33%が正解）✅
- **Mean Accuracy**: 0.5855 ± 0.1008
- **Optimal Threshold**: -0.235 ± 0.103

### 🔧 技術的改善

#### データリーク防止
- GroupKFoldの実装（同じ動画のシーケンスは同じFoldに配置）
- 完全なシード固定（`PYTHONHASHSEED`を含む）
- 各Foldでtrain/val動画の重複チェック

#### メトリクス計算の修正
- 全メトリクス（Accuracy, Precision, Recall, F1, Specificity）で最適閾値を使用
- 以前はargmax（50%閾値）を使用していたため不正確だった
- precision_recall_curveで最適閾値を自動計算

#### 損失関数の最適化
- 採用率ペナルティシステムを完全削除（負の損失値を引き起こしていた）
- Class Weights: Active 3x, Inactive 3x（両方のエラーに同等のペナルティ）
- Focal Loss: alpha=0.75, gamma=3.0
- TV Loss: 0.05x

#### 可視化の改善
- CE Loss vs TV Loss グラフの修正
  - CE Lossを左軸、TV Lossを右軸に分離
  - Val CE Lossが複数回描画される問題を修正
  - twin軸の適切なクリア処理を実装
- 6グラフシステムの完成
- HTMLビューアーのキャッシュバスティング（タイムスタンプ付きURL）

### 📁 新規ファイル
- `src/cut_selection/train_cut_selection_kfold.py` - K-Fold学習スクリプト
- `scripts/create_combined_data_for_kfold.py` - K-Fold用データ準備
- `configs/config_cut_selection_kfold.yaml` - K-Fold設定
- `train_cut_selection_kfold.bat` - K-Fold学習バッチファイル
- `docs/K_FOLD_CROSS_VALIDATION.md` - K-Fold詳細ドキュメント
- `docs/K_FOLD_FINAL_RESULTS.md` - 最終結果サマリー

### 📊 出力ファイル
- `checkpoints_cut_selection_kfold/kfold_summary.csv` - 統計サマリー
- `checkpoints_cut_selection_kfold/kfold_comparison.png` - 全Fold比較
- `checkpoints_cut_selection_kfold/fold_X/training_progress.png` - Fold別詳細（6グラフ）
- `checkpoints_cut_selection_kfold/view_training.html` - リアルタイムビューアー
- `checkpoints_cut_selection_kfold/inference_params.yaml` - 推論パラメータ

### 📝 ドキュメント更新
- README.md: K-Fold結果を反映
- docs/K_FOLD_FINAL_RESULTS.md: 詳細な結果分析

### 🐛 修正されたバグ
- Val CE Lossが複数回描画される問題
- TV Lossがグラフで見えない問題（スケールの違い）
- 採用率ペナルティによる負の損失値
- メトリクス計算の不整合（argmax vs 最適閾値）

---

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
