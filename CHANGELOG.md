# Changelog

## [2025-12-26] - 正しい評価方法による性能検証 ✅

### 🎯 重要な発見

**データリーク問題の発見と修正:**

以前報告した60.80% F1には**データリークの問題**がありました：
- 全データ（訓練+検証）で評価していた
- 各モデルが自分の訓練データも評価に含めていた
- 過大評価（楽観的な数値）

**正しい評価方法で再検証:**
- 各Foldで完全に未見のデータのみで評価
- データリークなし
- 真の汎化性能を測定

### 📊 最終結果（検証済み）

#### 真の汎化性能

| 指標 | 平均値 | 標準偏差 | 最良（Fold 1） |
|------|--------|----------|----------------|
| **F1 Score** | **42.30%** | ±5.75% | **49.42%** |
| **Accuracy** | 50.24% | ±14.92% | 73.63% |
| **Precision** | 29.83% | ±5.80% | 36.94% |
| **Recall** | **76.10%** | ±5.19% | 74.65% |

#### 各Foldの詳細結果

| Fold | Best Epoch | F1 Score | Accuracy | Precision | Recall | Threshold |
|------|-----------|----------|----------|-----------|--------|-----------|
| 1 | 4 | **49.42%** | 73.63% | 36.94% | 74.65% | -0.558 |
| 2 | 1 | 41.22% | 36.44% | 27.85% | 79.24% | -0.474 |
| 3 | 20 | 43.10% | 48.45% | 30.94% | 71.02% | -0.573 |
| 4 | 9 | 45.57% | 59.42% | 33.54% | 71.03% | -0.509 |
| 5 | 3 | 32.20% | 33.26% | 19.89% | 84.54% | -0.550 |
| **平均** | **7.4±6.8** | **42.30±5.75%** | **50.24±14.92%** | **29.83±5.80%** | **76.10±5.19%** | **-0.533±0.036** |

### ✅ 改善点

1. **正しい評価方法の確立**
   - K-Fold Cross Validationで各Foldの検証データのみで評価
   - データリークを完全に排除
   - 真の汎化性能を測定

2. **再現性の確保**
   - ランダムシード42で固定
   - 同じ設定で一貫した結果（42.30% ± 5.75%）

3. **ドキュメント整備**
   - `docs/FINAL_RESULTS.md`: 最終結果レポート
   - `docs/ENSEMBLE_60_80_PARAMS.md`: 60.80%記録時のパラメータ
   - README.md更新: 正確な性能を反映

### 📁 バックアップ

60.80%を記録したモデル（データリークあり）をバックアップ：
```
backups/2025-12-26_01-20-31_ensemble_60_80_percent/
```

### 🎯 目標達成状況

| 項目 | 目標 | 達成 | 状況 |
|------|------|------|------|
| F1スコア | 55% | 42.30% | ❌ 未達成 (-12.70pt) |
| Recall | 71% | 76.10% | ✅ 達成 (+5.10pt) |

### 💡 今後の改善方向

1. **データ拡張**: より多くの動画データ収集
2. **クラス不均衡対応**: より強力なFocal Loss設定
3. **特徴量エンジニアリング**: 長期的な時系列パターン追加
4. **モデルアーキテクチャ**: より深いTransformer検討

---

## [2025-12-25] - 時系列特徴量追加とK-Fold Cross Validation実装

### ⚠️ 重要な注意

この日のアンサンブル評価結果（60.80% F1）には**データリークの問題**がありました（2025-12-26に発見）。
正しい評価結果は上記の[2025-12-26]セクションを参照してください。

### ✅ 新機能

#### 1. 時系列特徴量の追加（83個）
- 移動統計量: MA5, MA10, MA30, MA60, MA120, STD5, STD30, STD120
- 変化率: DIFF1, DIFF2, DIFF30
- カットタイミング: time_since_prev, time_to_next, cut_duration, position_in_video, cut_density_10s
- CLIP類似度: clip_sim_prev, clip_sim_next, clip_sim_mean5
- 音声変化: audio_change_score, silence_to_speech, speech_to_silence, speaker_change, pitch_change
- 映像変化: visual_motion_change, face_count_change, saliency_movement
- 累積統計: cumulative_position, cumulative_adoption_rate

#### 2. K-Fold Cross Validation実装
- GroupKFoldで動画単位の分割
- データリーク防止（同じ動画は同じFoldに）
- 5-Fold評価で汎化性能を測定

#### 3. アンサンブル学習実装（データリークあり）
- Soft Voting（確率の平均）
- Hard Voting（多数決）
- Weighted Voting（F1スコアで重み付け）

### 📊 個別モデル性能（正しい評価）

| Fold | F1 Score |
|------|----------|
| 1 | 49.52% |
| 2 | 41.22% |
| 3 | 40.69% |
| 4 | 40.43% |
| 5 | 34.27% |
| **平均** | **41.23%** |

### 📁 新規ファイル

- `scripts/add_temporal_features.py`: 時系列特徴量追加スクリプト
- `src/cut_selection/train_cut_selection_kfold_enhanced.py`: K-Fold訓練スクリプト
- `src/cut_selection/ensemble_predictor.py`: アンサンブル予測器
- `src/cut_selection/evaluate_ensemble.py`: アンサンブル評価スクリプト
- `configs/config_cut_selection_kfold_enhanced.yaml`: 拡張モデル設定

---

## [2025-12-16] - マルチモーダル学習の成功

### 🎯 達成した成果

**100エポック訓練完了:**
- Val Loss: 0.0878（安定）
- Val Accuracy: 0.7586（75.86%）
- 過学習なし、安定した学習曲線

### ✅ 改善点

1. **学習率スケジューラの改善**
   - ReduceLROnPlateau導入
   - patience=5, factor=0.5
   - 学習の安定化

2. **Early Stopping実装**
   - patience=10
   - 過学習防止

3. **可視化の強化**
   - リアルタイム学習グラフ
   - 6つのメトリクスを同時表示

### 📁 生成ファイル

- `checkpoints_multimodal/best_model.pth`
- `checkpoints_multimodal/training_progress.png`
- `checkpoints_multimodal/view_training.html`

---

## [2025-12-15] - プロジェクト整理とドキュメント整備

### ✅ 改善点

1. **フォルダ構造の整理**
   - ルートディレクトリ: 30個以上 → 7個に削減
   - `batch/` フォルダ作成（バッチファイル整理）
   - `archive/` フォルダに古いファイル移動

2. **ドキュメント整備**
   - `PROJECT_STRUCTURE.md`: プロジェクト構造説明
   - `CLEANUP_SUMMARY.md`: 整理レポート
   - README.md更新

3. **Git管理**
   - 不要ファイルの削除
   - .gitignore更新
   - コミット & プッシュ完了

---

## [Initial Release] - 基本機能実装

### ✅ 実装済み機能

1. **データ準備パイプライン**
   - 特徴量抽出（音声・映像・テキスト）
   - ラベル抽出（Premiere Pro XML）
   - シーケンス分割

2. **モデル実装**
   - Transformer Encoder
   - マルチモーダル統合
   - Focal Loss

3. **訓練・推論**
   - 学習スクリプト
   - 推論スクリプト
   - XML生成

4. **可視化**
   - 学習曲線
   - メトリクス表示
