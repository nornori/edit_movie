# プロジェクト整理サマリー

## 📅 整理日時

- **初回整理**: 2025-12-15
- **ルート整理**: 2025-12-26
- **cut_selection整理**: 2025-12-26（最新）

---

## 🎯 整理の目的

1. ルートディレクトリをクリーンに保つ
2. ファイルを適切なフォルダに配置
3. 古いバージョンのファイルをアーカイブ
4. プロジェクト構造を明確にする
5. **モジュール内部を機能別に整理**（NEW）

---

## 📁 整理後のフォルダ構造

```
xmlai/
├── .git/                                      # Git管理
├── .kiro/                                     # Kiro設定
├── .venv/                                     # Python仮想環境
├── archive/                                   # アーカイブ（古いファイル）
│   ├── checkpoints_cut_selection_kfold/       # 旧K-Foldモデル
│   ├── checkpoints_cut_selection_kfold_enhanced_reset/
│   ├── checkpoints_cut_selection_kfold_enhanced_retrain/
│   ├── checkpoints_cut_selection_kfold_enhanced_v2/
│   ├── experiment_log_advanced.csv            # 実験ログ
│   └── (その他の古いファイル)
├── backups/                                   # バックアップ
├── batch/                                     # バッチファイル
│   ├── retrain_model.bat
│   ├── train_duration_constraint.bat
│   ├── train_fullvideo.bat
│   ├── train_reset.bat
│   └── (その他のバッチファイル)
├── checkpoints_cut_selection_fullvideo/       # Full Videoモデル（最新）
├── checkpoints_cut_selection_kfold_enhanced/  # K-Fold拡張モデル（最新）
├── configs/                                   # 設定ファイル
├── data/                                      # データ
├── docs/                                      # ドキュメント
├── models/                                    # モデル定義
├── outputs/                                   # 出力ファイル
├── preprocessed_data/                         # 前処理済みデータ
├── scripts/                                   # スクリプト
│   ├── generate_xml_from_inference.py         # XML生成スクリプト
│   └── (その他のスクリプト)
├── src/                                       # ソースコード
├── temp_features/                             # 一時特徴量
├── tests/                                     # テストコード
│   ├── test_inference_fullvideo.py            # Full Video推論テスト
│   ├── test_inference_simple.py               # シンプル推論テスト
│   ├── check_model.py                         # モデルチェック
│   └── (その他のテスト)
├── .gitignore                                 # Git除外設定
├── CHANGELOG.md                               # 変更履歴
├── CLEANUP_SUMMARY.md                         # このファイル
├── FEATURE_ENHANCEMENT_README.md              # 機能拡張README
├── LICENSE                                    # ライセンス
├── PROJECT_STRUCTURE.md                       # プロジェクト構造
├── README.md                                  # メインREADME
└── requirements.txt                           # 依存パッケージ
```

---

## 🔄 実行した整理作業（2025-12-26）

### 1. ファイル移動

#### テストスクリプト → `tests/`
- ✅ `test_inference_fullvideo.py` → `tests/`
- ✅ `test_inference_simple.py` → `tests/`
- ✅ `check_model.py` → `tests/`

#### 推論スクリプト → `scripts/`
- ✅ `generate_xml_from_inference.py` → `scripts/`

#### バッチファイル → `batch/`
- ✅ `retrain_model.bat` → `batch/`
- ✅ `train_duration_constraint.bat` → `batch/`
- ✅ `train_fullvideo.bat` → `batch/`
- ✅ `train_reset.bat` → `batch/`

#### ログファイル → `archive/`
- ✅ `experiment_log_advanced.csv` → `archive/`

### 2. 古いフォルダのアーカイブ

#### Checkpointsフォルダ → `archive/`
- ✅ `checkpoints_cut_selection_kfold/` → `archive/`
- ✅ `checkpoints_cut_selection_kfold_enhanced_reset/` → `archive/`
- ✅ `checkpoints_cut_selection_kfold_enhanced_retrain/` → `archive/`
- ✅ `checkpoints_cut_selection_kfold_enhanced_v2/` → `archive/`

### 3. 不要なフォルダの削除

- ✅ `None/` フォルダを削除

---

## 📊 整理前後の比較

### ルートディレクトリのファイル数

| 項目 | 整理前 | 整理後 | 削減数 |
|------|--------|--------|--------|
| ファイル数 | 18個 | 8個 | -10個 |
| フォルダ数 | 25個 | 20個 | -5個 |

### 整理後のルートディレクトリ（ファイルのみ）

1. `.gitignore` - Git除外設定
2. `CHANGELOG.md` - 変更履歴
3. `CLEANUP_SUMMARY.md` - 整理サマリー
4. `FEATURE_ENHANCEMENT_README.md` - 機能拡張README
5. `LICENSE` - ライセンス
6. `PROJECT_STRUCTURE.md` - プロジェクト構造
7. `README.md` - メインREADME
8. `requirements.txt` - 依存パッケージ

**結果**: ルートディレクトリがスッキリしました！✨

---

## 🎯 現在のアクティブなモデル

### 学習済みモデル（使用中）

1. **Full Video Model**
   - パス: `checkpoints_cut_selection_fullvideo/best_model.pth`
   - Epoch: 9
   - F1: 52.90%
   - 用途: per-video制約（90-200秒）推論

2. **K-Fold Enhanced Model**
   - パス: `checkpoints_cut_selection_kfold_enhanced/fold_1_best_model.pth`
   - Epoch: 4
   - F1: 49.42%
   - 用途: K-Fold CV評価

### アーカイブされたモデル（参考用）

- `archive/checkpoints_cut_selection_kfold/` - 初期K-Foldモデル
- `archive/checkpoints_cut_selection_kfold_enhanced_reset/` - リセット版
- `archive/checkpoints_cut_selection_kfold_enhanced_retrain/` - 再訓練版
- `archive/checkpoints_cut_selection_kfold_enhanced_v2/` - V2版

---

## 📝 各フォルダの役割

### コアフォルダ

| フォルダ | 役割 | 重要度 |
|---------|------|--------|
| `src/` | ソースコード | ⭐⭐⭐ |
| `configs/` | 設定ファイル | ⭐⭐⭐ |
| `docs/` | ドキュメント | ⭐⭐⭐ |
| `scripts/` | 実行スクリプト | ⭐⭐⭐ |
| `tests/` | テストコード | ⭐⭐ |

### データフォルダ

| フォルダ | 役割 | 重要度 |
|---------|------|--------|
| `data/` | 生データ | ⭐⭐⭐ |
| `preprocessed_data/` | 前処理済みデータ | ⭐⭐⭐ |
| `temp_features/` | 一時特徴量 | ⭐⭐ |

### モデルフォルダ

| フォルダ | 役割 | 重要度 |
|---------|------|--------|
| `checkpoints_cut_selection_fullvideo/` | Full Videoモデル | ⭐⭐⭐ |
| `checkpoints_cut_selection_kfold_enhanced/` | K-Fold拡張モデル | ⭐⭐⭐ |

### 補助フォルダ

| フォルダ | 役割 | 重要度 |
|---------|------|--------|
| `batch/` | バッチファイル | ⭐⭐ |
| `outputs/` | 出力ファイル | ⭐⭐ |
| `backups/` | バックアップ | ⭐ |
| `archive/` | アーカイブ | ⭐ |

---

## 🚀 よく使うファイルへのクイックアクセス

### ドキュメント

- メインREADME: `README.md`
- クイックスタート: `docs/QUICK_START.md`
- 最終結果: `docs/FINAL_RESULTS.md`
- 推論テスト結果: `docs/INFERENCE_TEST_RESULTS.md`
- 完全メトリクス: `docs/COMPLETE_METRICS_SUMMARY.md`

### スクリプト

- XML生成: `scripts/generate_xml_from_inference.py`
- 特徴量抽出: `scripts/extract_video_features_parallel.py`
- データ作成: `scripts/combine_sequences_enhanced.py`

### テスト

- Full Video推論: `tests/test_inference_fullvideo.py`
- シンプル推論: `tests/test_inference_simple.py`

### バッチファイル

- Full Video学習: `batch/train_fullvideo.bat`
- K-Fold学習: `batch/train_cut_selection_kfold_enhanced.bat`

### モデル

- Full Video: `checkpoints_cut_selection_fullvideo/best_model.pth`
- K-Fold Fold 1: `checkpoints_cut_selection_kfold_enhanced/fold_1_best_model.pth`

---

## 🔍 .gitignoreの設定

以下のフォルダ/ファイルはGit管理から除外されています：

```gitignore
# データ
data/
preprocessed_data/
temp_features/

# モデル
checkpoints*/
*.pth
*.pkl

# 出力
outputs/
archive/
backups/

# Python
.venv/
__pycache__/
*.pyc

# その他
.pytest_cache/
.vscode/
```

---

## 📋 今後のメンテナンス

### 定期的に実行すべきこと

1. **古いcheckpointsのアーカイブ**（月1回）
   ```bash
   # 使わなくなったモデルをarchiveに移動
   Move-Item checkpoints_old archive/
   ```

2. **temp_featuresのクリーンアップ**（週1回）
   ```bash
   # 不要な一時ファイルを削除
   Remove-Item temp_features/*.csv -Force
   ```

3. **outputsのバックアップ**（必要に応じて）
   ```bash
   # 重要な出力をbackupsに保存
   Copy-Item outputs/important.xml backups/
   ```

### 新しいモデルを追加する場合

1. 新しいcheckpointsフォルダを作成
2. 古いモデルをarchiveに移動
3. このドキュメントを更新

---

## ✅ 整理完了チェックリスト

### ルートディレクトリ整理（2025-12-26）

- [x] ルートディレクトリのファイルを整理
- [x] テストスクリプトをtests/に移動
- [x] バッチファイルをbatch/に移動
- [x] 古いcheckpointsをarchiveに移動
- [x] 不要なフォルダ（None）を削除
- [x] CLEANUP_SUMMARY.mdを更新
- [x] PROJECT_STRUCTURE.mdを更新

### src/cut_selection整理（2025-12-26）

- [x] 機能別にサブディレクトリを作成
  - [x] models/ - モデル定義
  - [x] datasets/ - データセットクラス
  - [x] training/ - 訓練スクリプト
  - [x] inference/ - 推論モジュール
  - [x] evaluation/ - 評価スクリプト
  - [x] utils/ - ユーティリティ
  - [x] archive/ - 旧バージョン
- [x] ファイルを適切なサブディレクトリに移動（14ファイル）
- [x] 旧バージョンをarchiveに移動（11ファイル）
- [x] 各サブディレクトリに__init__.pyを作成
- [x] メインモジュールの__init__.pyを更新
- [x] すべてのimport文を更新（25+ファイル）
  - [x] 訓練スクリプト（3ファイル）
  - [x] 評価スクリプト（3ファイル）
  - [x] 推論スクリプト（2ファイル）
  - [x] テストファイル（2ファイル）
  - [x] 外部スクリプト（3ファイル）
  - [x] アーカイブファイル（11ファイル）
  - [x] モデルファイル（1ファイル）
- [x] __pycache__をクリーンアップ
- [x] docs/CUT_SELECTION_REORGANIZATION.mdを作成
- [x] PROJECT_STRUCTURE.mdを更新

---

## 📊 整理の成果

### ルートディレクトリ

- **整理前**: 18ファイル、25フォルダ
- **整理後**: 8ファイル、20フォルダ
- **削減**: -10ファイル、-5フォルダ

### src/cut_selection/

- **整理前**: 27ファイル（フラット構造）
- **整理後**: 27ファイル（7サブディレクトリに整理）
- **アクティブファイル**: 14ファイル
- **アーカイブファイル**: 11ファイル
- **ユーティリティ**: 5ファイル

---

## 📞 問題が発生した場合

整理によって何か問題が発生した場合：

1. **ファイルが見つからない**
   - `archive/` フォルダを確認
   - `backups/` フォルダを確認

2. **スクリプトが動かない**
   - パスを確認（相対パスが変わった可能性）
   - `tests/` または `scripts/` フォルダを確認

3. **モデルが見つからない**
   - `checkpoints_cut_selection_fullvideo/` を確認
   - `checkpoints_cut_selection_kfold_enhanced/` を確認
   - `archive/` フォルダを確認

---

**最終更新**: 2025-12-26  
**整理担当**: AI開発チーム  
**ステータス**: ✅ 整理完了

