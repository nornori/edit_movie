# AI字幕生成のカスタマイズガイド

このガイドでは、AI字幕生成機能の感情検出感度とテキストパターンをカスタマイズする方法を説明します。

## 📋 目次

1. [基本的な使い方](#基本的な使い方)
2. [感情検出の感度調整](#感情検出の感度調整)
3. [カスタムテキストパターン](#カスタムテキストパターン)
4. [よくあるカスタマイズ例](#よくあるカスタマイズ例)
5. [トラブルシューティング](#トラブルシューティング)

---

## 基本的な使い方

### 1. デフォルト設定で実行

```bash
# AI字幕生成を有効にして推論実行
python src/inference/inference_pipeline.py video.mp4 --model models/editor_ai_model.pth --output temp.xml

# テロップをグラフィックに変換
python src/inference/fix_telop_simple.py temp.xml final.xml

# Premiere Proで final.xml を開く
```

### 2. カスタム設定ファイルを使用

```bash
# カスタム設定ファイルを指定
python src/inference/inference_pipeline.py video.mp4 \
  --model models/editor_ai_model.pth \
  --telop_config configs/config_telop_generation_custom_example.yaml \
  --output temp.xml
```

### 3. コマンドラインで機能を無効化

```bash
# 音声認識のみ（感情検出なし）
python src/inference/inference_pipeline.py video.mp4 --no-emotion --output temp.xml

# 感情検出のみ（音声認識なし）
python src/inference/inference_pipeline.py video.mp4 --no-speech --output temp.xml

# AI字幕生成を完全に無効化
# → configs/config_telop_generation.yaml で enabled: false に設定
```

---

## 感情検出の感度調整

### 全体的な感度調整

`configs/config_telop_generation.yaml`の`confidence_threshold`を変更：

```yaml
emotion:
  confidence_threshold: 0.5  # 0.0〜1.0
```

| 値 | 説明 | 効果 |
|---|---|---|
| **0.3** | 非常に敏感 | 多くの感情を検出（誤検出が増える） |
| **0.5** | バランス | 推奨設定 |
| **0.7** | 厳格 | 確実な感情のみ検出（見逃しが増える） |

### 笑い検出の感度調整

```yaml
laughter:
  enabled: true
  pitch_std_threshold: 40.0  # ピッチ変動の閾値 (Hz)
  energy_threshold: 0.25     # エネルギーの閾値 (0.0〜1.0)
```

#### `pitch_std_threshold`（ピッチ変動）

- **低い値（30.0〜40.0）**: 小さな笑いも検出（敏感）
- **デフォルト（50.0）**: バランス
- **高い値（60.0〜80.0）**: 大きな笑いのみ検出（厳格）

#### `energy_threshold`（エネルギー）

- **低い値（0.2〜0.25）**: 静かな笑いも検出（敏感）
- **デフォルト（0.3）**: バランス
- **高い値（0.4〜0.5）**: 大きな笑いのみ検出（厳格）

### 驚き検出の感度調整

```yaml
surprise:
  enabled: true
  pitch_delta_threshold: 80.0  # 急激なピッチ上昇の閾値 (Hz)
  max_duration: 1.0            # 最大持続時間（秒）
```

#### `pitch_delta_threshold`（ピッチ上昇）

- **低い値（60.0〜80.0）**: 小さな驚きも検出（敏感）
- **デフォルト（100.0）**: バランス
- **高い値（120.0〜150.0）**: 大きな驚きのみ検出（厳格）

### 悲しみ検出の感度調整

```yaml
sadness:
  enabled: true
  pitch_mean_threshold: 160.0  # 低ピッチの閾値 (Hz)
  energy_threshold: 0.12       # 低エネルギーの閾値 (0.0〜1.0)
```

#### `pitch_mean_threshold`（低ピッチ）

- **高い値（160.0〜180.0）**: 検出しやすい（敏感）
- **デフォルト（150.0）**: バランス
- **低い値（120.0〜140.0）**: 検出しにくい（厳格）

#### `energy_threshold`（低エネルギー）

- **高い値（0.12〜0.15）**: 検出しやすい（敏感）
- **デフォルト（0.1）**: バランス
- **低い値（0.05〜0.08）**: 検出しにくい（厳格）

---

## カスタムテキストパターン

### 笑いのテキストをカスタマイズ

```yaml
laughter:
  text_short: "w"        # 短い笑い (< 1秒)
  text_medium: "www"     # 中程度の笑い (1-2秒)
  text_long: "wwwww"     # 長い笑い (> 2秒)
```

#### カスタマイズ例

```yaml
# 日本語スタイル
laughter:
  text_short: "笑"
  text_medium: "笑笑"
  text_long: "爆笑"

# 英語スタイル
laughter:
  text_short: "lol"
  text_medium: "lol lol"
  text_long: "LMAO"

# 絵文字スタイル
laughter:
  text_short: "😄"
  text_medium: "😂"
  text_long: "🤣🤣"
```

### 驚きのテキストをカスタマイズ

```yaml
surprise:
  text: "！"           # メインテキスト
  text_alt: "えっ"     # 代替テキスト（将来の拡張用）
```

#### カスタマイズ例

```yaml
# 強調スタイル
surprise:
  text: "！！"

# 日本語スタイル
surprise:
  text: "えっ！？"

# 英語スタイル
surprise:
  text: "Wow!"

# 絵文字スタイル
surprise:
  text: "😲"
```

### 悲しみのテキストをカスタマイズ

```yaml
sadness:
  text: "..."          # メインテキスト
  text_alt: "悲"       # 代替テキスト（将来の拡張用）
```

#### カスタマイズ例

```yaml
# 日本語スタイル
sadness:
  text: "しょんぼり"

# 英語スタイル
sadness:
  text: "sad..."

# 絵文字スタイル
sadness:
  text: "😢"
```

---

## よくあるカスタマイズ例

### 例1: 笑いを検出しやすくする

```yaml
emotion:
  confidence_threshold: 0.4  # 全体的に敏感に

laughter:
  enabled: true
  pitch_std_threshold: 30.0  # 50.0 → 30.0（敏感）
  energy_threshold: 0.2      # 0.3 → 0.2（敏感）
  text_short: "w"
  text_medium: "www"
  text_long: "wwwww"
```

### 例2: 笑いのみ検出（他の感情は無効）

```yaml
emotion:
  enabled: true
  confidence_threshold: 0.5

laughter:
  enabled: true
  # ... 設定 ...

surprise:
  enabled: false  # 無効化

sadness:
  enabled: false  # 無効化
```

### 例3: 絵文字スタイルの字幕

```yaml
laughter:
  text_short: "😄"
  text_medium: "😂"
  text_long: "🤣"

surprise:
  text: "😲"

sadness:
  text: "😢"
```

### 例4: 厳格な検出（誤検出を減らす）

```yaml
emotion:
  confidence_threshold: 0.7  # 厳格

laughter:
  pitch_std_threshold: 60.0  # 50.0 → 60.0（厳格）
  energy_threshold: 0.4      # 0.3 → 0.4（厳格）

surprise:
  pitch_delta_threshold: 120.0  # 100.0 → 120.0（厳格）

sadness:
  pitch_mean_threshold: 140.0  # 150.0 → 140.0（厳格）
  energy_threshold: 0.08       # 0.1 → 0.08（厳格）
```

---

## トラブルシューティング

### 問題1: 感情が検出されない

**原因**: 感度が低すぎる

**解決策**:
```yaml
emotion:
  confidence_threshold: 0.4  # 0.6 → 0.4

laughter:
  pitch_std_threshold: 30.0  # 50.0 → 30.0
  energy_threshold: 0.2      # 0.3 → 0.2
```

### 問題2: 誤検出が多すぎる

**原因**: 感度が高すぎる

**解決策**:
```yaml
emotion:
  confidence_threshold: 0.7  # 0.6 → 0.7

laughter:
  pitch_std_threshold: 60.0  # 50.0 → 60.0
  energy_threshold: 0.4      # 0.3 → 0.4
```

### 問題3: 音声認識が遅い

**原因**: Whisperモデルが大きすぎる

**解決策**:
```yaml
speech:
  model_size: "tiny"  # "small" → "tiny"（高速だが精度低下）
```

### 問題4: 音声認識の精度が低い

**原因**: Whisperモデルが小さすぎる

**解決策**:
```yaml
speech:
  model_size: "medium"  # "small" → "medium"（高精度だが遅い）
```

### 問題5: 字幕が長すぎる/短すぎる

**原因**: セグメント処理の閾値が適切でない

**解決策**:
```yaml
speech:
  min_segment_duration: 0.3  # 0.5 → 0.3（短いセグメントを許可）
  max_segment_duration: 3.0  # 5.0 → 3.0（早めに分割）
```

---

## 設定ファイルの場所

- **デフォルト設定**: `configs/config_telop_generation.yaml`
- **カスタマイズ例**: `configs/config_telop_generation_custom_example.yaml`

---

## 参考情報

### 感情検出のアルゴリズム

- **笑い**: ピッチ変動が大きく、エネルギーが高い
- **驚き**: ピッチが急激に上昇する
- **悲しみ**: ピッチが低く、エネルギーが低い

### 推奨設定

| 用途 | confidence_threshold | 説明 |
|---|---|---|
| **エンタメ動画** | 0.4〜0.5 | 多めに検出 |
| **ビジネス動画** | 0.6〜0.7 | 控えめに検出 |
| **テスト/デバッグ** | 0.3 | 最大限検出 |

---

## サポート

問題が解決しない場合は、以下を確認してください：

1. 設定ファイルのYAML構文が正しいか
2. 音声が明瞭に録音されているか
3. ログファイルにエラーメッセージがないか

---

**Happy Editing! 🎬✨**
