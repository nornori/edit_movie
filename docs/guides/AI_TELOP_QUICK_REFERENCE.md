# AI字幕生成 クイックリファレンス

## 🚀 基本コマンド

```bash
# 1. AI字幕生成を有効にして推論
python src/inference/inference_pipeline.py video.mp4 --output temp.xml

# 2. テロップをグラフィックに変換
python src/inference/fix_telop_simple.py temp.xml final.xml

# 3. Premiere Proで final.xml を開く
```

---

## ⚙️ 感度調整（クイックガイド）

### 全体的な感度

`configs/config_telop_generation.yaml`を編集：

```yaml
emotion:
  confidence_threshold: 0.5  # 低い = 敏感、高い = 厳格
```

| 値 | 効果 |
|---|---|
| 0.3 | 非常に敏感（誤検出多い） |
| 0.5 | バランス（推奨） |
| 0.7 | 厳格（見逃し多い） |

### 笑い検出

```yaml
laughter:
  pitch_std_threshold: 40.0  # 低い = 敏感
  energy_threshold: 0.25     # 低い = 敏感
```

### 驚き検出

```yaml
surprise:
  pitch_delta_threshold: 80.0  # 低い = 敏感
```

### 悲しみ検出

```yaml
sadness:
  pitch_mean_threshold: 160.0  # 高い = 敏感
  energy_threshold: 0.12       # 高い = 敏感
```

---

## 🎨 テキストパターン

### 笑い

```yaml
laughter:
  text_short: "w"        # < 1秒
  text_medium: "www"     # 1-2秒
  text_long: "wwwww"     # > 2秒
```

**カスタマイズ例**:
- 日本語: `"笑"`, `"笑笑"`, `"爆笑"`
- 英語: `"lol"`, `"lol lol"`, `"LMAO"`
- 絵文字: `"😄"`, `"😂"`, `"🤣"`

### 驚き

```yaml
surprise:
  text: "！"
```

**カスタマイズ例**:
- `"！！"`, `"えっ！？"`, `"Wow!"`, `"😲"`

### 悲しみ

```yaml
sadness:
  text: "..."
```

**カスタマイズ例**:
- `"しょんぼり"`, `"sad..."`, `"😢"`

---

## 📝 よくある設定

### 笑いを検出しやすくする

```yaml
emotion:
  confidence_threshold: 0.4

laughter:
  pitch_std_threshold: 30.0
  energy_threshold: 0.2
```

### 誤検出を減らす（厳格）

```yaml
emotion:
  confidence_threshold: 0.7

laughter:
  pitch_std_threshold: 60.0
  energy_threshold: 0.4
```

### 絵文字スタイル

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

### 特定の感情を無効化

```yaml
surprise:
  enabled: false

sadness:
  enabled: false
```

---

## 🔧 コマンドラインオプション

```bash
# カスタム設定ファイルを使用
python src/inference/inference_pipeline.py video.mp4 \
  --telop_config configs/my_custom_config.yaml

# 音声認識のみ（感情検出なし）
python src/inference/inference_pipeline.py video.mp4 --no-emotion

# 感情検出のみ（音声認識なし）
python src/inference/inference_pipeline.py video.mp4 --no-speech

# Whisperモデルサイズを変更（設定ファイルで）
speech:
  model_size: "tiny"    # 高速、低精度
  model_size: "small"   # バランス（推奨）
  model_size: "medium"  # 高精度、低速
```

---

## 📂 ファイル構成

```
configs/
  ├── config_telop_generation.yaml              # デフォルト設定
  └── config_telop_generation_custom_example.yaml  # カスタマイズ例

docs/guides/
  ├── AI_TELOP_CUSTOMIZATION_GUIDE.md  # 詳細ガイド
  └── AI_TELOP_QUICK_REFERENCE.md      # このファイル

src/inference/
  ├── inference_pipeline.py      # メイン推論スクリプト
  ├── otio_xml_generator.py      # XML生成
  └── fix_telop_simple.py        # テロップ後処理
```

---

## 🐛 トラブルシューティング

| 問題 | 解決策 |
|---|---|
| 感情が検出されない | `confidence_threshold`を下げる（0.4） |
| 誤検出が多い | `confidence_threshold`を上げる（0.7） |
| 音声認識が遅い | `model_size: "tiny"` に変更 |
| 音声認識の精度が低い | `model_size: "medium"` に変更 |
| 字幕が長すぎる | `max_segment_duration: 3.0` に変更 |

---

## 📚 詳細情報

詳しいカスタマイズ方法は `docs/guides/AI_TELOP_CUSTOMIZATION_GUIDE.md` を参照してください。

---

**Happy Editing! 🎬✨**
