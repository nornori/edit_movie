"""
無効なテロップを削除
- 1文字だけのテロップ
- 記号だけのテロップ（!、?、...など）
"""
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

input_file = "outputs/test_ai_telop_final_fixed_optimized.xml"
output_file = "outputs/test_ai_telop_final_fixed_optimized_clean.xml"

logger.info(f"Reading {input_file}...")
with open(input_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 無効なテロップパターン
invalid_patterns = [
    r'!',
    r'\?',
    r'\.',
    r'…',
    r'、',
    r'。',
    r'！',
    r'？',
]

# グラフィックトラックを検出
telop_tracks = re.findall(r'<track[^>]*>.*?<mediaSource>GraphicAndType</mediaSource>.*?</track>', content, re.DOTALL)

logger.info(f"Found {len(telop_tracks)} telop tracks")

removed_count = 0
removed_clipitems = []

# 全てのclipitemをチェック
all_clipitems = re.findall(r'<clipitem[^>]*>.*?</clipitem>', content, re.DOTALL)

for clipitem in all_clipitems:
    # GraphicAndTypeを含むclipitemのみ処理
    if 'GraphicAndType' not in clipitem:
        continue
    
    # effectのnameを抽出（テロップテキスト）
    name_match = re.search(r'<effect>.*?<name>([^<]+)</name>', clipitem, re.DOTALL)
    
    if name_match:
        telop_text = name_match.group(1).strip()
        
        # 無効なテロップかチェック
        is_invalid = False
        
        # 1文字だけ
        if len(telop_text) == 1:
            is_invalid = True
        
        # 記号だけ
        if telop_text in ['!', '?', '.', '…', '、', '。', '！', '？', '・']:
            is_invalid = True
        
        # 空白だけ
        if not telop_text or telop_text.isspace():
            is_invalid = True
        
        if is_invalid:
            logger.info(f"  Removing invalid telop: '{telop_text}'")
            removed_clipitems.append(clipitem)
            removed_count += 1

# 無効なclipitemを削除
for clipitem in removed_clipitems:
    content = content.replace(clipitem, '', 1)

logger.info(f"✅ Removed {removed_count} invalid telop tracks")

# 保存
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(content)

logger.info(f"✅ Saved cleaned XML to {output_file}")

# トラック数を確認
import xml.etree.ElementTree as ET
tree = ET.parse(output_file)
root = tree.getroot()
video_tracks = root.findall('.//video/track')
print(f"\n📊 Result:")
print(f"   Removed: {removed_count} invalid telop tracks")
print(f"   Remaining tracks: {len(video_tracks)}")
