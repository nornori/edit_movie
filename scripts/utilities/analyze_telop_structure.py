"""
テロップのバイナリ構造を分析
"""
import re
import base64
import struct

# XMLファイルを読み込んで、テロップのvalueを抽出
with open('data/raw/editxml/bandicam 2025-03-03 22-34-57-492.xml', 'r', encoding='utf-8') as f:
    content = f.read()

# ソーステキストのパラメータを抽出
pattern = r'<name>ソーステキスト</name>.*?<value>(.*?)</value>'
matches = re.findall(pattern, content, re.DOTALL)

print(f'Found {len(matches)} telop values')
print()

for i, match in enumerate(matches[:10]):  # 最初の10個を分析
    # Base64デコード
    try:
        decoded = base64.b64decode(match)
        
        # フォント名を探す（HGPSoeiKakupoptai または SourceHanSansJP）
        font_patterns = [b'HGPSoeiKakupoptai', b'SourceHanSansJP']
        font_pos = -1
        font_name = ""
        
        for pattern_bytes in font_patterns:
            pos = decoded.find(pattern_bytes)
            if pos >= 0:
                font_pos = pos
                font_name = pattern_bytes.decode('ascii')
                break
        
        if font_pos < 0:
            continue
        
        # フォント名の後のテキストを探す
        search_start = font_pos + len(font_name.encode('ascii')) + 1
        
        # テキストセクションを探す
        text_found = None
        text_offset = -1
        
        for j in range(search_start, min(search_start + 100, len(decoded))):
            # 日本語のUTF-8バイト列を探す
            if decoded[j] >= 0xe0 and decoded[j] <= 0xef:
                # UTF-8デコードを試行
                for length in range(3, min(50, len(decoded) - j)):
                    try:
                        text = decoded[j:j+length].decode('utf-8')
                        # 日本語文字を含むか確認
                        if any(ord(c) > 127 for c in text):
                            # null文字または制御文字で終わる
                            for k, c in enumerate(text):
                                if c == '\x00' or ord(c) < 32:
                                    text = text[:k]
                                    break
                            if len(text) > 0:
                                text_found = text
                                text_offset = j
                                break
                    except:
                        pass
                if text_found:
                    break
        
        if text_found:
            text_bytes = text_found.encode('utf-8')
            text_len = len(text_bytes)
            
            print(f'Telop {i+1}:')
            print(f'  Font: "{font_name}"')
            print(f'  Text: "{text_found}"')
            print(f'  Text length: {text_len} bytes ({len(text_found)} chars)')
            print(f'  Font offset: {font_pos}')
            print(f'  Text offset: {text_offset}')
            print(f'  Distance: {text_offset - font_pos} bytes')
            print(f'  Total binary length: {len(decoded)} bytes')
            
            # テキスト位置の前のバイトを確認（長さ情報を探す）
            if text_offset >= 20:
                # テキスト長を探す
                found_lengths = []
                for offset_back in range(4, 32, 4):
                    if text_offset >= offset_back:
                        pos = text_offset - offset_back
                        val = struct.unpack('<I', decoded[pos:pos+4])[0]
                        if val == text_len or val == text_len + 4:
                            found_lengths.append((offset_back, pos, val))
                
                if found_lengths:
                    for offset_back, pos, val in found_lengths:
                        print(f'  Found length {val} at offset -{offset_back} (position {pos})')
                
                # 直前の20バイトを表示
                before = decoded[text_offset-20:text_offset]
                print(f'  20 bytes before text: {before.hex()}')
            
            print()
    except Exception as e:
        print(f'Telop {i+1}: Error - {e}')
        import traceback
        traceback.print_exc()
        print()


# 別のXMLファイルも分析（もっと長いテキストを探す）
print("\n" + "="*70)
print("Analyzing another XML file for longer texts...")
print("="*70 + "\n")

with open('data/raw/editxml/1.xml', 'r', encoding='utf-8') as f:
    content2 = f.read()

matches2 = re.findall(pattern, content2, re.DOTALL)
print(f'Found {len(matches2)} telop values in second file')
print()

for i, match in enumerate(matches2[:5]):
    try:
        decoded = base64.b64decode(match)
        
        # フォント名を探す
        font_patterns = [b'HGPSoeiKakupoptai', b'SourceHanSansJP']
        font_pos = -1
        font_name = ""
        
        for pattern_bytes in font_patterns:
            pos = decoded.find(pattern_bytes)
            if pos >= 0:
                font_pos = pos
                font_name = pattern_bytes.decode('ascii')
                break
        
        if font_pos < 0:
            continue
        
        search_start = font_pos + len(font_name.encode('ascii')) + 1
        
        text_found = None
        text_offset = -1
        
        for j in range(search_start, min(search_start + 150, len(decoded))):
            if decoded[j] >= 0xe0 and decoded[j] <= 0xef:
                for length in range(3, min(100, len(decoded) - j)):
                    try:
                        text = decoded[j:j+length].decode('utf-8')
                        if any(ord(c) > 127 for c in text):
                            for k, c in enumerate(text):
                                if c == '\x00' or ord(c) < 32:
                                    text = text[:k]
                                    break
                            if len(text) > 0:
                                text_found = text
                                text_offset = j
                                break
                    except:
                        pass
                if text_found:
                    break
        
        if text_found:
            text_bytes = text_found.encode('utf-8')
            text_len = len(text_bytes)
            
            print(f'Telop {i+1}:')
            print(f'  Font: "{font_name}"')
            print(f'  Text: "{text_found}"')
            print(f'  Text length: {text_len} bytes ({len(text_found)} chars)')
            print(f'  Font offset: {font_pos}')
            print(f'  Text offset: {text_offset}')
            print(f'  Distance: {text_offset - font_pos} bytes')
            print(f'  Total binary length: {len(decoded)} bytes')
            
            if text_offset >= 20:
                found_lengths = []
                for offset_back in range(4, 32, 4):
                    if text_offset >= offset_back:
                        pos = text_offset - offset_back
                        val = struct.unpack('<I', decoded[pos:pos+4])[0]
                        if val == text_len or val == text_len + 4:
                            found_lengths.append((offset_back, pos, val))
                
                if found_lengths:
                    for offset_back, pos, val in found_lengths:
                        print(f'  Found length {val} at offset -{offset_back} (position {pos})')
                
                before = decoded[text_offset-20:text_offset]
                print(f'  20 bytes before text: {before.hex()}')
            
            print()
    except Exception as e:
        print(f'Telop {i+1}: Error - {e}')
        print()
