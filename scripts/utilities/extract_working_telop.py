"""
動作する元のテロップデータを抽出
"""
import re
import base64

# 元のXMLから最初のテロップを抽出
with open('data/raw/editxml/bandicam 2025-03-03 22-34-57-492.xml', 'r', encoding='utf-8') as f:
    content = f.read()

pattern = r'<name>ソーステキスト</name>.*?<value>(.*?)</value>'
matches = re.findall(pattern, content, re.DOTALL)

if matches:
    # 最初のテロップ（"あ"）を取得
    first_telop = matches[0]
    decoded = base64.b64decode(first_telop)
    
    print(f"Original telop data length: {len(decoded)} bytes")
    print(f"Base64 length: {len(first_telop)} chars")
    print()
    
    # テキスト位置を探す
    text_pos = decoded.find('あ'.encode('utf-8'))
    if text_pos >= 0:
        print(f"Text 'あ' found at position: {text_pos}")
        print(f"Text bytes: {decoded[text_pos:text_pos+3].hex()}")
        print()
        
        # 前後のバイトを表示
        print("20 bytes before text:")
        print(decoded[text_pos-20:text_pos].hex())
        print()
        
        print("10 bytes after text:")
        print(decoded[text_pos+3:text_pos+13].hex())
        print()
        
        # 完全なバイナリをPythonコードとして出力
        print("# Template binary data (for 'あ'):")
        print("TEMPLATE_BINARY = bytes([")
        for i in range(0, len(decoded), 16):
            chunk = decoded[i:i+16]
            hex_str = ', '.join(f'0x{b:02x}' for b in chunk)
            print(f"    {hex_str},")
        print("])")
        print()
        print(f"# Text position: {text_pos}")
        print(f"# Text length position: {text_pos - 4}")
