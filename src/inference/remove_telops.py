"""
テロップトラックを完全に削除するスクリプト
"""
import re
import sys


def remove_telops(input_xml: str, output_xml: str):
    """
    XMLからテロップトラックを完全に削除
    
    Args:
        input_xml: 入力XMLファイルのパス
        output_xml: 出力XMLファイルのパス
    """
    with open(input_xml, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # テロップマーカーを削除
    content = re.sub(r'\[Telop\]\s*', '', content)
    content = re.sub(r'\[AI-Speech\]\s*', '', content)
    content = re.sub(r'\[AI-Emotion-\w+\]\s*', '', content)
    
    # file-2を含むtrackを完全に削除
    # パターン: <track>...</track>の中にfile-2がある場合
    content = re.sub(
        r'<track>(?:(?!<track>).)*?<file id="file-2"[^>]*/>(?:(?!</track>).)*?</track>\s*',
        '',
        content,
        flags=re.DOTALL
    )
    
    print(f"✅ Removed all telop tracks")
    
    # 保存
    with open(output_xml, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Output: {output_xml}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python remove_telops.py <input> <output>")
        sys.exit(1)
    
    remove_telops(sys.argv[1], sys.argv[2])
