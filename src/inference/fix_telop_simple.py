"""
テロップ修正スクリプト（OCR + AI字幕対応）
"""
import re
import sys

def fix_telops(input_xml, output_xml):
    with open(input_xml, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # すべてのテロップマーカーを削除
    content = re.sub(r'\[Telop\]\s*', '', content)
    content = re.sub(r'\[AI-Speech\]\s*', '', content)
    content = re.sub(r'\[AI-Emotion-\w+\]\s*', '', content)
    
    # file-2を参照しているclipitemを検出してカウント
    telop_matches = re.findall(r'<file id="file-2"', content)
    telop_counter = len(telop_matches)
    
    # file-2の最初の出現を完全なfile要素に置き換え
    first_file_pattern = r'<file id="file-2">.*?</file>'
    
    complete_file = '''<file id="file-2">
                                    <name>グラフィック</name>
                                    <mediaSource>GraphicAndType</mediaSource>
                                    <rate>
                                        <timebase>60</timebase>
                                        <ntsc>TRUE</ntsc>
                                    </rate>
                                    <timecode>
                                        <rate>
                                            <timebase>60</timebase>
                                            <ntsc>TRUE</ntsc>
                                        </rate>
                                        <string>00:00:00:00</string>
                                        <frame>0</frame>
                                        <displayformat>NDF</displayformat>
                                    </timecode>
                                    <media>
                                        <video>
                                            <samplecharacteristics>
                                                <rate>
                                                    <timebase>59</timebase>
                                                    <ntsc>TRUE</ntsc>
                                                </rate>
                                                <width>1080</width>
                                                <height>1920</height>
                                                <anamorphic>FALSE</anamorphic>
                                                <pixelaspectratio>square</pixelaspectratio>
                                                <fielddominance>none</fielddominance>
                                            </samplecharacteristics>
                                        </video>
                                    </media>
                                </file>'''
    
    # 最初のfile-2を完全な形式に置き換え
    content = re.sub(first_file_pattern, complete_file, content, count=1, flags=re.DOTALL)
    
    # 残りのfile-2を参照形式に置き換え
    content = re.sub(r'<file id="file-2">.*?</file>', '<file id="file-2"/>', content, flags=re.DOTALL)
    
    with open(output_xml, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Converted {telop_counter} telops to Premiere Pro graphics")
    print(f"Output: {output_xml}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fix_telop_simple.py <input> <output>")
        sys.exit(1)
    
    fix_telops(sys.argv[1], sys.argv[2])
