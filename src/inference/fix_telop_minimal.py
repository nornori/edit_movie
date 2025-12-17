"""
テロップ修正スクリプト（最小限の変換のみ）
OTIOの出力には忠実に、テロップのfile-2だけを変換
"""
import re
import sys
import logging

logger = logging.getLogger(__name__)


def fix_telops_minimal(input_xml: str, output_xml: str):
    """
    XMLのテロップ部分だけを最小限修正
    OTIOの出力には忠実に、テロップのfile-2だけを変換
    
    Args:
        input_xml: 入力XMLファイルのパス
        output_xml: 出力XMLファイルのパス
    """
    with open(input_xml, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. テロップマーカーを削除
    content = re.sub(r'\[Telop\]\s*', '', content)
    content = re.sub(r'\[AI-Speech\]\s*', '', content)
    content = re.sub(r'\[AI-Emotion-\w+\]\s*', '', content)
    
    # 2. file-2を参照しているclipitemを検出してカウント
    telop_matches = re.findall(r'<file id="file-2"', content)
    telop_counter = len(telop_matches)
    
    if telop_counter > 0:
        # 3. 最初のfile-2を完全な形式に置き換え
        first_file_pattern = r'(<file id="file-2">)(.*?)(</file>)'
        
        def replace_first_file(match):
            # 完全なfile要素を生成
            return '<file id="file-2"><name>グラフィック</name><mediaSource>GraphicAndType</mediaSource><rate><timebase>60</timebase><ntsc>TRUE</ntsc></rate><timecode><rate><timebase>60</timebase><ntsc>TRUE</ntsc></rate><string>00:00:00:00</string><frame>0</frame><displayformat>NDF</displayformat></timecode><media><video><samplecharacteristics><rate><timebase>59</timebase><ntsc>TRUE</ntsc></rate><width>1080</width><height>1920</height><anamorphic>FALSE</anamorphic><pixelaspectratio>square</pixelaspectratio><fielddominance>none</fielddominance></samplecharacteristics></video></media></file>'
        
        # 最初のfile-2を完全な形式に置き換え
        content = re.sub(first_file_pattern, replace_first_file, content, count=1, flags=re.DOTALL)
        
        # 4. 残りのfile-2を参照形式に置き換え
        remaining_file_pattern = r'<file id="file-2">.*?</file>'
        content = re.sub(remaining_file_pattern, '<file id="file-2"/>', content, flags=re.DOTALL)
        
        # 5. file-2を持つclipitemをグラフィックに変換
        def convert_telop_clipitem(match):
            clipitem_content = match.group(0)
            
            # file-2を含むかチェック
            if 'file id="file-2"' not in clipitem_content:
                return clipitem_content
            
            # clipitem内の各要素を抽出
            clipitem_attrs = re.search(r'<clipitem([^>]*)>', clipitem_content)
            name_match = re.search(r'<name>([^<]+)</name>', clipitem_content)
            rate_match = re.search(r'<rate>.*?</rate>', clipitem_content, re.DOTALL)
            duration_match = re.search(r'<duration>(\d+)</duration>', clipitem_content)
            start_match = re.search(r'<start>(\d+)</start>', clipitem_content)
            end_match = re.search(r'<end>(\d+)</end>', clipitem_content)
            in_match = re.search(r'<in>(\d+)</in>', clipitem_content)
            out_match = re.search(r'<out>(\d+)</out>', clipitem_content)
            
            if not all([name_match, rate_match, duration_match, start_match, end_match, in_match, out_match]):
                return clipitem_content
            
            telop_text = name_match.group(1)
            attrs = clipitem_attrs.group(1) if clipitem_attrs else ''
            rate_block = rate_match.group(0)
            duration = duration_match.group(1)
            start = start_match.group(1)
            end = end_match.group(1)
            in_val = in_match.group(1)
            out_val = out_match.group(1)
            
            # 正しい順序でclipitemを再構築
            new_clipitem = f'''<clipitem{attrs}>
                                <name>グラフィック</name>
                                <enabled>TRUE</enabled>
                                <duration>2582774</duration>
                                {rate_block}
                                <start>{start}</start>
                                <end>{end}</end>
                                <in>{in_val}</in>
                                <out>{out_val}</out>
                                <alphatype>straight</alphatype>
                                <pixelaspectratio>square</pixelaspectratio>
                                <anamorphic>FALSE</anamorphic>
                                <file id="file-2"/>
                                <filter>
                                    <effect>
                                        <name>{telop_text}</name>
                                        <effectid>GraphicAndType</effectid>
                                        <effectcategory>graphic</effectcategory>
                                        <effecttype>filter</effecttype>
                                        <mediatype>video</mediatype>
                                    </effect>
                                </filter>
                            </clipitem>'''
            
            return new_clipitem
        
        # file-2を含むclipitemを変換
        content = re.sub(
            r'<clipitem[^>]*>(?:(?!</clipitem>).)*?<file id="file-2"[^>]*/>(?:(?!</clipitem>).)*?</clipitem>',
            convert_telop_clipitem,
            content,
            flags=re.DOTALL
        )
        
        logger.info(f"  ✅ Converted {telop_counter} telops to Premiere Pro graphics")
    
    # 保存
    with open(output_xml, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"  Output: {output_xml}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fix_telop_minimal.py <input> <output>")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO)
    fix_telops_minimal(sys.argv[1], sys.argv[2])
