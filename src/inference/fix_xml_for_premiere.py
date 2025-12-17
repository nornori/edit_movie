"""
Premiere Pro互換のXMLに修正するスクリプト
OTIOが生成したXMLをPremiere Proが読み込める形式に変換
"""
import re
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def fix_xml_for_premiere(input_xml: str, output_xml: str, verbose: bool = True):
    """
    XMLをPremiere Pro互換の形式に修正
    
    Args:
        input_xml: 入力XMLファイルのパス
        output_xml: 出力XMLファイルのパス
        verbose: 詳細なログを出力するかどうか
    """
    if verbose:
        logger.info(f"  Reading XML from: {input_xml}")
    
    with open(input_xml, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. DOCTYPE宣言を追加（なければ）
    if '<!DOCTYPE xmeml>' not in content:
        content = content.replace(
            '<?xml version="1.0" ?>',
            '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>'
        )
        if verbose:
            logger.info("    ✓ Added DOCTYPE declaration")
    
    # 2. <project>タグを削除（Premiere Proは直接<sequence>を期待）
    if '<project>' in content:
        # <project>と</project>を削除
        content = re.sub(r'<project>\s*<name>[^<]*</name>\s*<children>', '', content)
        content = re.sub(r'</children>\s*</project>', '', content)
        if verbose:
            logger.info("    ✓ Removed <project> wrapper")
    
    # 3. 空の<format/>タグを詳細な形式に置き換え
    format_replacement = '''<format>
					<samplecharacteristics>
						<rate>
							<timebase>59</timebase>
							<ntsc>FALSE</ntsc>
						</rate>
						<codec>
							<name>Apple ProRes 422</name>
							<appspecificdata>
								<appname>Final Cut Pro</appname>
								<appmanufacturer>Apple Inc.</appmanufacturer>
								<appversion>7.0</appversion>
								<data>
									<qtcodec>
										<codecname>Apple ProRes 422</codecname>
										<codectypename>Apple ProRes 422</codectypename>
										<codectypecode>apcn</codectypecode>
										<codecvendorcode>appl</codecvendorcode>
										<spatialquality>1024</spatialquality>
										<temporalquality>0</temporalquality>
										<keyframerate>0</keyframerate>
										<datarate>0</datarate>
									</qtcodec>
								</data>
							</appspecificdata>
						</codec>
						<width>1080</width>
						<height>1920</height>
						<anamorphic>FALSE</anamorphic>
						<pixelaspectratio>square</pixelaspectratio>
						<fielddominance>none</fielddominance>
						<colordepth>24</colordepth>
					</samplecharacteristics>
				</format>'''
    
    content = re.sub(r'<format\s*/>', format_replacement, content)
    if verbose:
        logger.info("    ✓ Added detailed format specifications")
    
    # 4. 空の<video/>と<audio/>タグを詳細な形式に置き換え（file要素内）
    # ビデオの詳細
    video_replacement = '''<video>
									<samplecharacteristics>
										<rate>
											<timebase>59</timebase>
											<ntsc>FALSE</ntsc>
										</rate>
										<width>1920</width>
										<height>1080</height>
										<anamorphic>FALSE</anamorphic>
										<pixelaspectratio>square</pixelaspectratio>
										<fielddominance>none</fielddominance>
									</samplecharacteristics>
								</video>'''
    
    # 音声の詳細
    audio_replacement = '''<audio>
									<samplecharacteristics>
										<depth>16</depth>
										<samplerate>48000</samplerate>
									</samplecharacteristics>
									<channelcount>2</channelcount>
								</audio>'''
    
    # file要素内のmedia要素を置き換え
    def replace_media_in_file(match):
        media_content = match.group(0)
        # <video/>を詳細な形式に
        media_content = re.sub(r'<video\s*/>', video_replacement, media_content)
        # <audio/>を詳細な形式に
        media_content = re.sub(r'<audio\s*/>', audio_replacement, media_content)
        return media_content
    
    # file要素内のmediaタグを処理
    content = re.sub(
        r'<media>.*?</media>',
        replace_media_in_file,
        content,
        flags=re.DOTALL
    )
    print("  ✓ Added detailed media specifications")
    
    # 5. clipitemにmasterclipidを追加（なければ）
    def add_masterclipid(match):
        clipitem = match.group(0)
        clipitem_id = match.group(1)
        
        # すでにmasterclipidがある場合はスキップ
        if '<masterclipid>' in clipitem:
            return clipitem
        
        # file idを取得
        file_match = re.search(r'<file id="([^"]+)"', clipitem)
        if file_match:
            file_id = file_match.group(1)
            # masterclipidを追加（clipitem開始タグの直後）
            clipitem = clipitem.replace(
                f'<clipitem frameBlend="FALSE" id="{clipitem_id}">',
                f'<clipitem frameBlend="FALSE" id="{clipitem_id}">\n\t\t\t\t\t\t<masterclipid>{file_id}</masterclipid>',
                1
            )
        
        return clipitem
    
    content = re.sub(
        r'<clipitem frameBlend="FALSE" id="([^"]+)">.*?</clipitem>',
        add_masterclipid,
        content,
        flags=re.DOTALL
    )
    print("  ✓ Added masterclipid to clipitems")
    
    # 6. clipitemにenabled, alphatype, pixelaspectratio, anamorphicを追加
    def add_clipitem_attributes(match):
        clipitem = match.group(0)
        
        # すでに属性がある場合はスキップ
        if '<enabled>' in clipitem:
            return clipitem
        
        # <file>タグの前に属性を追加
        if '<file id=' in clipitem:
            clipitem = clipitem.replace(
                '<file id=',
                '''<enabled>TRUE</enabled>
						<alphatype>none</alphatype>
						<pixelaspectratio>square</pixelaspectratio>
						<anamorphic>FALSE</anamorphic>
						<file id=''',
                1
            )
        
        return clipitem
    
    content = re.sub(
        r'<clipitem frameBlend="FALSE" id="[^"]+">.*?</clipitem>',
        add_clipitem_attributes,
        content,
        flags=re.DOTALL
    )
    print("  ✓ Added clipitem attributes")
    
    # 7. テロップマーカーを削除
    content = re.sub(r'\[Telop\]\s*', '', content)
    content = re.sub(r'\[AI-Speech\]\s*', '', content)
    content = re.sub(r'\[AI-Emotion-\w+\]\s*', '', content)
    print("  ✓ Removed telop markers")
    
    # 8. file-2（テロップ用）を正しいグラフィック形式に変換
    telop_matches = re.findall(r'<file id="file-2"', content)
    telop_counter = len(telop_matches)
    
    if telop_counter > 0:
        first_file_pattern = r'<file id="file-2">.*?</file>'
        
        complete_file = '''<file id="file-2">
							<name>グラフィック</name>
							<pathurl></pathurl>
							<rate>
								<timebase>59</timebase>
								<ntsc>FALSE</ntsc>
							</rate>
							<duration>1000</duration>
							<timecode>
								<rate>
									<timebase>59</timebase>
									<ntsc>FALSE</ntsc>
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
											<ntsc>FALSE</ntsc>
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
        
        print(f"  ✓ Converted {telop_counter} telops to graphics")
    
    # 9. インデントを整形（タブを使用）
    # 既にインデントされているので、この処理はスキップ
    
    # 保存
    output_path = Path(output_xml)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_xml, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n✅ Fixed XML saved to: {output_xml}")
    print(f"   Total telops converted: {telop_counter}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fix_xml_for_premiere.py <input.xml> <output.xml>")
        sys.exit(1)
    
    fix_xml_for_premiere(sys.argv[1], sys.argv[2])
