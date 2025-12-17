"""
テロップを完全にPremiere Pro互換のグラフィックに変換
【バックアップ版 - test_ai_telop_final_fixed.xml作成時のコード】

このコードは以下の特徴があります：
- ✅ Premiere Proで読み込める
- ✅ テロップがグラフィックレイヤーとして表示される
- ✅ 各テロップに日本語テキストが表示される（最大5文字）
- ⚠️ 各テロップが個別のトラックになる（705個のテロップ = 705個のトラック）

使用方法：
python src/inference/fix_telop_complete_BACKUP_WORKING.py outputs/test_ai_telop_final.xml outputs/test_ai_telop_final_fixed.xml
"""
import re
import sys
import logging
import base64

logger = logging.getLogger(__name__)


def encode_text_for_premiere(text: str) -> str:
    """
    テキストをPremiere Pro互換のBase64形式にエンコード
    
    戦略：参考XMLのテンプレートを使い、テキスト部分だけを上書きする
    バイナリ構造の長さは変更せず、固定長（15バイト）で処理する
    
    制限：テキストは最大15バイト（日本語で約5文字）
    """
    # 参考XMLの「すいません」のBase64データをデコード
    template_b64 = '5AEAAAAAAABEMyIRDAAAAAAABgAKAAQABgAAAGQAAAAAAF4ASAAQAAwARABAADwAOAAAAAAAAAAAADQAMwAsAAAAKAAkACAAAAAAAAAAAAAAAAAAHAAAAAAAGgAAAAAAAAAUAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAAAABsABwBeAAAAAAAAAUAAAABQAAAAbAAAAAIAAAAAAAEAAQAAAAAAQEEAAMBAAABAQAAAyEIAAAABIAAAAAIAAAACAAAAAAA2QwCASkTw/v//9P7///j+//8S////AAAAAAEAAAAEAAAAEQAAAEhHUFNvZWlLYWt1cG9wdGFpAAAAAQAAAAwAAAAIAAwABAAIAAgAAAAIAAAAUAAAAA8AAADjgZnjgYTjgb7jgZvjgpMAAAA2ACQAAAAgABwAAAAAABsAFAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAADAAAAAgABAA2AAAAAgAAABwAAAAcAAAAHAAAAAAAIEEAAAABaAAAAE3PAEOw////tP///wEAAAAUAAAAEAAaAAgABwAMAAAAEAAUABAAAAAAAAABHAAAAAAAoEEcAAAAAgAAAAAACgAIAAUABgAHAAoAAAAADAD//P///wQABAAEAAAACAAIAAYABwAIAAAAAAAMAA=='
    template_data = bytearray(base64.b64decode(template_b64))
    
    # 元のテキスト「すいません」のUTF-8バイト列
    original_text = 'すいません'
    original_bytes = original_text.encode('utf-8')  # 15バイト
    original_len = len(original_bytes)
    
    # 新しいテキストのUTF-8バイト列
    new_text_bytes = text.encode('utf-8')
    new_text_len = len(new_text_bytes)
    
    # テンプレート内でテキストが出現する位置を検索
    text_position = template_data.find(original_bytes)
    
    if text_position == -1:
        logger.warning(f"Could not find text position in template")
        return template_b64
    
    # テキストの長さ情報の位置（テキストの4バイト前）
    length_position = text_position - 4
    
    # 固定長（15バイト）に調整
    if new_text_len > original_len:
        # 長すぎる場合はトリミング（文字境界を考慮）
        adjusted_bytes = new_text_bytes[:original_len]
        # UTF-8の文字境界で切れている可能性があるので、デコード可能な部分まで削る
        while len(adjusted_bytes) > 0:
            try:
                adjusted_bytes.decode('utf-8')
                break
            except UnicodeDecodeError:
                adjusted_bytes = adjusted_bytes[:-1]
        new_text_bytes = adjusted_bytes
        new_text_len = len(new_text_bytes)
    elif new_text_len < original_len:
        # 短い場合はNULLバイトでパディング
        padding_needed = original_len - new_text_len
        new_text_bytes = new_text_bytes + b'\x00' * padding_needed
        new_text_len = len(new_text_bytes)
    
    # テンプレートを修正（長さは変更しない）
    # 1. テキスト長を更新
    template_data[length_position] = new_text_len & 0xFF
    
    # 2. テキスト部分を上書き（長さは同じなので挿入/削除不要）
    for i, byte in enumerate(new_text_bytes):
        template_data[text_position + i] = byte
    
    # Base64エンコードして返す
    return base64.b64encode(bytes(template_data)).decode('ascii')


def fix_telops_complete(input_xml: str, output_xml: str):
    """
    XMLのテロップを完全にPremiere Pro互換のグラフィックに変換
    
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
    
    # 2. file-2を参照しているtrackを検出
    telop_tracks = re.findall(r'<track>.*?<file id="file-2".*?</track>', content, re.DOTALL)
    telop_counter = len(telop_tracks)
    
    if telop_counter > 0:
        logger.info(f"  Found {telop_counter} telop tracks")
        
        # 3. 各テロップトラックを完全なグラフィックトラックに置き換え
        file_id_counter = 2
        
        def convert_telop_track(match):
            nonlocal file_id_counter
            
            track_content = match.group(0)
            
            # file-2を含まないtrackはそのまま返す
            if 'file id="file-2"' not in track_content:
                return track_content
            
            # clipitem内の情報を抽出
            clipitem_match = re.search(r'<clipitem([^>]*)>(.*?)</clipitem>', track_content, re.DOTALL)
            if not clipitem_match:
                return track_content
            
            clipitem_attrs = clipitem_match.group(1)
            clipitem_inner = clipitem_match.group(2)
            
            # 各要素を抽出
            name_match = re.search(r'<name>([^<]+)</name>', clipitem_inner)
            rate_match = re.search(r'<rate>(.*?)</rate>', clipitem_inner, re.DOTALL)
            start_match = re.search(r'<start>(\d+)</start>', clipitem_inner)
            end_match = re.search(r'<end>(\d+)</end>', clipitem_inner)
            in_match = re.search(r'<in>(\d+)</in>', clipitem_inner)
            out_match = re.search(r'<out>(\d+)</out>', clipitem_inner)
            
            if not all([name_match, rate_match, start_match, end_match, in_match, out_match]):
                return track_content
            
            telop_text = name_match.group(1)
            rate_block = rate_match.group(1)
            start = start_match.group(1)
            end = end_match.group(1)
            in_val = in_match.group(1)
            out_val = out_match.group(1)
            
            # 現在のfile IDを取得
            current_file_id = f"file-{file_id_counter}"
            file_id_counter += 1
            
            # テキストをPremiere Pro形式にエンコード
            encoded_text = encode_text_for_premiere(telop_text)
            
            # 完全なグラフィックトラックを生成
            new_track = f'''<track TL.SQTrackShy="0" TL.SQTrackExpandedHeight="25" TL.SQTrackExpanded="0" MZ.TrackTargeted="0">
							<clipitem{clipitem_attrs}>
								<masterclipid>masterclip-{file_id_counter}</masterclipid>
								<name>グラフィック</name>
								<enabled>TRUE</enabled>
								<duration>2582774</duration>
								<rate>
									{rate_block}
								</rate>
								<start>{start}</start>
								<end>{end}</end>
								<in>{in_val}</in>
								<out>{out_val}</out>
								<pproTicksIn>914456774963789</pproTicksIn>
								<pproTicksOut>914677708529577</pproTicksOut>
								<alphatype>straight</alphatype>
								<pixelaspectratio>square</pixelaspectratio>
								<anamorphic>FALSE</anamorphic>
								<file id="{current_file_id}">
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
													<timebase>60</timebase>
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
								</file>
								<filter>
									<effect>
										<name>{telop_text}</name>
										<effectid>GraphicAndType</effectid>
										<effectcategory>graphic</effectcategory>
										<effecttype>filter</effecttype>
										<mediatype>video</mediatype>
										<pproBypass>false</pproBypass>
										<parameter authoringApp="PremierePro">
											<parameterid>1</parameterid>
											<name>ソーステキスト</name>
											<hash>025b1101-ead6-0ca2-1feb-caf3000001fc</hash>
											<value>{encoded_text}</value>
										</parameter>
										<parameter authoringApp="PremierePro">
											<parameterid>2</parameterid>
											<name>トランスフォーム</name>
											<IsTimeVarying>false</IsTimeVarying>
											<ParameterControlType>11</ParameterControlType>
											<LowerBound>false</LowerBound>
											<UpperBound>false</UpperBound>
											<value>-91445760000000000,false,0,0,0,0,0,0</value>
										</parameter>
										<parameter authoringApp="PremierePro">
											<parameterid>3</parameterid>
											<name>位置</name>
											<IsTimeVarying>false</IsTimeVarying>
											<value>-91445760000000000,0.5:0.5,0,0,0,0,0,0,5,4,0,0,0,0</value>
										</parameter>
										<parameter authoringApp="PremierePro">
											<parameterid>4</parameterid>
											<name>スケール</name>
											<IsTimeVarying>false</IsTimeVarying>
											<ParameterControlType>2</ParameterControlType>
											<LowerBound>0</LowerBound>
											<UpperBound>4000</UpperBound>
											<value>-91445760000000000,100.,0,0,0,0,0,0</value>
										</parameter>
										<parameter authoringApp="PremierePro">
											<parameterid>8</parameterid>
											<name>不透明度</name>
											<IsTimeVarying>false</IsTimeVarying>
											<ParameterControlType>2</ParameterControlType>
											<LowerBound>0</LowerBound>
											<UpperBound>100</UpperBound>
											<value>-91445760000000000,100.,0,0,0,0,0,0</value>
										</parameter>
									</effect>
								</filter>
								<logginginfo>
									<description></description>
									<scene></scene>
									<shottake></shottake>
									<lognote></lognote>
									<good></good>
									<originalvideofilename></originalvideofilename>
									<originalaudiofilename></originalaudiofilename>
								</logginginfo>
								<colorinfo>
									<lut></lut>
									<lut1></lut1>
									<asc_sop></asc_sop>
									<asc_sat></asc_sat>
									<lut2></lut2>
								</colorinfo>
								<labels>
								</labels>
							</clipitem>
							<enabled>TRUE</enabled>
							<locked>FALSE</locked>
						</track>'''
            
            return new_track
        
        # 全てのtrackを処理
        content = re.sub(
            r'<track>.*?</track>',
            convert_telop_track,
            content,
            flags=re.DOTALL
        )
        
        logger.info(f"  ✅ Converted {telop_counter} telops to complete Premiere Pro graphics")
    
    # 保存
    with open(output_xml, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"  Output: {output_xml}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fix_telop_complete_BACKUP_WORKING.py <input> <output>")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO)
    fix_telops_complete(sys.argv[1], sys.argv[2])
