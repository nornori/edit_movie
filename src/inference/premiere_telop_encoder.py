"""
Premiere Pro テロップエンコーダー

Premiere ProのMOGRT形式のBase64エンコードされたテロップデータを生成
"""
import base64
import struct
import logging

logger = logging.getLogger(__name__)


def encode_telop_text(text: str) -> str:
    """
    テロップテキストをPremiere Pro形式のBase64エンコードされた値に変換
    
    Args:
        text: テロップテキスト
    
    Returns:
        Base64エンコードされた文字列
    """
    # UTF-8エンコード
    text_bytes = text.encode('utf-8')
    text_len = len(text_bytes)
    
    # パディング後のテキスト長（4バイトアライメント）
    text_padded_len = ((text_len + 3) // 4) * 4
    
    # 全体のサイズを計算
    # 基本サイズ + テキスト長 + パディング
    total_size = 480 + text_padded_len - 3  # 480は1文字（3バイト）の場合の基本サイズ
    
    # バイナリデータを構築
    data = bytearray()
    
    # ヘッダー部分（固定）
    data.extend(b'\xd4\x01\x00\x00\x00\x00\x00\x00')  # マジックナンバー
    data.extend(b'D3"\x11\x0c\x00\x00\x00\x00\x00')  # バージョン情報
    
    # FlatBuffersヘッダー
    data.extend(b'\x06\x00\n\x00\x04\x00\x06\x00\x00\x00')
    data.extend(b'd\x00\x00\x00\x00\x00')
    
    # オフセット情報
    data.extend(b'^\x00H\x00\x10\x00\x0c\x00')
    data.extend(b'D\x00@\x00<\x008\x00\x00\x00\x00\x00\x00\x00\x00\x00')
    data.extend(b'4\x003\x00,\x00\x00\x00(\x00$\x00 \x00')
    data.extend(b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
    data.extend(b'\x1c\x00\x00\x00\x00\x00\x1a\x00\x00\x00\x00\x00\x00\x00')
    data.extend(b'\x14\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
    data.extend(b'\x00\x00\x00\x00\x08\x00\x00\x00\x00\x00')
    data.extend(b'\x1b\x00\x07\x00^\x00\x00\x00\x00\x00\x00\x01')
    
    # サイズ情報
    data.extend(b'@\x00\x00\x00P\x00\x00\x00l\x00\x00\x00')
    
    # フラグ
    data.extend(b'\x02\x00\x00\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00\x00')
    
    # デフォルト値（フォントサイズ、色など）
    data.extend(b'@A\x00\x00\xc0@\x00\x00@@\x00\x00\xc8B')
    data.extend(b'\x00\x00\x00\x01 \x00\x00\x00\x02\x00\x00\x00\x02\x00\x00\x00\x00\x00')
    data.extend(b'\x10D\x00\x80JD\x00')
    
    # 色情報
    data.extend(b'\xff\xff\xff\x04\xff\xff\xff\x08\xff\xff\xff"\xff\xff\xff\x00')
    data.extend(b'\x00\x00\x00\x01\x00\x00\x00\x04\x00\x00\x00')
    
    # フォント名長さ
    data.extend(b'\x11\x00\x00\x00')
    
    # フォント名（HGPSoeiKakupoptai）
    data.extend(b'HGPSoeiKakupoptai\x00\x00\x00')
    
    # テキストセクションヘッダー
    data.extend(b'\x01\x00\x00\x00\x0c\x00\x00\x00')
    data.extend(b'\x08\x00\x0c\x00\x04\x00\x08\x00\x08\x00\x00\x00')
    data.extend(b'\x08\x00\x00\x00')
    
    # テキストブロック全体のサイズ（テキスト長 + 4）
    text_block_size = text_len + 4
    data.extend(struct.pack('<I', text_block_size))
    
    # テキスト長
    data.extend(struct.pack('<I', text_len))
    
    # テキストデータ
    data.extend(text_bytes)
    
    # パディング（4バイトアライメント）
    padding = (4 - (text_len % 4)) % 4
    data.extend(b'\x00' * padding)
    
    # フッター部分
    data.extend(b'4\x00 \x00\x00\x00\x1c\x00\x18\x00\x00\x00\x00\x00')
    data.extend(b'\x17\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
    data.extend(b'\x00\x00\x00\x00\x00\x00\x0c\x00\x00\x00\x00\x00\x00\x00')
    data.extend(b'\x08\x00\x00\x00\x04\x004\x00\x00\x00')
    data.extend(b'\x1c\x00\x00\x00\x1c\x00\x00\x00\x1c\x00\x00\x00\x00\x00')
    data.extend(b'pA\x00\x00\x00\x01h\x00\x00\x00')
    data.extend(b'\x1d\xff\x9eC\xb0\xff\xff\xff\xb4\xff\xff\xff')
    data.extend(b'\x01\x00\x00\x00\x14\x00\x00\x00')
    data.extend(b'\x10\x00\x1a\x00\x08\x00\x07\x00\x0c\x00\x00\x00')
    data.extend(b'\x10\x00\x14\x00\x10\x00\x00\x00\x00\x00\x00\x01')
    data.extend(b'\x1c\x00\x00\x00\x00\x00\x88A\x1c\x00\x00\x00')
    data.extend(b'\x02\x00\x00\x00\x00\x00\n\x00\x08\x00\x05\x00\x06\x00\x07\x00\n\x00')
    data.extend(b'\x00\x00\x00\xfdo\xff\xfc\xff\xff\xff')
    data.extend(b'\x04\x00\x04\x00\x04\x00\x00\x00')
    data.extend(b'\x08\x00\x08\x00\x06\x00\x07\x00\x08\x00\x00\x00\x00\x00\xfdo')
    
    # Base64エンコード
    encoded = base64.b64encode(data).decode('ascii')
    
    logger.debug(f"Encoded telop: text_len={text_len}, padded={text_padded_len}, total={len(data)} bytes")
    
    return encoded


def test_encoder():
    """エンコーダーのテスト"""
    test_texts = [
        "あっ",
        "おおっちゃ",
        "テスト",
        "Hello World"
    ]
    
    for text in test_texts:
        encoded = encode_telop_text(text)
        print(f"\nText: {text}")
        print(f"Encoded length: {len(encoded)}")
        print(f"First 100 chars: {encoded[:100]}")
        
        # デコードしてテキストが含まれているか確認
        decoded = base64.b64decode(encoded)
        if text.encode('utf-8') in decoded:
            print("✅ Text found in encoded data")
        else:
            print("❌ Text NOT found in encoded data")


if __name__ == "__main__":
    test_encoder()
