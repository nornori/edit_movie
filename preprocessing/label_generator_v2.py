import os
import json
import math
import glob # ファイル検索のためにglobモジュールを追加
from typing import List, Dict, Any, Set

# 定数
TIME_STEP = 0.1 # サンプリング刻み幅（秒）

# グラフィックとオーディオの判定に使う名称（プロジェクトに合わせて調整可能）
GRAPHIC_CLIP_NAME = "グラフィック"
AUDIO_FILE_EXTENSIONS = {".mp3", ".wav", ".aac", ".ogg"}


def load_xml_json(json_path: str) -> Dict[str, Any]:
    """*.xml.jsonファイルを読み込む"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_base_video_name(meta: Dict[str, Any]) -> str:
    """
    meta["video_path"]からOS依存なくファイル名だけを抜き出す。
    メイン動画クリップのname判定に使用。
    """
    video_path = meta.get("video_path", "")
    # Windowsパス (\\) を / に変換し、ファイル名を取得
    v = video_path.replace("\\", "/")
    return v.split("/")[-1]


def is_audio_file(filename: str) -> bool:
    """ファイル名がオーディオ拡張子かどうかを判定するヘルパー関数"""
    return any(filename.lower().endswith(ext) for ext in AUDIO_FILE_EXTENSIONS)


def build_labels_with_layout(
    data: Dict[str, Any],
    time_step: float = TIME_STEP,
) -> List[Dict[str, Any]]:
    """
    1つの *.xml.json から、指定time_step刻みで時系列ラベルを生成する。
    （この関数のロジックはV2から変更なし）
    """
    meta = data["meta"]
    clips = data["ground_truth_clips"]
    base_name = get_base_video_name(meta)

    # タイムラインの最大長を計算
    max_time = 0.0
    for c in clips:
        max_time = max(max_time, float(c.get("timeline_end", 0.0)))

    if max_time <= 0:
        return []

    num_steps = int(math.ceil(max_time / time_step))
    rows: List[Dict[str, Any]] = []

    for i in range(num_steps + 1):
        t = round(i * time_step, 6)

        # この時刻にアクティブな全クリップを抽出
        active_clips = [
            c for c in clips
            if float(c.get("timeline_start", 0.0)) <= t < float(c.get("timeline_end", 0.0))
        ]
        
        # ==========================================================
        # 1. メイン動画 (Cut / Zoom / Pos) の抽出
        # ==========================================================
        main_video_candidates = [
            c for c in active_clips
            if c.get("track_kind") == "video" and c.get("name") == base_name
        ]
        is_used = 1 if main_video_candidates else 0

        # is_used=0の場合は0.0をデフォルトとする
        main_scale, main_pos_x, main_pos_y = 0.0, 0.0, 0.0 
        main_track_index = -1

        if main_video_candidates:
            main_clip = sorted(
                main_video_candidates,
                key=lambda c: int(c.get("track_index", 9999))
            )[0]
            main_track_index = int(main_clip.get("track_index", -1))
            
            main_scale = float(main_clip.get("scale", 1.0))
            main_pos_x = float(main_clip.get("pos_x", 0.0))
            main_pos_y = float(main_clip.get("pos_y", 0.0))

        # ==========================================================
        # 2. コンテンツの種類別活性状態の抽出 (トラックインデックスに非依存)
        # ==========================================================
        
        is_graphic_active = any(
            c for c in active_clips
            if c.get("track_kind") == "video" and c.get("name") == GRAPHIC_CLIP_NAME
        )
        
        is_broll_active = any(
            c for c in active_clips
            if c.get("track_kind") == "video" and c.get("name") != base_name and c.get("name") != GRAPHIC_CLIP_NAME
        )
        
        is_any_audio_active = any(
            c for c in active_clips
            if c.get("track_kind") == "audio"
        )
        
        # ==========================================================
        # 3. layoutリストの作成 (詳細なトラック構造)
        # ==========================================================
        
        layout = []
        active_track_indices: Set[int] = set()
        
        for c in active_clips:
            ti = int(c.get("track_index", -1))
            active_track_indices.add(ti)
            name = c.get("name", "")
            kind = c.get("track_kind", "")
            
            is_main = bool(
                main_track_index != -1 and 
                ti == main_track_index and 
                name == base_name
            )
            is_graphic = bool(name == GRAPHIC_CLIP_NAME)
            
            # オーディオファイル名の判定をヘルパー関数で実行
            is_music_or_sfx = bool(kind == "audio" and is_audio_file(name))
            
            layout.append(
                {
                    "track_index": ti,
                    "kind": kind,
                    "name": name,
                    "is_main_video": is_main,
                    "is_graphic": is_graphic,
                    "is_music_or_sfx": is_music_or_sfx, # 追加
                    "scale": float(c.get("scale", float("nan"))),
                    "pos_x": float(c.get("pos_x", float("nan"))),
                    "pos_y": float(c.get("pos_y", float("nan"))),
                    "opacity": float(c.get("opacity", float("nan"))),
                }
            )

        # ==========================================================
        # 4. 結果行の統合
        # ==========================================================
        
        row = {
            "time": round(t, 3),
            
            # 基本のCut/Zoom/Layoutラベル
            "is_used": is_used,
            "main_scale": main_scale,
            "main_pos_x": main_pos_x,
            "main_pos_y": main_pos_y,
            "main_track_index": main_track_index,
            
            # コンテンツの種類別活性状態
            "is_graphic_active": 1 if is_graphic_active else 0,
            "is_broll_active": 1 if is_broll_active else 0,
            "is_any_audio_active": 1 if is_any_audio_active else 0,
            
            # 詳細なトラック構造
            "active_track_count": len(active_track_indices),
            "layout": layout, 
            "active_track_indices": sorted(list(active_track_indices)),
        }
        rows.append(row)

    return rows


def main():
    # --- 【重要】バッチ処理の設定 ---
    
    # 1. 入力ディレクトリ: *.xml.jsonファイルが格納されているフォルダを指定
    INPUT_DIR = "./input_jsons" 
    
    # 2. 出力ディレクトリ: 生成されたラベルJSONを保存するフォルダを指定
    OUTPUT_DIR = "./output_labels"
    
    # 3. ファイル検索パターン: INPUT_DIR内の全ての *.xml.json を検索
    json_paths = glob.glob(os.path.join(INPUT_DIR, "*.xml.json"))
    
    if not json_paths:
        print(f"[ERROR] '{INPUT_DIR}'内に処理対象の *.xml.json ファイルが見つかりませんでした。")
        print("INPUT_DIRのパスを確認してください。")
        return

    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"[INFO] 処理対象ファイル数: {len(json_paths)} 件")
    
    # 全ファイルをループして処理
    for idx, json_path in enumerate(json_paths):
        base_filename = os.path.basename(json_path)
        output_filename = base_filename.replace(".xml.json", "_labels.json")
        out_path = os.path.join(OUTPUT_DIR, output_filename)
        
        print(f"--- ({idx+1}/{len(json_paths)}) 処理中: {base_filename}")
        
        try:
            data = load_xml_json(json_path)
            rows = build_labels_with_layout(data)

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)

            print(f"[SUCCESS] ラベル {len(rows)} ステップを保存しました: {out_path}")
            
        except Exception as e:
            print(f"[ERROR] ファイル '{base_filename}' の処理中にエラーが発生しました: {e}")
            continue # エラーが発生した場合は次のファイルへ

    print("\n====================================")
    print("[完了] 全バッチ処理が終了しました。")
    print(f"結果は '{OUTPUT_DIR}' フォルダに保存されています。")


if __name__ == "__main__":
    main()