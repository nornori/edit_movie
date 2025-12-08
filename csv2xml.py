import pandas as pd
import opentimelineio as otio
import os
import argparse

def create_otio_timeline(csv_path, video_path, output_xml_path, fps=30.0):
    # 1. CSV読み込み
    df = pd.read_csv(csv_path)
    
    # 2. タイムラインの作成
    timeline = otio.schema.Timeline(name="AI_Edited_Timeline")
    
    # ★修正点: ビデオ用とオーディオ用の2つのトラックを作る
    video_track = otio.schema.Track(name="Main Video", kind=otio.schema.TrackKind.Video)
    audio_track = otio.schema.Track(name="Main Audio", kind=otio.schema.TrackKind.Audio)
    
    timeline.tracks.append(video_track)
    timeline.tracks.append(audio_track)

    # 3. メディアリファレンスの作成
    abs_video_path = os.path.abspath(video_path)
    
    # 動画ファイル自体をメディアとして登録（ここから映像と音声を両方引っ張る）
    media_reference = otio.schema.ExternalReference(
        target_url=abs_video_path,
        available_range=otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(0, fps),
            duration=otio.opentime.RationalTime(1000000, fps) # 仮の長尺
        )
    )

    # 4. クリップの結合処理 (KEEP判定の連続部分をまとめる)
    if len(df) > 1:
        interval = df.iloc[1]['time'] - df.iloc[0]['time']
    else:
        interval = 0.1
    
    current_start = None
    current_end = None
    clips_to_add = []

    print("Processing cuts...")
    for index, row in df.iterrows():
        is_keep = (row['decision'] == 'KEEP')
        time = row['time']
        
        if is_keep:
            if current_start is None:
                current_start = time
                current_end = time
            else:
                current_end = time
        else:
            if current_start is not None:
                clips_to_add.append((current_start, current_end))
                current_start = None
                current_end = None

    if current_start is not None:
        clips_to_add.append((current_start, current_end))

    # 5. クリップ生成と追加
    print(f"Generating {len(clips_to_add)} clips (Video & Audio) via OTIO...")
    
    for start_sec, end_sec in clips_to_add:
        # 時間計算
        duration_sec = (end_sec - start_sec) + interval
        
        start_frame = int(round(start_sec * fps))
        duration_frame = int(round(duration_sec * fps))
        
        # 共通のソース範囲（動画ファイルのどこを使うか）
        source_range = otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(start_frame, fps),
            duration=otio.opentime.RationalTime(duration_frame, fps)
        )
        
        # --- ビデオクリップ作成 ---
        video_clip = otio.schema.Clip(
            name=f"Video_{start_frame}",
            media_reference=media_reference,
            source_range=source_range
        )
        video_track.append(video_clip)
        
        # --- ★修正点: オーディオクリップ作成 ---
        # 全く同じ設定でオーディオトラックにも追加する
        audio_clip = otio.schema.Clip(
            name=f"Audio_{start_frame}",
            media_reference=media_reference,
            source_range=source_range
        )
        audio_track.append(audio_clip)

    # 6. 書き出し
    try:
        otio.adapters.write_to_file(timeline, output_xml_path, adapter_name='fcp_xml')
        print(f"✅ Successfully exported XML with Audio to: {output_xml_path}")
    except Exception as e:
        print(f"❌ Export failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to final_timeline.csv")
    parser.add_argument("video_path", help="Path to original video file")
    parser.add_argument("--output", default="timeline_otio.xml", help="Output XML filename")
    parser.add_argument("--fps", type=float, default=30.0, help="Frame rate")
    
    args = parser.parse_args()
    
    create_otio_timeline(args.csv_path, args.video_path, args.output, args.fps)