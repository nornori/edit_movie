"""
トレーニングシーケンスの作成

ラベルCSVファイルをNPZ形式に変換し、train/valに分割します。
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_label_csv(csv_path: str, max_seq_len: int = 1000) -> list:
    """
    ラベルCSVを読み込んでシーケンスに変換
    長いシーケンスは複数のチャンクに分割
    
    Args:
        csv_path: CSVファイルのパス
        max_seq_len: 最大シーケンス長（タイムステップ）
    
    Returns:
        List of (sequence, mask, video_id) tuples
    """
    df = pd.read_csv(csv_path)
    
    # video_idを取得
    video_id = df['video_id'].iloc[0] if 'video_id' in df.columns else Path(csv_path).stem.replace('_tracks', '')
    
    # トラックパラメータのカラムを抽出
    # active, asset_id, scale, pos_x, pos_y, anchor_x, anchor_y, rotation, crop_l, crop_r, crop_t, crop_b (12 parameters)
    param_cols = ['active', 'asset_id', 'scale', 'pos_x', 'pos_y', 'anchor_x', 'anchor_y', 
                  'rotation', 'crop_l', 'crop_r', 'crop_t', 'crop_b']
    
    # タイムステップ数を取得
    unique_times = df['time'].unique()
    num_timesteps = len(unique_times)
    
    # トラック数を取得
    num_tracks = df['track'].max() + 1
    
    # シーケンスを初期化 (num_timesteps, num_tracks, 12)
    sequence = np.zeros((num_timesteps, num_tracks, len(param_cols)), dtype=np.float32)
    
    # データを埋める
    for _, row in df.iterrows():
        t_idx = int(row['time'] / 0.1)  # 0.1秒間隔
        if t_idx >= num_timesteps:
            continue
        track_idx = int(row['track'])
        if track_idx >= num_tracks:
            continue
        
        for param_idx, param in enumerate(param_cols):
            if param in row:
                sequence[t_idx, track_idx, param_idx] = row[param]
    
    # シーケンスをフラット化 (num_timesteps, num_tracks * 12 = 240)
    sequence_flat = sequence.reshape(num_timesteps, -1)
    
    # 長いシーケンスを複数のチャンクに分割
    chunks = []
    num_chunks = (num_timesteps + max_seq_len - 1) // max_seq_len  # 切り上げ除算
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * max_seq_len
        end_idx = min(start_idx + max_seq_len, num_timesteps)
        
        chunk_seq = sequence_flat[start_idx:end_idx]
        chunk_mask = np.ones(len(chunk_seq), dtype=bool)
        
        # チャンクIDを含むvideo_id
        if num_chunks > 1:
            chunk_video_id = f"{video_id}_chunk{chunk_idx}"
        else:
            chunk_video_id = video_id
        
        chunks.append((chunk_seq, chunk_mask, chunk_video_id))
    
    return chunks


def main():
    logger.info("="*70)
    logger.info("トレーニングシーケンスの作成（元動画全体版）")
    logger.info("="*70)
    
    # ディレクトリ設定 - 元動画全体のラベルを使用
    labels_dir = Path("data/processed/output_labels_full")
    output_dir = Path("preprocessed_data")
    output_dir.mkdir(exist_ok=True)
    
    # ラベルCSVファイルを取得
    csv_files = list(labels_dir.glob("*_tracks.csv"))
    logger.info(f"\nラベルCSVファイル数: {len(csv_files)}")
    
    if not csv_files:
        logger.error("ラベルCSVファイルが見つかりません")
        return
    
    # 全てのシーケンスを読み込み（チャンクに分割）
    sequences = []
    masks = []
    video_ids = []
    
    max_seq_len = 1000  # 最大1000タイムステップ（100秒）
    
    for csv_file in csv_files:
        try:
            chunks = load_label_csv(str(csv_file), max_seq_len=max_seq_len)
            for chunk_seq, chunk_mask, chunk_video_id in chunks:
                sequences.append(chunk_seq)
                masks.append(chunk_mask)
                video_ids.append(chunk_video_id)
            
            video_id = Path(csv_file).stem.replace('_tracks', '')
            logger.info(f"✅ {video_id}: {len(chunks)} chunks, shapes: {[s.shape for s, _, _ in chunks]}")
        except Exception as e:
            logger.error(f"❌ {csv_file.name}: {e}")
    
    logger.info(f"\n読み込み完了: {len(sequences)}個のチャンク（元の動画: {len(csv_files)}個）")
    
    # パディング（最大長に合わせる）
    max_len = max(seq.shape[0] for seq in sequences)
    logger.info(f"最大シーケンス長: {max_len}")
    
    padded_sequences = []
    padded_masks = []
    
    for seq, mask in zip(sequences, masks):
        if seq.shape[0] < max_len:
            # パディング
            pad_len = max_len - seq.shape[0]
            seq_padded = np.pad(seq, ((0, pad_len), (0, 0)), mode='constant')
            mask_padded = np.pad(mask, (0, pad_len), mode='constant', constant_values=False)
        else:
            seq_padded = seq
            mask_padded = mask
        
        padded_sequences.append(seq_padded)
        padded_masks.append(mask_padded)
    
    # NumPy配列に変換
    sequences_array = np.array(padded_sequences, dtype=np.float32)
    masks_array = np.array(padded_masks, dtype=bool)
    video_ids_array = np.array(video_ids)
    
    logger.info(f"\nシーケンス配列: {sequences_array.shape}")
    logger.info(f"マスク配列: {masks_array.shape}")
    
    # Train/Val分割（80/20）
    indices = np.arange(len(sequences_array))
    train_indices, val_indices = train_test_split(
        indices, test_size=0.2, random_state=42
    )
    
    logger.info(f"\nTrain: {len(train_indices)}個")
    logger.info(f"Val: {len(val_indices)}個")
    
    # Train NPZを保存
    train_path = output_dir / "train_sequences.npz"
    np.savez_compressed(
        train_path,
        sequences=sequences_array[train_indices],
        masks=masks_array[train_indices],
        video_ids=video_ids_array[train_indices],
        source_video_names=video_ids_array[train_indices]
    )
    logger.info(f"\n✅ Train NPZ保存: {train_path}")
    
    # Val NPZを保存
    val_path = output_dir / "val_sequences.npz"
    np.savez_compressed(
        val_path,
        sequences=sequences_array[val_indices],
        masks=masks_array[val_indices],
        video_ids=video_ids_array[val_indices],
        source_video_names=video_ids_array[val_indices]
    )
    logger.info(f"✅ Val NPZ保存: {val_path}")
    
    logger.info("\n" + "="*70)
    logger.info("トレーニングシーケンス作成完了")
    logger.info("="*70)


if __name__ == "__main__":
    main()
