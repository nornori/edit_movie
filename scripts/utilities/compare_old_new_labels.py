"""
古いラベルと新しいラベルを比較

元のモデルが何を学習していたかを確認します。
"""
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compare_labels(old_path: Path, new_path: Path, video_name: str):
    """
    古いラベルと新しいラベルを比較
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"動画: {video_name}")
    logger.info(f"{'='*70}")
    
    # 古いラベルを読み込み
    df_old = pd.read_csv(old_path)
    old_timesteps = len(df_old['time'].unique())
    old_duration = df_old['time'].max()
    old_active_frames = (df_old['active'] > 0.5).sum()
    old_total_frames = len(df_old)
    
    logger.info(f"\n【古いラベル】 (data/processed/output_labels)")
    logger.info(f"  タイムステップ数: {old_timesteps}")
    logger.info(f"  動画時間: {old_duration:.1f}秒")
    logger.info(f"  総フレーム数: {old_total_frames}")
    logger.info(f"  アクティブフレーム数: {old_active_frames}")
    logger.info(f"  アクティブ率: {100 * old_active_frames / old_total_frames:.1f}%")
    
    # 新しいラベルを読み込み
    df_new = pd.read_csv(new_path)
    new_timesteps = len(df_new['time'].unique())
    new_duration = df_new['time'].max()
    new_active_frames = (df_new['active'] > 0.5).sum()
    new_total_frames = len(df_new)
    
    logger.info(f"\n【新しいラベル】 (data/processed/output_labels_full)")
    logger.info(f"  タイムステップ数: {new_timesteps}")
    logger.info(f"  動画時間: {new_duration:.1f}秒")
    logger.info(f"  総フレーム数: {new_total_frames}")
    logger.info(f"  アクティブフレーム数: {new_active_frames}")
    logger.info(f"  アクティブ率: {100 * new_active_frames / new_total_frames:.1f}%")
    
    # 差分を計算
    logger.info(f"\n【差分】")
    logger.info(f"  タイムステップ差: {new_timesteps - old_timesteps} ({new_timesteps / old_timesteps:.2f}x)")
    logger.info(f"  動画時間差: {new_duration - old_duration:.1f}秒")
    logger.info(f"  カットされた時間: {new_duration - old_duration:.1f}秒 ({100 * (new_duration - old_duration) / new_duration:.1f}%)")
    
    # 結論
    logger.info(f"\n【結論】")
    if new_timesteps > old_timesteps * 1.5:
        logger.info(f"  ✅ 古いラベルは編集後の動画（{old_duration:.0f}秒）のみ")
        logger.info(f"  ✅ 新しいラベルは元動画全体（{new_duration:.0f}秒）")
        logger.info(f"  ✅ 約{new_duration - old_duration:.0f}秒分のカット判断を学習できるようになった！")
    else:
        logger.info(f"  ⚠️  差が小さい - 両方とも同じ動画の可能性")


def main():
    logger.info("="*70)
    logger.info("古いラベルと新しいラベルの比較")
    logger.info("="*70)
    
    # サンプル動画で比較
    sample_videos = [
        "bandicam 2025-06-11 22-48-40-994",
        "bandicam 2025-03-26 13-09-28-006",
        "bandicam 2025-05-11 19-25-14-768"
    ]
    
    old_dir = Path("data/processed/output_labels")
    new_dir = Path("data/processed/output_labels_full")
    
    for video_name in sample_videos:
        old_path = old_dir / f"{video_name}_tracks.csv"
        new_path = new_dir / f"{video_name}_tracks.csv"
        
        if old_path.exists() and new_path.exists():
            compare_labels(old_path, new_path, video_name)
        else:
            logger.warning(f"\n⚠️  {video_name}: ファイルが見つかりません")
            if not old_path.exists():
                logger.warning(f"    古いラベルなし: {old_path}")
            if not new_path.exists():
                logger.warning(f"    新しいラベルなし: {new_path}")
    
    logger.info("\n" + "="*70)
    logger.info("比較完了")
    logger.info("="*70)


if __name__ == "__main__":
    main()
