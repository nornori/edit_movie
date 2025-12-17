"""
データ整合性確認スクリプト

特徴量CSVとラベルCSVの整合性を確認します。
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def verify_single_video(features_path: Path, labels_path: Path, video_name: str) -> dict:
    """
    1つの動画の特徴量とラベルの整合性を確認
    
    Returns:
        dict: 検証結果
    """
    result = {
        'video_name': video_name,
        'status': 'OK',
        'issues': []
    }
    
    try:
        # 特徴量を読み込み
        df_features = pd.read_csv(features_path)
        features_timesteps = len(df_features)
        features_duration = df_features['time'].max() if 'time' in df_features.columns else 0
        
        # ラベルを読み込み
        df_labels = pd.read_csv(labels_path)
        # ラベルは20トラック分あるので、タイムステップ数 = 行数 / 20
        labels_timesteps = len(df_labels['time'].unique())
        labels_duration = df_labels['time'].max() if 'time' in df_labels.columns else 0
        
        result['features_timesteps'] = features_timesteps
        result['features_duration'] = features_duration
        result['labels_timesteps'] = labels_timesteps
        result['labels_duration'] = labels_duration
        
        # タイムステップ数の差を確認
        timestep_diff = abs(features_timesteps - labels_timesteps)
        result['timestep_diff'] = timestep_diff
        
        # 許容範囲: 5%以内
        tolerance = max(features_timesteps, labels_timesteps) * 0.05
        
        if timestep_diff > tolerance:
            result['status'] = 'WARNING'
            result['issues'].append(f"タイムステップ数の差が大きい: {timestep_diff} (許容: {tolerance:.0f})")
        
        # 時間の差を確認
        duration_diff = abs(features_duration - labels_duration)
        result['duration_diff'] = duration_diff
        
        if duration_diff > 5.0:  # 5秒以上の差
            result['status'] = 'WARNING'
            result['issues'].append(f"動画時間の差が大きい: {duration_diff:.1f}秒")
        
        # 特徴量の列数を確認
        expected_audio_cols = ['time', 'audio_energy_rms', 'audio_is_speaking', 'silence_duration_ms', 
                              'speaker_id', 'text_is_active', 'text_word', 'telop_active', 'telop_text']
        expected_visual_cols = ['scene_change', 'visual_motion', 'saliency_x', 'saliency_y',
                               'face_count', 'face_center_x', 'face_center_y', 'face_size',
                               'face_mouth_open', 'face_eyebrow_raise']
        clip_cols = [f'clip_{i}' for i in range(512)]
        
        missing_audio = [col for col in expected_audio_cols if col not in df_features.columns]
        missing_visual = [col for col in expected_visual_cols if col not in df_features.columns]
        missing_clip = [col for col in clip_cols if col not in df_features.columns]
        
        if missing_audio:
            result['status'] = 'ERROR'
            result['issues'].append(f"音声特徴量が不足: {missing_audio}")
        
        if missing_visual:
            result['status'] = 'ERROR'
            result['issues'].append(f"視覚特徴量が不足: {missing_visual}")
        
        if len(missing_clip) > 0:
            result['status'] = 'ERROR'
            result['issues'].append(f"CLIP特徴量が不足: {len(missing_clip)}個")
        
        # ラベルの列数を確認
        expected_label_cols = ['time', 'track', 'active', 'asset_id', 'scale', 'pos_x', 'pos_y',
                              'anchor_x', 'anchor_y', 'rotation', 'crop_l', 'crop_r', 'crop_t', 'crop_b']
        missing_labels = [col for col in expected_label_cols if col not in df_labels.columns]
        
        if missing_labels:
            result['status'] = 'ERROR'
            result['issues'].append(f"ラベル列が不足: {missing_labels}")
        
        # トラック数を確認
        num_tracks = df_labels['track'].max() + 1 if 'track' in df_labels.columns else 0
        result['num_tracks'] = num_tracks
        
        if num_tracks != 20:
            result['status'] = 'WARNING'
            result['issues'].append(f"トラック数が20ではない: {num_tracks}")
        
        # アクティブフレーム数を確認
        if 'active' in df_labels.columns:
            active_frames = (df_labels['active'] > 0.5).sum()
            total_frames = len(df_labels)
            active_ratio = active_frames / total_frames if total_frames > 0 else 0
            result['active_frames'] = active_frames
            result['active_ratio'] = active_ratio
        
    except Exception as e:
        result['status'] = 'ERROR'
        result['issues'].append(f"エラー: {str(e)}")
    
    return result


def main():
    logger.info("="*70)
    logger.info("データ整合性確認")
    logger.info("="*70)
    
    # ディレクトリ設定
    features_dir = Path("data/processed/input_features")
    labels_dir = Path("data/processed/output_labels_full")
    
    # 統計情報を読み込み
    stats_file = labels_dir / "generation_stats.json"
    with open(stats_file, 'r', encoding='utf-8') as f:
        label_stats = json.load(f)
    
    # 検証結果を保存
    verification_results = []
    
    # 各動画を検証
    ok_count = 0
    warning_count = 0
    error_count = 0
    missing_features_count = 0
    
    for video_info in label_stats['videos']:
        if 'error' in video_info:
            # ラベル生成に失敗した動画はスキップ
            continue
        
        video_name = video_info['name']
        
        # 特徴量ファイルを探す
        # ラベルのsource_pathから元動画のファイル名を取得
        source_path = Path(video_info['source_path'])
        source_stem = source_path.stem
        
        features_path = features_dir / f"{source_stem}_features.csv"
        labels_path = labels_dir / f"{video_name}_tracks.csv"
        
        if not features_path.exists():
            logger.warning(f"⚠️  特徴量ファイルが見つかりません: {source_stem}")
            missing_features_count += 1
            verification_results.append({
                'video_name': video_name,
                'source_stem': source_stem,
                'status': 'MISSING_FEATURES',
                'issues': ['特徴量ファイルが存在しない']
            })
            continue
        
        # 検証実行
        result = verify_single_video(features_path, labels_path, video_name)
        result['source_stem'] = source_stem
        verification_results.append(result)
        
        # ステータスをカウント
        if result['status'] == 'OK':
            ok_count += 1
            logger.info(f"✅ {video_name}: OK (特徴量: {result['features_timesteps']}, ラベル: {result['labels_timesteps']}, 差: {result['timestep_diff']})")
        elif result['status'] == 'WARNING':
            warning_count += 1
            logger.warning(f"⚠️  {video_name}: WARNING")
            for issue in result['issues']:
                logger.warning(f"    - {issue}")
        else:
            error_count += 1
            logger.error(f"❌ {video_name}: ERROR")
            for issue in result['issues']:
                logger.error(f"    - {issue}")
    
    # サマリー
    logger.info("\n" + "="*70)
    logger.info("検証完了")
    logger.info("="*70)
    logger.info(f"総動画数: {len(verification_results) + missing_features_count}")
    logger.info(f"OK: {ok_count}")
    logger.info(f"WARNING: {warning_count}")
    logger.info(f"ERROR: {error_count}")
    logger.info(f"特徴量なし: {missing_features_count}")
    
    # 統計情報
    if verification_results:
        timestep_diffs = [r['timestep_diff'] for r in verification_results if 'timestep_diff' in r]
        if timestep_diffs:
            logger.info(f"\nタイムステップ差の統計:")
            logger.info(f"  平均: {np.mean(timestep_diffs):.1f}")
            logger.info(f"  中央値: {np.median(timestep_diffs):.1f}")
            logger.info(f"  最大: {np.max(timestep_diffs):.0f}")
            logger.info(f"  最小: {np.min(timestep_diffs):.0f}")
        
        duration_diffs = [r['duration_diff'] for r in verification_results if 'duration_diff' in r]
        if duration_diffs:
            logger.info(f"\n動画時間差の統計:")
            logger.info(f"  平均: {np.mean(duration_diffs):.2f}秒")
            logger.info(f"  中央値: {np.median(duration_diffs):.2f}秒")
            logger.info(f"  最大: {np.max(duration_diffs):.2f}秒")
            logger.info(f"  最小: {np.min(duration_diffs):.2f}秒")
    
    # 結果を保存
    output_file = Path("data_integrity_report.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total': len(verification_results) + missing_features_count,
                'ok': ok_count,
                'warning': warning_count,
                'error': error_count,
                'missing_features': missing_features_count
            },
            'results': verification_results
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n詳細レポート保存: {output_file}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
