"""
Generate Premiere Pro XML from inference results
"""
import torch
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import argparse
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_features(video_name):
    """Load features from CSV"""
    csv_path = Path(f"temp_features/{video_name}_features_enhanced.csv")
    
    if not csv_path.exists():
        logger.error(f"❌ Feature file not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    logger.info(f"✅ Loaded CSV: {len(df)} frames")
    
    # Define feature columns
    audio_cols = []
    for col in ['audio_energy_rms', 'audio_is_speaking', 'silence_duration_ms', 'speaker_id']:
        if col in df.columns:
            audio_cols.append(col)
    
    speaker_emb_cols = [col for col in df.columns if col.startswith('speaker_emb_')]
    audio_cols.extend(speaker_emb_cols)
    
    acoustic_cols = [col for col in df.columns if col.startswith(('pitch_', 'spectral_', 'zcr', 'mfcc_'))]
    audio_cols.extend(acoustic_cols)
    
    if 'text_is_active' in df.columns:
        audio_cols.append('text_is_active')
    if 'telop_active' in df.columns:
        audio_cols.append('telop_active')
    
    audio_temporal_cols = [col for col in df.columns if any(col.startswith(prefix) for prefix in [
        'audio_energy_rms_ma', 'audio_change_score', 'speaker_change'
    ])]
    audio_cols.extend(audio_temporal_cols)
    
    visual_cols = []
    for col in ['scene_change', 'visual_motion', 'saliency_x', 'saliency_y']:
        if col in df.columns:
            visual_cols.append(col)
    
    face_cols = [col for col in df.columns if col.startswith('face_')]
    visual_cols.extend(face_cols)
    
    clip_cols = [col for col in df.columns if col.startswith('clip_') and col[5:].isdigit()]
    visual_cols.extend(sorted(clip_cols, key=lambda x: int(x.split('_')[1])))
    
    visual_temporal_cols = [col for col in df.columns if any(col.startswith(prefix) for prefix in [
        'visual_motion_ma', 'visual_motion_change', 'face_count_change', 'clip_sim_'
    ])]
    visual_cols.extend(visual_temporal_cols)
    
    temporal_cols = ['time_since_prev', 'time_to_next', 'cut_duration', 
                     'position_in_video', 'cut_density_10s', 'cumulative_adoption_rate']
    temporal_cols = [col for col in temporal_cols if col in df.columns]
    
    audio_features = df[audio_cols].values.astype(np.float32)
    visual_features = df[visual_cols].values.astype(np.float32)
    temporal_features = df[temporal_cols].values.astype(np.float32)
    
    audio_features = np.nan_to_num(audio_features, nan=0.0)
    visual_features = np.nan_to_num(visual_features, nan=0.0)
    temporal_features = np.nan_to_num(temporal_features, nan=0.0)
    
    return {
        'audio': audio_features,
        'visual': visual_features,
        'temporal': temporal_features,
        'df': df
    }


def load_model():
    """Load the trained fullvideo model"""
    model_path = Path("checkpoints_cut_selection_fullvideo/best_model.pth")
    
    if not model_path.exists():
        logger.error(f"❌ Model file not found: {model_path}")
        return None
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    logger.info(f"✅ Model loaded (Epoch {checkpoint['epoch']})")
    
    return checkpoint


def run_inference(features, checkpoint):
    """Run model inference"""
    from src.cut_selection.models.cut_model_enhanced import EnhancedCutSelectionModel
    
    config = checkpoint['config']
    
    model = EnhancedCutSelectionModel(
        audio_features=config['audio_features'],
        visual_features=config['visual_features'],
        temporal_features=config['temporal_features'],
        d_model=config.get('d_model', 256),
        nhead=config.get('nhead', 8),
        num_encoder_layers=config.get('num_encoder_layers', 6),
        dim_feedforward=config.get('dim_feedforward', 1024),
        dropout=config.get('dropout', 0.1)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    audio_features = features['audio']
    visual_features = features['visual']
    temporal_features = features['temporal']
    
    expected_audio = config['audio_features']
    expected_visual = config['visual_features']
    expected_temporal = config['temporal_features']
    
    if audio_features.shape[1] != expected_audio:
        if audio_features.shape[1] < expected_audio:
            padding = np.zeros((len(audio_features), expected_audio - audio_features.shape[1]), dtype=np.float32)
            audio_features = np.concatenate([audio_features, padding], axis=1)
        else:
            audio_features = audio_features[:, :expected_audio]
    
    if visual_features.shape[1] != expected_visual:
        if visual_features.shape[1] < expected_visual:
            padding = np.zeros((len(visual_features), expected_visual - visual_features.shape[1]), dtype=np.float32)
            visual_features = np.concatenate([visual_features, padding], axis=1)
        else:
            visual_features = visual_features[:, :expected_visual]
    
    if temporal_features.shape[1] != expected_temporal:
        if temporal_features.shape[1] < expected_temporal:
            padding = np.zeros((len(temporal_features), expected_temporal - temporal_features.shape[1]), dtype=np.float32)
            temporal_features = np.concatenate([temporal_features, padding], axis=1)
        else:
            temporal_features = temporal_features[:, :expected_temporal]
    
    import pickle
    scaler_dir = Path('preprocessed_data')
    
    audio_scaler_path = scaler_dir / 'audio_scaler_cut_selection_enhanced_fullvideo.pkl'
    visual_scaler_path = scaler_dir / 'visual_scaler_cut_selection_enhanced_fullvideo.pkl'
    temporal_scaler_path = scaler_dir / 'temporal_scaler_cut_selection_enhanced_fullvideo.pkl'
    
    if audio_scaler_path.exists():
        with open(audio_scaler_path, 'rb') as f:
            audio_scaler = pickle.load(f)
        audio_features = audio_scaler.transform(audio_features)
    
    if visual_scaler_path.exists():
        with open(visual_scaler_path, 'rb') as f:
            visual_scaler = pickle.load(f)
        visual_features = visual_scaler.transform(visual_features)
    
    if temporal_scaler_path.exists():
        with open(temporal_scaler_path, 'rb') as f:
            temporal_scaler = pickle.load(f)
        temporal_features = temporal_scaler.transform(temporal_features)
    
    audio_tensor = torch.from_numpy(audio_features).float().unsqueeze(0)
    visual_tensor = torch.from_numpy(visual_features).float().unsqueeze(0)
    temporal_tensor = torch.from_numpy(temporal_features).float().unsqueeze(0)
    
    logger.info(f"   Running inference on {len(audio_features)} frames...")
    with torch.no_grad():
        outputs = model(audio_tensor, visual_tensor, temporal_tensor)
        active_logits = outputs['active']
        
        probs = torch.softmax(active_logits, dim=-1)
        active_probs = probs[..., 1].squeeze(0).numpy()
        inactive_probs = probs[..., 0].squeeze(0).numpy()
        
        confidence_scores = active_probs - inactive_probs
    
    logger.info(f"✅ Inference completed")
    
    return {
        'confidence_scores': confidence_scores,
        'active_probs': active_probs,
        'df': features['df']
    }


def optimize_threshold(predictions, min_duration=90.0, max_duration=200.0):
    """Find optimal threshold"""
    confidence_scores = predictions['confidence_scores']
    df = predictions['df']
    
    fps = 10.0
    video_duration = len(confidence_scores) / fps
    
    logger.info(f"   Video duration: {video_duration:.1f}s")
    
    if video_duration < min_duration:
        logger.info(f"   Video < {min_duration}s, accepting all frames")
        best_threshold = confidence_scores.min() - 1.0
        active_binary = np.ones(len(confidence_scores), dtype=int)
        
        return {
            'threshold': best_threshold,
            'active_binary': active_binary,
            'duration': video_duration
        }
    
    thresholds = np.linspace(confidence_scores.min(), confidence_scores.max(), 100)
    
    best_threshold = 0.0
    best_metrics = None
    
    for thresh in thresholds:
        active_binary = (confidence_scores >= thresh).astype(int)
        pred_duration = active_binary.sum() / fps
        
        if pred_duration < min_duration or pred_duration > max_duration:
            continue
        
        if best_threshold == 0.0:
            best_threshold = thresh
            best_metrics = {
                'duration': pred_duration,
                'active_ratio': active_binary.sum() / len(active_binary)
            }
    
    if best_metrics is None:
        logger.warning(f"   No threshold satisfies constraints, using closest to 150s")
        target_duration = 150.0
        best_diff = float('inf')
        
        for thresh in thresholds:
            active_binary = (confidence_scores >= thresh).astype(int)
            pred_duration = active_binary.sum() / fps
            diff = abs(pred_duration - target_duration)
            
            if diff < best_diff:
                best_diff = diff
                best_threshold = thresh
                best_metrics = {
                    'duration': pred_duration,
                    'active_ratio': active_binary.sum() / len(active_binary)
                }
    
    active_binary = (confidence_scores >= best_threshold).astype(int)
    
    logger.info(f"✅ Optimal threshold: {best_threshold:.4f}")
    logger.info(f"   Duration: {best_metrics['duration']:.1f}s")
    logger.info(f"   Active ratio: {best_metrics['active_ratio']*100:.1f}%")
    
    return {
        'threshold': best_threshold,
        'active_binary': active_binary,
        **best_metrics
    }


def extract_clips(active_binary, min_clip_duration=3.0):
    """Extract clips from binary predictions"""
    fps = 10.0
    
    clips = []
    in_clip = False
    start_idx = 0
    
    for i, active in enumerate(active_binary):
        if active == 1 and not in_clip:
            start_idx = i
            in_clip = True
        elif active == 0 and in_clip:
            end_idx = i
            start_time = start_idx / fps
            end_time = end_idx / fps
            if (end_time - start_time) >= min_clip_duration:
                clips.append((start_time, end_time))
            in_clip = False
    
    if in_clip:
        end_idx = len(active_binary)
        start_time = start_idx / fps
        end_time = end_idx / fps
        if (end_time - start_time) >= min_clip_duration:
            clips.append((start_time, end_time))
    
    total_duration = sum(e - s for s, e in clips)
    
    logger.info(f"✅ Extracted {len(clips)} clips")
    logger.info(f"   Total duration: {total_duration:.1f}s")
    
    return clips


def generate_xml(clips, video_path, output_path):
    """Generate Premiere Pro XML"""
    from src.inference.direct_xml_generator import create_premiere_xml_direct
    
    fps = 10.0
    tracks_data = []
    for start_time, end_time in clips:
        tracks_data.append({
            'start_frame': int(start_time * fps),
            'end_frame': int(end_time * fps)
        })
    
    total_frames = int(max(end_time for _, end_time in clips) * fps)
    video_name = Path(video_path).stem
    
    logger.info(f"   Generating XML for {len(clips)} clips...")
    
    xml_path = create_premiere_xml_direct(
        video_path=video_path,
        video_name=video_name,
        total_frames=total_frames,
        fps=fps,
        tracks_data=tracks_data,
        telops=[],
        output_path=output_path,
        ai_telops=[]
    )
    
    logger.info(f"✅ XML generated: {xml_path}")
    
    return xml_path


def main():
    parser = argparse.ArgumentParser(description='Generate XML from inference')
    parser.add_argument('video_path', type=str, help='Path to video file')
    parser.add_argument('--output', type=str, default=None, help='Output XML path')
    
    args = parser.parse_args()
    
    video_path = args.video_path
    video_name = Path(video_path).stem
    output_xml = args.output if args.output else f"outputs/{video_name}_output.xml"
    
    logger.info("🚀 Generating XML from Full Video Model")
    logger.info(f"   Video: {video_path}")
    logger.info(f"   Output: {output_xml}")
    
    # Load features
    logger.info("\n" + "="*80)
    logger.info("STEP 1: Load Features")
    logger.info("="*80)
    features = load_features(video_name)
    if features is None:
        sys.exit(1)
    
    # Load model
    logger.info("\n" + "="*80)
    logger.info("STEP 2: Load Model")
    logger.info("="*80)
    checkpoint = load_model()
    if checkpoint is None:
        sys.exit(1)
    
    # Run inference
    logger.info("\n" + "="*80)
    logger.info("STEP 3: Run Inference")
    logger.info("="*80)
    predictions = run_inference(features, checkpoint)
    if predictions is None:
        sys.exit(1)
    
    # Optimize threshold
    logger.info("\n" + "="*80)
    logger.info("STEP 4: Optimize Threshold")
    logger.info("="*80)
    result = optimize_threshold(predictions, min_duration=90.0, max_duration=200.0)
    if result is None:
        sys.exit(1)
    
    # Extract clips
    logger.info("\n" + "="*80)
    logger.info("STEP 5: Extract Clips")
    logger.info("="*80)
    clips = extract_clips(result['active_binary'], min_clip_duration=3.0)
    if not clips:
        logger.error("❌ No clips extracted")
        sys.exit(1)
    
    # Generate XML
    logger.info("\n" + "="*80)
    logger.info("STEP 6: Generate XML")
    logger.info("="*80)
    xml_path = generate_xml(clips, video_path, output_xml)
    
    logger.info("\n" + "="*80)
    logger.info("✅ COMPLETED")
    logger.info("="*80)
    logger.info(f"   Clips: {len(clips)}")
    logger.info(f"   Duration: {result['duration']:.1f}s")
    logger.info(f"   XML: {xml_path}")


if __name__ == "__main__":
    main()
