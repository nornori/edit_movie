# Feature Enhancement for Cut Selection Model

## Overview

This enhancement adds **temporal and contextual features** to improve the cut selection model's F1 score from ~45% to the target of 55%.

## What's New

### Added Features (~60-80 new dimensions)

#### 1. **Temporal Statistics** (Moving Averages, Variance, Change Rates)
- Moving averages (3, 5, 10 frame windows) for:
  - `audio_energy_rms`
  - `visual_motion`
  - `face_count`
  - `pitch_f0`
  - `spectral_centroid`
  - `scene_change`
- Moving variance (5 frame window)
- Change rates (1st and 2nd order differences)

#### 2. **Cut Timing Information**
- `time_since_prev`: Time since previous cut
- `time_to_next`: Time until next cut
- `cut_duration`: Duration of current cut
- `position_in_video`: Normalized position (0-1)
- `cut_density_10s`: Cut density in 10-second window

#### 3. **Scene Similarity** (CLIP-based)
- `clip_sim_prev`: Similarity with previous frame
- `clip_sim_next`: Similarity with next frame
- `clip_sim_mean5`: Mean similarity with surrounding 5 frames

#### 4. **Audio Change Detection**
- `audio_change_score`: Magnitude of audio energy change
- `silence_to_speech`: Transition from silence to speech
- `speech_to_silence`: Transition from speech to silence
- `speaker_change`: Speaker ID change detection
- `pitch_change`: Pitch variation

#### 5. **Visual Change Detection**
- `visual_motion_change`: Change in motion intensity
- `face_count_change`: Change in number of faces
- `saliency_movement`: Movement of attention point

#### 6. **Cumulative Statistics**
- `cumulative_position`: Frame position in sequence
- `cumulative_adoption_rate`: Running average of adoption rate

## File Structure

```
scripts/
├── add_temporal_features.py              # Add temporal features to CSVs
├── create_cut_selection_data_enhanced.py # Create training data with enhanced features
└── ...

src/cut_selection/
├── cut_model_enhanced.py                 # Model with 3 modalities (audio, visual, temporal)
├── train_cut_selection_kfold_enhanced.py # Training script for enhanced model
└── ...

configs/
└── config_cut_selection_kfold_enhanced.yaml  # Configuration for enhanced model

data/processed/
├── source_features/                      # Original features
└── source_features_enhanced/             # Enhanced features (output)

preprocessed_data/
├── train_sequences_cut_selection_enhanced.npz  # Enhanced training data
└── val_sequences_cut_selection_enhanced.npz    # Enhanced validation data
```

## Usage

### Step 1: Add Temporal Features to Existing CSVs

```bash
python scripts/add_temporal_features.py
```

This will:
- Read CSVs from `data/processed/source_features/`
- Add ~60-80 temporal and contextual features
- Save enhanced CSVs to `data/processed/source_features_enhanced/`

**Expected output:**
```
Found 68 feature files
Processing videos...
  video1_features.csv
  ✓ 737 → 817 columns (+80)
  ...
Successfully processed: 68/68 videos
```

### Step 2: Create Enhanced Training Data

```bash
python scripts/create_cut_selection_data_enhanced.py
```

This will:
- Load enhanced features from `data/processed/source_features_enhanced/`
- Merge with active labels
- Create sequences with 3 modalities: audio, visual, temporal
- Split into train/val sets
- Save to `preprocessed_data/`

**Expected output:**
```
Feature dimensions:
  Audio: 250 columns
  Visual: 570 columns
  Temporal: 7 columns
  Total: 827 columns

Train sequences: 241 from 54 videos
Val sequences: 60 from 14 videos
```

### Step 3: Train Enhanced Model

```bash
python src/cut_selection/train_cut_selection_kfold_enhanced.py --config configs/config_cut_selection_kfold_enhanced.yaml
```

Or use the batch file:
```bash
train_cut_selection_enhanced.bat
```

### Quick Start (All Steps)

```bash
enhance_features.bat
```

This runs both Step 1 and Step 2 automatically.

## Model Architecture Changes

### Original Model (2 Modalities)
```
Audio (215) ──┐
              ├─> Fusion ─> Transformer ─> Active Prediction
Visual (522) ─┘
```

### Enhanced Model (3 Modalities)
```
Audio (250) ───┐
               │
Visual (570) ──┼─> 3-Way Fusion ─> Transformer ─> Active Prediction
               │
Temporal (7) ──┘
```

**Key improvements:**
- Added `ThreeModalityFusion` module with gated fusion
- Separate embedding for temporal features
- Increased model capacity to handle richer features

## Expected Performance Improvement

| Metric | Before | Target | Improvement |
|--------|--------|--------|-------------|
| F1 Score | 45.52% | 55%+ | +9.48 pp |
| Precision | ~31% | ~40%+ | +9 pp |
| Recall | ~83% | ~75-80% | Balanced |

**Why this should work:**
1. **Temporal context**: Model now sees trends, not just snapshots
2. **Cut timing**: Explicit timing information helps identify patterns
3. **Scene similarity**: Detects repetitive vs unique content
4. **Change detection**: Identifies important transitions
5. **Cumulative stats**: Understands position in video timeline

## Configuration

Edit `configs/config_cut_selection_kfold_enhanced.yaml` to adjust:

```yaml
# Feature dimensions (auto-detected, but can override)
audio_features: 250
visual_features: 570
temporal_features: 7

# Model size
d_model: 256
num_encoder_layers: 6

# Training
learning_rate: 0.0001
batch_size: 16
dropout: 0.15

# Loss weights
focal_alpha: 0.5
focal_gamma: 2.0
tv_weight: 0.02
adoption_penalty_weight: 10.0
```

## Troubleshooting

### Issue: "No feature files found"
**Solution:** Make sure you have CSV files in `data/processed/source_features/`

### Issue: "No matching timestamps"
**Solution:** Check that active label files exist in `data/processed/active_labels/`

### Issue: "CUDA out of memory"
**Solution:** Reduce `batch_size` in config file (try 8 or 4)

### Issue: "Feature dimension mismatch"
**Solution:** Delete old `.npz` files and regenerate with enhanced features

## Next Steps After Training

1. **Compare Results**: Check if F1 score improved
2. **Analyze Features**: Use feature importance analysis to see which new features help most
3. **Further Optimization**: If still below 55%, consider:
   - Data augmentation
   - Ensemble methods
   - Architecture changes (deeper network, attention mechanisms)

## Technical Details

### Feature Computation

**Moving Average (window=5):**
```python
df['audio_energy_rms_ma5'] = df['audio_energy_rms'].rolling(
    window=5, center=True, min_periods=1
).mean()
```

**CLIP Similarity:**
```python
current_emb = clip_embeddings[idx:idx+1]
prev_emb = clip_embeddings[idx-1:idx]
similarity = cosine_similarity(current_emb, prev_emb)[0, 0]
```

**Cut Density:**
```python
window_sec = 10.0
mask = (df['time'] >= current_time - window_sec) & 
       (df['time'] <= current_time + window_sec)
density = mask.sum() / max_density
```

### Data Normalization

All features are normalized using `StandardScaler`:
```python
audio_scaler = StandardScaler()
audio_features = audio_scaler.fit_transform(audio_features)
```

Scalers are saved for inference:
- `audio_scaler_cut_selection_enhanced.pkl`
- `visual_scaler_cut_selection_enhanced.pkl`
- `temporal_scaler_cut_selection_enhanced.pkl`

## References

- **Moving Averages**: Smooth time series, reduce noise
- **Cosine Similarity**: Measure semantic similarity between embeddings
- **Focal Loss**: Handle class imbalance (23% adoption rate)
- **Transformer**: Capture long-range dependencies in sequences

## Questions?

If you encounter issues or have questions about the feature enhancement:
1. Check the logs in console output
2. Verify CSV files have the expected columns
3. Ensure data paths in config files are correct
4. Check GPU memory usage if training fails
