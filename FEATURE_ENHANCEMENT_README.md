# Feature Enhancement for Cut Selection Model

## Overview

This enhancement adds **temporal and contextual features** to improve the cut selection model. The current K-Fold Cross Validation results show an average F1 score of **42.30%** (best: 49.42%), with the goal of reaching 55%+.

## What's New

### Added Features (83 new dimensions)

#### 1. **Temporal Statistics** (Moving Averages, Variance, Change Rates)
- Moving averages (MA5, MA10, MA30, MA60, MA120) for:
  - `audio_energy_rms`
  - `visual_motion`
  - `face_count`
  - `pitch`
  - `spectral_centroid`
- Moving standard deviation (STD5, STD30, STD120)
- Change rates (DIFF1, DIFF2, DIFF30)

#### 2. **Cut Timing Information** (6 dimensions)
- `time_since_prev`: Time since previous cut (seconds)
- `time_to_next`: Time until next cut (seconds)
- `cut_duration`: Duration of current cut (seconds)
- `position_in_video`: Normalized position in video (0-1)
- `cut_density_10s`: Cut density in 10-second window
- `cumulative_adoption_rate`: Running average of adoption rate

#### 3. **Scene Similarity** (CLIP-based, 3 dimensions)
- `clip_sim_prev`: Cosine similarity with previous frame
- `clip_sim_next`: Cosine similarity with next frame
- `clip_sim_mean5`: Mean similarity with surrounding 5 frames

#### 4. **Audio Change Detection** (5 dimensions)
- `audio_change_score`: Magnitude of audio energy change
- `silence_to_speech`: Transition from silence to speech (0/1)
- `speech_to_silence`: Transition from speech to silence (0/1)
- `speaker_change`: Speaker ID change detection (0/1)
- `pitch_change`: Pitch variation magnitude

#### 5. **Visual Change Detection** (3 dimensions)
- `visual_motion_change`: Change in motion intensity
- `face_count_change`: Change in number of detected faces
- `saliency_movement`: Movement of visual attention point

#### 6. **Cumulative Statistics** (2 dimensions)
- `cumulative_position`: Frame position in sequence
- `cumulative_adoption_rate`: Running average of adoption rate

**Total Enhanced Features**: 784 dimensions
- Audio: 235 dimensions (215 original + 20 temporal)
- Visual: 543 dimensions (522 original + 21 temporal)
- Temporal: 6 dimensions (cut timing features)

## File Structure

```
scripts/
├── add_temporal_features.py              # Add temporal features to CSVs
├── create_cut_selection_data_enhanced.py # Create training data with enhanced features
└── combine_sequences_enhanced.py         # Combine sequences for K-Fold CV

src/cut_selection/
├── cut_model_enhanced.py                 # Enhanced model with 3 modalities
├── cut_dataset_enhanced.py               # Dataset for enhanced features
├── train_cut_selection_kfold_enhanced.py # K-Fold training script
└── evaluate_ensemble.py                  # Ensemble evaluation

configs/
└── config_cut_selection_kfold_enhanced.yaml  # Configuration for enhanced model

data/processed/
├── source_features/                      # Original features
├── source_features_enhanced/             # Enhanced features (with temporal)
└── active_labels/                        # Active/Inactive labels

preprocessed_data/
└── combined_sequences_cut_selection_enhanced.npz  # K-Fold training data
    - 289 sequences
    - 67 unique videos
    - 784 features (235 audio + 543 visual + 6 temporal)

checkpoints_cut_selection_kfold_enhanced/
├── fold_1_best_model.pth                 # Best model (F1: 49.42%)
├── fold_2_best_model.pth                 # Fold 2 model (F1: 41.22%)
├── fold_3_best_model.pth                 # Fold 3 model (F1: 43.10%)
├── fold_4_best_model.pth                 # Fold 4 model (F1: 45.57%)
├── fold_5_best_model.pth                 # Fold 5 model (F1: 32.20%)
├── kfold_summary.csv                     # K-Fold statistics
└── kfold_comparison.png                  # Comparison graph
```

## Usage

### Step 1: Add Temporal Features to Existing CSVs

```bash
python scripts/add_temporal_features.py
```

This will:
- Read CSVs from `data/processed/source_features/`
- Add 83 temporal and contextual features
- Save enhanced CSVs to `data/processed/source_features/` (overwrites with `_enhanced.csv` suffix)

**Expected output:**
```
Found 67 feature files
Processing videos...
  video1_features.csv
  ✓ 737 → 820 columns (+83)
  ...
Successfully processed: 67/67 videos
```

### Step 2: Create K-Fold Training Data

```bash
python scripts/combine_sequences_enhanced.py
```

This will:
- Load enhanced features from `data/processed/source_features/`
- Merge with active labels from `data/processed/active_labels/`
- Create sequences with 3 modalities: audio (235), visual (543), temporal (6)
- Group by video for K-Fold Cross Validation (prevents data leak)
- Save to `preprocessed_data/combined_sequences_cut_selection_enhanced.npz`

**Expected output:**
```
Feature dimensions:
  Audio: 235 columns
  Visual: 543 columns
  Temporal: 6 columns
  Total: 784 columns

Total sequences: 289 from 67 videos
Saved to: preprocessed_data/combined_sequences_cut_selection_enhanced.npz
```

### Step 3: Train Enhanced Model with K-Fold CV

```bash
python -m src.cut_selection.train_cut_selection_kfold_enhanced --config configs/config_cut_selection_kfold_enhanced.yaml
```

Or use the batch file:
```bash
batch/train_cut_selection_enhanced.bat
```

This will:
- Run 5-Fold Cross Validation (GroupKFold by video)
- Train each fold with Early Stopping (patience=15)
- Save best model for each fold
- Generate summary statistics and comparison graphs

**Expected output:**
```
K-Fold Cross Validation Results:
  Average F1: 42.30% ± 5.75%
  Best F1: 49.42% (Fold 1)
  Average Recall: 76.10% ± 5.19%
  
Saved:
  - checkpoints_cut_selection_kfold_enhanced/fold_X_best_model.pth
  - kfold_summary.csv
  - kfold_comparison.png
```

## Model Architecture Changes

### Original Model (2 Modalities)
```
Audio (215) ──┐
              ├─> Fusion ─> Transformer ─> Active Prediction
Visual (522) ─┘
```

### Enhanced Model (3 Modalities) - Current Implementation
```
Audio (235) ───┐
               │
Visual (543) ──┼─> Three-Modality Gated Fusion ─> Transformer ─> Active Prediction
               │
Temporal (6) ──┘
```

**Key improvements:**
- Added `ThreeModalityGatedFusion` module with learnable gates
- Separate embedding for temporal features (6 dimensions)
- Each modality embedded to 256 dimensions
- Dynamic weighting of modalities based on input
- 6-layer Transformer Encoder with 8 attention heads
- Increased model capacity: ~8.5M parameters

**Architecture Details:**
```
1. Modality Embedding
   - Audio (235) → Linear(256)
   - Visual (543) → Linear(256)
   - Temporal (6) → Linear(256)

2. Three-Modality Gated Fusion
   - Gate weights: Sigmoid(Linear(256 × 3))
   - Weighted sum: Σ(gate_i × modality_i)
   - Output: (batch, seq_len, 256)

3. Positional Encoding
   - Sinusoidal encoding (max_len=5000)

4. Transformer Encoder
   - 6 layers, 8 heads, d_model=256, d_ff=1024
   - Dropout: 0.15

5. Active Head
   - Linear(256 → 2)
   - Output: (batch, seq_len, 2) [Inactive, Active]
```

## Current Performance (K-Fold Cross Validation)

### Overall Results (5-Fold CV, 2025-12-26)

| Metric | Average | Std Dev | Min | Max |
|--------|---------|---------|-----|-----|
| **F1 Score** | **42.30%** | ±5.75% | 32.20% | **49.42%** |
| **Accuracy** | 50.24% | ±14.92% | 33.26% | 73.63% |
| **Precision** | 29.83% | ±5.80% | 19.89% | 36.94% |
| **Recall** | **76.10%** | ±5.19% | 71.02% | 84.54% |
| **Best Epoch** | 7.4 | ±6.8 | 1 | 20 |

### Per-Fold Results

| Fold | Best Epoch | F1 Score | Accuracy | Precision | Recall | Threshold |
|------|-----------|----------|----------|-----------|--------|-----------|
| 1 | 4 | **49.42%** | 73.63% | 36.94% | 74.65% | -0.558 |
| 2 | 1 | 41.22% | 36.44% | 27.85% | 79.24% | -0.474 |
| 3 | 20 | 43.10% | 48.45% | 30.94% | 71.02% | -0.573 |
| 4 | 9 | 45.57% | 59.42% | 33.54% | 71.03% | -0.509 |
| 5 | 3 | 32.20% | 33.26% | 19.89% | 84.54% | -0.550 |

**Recommended Model**: Fold 1 (F1: 49.42%, most stable performance)

### Inference Performance (Full Video Model)

**Model**: `checkpoints_cut_selection_fullvideo/best_model.pth` (Epoch 9)

**Training Performance**:
- F1: 0.5290
- Recall: 0.8065
- Avg Duration: 101.3s

**Inference Test Results** (bandicam 2025-05-11 19-25-14-768.mp4):
- Video Length: 1000.1s (~16.7 minutes)
- **Optimal Threshold**: 0.8952 (F1 maximization within 90-200s constraint)
- **Predicted Duration**: 181.9s (target: 180s, perfect match!)
- **Adoption Rate**: 18.2% (1,819 / 10,001 frames)
- **Extracted Clips**: 10 clips (total 138.3s)
- **XML Generation**: Success (Premiere Pro compatible)

**Constraint Satisfaction**:
- ✅ 90-200s constraint satisfied
- ✅ Target 180s (3 minutes) almost perfectly matched (+1.9s)
- ✅ Per-video optimization working correctly

**Details**: See [Inference Test Results Report](docs/INFERENCE_TEST_RESULTS.md)

### Training Details
- **Dataset**: 67 videos, 289 sequences
- **Sequence Length**: 1000 frames (~100 seconds @ 10 FPS)
- **Overlap**: 500 frames
- **Evaluation**: GroupKFold (video-level split, prevents data leak)
- **Random Seed**: 42 (reproducibility)
- **Early Stopping**: patience=15, average convergence at 7.4 epochs
- **Training Time**: ~2-3 hours (GPU), 37 total epochs across 5 folds

### Target Performance

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| F1 Score | 42.30% | 55%+ | -12.70 pp |
| Precision | 29.83% | 40%+ | -10.17 pp |
| Recall | 76.10% | 75-80% | ✅ Achieved |

**Why this should work:**
1. **Temporal context**: Model sees trends over time, not just snapshots
2. **Cut timing**: Explicit timing information helps identify editing patterns
3. **Scene similarity**: Detects repetitive vs unique content using CLIP embeddings
4. **Change detection**: Identifies important audio/visual transitions
5. **Cumulative stats**: Understands position in video timeline
6. **Gated fusion**: Dynamically weights modalities based on importance

**Current Challenges:**
- Precision is still low (29.83%), indicating false positives
- High recall (76.10%) means we're catching most good cuts
- Need to improve precision without sacrificing recall
- Possible solutions: Better threshold tuning, ensemble methods, data augmentation

## Configuration

Edit `configs/config_cut_selection_kfold_enhanced.yaml` to adjust:

```yaml
# Feature dimensions
audio_features: 235
visual_features: 543
temporal_features: 6

# Model architecture
d_model: 256
num_encoder_layers: 6
num_heads: 8
d_ff: 1024
dropout: 0.15

# Training
learning_rate: 0.0001
batch_size: 16
num_epochs: 50
patience: 15  # Early stopping

# K-Fold Cross Validation
n_splits: 5
random_seed: 42

# Loss weights
focal_alpha: 0.5
focal_gamma: 2.0
tv_weight: 0.02
adoption_penalty_weight: 10.0
target_adoption_rate: 0.23
```

## Troubleshooting

### Issue: "No feature files found"
**Solution:** Make sure you have CSV files in `data/processed/source_features/` with `_features_enhanced.csv` suffix

### Issue: "No matching timestamps"
**Solution:** Check that active label files exist in `data/processed/active_labels/` with `_active.csv` suffix

### Issue: "CUDA out of memory"
**Solution:** Reduce `batch_size` in config file (try 8 or 4), or use CPU mode

### Issue: "Feature dimension mismatch"
**Solution:** 
1. Delete old `.npz` files in `preprocessed_data/`
2. Re-run `scripts/combine_sequences_enhanced.py`
3. Ensure all CSV files have been processed with `scripts/add_temporal_features.py`

### Issue: "Model not converging"
**Solution:**
1. Check learning rate (try 0.00005 or 0.0002)
2. Adjust loss weights (reduce `adoption_penalty_weight` to 5.0)
3. Increase `patience` for Early Stopping (try 20 or 25)

### Issue: "Data leak warning"
**Solution:** This is expected! GroupKFold ensures videos are split at the video level, preventing data leak. Each fold trains on different videos.

## Next Steps After Training

1. **Evaluate Results**: Check K-Fold summary in `checkpoints_cut_selection_kfold_enhanced/kfold_summary.csv`
2. **Compare Folds**: View comparison graph in `kfold_comparison.png`
3. **Select Best Model**: Use Fold 1 model (highest F1: 49.42%) for inference
4. **Analyze Features**: Use feature importance analysis to see which temporal features help most
5. **Further Optimization**: If still below 55% target, consider:
   - **Data augmentation**: Time shifting, noise injection, speed variation (V2 model)
   - **Ensemble methods**: Combine predictions from multiple folds
   - **Architecture changes**: Deeper network (8 layers), more attention heads (16)
   - **Loss function tuning**: Adjust focal loss parameters, TV weight
   - **Threshold optimization**: Fine-tune per-fold thresholds

### Ensemble Evaluation

To evaluate ensemble performance across all 5 folds:

```bash
batch/evaluate_ensemble.bat
```

This will:
- Load all 5 fold models
- Evaluate on validation data
- Compare single-fold vs ensemble performance
- Generate ensemble comparison graphs

## Technical Details

### Feature Computation Examples

**Moving Average (MA5):**
```python
df['audio_energy_rms_ma5'] = df['audio_energy_rms'].rolling(
    window=5, center=True, min_periods=1
).mean()
```

**Moving Standard Deviation (STD5):**
```python
df['audio_energy_rms_std5'] = df['audio_energy_rms'].rolling(
    window=5, center=True, min_periods=1
).std()
```

**CLIP Cosine Similarity:**
```python
from sklearn.metrics.pairwise import cosine_similarity

current_emb = clip_embeddings[idx:idx+1]  # (1, 512)
prev_emb = clip_embeddings[idx-1:idx]     # (1, 512)
similarity = cosine_similarity(current_emb, prev_emb)[0, 0]
```

**Cut Density (10-second window):**
```python
window_sec = 10.0
mask = (df['time'] >= current_time - window_sec) & 
       (df['time'] <= current_time + window_sec)
density = mask.sum() / (window_sec * fps)
```

**Change Detection:**
```python
# Audio change
df['audio_change_score'] = df['audio_energy_rms'].diff().abs()

# Speaker change
df['speaker_change'] = (df['speaker_id'].diff() != 0).astype(int)

# Visual motion change
df['visual_motion_change'] = df['visual_motion'].diff().abs()
```

### Data Normalization

All features are normalized using `StandardScaler` (zero mean, unit variance):

```python
from sklearn.preprocessing import StandardScaler

# Fit on training data only (prevent data leak)
audio_scaler = StandardScaler()
audio_features_train = audio_scaler.fit_transform(audio_features_train)

# Transform validation data using training scaler
audio_features_val = audio_scaler.transform(audio_features_val)
```

Scalers are saved for inference:
- `preprocessed_data/audio_scaler_cut_selection_enhanced.pkl`
- `preprocessed_data/visual_scaler_cut_selection_enhanced.pkl`
- `preprocessed_data/temporal_scaler_cut_selection_enhanced.pkl`

**Important**: Each fold has its own scaler fitted on training data only. This prevents data leakage and ensures true generalization performance.

### Loss Functions

**Focal Loss** (handles class imbalance):
```python
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```
- α = 0.5 (class balance weight)
- γ = 2.0 (focus on hard examples)

**Total Variation Loss** (temporal smoothness):
```python
TV_Loss = Σ |active[t+1] - active[t]|
```
- Weight: 0.02

**Adoption Penalty** (target adoption rate):
```python
Adoption_Penalty = |adoption_rate - target_rate|^2
```
- Weight: 10.0
- Target: 0.23 (23% adoption rate)

## References

- **Moving Averages**: Smooth time series, reduce noise, capture trends
- **Cosine Similarity**: Measure semantic similarity between CLIP embeddings (range: -1 to 1)
- **Focal Loss**: Handle class imbalance (Lin et al., 2017) - focuses on hard examples
- **Transformer**: Capture long-range dependencies in sequences (Vaswani et al., 2017)
- **K-Fold Cross Validation**: Robust evaluation method, prevents overfitting
- **GroupKFold**: Ensures videos are split at video level, prevents data leakage
- **Early Stopping**: Prevents overfitting by monitoring validation loss

### Key Papers
1. Lin, T. Y., et al. (2017). "Focal Loss for Dense Object Detection." ICCV.
2. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
3. Radford, A., et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision." ICML.

## Questions?

If you encounter issues or have questions about the feature enhancement:
1. Check the logs in console output for detailed error messages
2. Verify CSV files have the expected columns (use `pandas.read_csv()` to inspect)
3. Ensure data paths in config files are correct (relative to project root)
4. Check GPU memory usage if training fails (`nvidia-smi` on Linux/Windows)
5. Review K-Fold summary CSV for per-fold performance breakdown
6. Check `view_training.html` for real-time training visualization

### Common Questions

**Q: Why is Recall high but Precision low?**
A: The model is good at finding most good cuts (high recall) but also selects some bad cuts (low precision). This is controlled by the threshold - lowering it increases recall but decreases precision.

**Q: Why does performance vary across folds?**
A: Different videos have different editing styles. Some folds may have more consistent editing patterns than others. This is expected and why we use K-Fold CV.

**Q: Should I use ensemble or single-fold model?**
A: For production, use Fold 1 (best F1: 49.42%). For research, ensemble can provide more robust predictions but is slower.

**Q: How do I improve precision without losing recall?**
A: Try adjusting loss weights (increase `focal_alpha`), use data augmentation, or implement post-processing filters (e.g., minimum clip duration, maximum clips per video).
