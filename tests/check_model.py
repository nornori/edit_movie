"""Check model details"""
import torch
import pandas as pd

# Load checkpoint
checkpoint = torch.load('checkpoints_cut_selection_kfold_enhanced/fold_1_best_model.pth', 
                        map_location='cpu', weights_only=False)

print('=' * 80)
print('MODEL CHECKPOINT INFO')
print('=' * 80)
print(f'Fold: {checkpoint.get("fold", "N/A")}')
print(f'Epoch: {checkpoint.get("epoch", "N/A")}')

print()
print('=' * 80)
print('VALIDATION METRICS')
print('=' * 80)
val_metrics = checkpoint.get('val_metrics', {})
for k, v in val_metrics.items():
    if isinstance(v, float):
        print(f'{k}: {v:.4f}')
    else:
        print(f'{k}: {v}')

print()
print('=' * 80)
print('MODEL CONFIG')
print('=' * 80)
config = checkpoint.get('config', {})
for k, v in config.items():
    print(f'{k}: {v}')

print()
print('=' * 80)
print('K-FOLD SUMMARY')
print('=' * 80)
df = pd.read_csv('checkpoints_cut_selection_kfold_enhanced/kfold_summary.csv')
print(df.to_string())

print()
print('=' * 80)
print('ANALYSIS')
print('=' * 80)
print(f'Best threshold: {checkpoint.get("val_metrics", {}).get("best_threshold", "N/A")}')
print(f'This threshold was optimized on validation data')
print(f'')
print(f'Inference results:')
print(f'  Video 1: 99.8% above threshold (2 clips, 130s)')
print(f'  Video 2: 100% above threshold (1 clip, 1000s)')
print(f'')
print(f'Possible issues:')
print(f'  1. Threshold too low (-0.558) - most frames pass')
print(f'  2. Model overfitting to training data distribution')
print(f'  3. Inference videos are genuinely high quality')
print(f'  4. Feature extraction differences between train/inference')
