"""Check model parameters from checkpoint"""
import torch
from pathlib import Path

checkpoint_path = Path('checkpoints_cut_selection_kfold_enhanced/fold_1_best_model.pth')
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print('='*80)
print('60.80% F1 Score - Model Configuration')
print('='*80)
print()

print('Model Architecture:')
print('-'*80)
config = checkpoint['config']
for key in sorted(config.keys()):
    value = config[key]
    print(f'  {key:30s}: {value}')

print()
print('Validation Metrics (Fold 1 - Best Individual Model):')
print('-'*80)
val_metrics = checkpoint['val_metrics']
for key in sorted(val_metrics.keys()):
    value = val_metrics[key]
    if isinstance(value, float):
        print(f'  {key:30s}: {value:.6f}')
    else:
        print(f'  {key:30s}: {value}')

print()
print('Training Info:')
print('-'*80)
print(f'  Best Epoch: {checkpoint["epoch"]}')

print()
print('='*80)
print('Ensemble Configuration (60.80% F1):')
print('='*80)
print('  Strategy: Soft Voting (probability averaging)')
print('  Number of Models: 5 (all folds)')
print('  Optimal Threshold: -0.4477')
print('  Min Recall Constraint: 0.71 (71%)')
print()
print('Results:')
print('  F1 Score: 60.80%')
print('  Accuracy: 78.69%')
print('  Precision: 52.90%')
print('  Recall: 71.45%')
print('  Specificity: 80.87%')
