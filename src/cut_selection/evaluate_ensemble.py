"""
Evaluate Ensemble Model on K-Fold Cross Validation

Tests all three voting strategies and compares with individual models.
"""
import torch
from torch.utils.data import DataLoader
import yaml
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.cut_selection.cut_dataset_enhanced import EnhancedCutSelectionDataset
from src.cut_selection.ensemble_predictor import EnsemblePredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_all_strategies(checkpoint_dir: str, data_path: str, device: str = 'cuda'):
    """
    Evaluate all ensemble strategies and compare with individual models
    
    Args:
        checkpoint_dir: Directory with fold models
        data_path: Path to dataset
        device: Device to use
    """
    logger.info("="*80)
    logger.info("ENSEMBLE MODEL EVALUATION")
    logger.info("="*80)
    
    # Load dataset
    logger.info(f"\nLoading dataset from {data_path}")
    dataset = EnhancedCutSelectionDataset(data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Load K-Fold summary to get individual model performance
    checkpoint_path = Path(checkpoint_dir)
    summary_path = checkpoint_path / 'kfold_summary.csv'
    
    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        individual_f1_scores = summary_df[summary_df['fold'] != 'Mean ± Std']['best_val_f1'].astype(float).tolist()
        mean_individual_f1 = np.mean(individual_f1_scores)
        std_individual_f1 = np.std(individual_f1_scores)
        
        logger.info(f"\n📊 Individual Model Performance:")
        logger.info(f"   Mean F1: {mean_individual_f1:.4f} ± {std_individual_f1:.4f}")
        for i, f1 in enumerate(individual_f1_scores, 1):
            logger.info(f"   Fold {i}: {f1:.4f}")
    else:
        logger.warning(f"⚠️  K-Fold summary not found: {summary_path}")
        individual_f1_scores = []
        mean_individual_f1 = 0.0
    
    # Test all voting strategies
    strategies = ['soft', 'hard', 'weighted']
    results = {}
    
    for strategy in strategies:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing {strategy.upper()} voting strategy")
        logger.info(f"{'='*80}")
        
        # Create ensemble
        ensemble = EnsemblePredictor(
            checkpoint_dir=checkpoint_dir,
            n_folds=5,
            device=device,
            voting_strategy=strategy
        )
        
        # Find optimal threshold
        optimal_threshold = ensemble.find_optimal_threshold(dataloader, min_recall=0.71)
        
        # Evaluate with optimal threshold
        metrics = ensemble.evaluate(dataloader, threshold=optimal_threshold)
        
        results[strategy] = {
            'threshold': optimal_threshold,
            'metrics': metrics
        }
    
    # Compare results
    logger.info(f"\n{'='*80}")
    logger.info("COMPARISON OF ENSEMBLE STRATEGIES")
    logger.info(f"{'='*80}")
    
    comparison_data = []
    
    # Add individual models baseline
    if individual_f1_scores:
        comparison_data.append({
            'Strategy': 'Individual (Mean)',
            'F1 Score': mean_individual_f1,
            'Accuracy': '-',
            'Precision': '-',
            'Recall': '-',
            'Specificity': '-',
            'Threshold': '-'
        })
    
    # Add ensemble results
    for strategy, result in results.items():
        metrics = result['metrics']
        comparison_data.append({
            'Strategy': f'Ensemble ({strategy})',
            'F1 Score': metrics['f1'],
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'Specificity': metrics['specificity'],
            'Threshold': result['threshold']
        })
    
    # Create comparison table
    comparison_df = pd.DataFrame(comparison_data)
    
    logger.info("\n" + comparison_df.to_string(index=False))
    
    # Find best strategy
    best_strategy = max(results.items(), key=lambda x: x[1]['metrics']['f1'])
    best_strategy_name = best_strategy[0]
    best_f1 = best_strategy[1]['metrics']['f1']
    
    improvement = best_f1 - mean_individual_f1 if individual_f1_scores else 0.0
    
    logger.info(f"\n{'='*80}")
    logger.info(f"🏆 BEST ENSEMBLE STRATEGY: {best_strategy_name.upper()}")
    logger.info(f"{'='*80}")
    logger.info(f"   F1 Score: {best_f1:.4f}")
    if individual_f1_scores:
        logger.info(f"   Improvement over individual: +{improvement:.4f} ({improvement/mean_individual_f1*100:+.2f}%)")
    logger.info(f"   Threshold: {best_strategy[1]['threshold']:.4f}")
    
    # Save results
    output_dir = Path(checkpoint_dir)
    comparison_df.to_csv(output_dir / 'ensemble_comparison.csv', index=False)
    logger.info(f"\n💾 Results saved to: {output_dir / 'ensemble_comparison.csv'}")
    
    # Create visualization
    create_comparison_plot(comparison_df, output_dir, mean_individual_f1)
    
    return results, best_strategy_name


def create_comparison_plot(comparison_df: pd.DataFrame, output_dir: Path, baseline_f1: float):
    """Create visualization comparing ensemble strategies"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: F1 Score comparison
    ax = axes[0]
    strategies = comparison_df['Strategy'].tolist()
    f1_scores = comparison_df['F1 Score'].tolist()
    
    colors = ['gray'] + ['skyblue', 'lightcoral', 'lightgreen'][:len(f1_scores)-1]
    bars = ax.bar(range(len(strategies)), f1_scores, color=colors, edgecolor='black', alpha=0.7)
    
    # Add baseline line
    if baseline_f1 > 0:
        ax.axhline(y=baseline_f1, color='red', linestyle='--', linewidth=2, label=f'Individual Mean: {baseline_f1:.4f}')
    
    # Add values on bars
    for i, (strategy, f1) in enumerate(zip(strategies, f1_scores)):
        ax.text(i, f1 + 0.01, f'{f1:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=15, ha='right')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(f1_scores) * 1.15])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Metrics radar chart (ensemble only)
    ax = axes[1]
    
    ensemble_rows = comparison_df[comparison_df['Strategy'].str.contains('Ensemble')]
    
    if len(ensemble_rows) > 0:
        metrics_names = ['F1 Score', 'Accuracy', 'Precision', 'Recall', 'Specificity']
        
        for idx, row in ensemble_rows.iterrows():
            strategy = row['Strategy']
            values = [
                row['F1 Score'],
                row['Accuracy'],
                row['Precision'],
                row['Recall'],
                row['Specificity']
            ]
            
            x = np.arange(len(metrics_names))
            ax.plot(x, values, marker='o', linewidth=2, label=strategy, markersize=8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names, rotation=15, ha='right')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Ensemble Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = output_dir / 'ensemble_comparison.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"📊 Comparison plot saved to: {save_path}")
    
    plt.close(fig)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Ensemble Model')
    parser.add_argument('--checkpoint_dir', type=str, 
                       default='checkpoints_cut_selection_kfold_enhanced',
                       help='Directory with fold models')
    parser.add_argument('--data_path', type=str,
                       default='preprocessed_data/combined_sequences_cut_selection_enhanced.npz',
                       help='Path to dataset')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Run evaluation
    results, best_strategy = evaluate_all_strategies(
        args.checkpoint_dir,
        args.data_path,
        args.device
    )
    
    logger.info(f"\n✅ Ensemble evaluation complete!")
    logger.info(f"   Best strategy: {best_strategy}")


if __name__ == "__main__":
    main()
