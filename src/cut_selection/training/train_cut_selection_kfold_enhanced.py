"""
K-Fold Cross Validation training script for ENHANCED cut selection model

Trains the model using K-Fold CV with ENHANCED FEATURES (audio + visual + temporal)
Uses GroupKFold to prevent data leakage (same video clips stay in same fold)
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import yaml
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
import random

from src.cut_selection.datasets.cut_dataset_enhanced import EnhancedCutSelectionDataset
from src.cut_selection.models.cut_model_enhanced import EnhancedCutSelectionModel
from src.cut_selection.utils.losses import CombinedCutSelectionLoss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 日本語フォント対応
try:
    import japanize_matplotlib
    logger.info("✅ japanize_matplotlib loaded")
except ImportError:
    logger.warning("⚠️  japanize_matplotlib not installed")


def set_seed(seed: int):
    """完全な再現性のためにシードを固定"""
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)  # Pythonハッシュシードを固定
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"🎲 Random seed set to {seed} for reproducibility")


class FoldVisualizer:
    """各Foldの詳細な学習状況を可視化（6個のグラフ）"""
    
    def __init__(self, checkpoint_dir: Path, num_epochs: int):
        self.checkpoint_dir = checkpoint_dir
        self.num_epochs = num_epochs
        
        # 学習履歴を保存
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_ce_loss': [],
            'train_tv_loss': [],
            'val_loss': [],
            'val_ce_loss': [],
            'val_tv_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_specificity': [],
            'val_pred_active_ratio': [],
            'val_pred_inactive_ratio': [],
            'optimal_threshold': [],
            'pred_total_duration': []  # Add predicted duration tracking
        }
        
        self.fig = None
        self.axes = None
        self.setup_plot()
    
    def setup_plot(self):
        """グラフの初期設定"""
        self.fig, self.axes = plt.subplots(3, 2, figsize=(15, 14))
        self.fig.suptitle('Fold詳細 - 学習状況', fontsize=16, fontweight='bold')
        plt.tight_layout()
    
    def update(self, epoch: int, train_losses: dict, val_metrics: dict):
        """学習データを更新してグラフを再描画"""
        # データを追加
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_losses['total_loss'])
        self.history['train_ce_loss'].append(train_losses['ce_loss'])
        self.history['train_tv_loss'].append(train_losses['tv_loss'])
        self.history['val_loss'].append(val_metrics['total_loss'])
        self.history['val_ce_loss'].append(val_metrics['ce_loss'])
        self.history['val_tv_loss'].append(val_metrics['tv_loss'])
        self.history['val_accuracy'].append(val_metrics['accuracy'])
        self.history['val_precision'].append(val_metrics['precision'])
        self.history['val_recall'].append(val_metrics['recall'])
        self.history['val_f1'].append(val_metrics['f1'])
        self.history['val_specificity'].append(val_metrics['specificity'])
        self.history['val_pred_active_ratio'].append(val_metrics['pred_active_ratio'] * 100)
        self.history['val_pred_inactive_ratio'].append(val_metrics['pred_inactive_ratio'] * 100)
        self.history['optimal_threshold'].append(val_metrics.get('optimal_threshold', 0.0))
        self.history['pred_total_duration'].append(val_metrics.get('pred_total_duration', 0.0))
        
        # グラフを完全にクリア
        for ax in self.axes.flat:
            # 全ての子要素（線、テキストなど）を削除
            ax.cla()
        
        epochs = self.history['epoch']
        
        # 1. 損失関数
        ax = self.axes[0, 0]
        ax.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax.set_title('損失関数（Loss）')
        ax.set_xlabel('エポック')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 損失の内訳（CE Lossを左軸、TV Lossを右軸）
        ax = self.axes[0, 1]
        
        # 既存のtwin軸を削除
        for twin_ax in list(self.fig.axes):
            if twin_ax is not ax and twin_ax.bbox.bounds == ax.bbox.bounds:
                twin_ax.remove()
        
        ax2 = ax.twinx()  # 右側に第2のY軸を作成
        
        # CE Loss（左軸、Train=青、Val=赤）
        line1 = ax.plot(epochs, self.history['train_ce_loss'], 'b-', label='Train CE', linewidth=2.5, marker='o', markersize=5)
        line2 = ax.plot(epochs, self.history['val_ce_loss'], 'r-', label='Val CE', linewidth=2.5, marker='o', markersize=5)
        ax.set_xlabel('エポック')
        ax.set_ylabel('CE Loss', color='black', fontweight='bold')
        ax.tick_params(axis='y')
        
        # TV Loss（右軸、Train=青点線、Val=赤点線）
        line3 = ax2.plot(epochs, self.history['train_tv_loss'], 'b--', label='Train TV', linewidth=2, marker='s', markersize=4)
        line4 = ax2.plot(epochs, self.history['val_tv_loss'], 'r--', label='Val TV', linewidth=2, marker='s', markersize=4)
        ax2.set_ylabel('TV Loss', color='green', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='green')
        
        # 凡例を統合
        lines = line1 + line2 + line3 + line4
        labels = [l.get_label() for l in lines]
        legend = ax.legend(lines, labels, loc='upper right', fontsize=9, framealpha=0.9)
        legend.get_frame().set_facecolor('white')
        
        ax.set_title('損失の内訳（CE=左軸、TV=右軸）', fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle=':')
        
        # 3. 分類性能
        ax = self.axes[1, 0]
        ax.plot(epochs, self.history['val_accuracy'], 'g-', label='Accuracy', linewidth=2, marker='o')
        ax.plot(epochs, self.history['val_f1'], 'purple', label='F1 Score', linewidth=2, marker='s')
        ax.set_title('分類性能（Accuracy & F1）')
        ax.set_xlabel('エポック')
        ax.set_ylabel('スコア')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Precision, Recall, Specificity
        ax = self.axes[1, 1]
        ax.plot(epochs, self.history['val_precision'], 'b-', label='Precision', linewidth=2, marker='o')
        ax.plot(epochs, self.history['val_recall'], 'r-', label='Recall', linewidth=2, marker='s')
        ax.plot(epochs, self.history['val_specificity'], 'orange', label='Specificity', linewidth=2, marker='^')
        ax.set_title('Precision, Recall, Specificity')
        ax.set_xlabel('エポック')
        ax.set_ylabel('スコア')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. 最適な閾値の推移
        ax = self.axes[2, 0]
        ax.plot(epochs, self.history['optimal_threshold'], 'purple', label='最適閾値', linewidth=2, marker='o')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='0 (Active=Inactive)')
        ax.set_title('最適な閾値の推移')
        ax.set_xlabel('エポック')
        ax.set_ylabel('閾値 (Confidence Threshold)')
        ax.set_ylim([-1, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        if len(epochs) > 0:
            current_threshold = self.history['optimal_threshold'][-1]
            ax.text(0.02, 0.98, f'現在: {current_threshold:.3f}', 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 6. 予測の採用/不採用割合
        ax = self.axes[2, 1]
        ax.plot(epochs, self.history['val_pred_active_ratio'], 'g-', label='採用 (Active)', linewidth=2, marker='o')
        ax.plot(epochs, self.history['val_pred_inactive_ratio'], 'r-', label='不採用 (Inactive)', linewidth=2, marker='s')
        if len(epochs) > 0:
            true_active_ratio = val_metrics.get('true_active_ratio', 0) * 100
            true_inactive_ratio = val_metrics.get('true_inactive_ratio', 0) * 100
            ax.axhline(y=true_active_ratio, color='g', linestyle='--', alpha=0.5, label=f'正解採用 ({true_active_ratio:.1f}%)')
            ax.axhline(y=true_inactive_ratio, color='r', linestyle='--', alpha=0.5, label=f'正解不採用 ({true_inactive_ratio:.1f}%)')
        ax.set_title('予測の採用/不採用割合')
        ax.set_xlabel('エポック')
        ax.set_ylabel('割合 (%)')
        ax.set_ylim([0, 105])
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add predicted duration text on the graph
        if len(epochs) > 0:
            current_duration = self.history['pred_total_duration'][-1]
            duration_text = f'予測合計時間: {current_duration:.1f}秒'
            if current_duration < 90:
                color = 'red'
                status = '⚠️ 短すぎ'
            elif current_duration <= 180:
                color = 'green'
                status = '✅ 理想的'
            elif current_duration <= 216:
                color = 'orange'
                status = '⚠️ 少し超過'
            else:
                color = 'red'
                status = '❌ 超過'
            
            ax.text(0.02, 0.02, f'{duration_text}\n{status}', 
                   transform=ax.transAxes, fontsize=11, verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.3, edgecolor=color))
        
        # タイトル更新
        best_f1 = max(self.history['val_f1'])
        best_epoch = self.history['epoch'][self.history['val_f1'].index(best_f1)]
        self.fig.suptitle(
            f'Fold詳細 - Epoch {epoch}/{self.num_epochs} | Best F1: {best_f1:.4f} (Epoch {best_epoch})',
            fontsize=16,
            fontweight='bold'
        )
        
        plt.tight_layout()
        
        # 保存
        save_path = self.checkpoint_dir / 'training_progress.png'
        self.fig.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"📊 Training visualization saved: {save_path}")
    
    def save_final(self):
        """最終的なグラフを高解像度で保存"""
        final_path = self.checkpoint_dir / 'training_final.png'
        self.fig.savefig(final_path, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Final training visualization saved: {final_path}")
        
        # CSVとしても保存
        import pandas as pd
        df = pd.DataFrame(self.history)
        csv_path = self.checkpoint_dir / 'training_history.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"📊 Training history saved: {csv_path}")
        
        plt.close(self.fig)


class KFoldVisualizer:
    """K-Fold学習の可視化"""
    
    def __init__(self, checkpoint_dir: Path, n_folds: int):
        self.checkpoint_dir = checkpoint_dir
        self.n_folds = n_folds
        
        # 各Foldの履歴
        self.fold_histories = []
        
        # 集約された統計
        self.summary = {
            'fold': [],
            'best_epoch': [],
            'best_val_f1': [],
            'best_val_accuracy': [],
            'best_val_precision': [],
            'best_val_recall': [],
            'optimal_threshold': []
        }
        
        # リアルタイム更新用
        self.current_fold_data = {}  # {fold_num: {'epoch': [], 'f1': [], 'loss': [], 'recall': [], 'specificity': []}}
    
    def update_realtime(self, fold: int, epoch: int, val_f1: float, val_loss: float, val_recall: float, val_specificity: float):
        """リアルタイムで進捗を更新（採用/不採用の正確性を含む）"""
        if fold not in self.current_fold_data:
            self.current_fold_data[fold] = {
                'epoch': [],
                'f1': [],
                'loss': [],
                'recall': [],  # 採用の正確性（True Positive Rate）
                'specificity': []  # 不採用の正確性（True Negative Rate）
            }
        
        self.current_fold_data[fold]['epoch'].append(epoch)
        self.current_fold_data[fold]['f1'].append(val_f1)
        self.current_fold_data[fold]['loss'].append(val_loss)
        self.current_fold_data[fold]['recall'].append(val_recall)
        self.current_fold_data[fold]['specificity'].append(val_specificity)
        
        # リアルタイムグラフを更新（エポックごと）
        self.plot_realtime_progress()
    
    def plot_realtime_progress(self):
        """リアルタイムで全Foldの進捗を表示"""
        if not self.current_fold_data:
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 14))
        fig.suptitle(f'K-Fold Cross Validation - リアルタイム進捗', fontsize=16, fontweight='bold')
        
        # 1. F1スコアの推移
        ax = axes[0, 0]
        for fold_num, data in self.current_fold_data.items():
            if data['epoch']:
                ax.plot(data['epoch'], data['f1'], 
                       label=f'Fold {fold_num}', linewidth=2, marker='o', markersize=3)
        ax.set_title('F1スコアの推移（各Fold）')
        ax.set_xlabel('エポック')
        ax.set_ylabel('F1 Score')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Lossの推移
        ax = axes[0, 1]
        for fold_num, data in self.current_fold_data.items():
            if data['epoch']:
                ax.plot(data['epoch'], data['loss'], 
                       label=f'Fold {fold_num}', linewidth=2, marker='o', markersize=3)
        ax.set_title('Validation Lossの推移（各Fold）')
        ax.set_xlabel('エポック')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 現在のF1スコア（棒グラフ）
        ax = axes[1, 0]
        if self.current_fold_data:
            folds = []
            current_f1s = []
            for fold_num, data in sorted(self.current_fold_data.items()):
                if data['f1']:
                    folds.append(f'Fold {fold_num}')
                    current_f1s.append(data['f1'][-1])
            if folds:
                colors = plt.cm.viridis(np.linspace(0, 1, len(folds)))
                bars = ax.bar(folds, current_f1s, color=colors, alpha=0.7, edgecolor='black')
                for i, (fold, f1) in enumerate(zip(folds, current_f1s)):
                    ax.text(i, f1 + 0.02, f'{f1:.4f}', ha='center', va='bottom', fontsize=9)
        ax.set_title('現在のF1スコア（各Fold）')
        ax.set_ylabel('F1 Score')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. 進捗状況
        ax = axes[1, 1]
        ax.axis('off')
        ax.set_title('進捗状況')
        progress_text = ""
        for fold_num in sorted(self.current_fold_data.keys()):
            data = self.current_fold_data[fold_num]
            if data['epoch']:
                current_epoch = data['epoch'][-1]
                current_f1 = data['f1'][-1]
                max_f1 = max(data['f1'])
                progress_text += f"Fold {fold_num}: Epoch {current_epoch}\n"
                progress_text += f"  現在のF1: {current_f1:.4f}\n"
                progress_text += f"  最良F1: {max_f1:.4f}\n\n"
        ax.text(0.1, 0.9, progress_text, transform=ax.transAxes, 
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 5. 最良F1の推移
        ax = axes[2, 0]
        for fold_num, data in self.current_fold_data.items():
            if data['f1']:
                best_f1s = [max(data['f1'][:i+1]) for i in range(len(data['f1']))]
                ax.plot(data['epoch'], best_f1s, 
                       label=f'Fold {fold_num}', linewidth=2, marker='s', markersize=3)
        ax.set_title('最良F1スコアの推移（各Fold）')
        ax.set_xlabel('エポック')
        ax.set_ylabel('Best F1 Score')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. 採用/不採用の正確性（Recall & Specificity）
        ax = axes[2, 1]
        for fold_num, data in self.current_fold_data.items():
            if data.get('recall') and data.get('specificity'):
                ax.plot(data['epoch'], data['recall'], 
                       label=f'Fold {fold_num} - 採用正確性', linewidth=2, marker='o', markersize=3, linestyle='-')
                ax.plot(data['epoch'], data['specificity'], 
                       label=f'Fold {fold_num} - 不採用正確性', linewidth=2, marker='s', markersize=3, linestyle='--')
        ax.set_title('採用/不採用の正確性（Recall & Specificity）')
        ax.set_xlabel('エポック')
        ax.set_ylabel('正確性')
        ax.set_ylim([0, 1])
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='50%')
        
        plt.tight_layout()
        
        # 保存
        save_path = self.checkpoint_dir / 'kfold_realtime_progress.png'
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        # HTMLビューアーは別途呼び出される（generate_html_viewer()）
    
    def generate_html_viewer(self):
        """HTMLビューアーを生成（自動更新付き、キャッシュ対策込み）"""
        import time
        timestamp = int(time.time() * 1000)  # ミリ秒単位のタイムスタンプ
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="2">
    <title>K-Fold Cross Validation - Real-time Progress</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .container {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
        }}
        .main-graph {{
            width: 100%;
            max-width: 1400px;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .fold-graph {{
            width: 45%;
            min-width: 600px;
            background: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        img {{
            width: 100%;
            height: auto;
            border-radius: 5px;
        }}
        .info {{
            text-align: center;
            color: #666;
            margin-top: 20px;
        }}
        .timestamp {{
            text-align: center;
            color: #999;
            font-size: 14px;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <h1>🔄 K-Fold Cross Validation - リアルタイム進捗</h1>
    <div class="info">
        <p>このページは2秒ごとに自動更新されます</p>
    </div>
    
    <div class="container">
        <div class="main-graph">
            <h2>📊 全体の進捗（全Fold統合）</h2>
            <img src="kfold_realtime_progress.png?t={timestamp}" alt="K-Fold Real-time Progress">
        </div>
"""
        
        # 各Foldの詳細グラフを追加（キャッシュ対策）
        for fold_num in range(1, self.n_folds + 1):
            fold_dir = self.checkpoint_dir / f"fold_{fold_num}"
            if fold_dir.exists() and (fold_dir / "training_progress.png").exists():
                html_content += f"""
        <div class="fold-graph">
            <h3>📈 Fold {fold_num} 詳細</h3>
            <img src="fold_{fold_num}/training_progress.png?t={timestamp}" alt="Fold {fold_num} Progress">
        </div>
"""
        
        html_content += """
    </div>
    
    <div class="timestamp">
        <p>最終更新: <span id="timestamp"></span></p>
    </div>
    
    <script>
        document.getElementById('timestamp').textContent = new Date().toLocaleString('ja-JP');
    </script>
</body>
</html>
"""
        
        html_path = self.checkpoint_dir / 'view_training.html'
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def add_fold_result(self, fold: int, history: dict, best_metrics: dict):
        """Foldの結果を追加"""
        self.fold_histories.append({
            'fold': fold,
            'history': history,
            'best_metrics': best_metrics
        })
        
        self.summary['fold'].append(fold)
        self.summary['best_epoch'].append(best_metrics['epoch'])
        self.summary['best_val_f1'].append(best_metrics['f1'])
        self.summary['best_val_accuracy'].append(best_metrics['accuracy'])
        self.summary['best_val_precision'].append(best_metrics['precision'])
        self.summary['best_val_recall'].append(best_metrics['recall'])
        self.summary['optimal_threshold'].append(best_metrics['optimal_threshold'])
    
    def plot_fold_comparison(self):
        """全Foldの比較グラフを作成"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'K-Fold Cross Validation Results (K={self.n_folds})', 
                    fontsize=16, fontweight='bold')
        
        # 1. F1スコアの推移（全Fold）
        ax = axes[0, 0]
        for fold_data in self.fold_histories:
            fold = fold_data['fold']
            history = fold_data['history']
            ax.plot(history['epoch'], history['val_f1'], 
                   label=f'Fold {fold}', linewidth=2, marker='o', markersize=3)
        ax.set_title('F1スコアの推移（全Fold）')
        ax.set_xlabel('エポック')
        ax.set_ylabel('F1 Score')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 各Foldの最良F1スコア
        ax = axes[0, 1]
        folds = self.summary['fold']
        f1_scores = self.summary['best_val_f1']
        colors = plt.cm.viridis(np.linspace(0, 1, len(folds)))
        bars = ax.bar(folds, f1_scores, color=colors, alpha=0.7, edgecolor='black')
        
        # 平均値と標準偏差を表示
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        ax.axhline(y=mean_f1, color='red', linestyle='--', linewidth=2, 
                  label=f'平均: {mean_f1:.4f} ± {std_f1:.4f}')
        
        ax.set_title('各Foldの最良F1スコア')
        ax.set_xlabel('Fold')
        ax.set_ylabel('Best F1 Score')
        ax.set_ylim([0, 1])
        ax.set_xticks(folds)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 各バーに値を表示
        for i, (fold, f1) in enumerate(zip(folds, f1_scores)):
            ax.text(fold, f1 + 0.02, f'{f1:.4f}', 
                   ha='center', va='bottom', fontsize=9)
        
        # 3. Precision vs Recall（各Foldの最良値）
        ax = axes[1, 0]
        precisions = self.summary['best_val_precision']
        recalls = self.summary['best_val_recall']
        colors = plt.cm.viridis(np.linspace(0, 1, len(folds)))
        
        for i, (fold, prec, rec, color) in enumerate(zip(folds, precisions, recalls, colors)):
            ax.scatter(rec, prec, s=200, color=color, alpha=0.7, 
                      edgecolor='black', linewidth=2, label=f'Fold {fold}', zorder=3)
            ax.text(rec, prec, f'{fold}', ha='center', va='center', 
                   fontsize=10, fontweight='bold', color='white')
        
        # 平均値をプロット
        mean_prec = np.mean(precisions)
        mean_rec = np.mean(recalls)
        ax.scatter(mean_rec, mean_prec, s=300, color='red', alpha=0.8, 
                  edgecolor='black', linewidth=3, marker='*', label='平均', zorder=4)
        
        ax.set_title('Precision vs Recall（各Foldの最良値）')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 4. 最適閾値（各Fold）
        ax = axes[1, 1]
        thresholds = self.summary['optimal_threshold']
        colors = plt.cm.viridis(np.linspace(0, 1, len(folds)))
        bars = ax.bar(folds, thresholds, color=colors, alpha=0.7, edgecolor='black')
        
        # 平均値と標準偏差を表示
        mean_threshold = np.mean(thresholds)
        std_threshold = np.std(thresholds)
        ax.axhline(y=mean_threshold, color='red', linestyle='--', linewidth=2,
                  label=f'平均: {mean_threshold:.3f} ± {std_threshold:.3f}')
        
        ax.set_title('最適閾値（各Fold）')
        ax.set_xlabel('Fold')
        ax.set_ylabel('Confidence Threshold')
        ax.set_xticks(folds)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 各バーに値を表示
        for i, (fold, th) in enumerate(zip(folds, thresholds)):
            ax.text(fold, th + 0.02, f'{th:.3f}', 
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # 保存
        save_path = self.checkpoint_dir / 'kfold_comparison.png'
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"📊 K-Fold comparison saved: {save_path}")
        plt.close(fig)
    
    def save_summary(self):
        """サマリーをCSVとして保存"""
        import pandas as pd
        
        df = pd.DataFrame(self.summary)
        
        # 統計を追加
        stats_row = {
            'fold': 'Mean ± Std',
            'best_epoch': f"{np.mean(self.summary['best_epoch']):.1f} ± {np.std(self.summary['best_epoch']):.1f}",
            'best_val_f1': f"{np.mean(self.summary['best_val_f1']):.4f} ± {np.std(self.summary['best_val_f1']):.4f}",
            'best_val_accuracy': f"{np.mean(self.summary['best_val_accuracy']):.4f} ± {np.std(self.summary['best_val_accuracy']):.4f}",
            'best_val_precision': f"{np.mean(self.summary['best_val_precision']):.4f} ± {np.std(self.summary['best_val_precision']):.4f}",
            'best_val_recall': f"{np.mean(self.summary['best_val_recall']):.4f} ± {np.std(self.summary['best_val_recall']):.4f}",
            'optimal_threshold': f"{np.mean(self.summary['optimal_threshold']):.3f} ± {np.std(self.summary['optimal_threshold']):.3f}"
        }
        
        df = pd.concat([df, pd.DataFrame([stats_row])], ignore_index=True)
        
        csv_path = self.checkpoint_dir / 'kfold_summary.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"📊 K-Fold summary saved: {csv_path}")
        
        # コンソールに表示
        logger.info("\n" + "="*80)
        logger.info("K-FOLD CROSS VALIDATION SUMMARY")
        logger.info("="*80)
        logger.info(f"\nMean F1 Score: {np.mean(self.summary['best_val_f1']):.4f} ± {np.std(self.summary['best_val_f1']):.4f}")
        logger.info(f"Mean Accuracy: {np.mean(self.summary['best_val_accuracy']):.4f} ± {np.std(self.summary['best_val_accuracy']):.4f}")
        logger.info(f"Mean Precision: {np.mean(self.summary['best_val_precision']):.4f} ± {np.std(self.summary['best_val_precision']):.4f}")
        logger.info(f"Mean Recall: {np.mean(self.summary['best_val_recall']):.4f} ± {np.std(self.summary['best_val_recall']):.4f}")
        logger.info(f"Mean Optimal Threshold: {np.mean(self.summary['optimal_threshold']):.3f} ± {np.std(self.summary['optimal_threshold']):.3f}")
        logger.info("="*80 + "\n")


def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None, max_grad_norm=1.0):
    """Train for one epoch with 3 modalities (audio, visual, temporal)"""
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_tv_loss = 0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        audio = batch['audio'].to(device)
        visual = batch['visual'].to(device)
        temporal = batch.get('temporal', torch.zeros_like(audio[:, :, :7])).to(device)  # Fallback if no temporal
        active_labels = batch['active'].to(device)
        
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(audio, visual, temporal)
                active_logits = outputs['active']
                loss, loss_dict = criterion(active_logits, active_labels)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(audio, visual, temporal)
            active_logits = outputs['active']
            loss, loss_dict = criterion(active_logits, active_labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        
        total_loss += loss_dict['total_loss']
        total_ce_loss += loss_dict['ce_loss']
        total_tv_loss += loss_dict['tv_loss']
    
    num_batches = len(dataloader)
    return {
        'total_loss': total_loss / num_batches,
        'ce_loss': total_ce_loss / num_batches,
        'tv_loss': total_tv_loss / num_batches
    }


def validate(model, dataloader, criterion, device, config=None):
    """Validate the model with 3 modalities (audio, visual, temporal)"""
    model.eval()
    total_loss = 0
    total_ce_loss = 0
    total_tv_loss = 0
    
    all_predictions = []
    all_labels = []
    all_confidence_scores = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            audio = batch['audio'].to(device)
            visual = batch['visual'].to(device)
            temporal = batch.get('temporal', torch.zeros_like(audio[:, :, :7])).to(device)  # Fallback if no temporal
            active_labels = batch['active'].to(device)
            
            outputs = model(audio, visual, temporal)
            active_logits = outputs['active']
            
            loss, loss_dict = criterion(active_logits, active_labels)
            total_loss += loss_dict['total_loss']
            total_ce_loss += loss_dict['ce_loss']
            total_tv_loss += loss_dict['tv_loss']
            
            predictions = torch.argmax(active_logits, dim=-1)
            probs = torch.softmax(active_logits, dim=-1)
            confidence_scores = probs[..., 1] - probs[..., 0]
            
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(active_labels.cpu().numpy())
            all_confidence_scores.append(confidence_scores.cpu().numpy())
    
    all_predictions = np.concatenate([p.flatten() for p in all_predictions])
    all_labels = np.concatenate([l.flatten() for l in all_labels])
    all_confidence_scores = np.concatenate([s.flatten() for s in all_confidence_scores])
    
    # Calculate average losses (actual values, not estimates)
    num_batches = len(dataloader)
    avg_total_loss = total_loss / num_batches
    avg_ce_loss = total_ce_loss / num_batches
    avg_tv_loss = total_tv_loss / num_batches
    
    # Temperature Scaling (optional calibration)
    temperature = 1.02  # >1 makes predictions less confident, <1 makes them more confident
    all_confidence_scores_calibrated = all_confidence_scores / temperature
    
    # Find optimal threshold using F1 + Specificity composite score with Recall constraint
    from sklearn.metrics import precision_recall_curve
    
    # Get duration constraint parameters
    max_total_duration = config.get('max_total_duration', 180.0) if config else 180.0  # Target: 3 minutes
    hard_max_duration = config.get('hard_max_duration', 200.0) if config else 200.0  # Hard max: target + 20s
    min_total_duration = config.get('min_total_duration', 90.0) if config else 90.0  # Min: 1.5 minutes (50% of target)
    fps = config.get('inference_fps', 10.0) if config else 10.0  # Default: 10 fps
    min_clip_duration = config.get('min_clip_duration', 3.0) if config else 3.0  # Default: 3 seconds
    
    def extract_clips_from_predictions(predictions, fps=10.0, min_duration=3.0):
        """Extract clips from binary predictions"""
        clips = []
        in_clip = False
        start_idx = 0
        
        for i, active in enumerate(predictions):
            if active == 1 and not in_clip:
                start_idx = i
                in_clip = True
            elif active == 0 and in_clip:
                end_idx = i
                start_time = start_idx / fps
                end_time = end_idx / fps
                if (end_time - start_time) >= min_duration:
                    clips.append((start_time, end_time))
                in_clip = False
        
        # Handle last clip
        if in_clip:
            end_idx = len(predictions)
            start_time = start_idx / fps
            end_time = end_idx / fps
            if (end_time - start_time) >= min_duration:
                clips.append((start_time, end_time))
        
        return clips
    
    # Calculate total video duration
    total_frames = len(all_labels)
    total_video_duration = total_frames / fps
    
    # 🔒 SPECIAL CASE: If video is shorter than min_total_duration, accept everything
    if total_video_duration < min_total_duration:
        logger.info(f"📹 Video duration ({total_video_duration:.1f}s) < min ({min_total_duration:.1f}s) → Accepting all frames")
        optimal_predictions = np.ones_like(all_labels, dtype=int)
        best_threshold = -999.0  # Very low threshold to accept everything
    else:
        # Calculate precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(all_labels, all_confidence_scores_calibrated)
        
        # FAST APPROXIMATION: Pre-calculate approximate durations for all thresholds
        # This avoids expensive clip extraction in the loop
        logger.debug("Pre-calculating approximate durations for threshold search...")
        
        # Approximate duration = (number of active frames) / fps
        # This is much faster than extracting clips
        approx_durations = []
        for thresh in thresholds:
            candidate_predictions = (all_confidence_scores_calibrated >= thresh).astype(int)
            active_frames = np.sum(candidate_predictions)
            approx_duration = active_frames / fps
            approx_durations.append(approx_duration)
        
        approx_durations = np.array(approx_durations)
        
        # Find threshold that maximizes Recall while maintaining min <= duration <= hard_max
        best_threshold = thresholds[0] if len(thresholds) > 0 else 0.0
        best_recall = 0.0
        best_f1 = 0.0
        best_duration = 0.0
        
        for i, (prec, rec, thresh, approx_dur) in enumerate(zip(precisions, recalls, thresholds, approx_durations)):
            # 🔒 HARD CONSTRAINTS: Use approximate duration for fast filtering
            # Use stricter thresholds to ensure we don't violate constraints after clip extraction
            if approx_dur < min_total_duration * 0.95:  # 95% threshold (stricter)
                continue  # Likely too short
            if approx_dur > hard_max_duration * 1.05:  # 105% threshold (stricter)
                continue  # Likely too long
            
            # Maximize Recall within approximate constraints
            if rec > best_recall:
                best_recall = rec
                best_threshold = thresh
                best_duration = approx_dur
                # Calculate F1 for logging
                best_f1 = 2 * (prec * rec) / (prec + rec + 1e-10)
        
        # If no threshold satisfies constraints
        if best_recall == 0.0:
            logger.error(f"❌ Cannot find threshold with {min_total_duration:.1f}s <= duration <= {hard_max_duration:.1f}s")
            # Use lowest threshold as fallback
            best_idx = 0
            best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.0
            best_recall = recalls[best_idx]
        else:
            logger.debug(f"Optimal threshold: {best_threshold:.4f} (Recall={best_recall:.2%}, F1={best_f1:.4f}, Approx Duration={best_duration:.1f}s)")
        
        # Use OPTIMAL THRESHOLD for all metrics
        optimal_predictions = (all_confidence_scores_calibrated >= best_threshold).astype(int)
    
    # Extract clips with optimal threshold
    clips = extract_clips_from_predictions(optimal_predictions, fps, min_clip_duration)
    total_duration = sum(e - s for s, e in clips)
    
    # HARD CONSTRAINT ENFORCEMENT: Adjust threshold if needed
    # Only enforce minimum constraint (90s) - maximum is already enforced in threshold selection
    needs_adjustment = total_duration < min_total_duration
    
    if needs_adjustment and total_video_duration >= min_total_duration:
        # Only adjust if video is long enough
        logger.debug(f"⚠️  Duration ({total_duration:.1f}s) < min ({min_total_duration:.1f}s), decreasing threshold...")
        # Too short: decrease threshold to increase duration
        low_threshold = all_confidence_scores_calibrated.min()
        high_threshold = best_threshold
        target_duration = min_total_duration
        
        adjusted_threshold = best_threshold
        adjusted_predictions = optimal_predictions.copy()
        best_duration_diff = abs(total_duration - target_duration)
        
        for iteration in range(20):  # Max 20 iterations for precision
            mid_threshold = (low_threshold + high_threshold) / 2
            test_predictions = (all_confidence_scores_calibrated >= mid_threshold).astype(int)
            test_clips = extract_clips_from_predictions(test_predictions, fps, min_clip_duration)
            test_duration = sum(e - s for s, e in test_clips)
            
            # Check if this threshold satisfies minimum constraint
            if test_duration >= min_total_duration:
                # Valid! Save this as a candidate
                duration_diff = abs(test_duration - target_duration)
                if duration_diff < best_duration_diff:
                    adjusted_threshold = mid_threshold
                    adjusted_predictions = test_predictions
                    best_duration_diff = duration_diff
                
                # If we're close enough to target, stop
                if duration_diff < 5.0:  # Within 5 seconds of target
                    break
                
                # Try to get closer to minimum (not too much over)
                low_threshold = mid_threshold
            else:
                # Still too short, go lower
                high_threshold = mid_threshold
            
            # Stop if range is too small
            if abs(high_threshold - low_threshold) < 0.0001:
                break
        
        optimal_predictions = adjusted_predictions
        clips = extract_clips_from_predictions(optimal_predictions, fps, min_clip_duration)
        total_duration = sum(e - s for s, e in clips)
        
        # Final check: ensure minimum constraint is met
        if total_duration < min_total_duration:
            logger.warning(f"❌ HARD CONSTRAINT VIOLATION: Duration ({total_duration:.1f}s) < min ({min_total_duration:.1f}s)")
        else:
            logger.info(f"✅ Minimum constraint satisfied: {total_duration:.1f}s >= {min_total_duration:.1f}s")
        
        logger.info(f"⚠️  Threshold adjusted: {best_threshold:.4f} → {adjusted_threshold:.4f} (duration: {total_duration:.1f}s, clips: {len(clips)})")
        best_threshold = adjusted_threshold  # Update threshold for saving
    
    # Check hard maximum constraint
    if total_duration > hard_max_duration:
        logger.warning(f"❌ HARD MAX CONSTRAINT VIOLATION: Duration ({total_duration:.1f}s) > hard_max ({hard_max_duration:.1f}s)")
    elif total_duration > max_total_duration:
        # Soft constraint violation (target exceeded but within hard max)
        logger.info(f"⚠️  Soft constraint: Duration ({total_duration:.1f}s) > target ({max_total_duration:.1f}s) but <= hard_max ({hard_max_duration:.1f}s) - acceptable")
    
    # Calculate metrics using optimal threshold (with duration constraint applied)
    tp = np.sum((optimal_predictions == 1) & (all_labels == 1))
    fp = np.sum((optimal_predictions == 1) & (all_labels == 0))
    fn = np.sum((optimal_predictions == 0) & (all_labels == 1))
    tn = np.sum((optimal_predictions == 0) & (all_labels == 0))
    
    accuracy = (optimal_predictions == all_labels).mean()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Specificity (for Inactive class) using optimal threshold
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Prediction ratios using OPTIMAL THRESHOLD
    total_samples = len(optimal_predictions)
    pred_active_ratio = np.sum(optimal_predictions == 1) / total_samples
    pred_inactive_ratio = np.sum(optimal_predictions == 0) / total_samples
    true_active_ratio = np.sum(all_labels == 1) / total_samples
    true_inactive_ratio = np.sum(all_labels == 0) / total_samples
    
    # Calculate predicted total duration (for display in graph)
    pred_clips = extract_clips_from_predictions(optimal_predictions, fps, min_clip_duration)
    pred_total_duration = sum(e - s for s, e in pred_clips)
    
    metrics = {
        'loss': avg_total_loss,
        'ce_loss': avg_ce_loss,
        'tv_loss': avg_tv_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'optimal_threshold': best_threshold,  # Use optimal F1 threshold
        'pred_active_ratio': pred_active_ratio,
        'pred_inactive_ratio': pred_inactive_ratio,
        'true_active_ratio': true_active_ratio,
        'true_inactive_ratio': true_inactive_ratio,
        'pred_total_duration': pred_total_duration  # Add predicted duration
    }
    
    return metrics


def train_single_fold(fold, train_indices, val_indices, full_dataset, config, device, checkpoint_dir, visualizer):
    """Train a single fold"""
    logger.info(f"\n{'='*80}")
    logger.info(f"FOLD {fold + 1}/{config['n_folds']}")
    logger.info(f"{'='*80}")
    
    # Create fold-specific visualizer（6個のグラフ構成）
    fold_checkpoint_dir = checkpoint_dir / f"fold_{fold+1}"
    fold_checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # Fold専用のビジュアライザー（6個のグラフ）
    fold_visualizer = FoldVisualizer(fold_checkpoint_dir, config['num_epochs'])
    logger.info(f"📊 Fold {fold+1} visualizer initialized (6 graphs)")
    
    # Create data loaders
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    
    train_loader = DataLoader(
        train_subset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True if device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True if device == 'cuda' else False
    )
    
    logger.info(f"Train samples: {len(train_indices)}")
    logger.info(f"Val samples: {len(val_indices)}")
    
    # クラス不均衡の自動補正（データリーク防止のため学習データのみで計算）
    logger.info("Calculating class weights from training data...")
    train_active_count = 0
    train_inactive_count = 0
    for idx in train_indices:
        active_labels = full_dataset[idx]['active'].numpy()
        train_active_count += np.sum(active_labels == 1)
        train_inactive_count += np.sum(active_labels == 0)
    
    total_train_samples = train_active_count + train_inactive_count
    
    # Focal Loss使用時: バランスの取れたクラスウェイト
    # Focal Lossが自動的に難しいサンプルにフォーカスするため、過度な重み付けは不要
    base_weight_active = total_train_samples / (2 * train_active_count) if train_active_count > 0 else 1.0
    base_weight_inactive = total_train_samples / (2 * train_inactive_count) if train_inactive_count > 0 else 1.0
    
    # Activeクラスを適度に重視（Focal Lossと組み合わせて効果的）
    weight_active = base_weight_active * 3.0  # Activeを適度に重視
    weight_inactive = base_weight_inactive * 1.0  # Inactiveは標準

    class_weights = torch.tensor([weight_inactive, weight_active], device=device)

    logger.info(f"  Train Active: {train_active_count:,} ({train_active_count/total_train_samples*100:.2f}%)")
    logger.info(f"  Train Inactive: {train_inactive_count:,} ({train_inactive_count/total_train_samples*100:.2f}%)")
    logger.info(f"  Base weights: [inactive={base_weight_inactive:.4f}, active={base_weight_active:.4f}]")
    logger.info(f"  Enhanced weights: [inactive={weight_inactive:.4f}, active={weight_active:.4f}]")
    logger.info(f"  Focal Loss: alpha={config.get('focal_alpha', 0.75)}, gamma={config.get('focal_gamma', 2.0)} (auto-focus on hard examples)")
    logger.info(f"  Duration Constraint: target={config.get('max_total_duration', 180.0):.1f}s, penalty_weight={config.get('duration_penalty_weight', 0.5)}")
    logger.info(f"  Min Clip Duration: {config.get('min_clip_duration', 3.0):.1f}s @ {config.get('inference_fps', 10.0):.1f}fps")
    
    # Create model with 3 modalities
    model = EnhancedCutSelectionModel(
        audio_features=config['audio_features'],
        visual_features=config['visual_features'],
        temporal_features=config.get('temporal_features', 7),  # Default 7 temporal features
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout']
    ).to(device)
    
    # Loss and optimizer (class_weightsを設定)
    criterion = CombinedCutSelectionLoss(
        class_weights=class_weights,  # 自動計算された重みを使用
        tv_weight=config.get('tv_weight', 0.1),
        label_smoothing=config.get('label_smoothing', 0.0),
        use_focal=config.get('use_focal_loss', False),
        focal_alpha=config.get('focal_alpha', 0.25),
        focal_gamma=config.get('focal_gamma', 2.0),
        target_duration=config.get('max_total_duration', 180.0),
        duration_penalty_weight=config.get('duration_penalty_weight', 1.0),
        recall_reward_weight=config.get('recall_reward_weight', 2.0),
        fps=config.get('inference_fps', 10.0),
        min_clip_duration=config.get('min_clip_duration', 3.0)
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0.0)
    )
    
    # Training history
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }
    
    best_val_f1 = 0.0
    best_metrics = None
    patience_counter = 0
    use_amp = config.get('use_amp', False) and device == 'cuda'
    
    # GradScalerをループの外で作成（AMPの安定性向上）
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if scaler:
        logger.info("📊 Mixed Precision Training enabled with GradScaler")
    
    # Get max gradient norm from config
    max_grad_norm = config.get('max_grad_norm', 1.0)
    logger.info(f"📊 Gradient clipping: max_norm={max_grad_norm}")
    
    # Training loop
    for epoch in range(config['num_epochs']):
        # Calculate overall progress
        total_epochs = config['n_folds'] * config['num_epochs']
        current_overall_epoch = fold * config['num_epochs'] + epoch + 1
        
        # Train
        train_losses = train_epoch(model, train_loader, optimizer, criterion, device, scaler, max_grad_norm)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, config)
        
        # Record history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_losses['total_loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        
        # Update fold visualizer (エポックごとに更新)
        val_metrics_viz = {
            'total_loss': val_metrics['loss'],
            'ce_loss': val_metrics['loss'],
            'tv_loss': 0.0,
            'accuracy': val_metrics['accuracy'],
            'precision': val_metrics['precision'],
            'recall': val_metrics['recall'],
            'f1': val_metrics['f1'],
            'specificity': val_metrics['specificity'],
            'pred_active_ratio': val_metrics['pred_active_ratio'],
            'pred_inactive_ratio': val_metrics['pred_inactive_ratio'],
            'true_active_ratio': val_metrics['true_active_ratio'],
            'true_inactive_ratio': val_metrics['true_inactive_ratio'],
            'optimal_threshold': val_metrics['optimal_threshold']
        }
        fold_visualizer.update(epoch + 1, train_losses, val_metrics_viz)
        
        # Update global K-Fold visualizer (realtime)
        visualizer.update_realtime(
            fold + 1, 
            epoch + 1, 
            val_metrics['f1'], 
            val_metrics['loss'],
            val_metrics['recall'],
            val_metrics['specificity']
        )
        
        # HTMLビューアーを更新（各Foldの詳細グラフも含む）
        visualizer.generate_html_viewer()
        
        logger.info(f"Fold {fold+1}/{config['n_folds']} | "
                   f"Epoch {epoch+1}/{config['num_epochs']} "
                   f"(Overall: {current_overall_epoch}/{total_epochs}) - "
                   f"Train Loss: {train_losses['total_loss']:.4f}, "
                   f"Val Loss: {val_metrics['loss']:.4f}, "
                   f"Val F1: {val_metrics['f1']:.4f}")
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_metrics = {
                'epoch': epoch + 1,
                'accuracy': val_metrics['accuracy'],
                'precision': val_metrics['precision'],
                'recall': val_metrics['recall'],
                'f1': val_metrics['f1'],
                'optimal_threshold': val_metrics['optimal_threshold']
            }
            patience_counter = 0
            
            # Save model
            model_path = checkpoint_dir / f"fold_{fold+1}_best_model.pth"
            torch.save({
                'fold': fold + 1,
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, model_path)
        else:
            patience_counter += 1
            if patience_counter >= config.get('early_stopping_patience', 10):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Save final visualization for this fold
    fold_visualizer.save_final()
    
    logger.info(f"\nFold {fold+1} Best F1: {best_val_f1:.4f} at epoch {best_metrics['epoch']}")
    
    return history, best_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_cut_selection_kfold.yaml')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 完全な再現性のためにシードを固定
    random_state = config.get('random_state', 42)
    set_seed(random_state)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() and not config.get('cpu', False) else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(config['checkpoint_dir'])
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # Load full dataset
    logger.info(f"\nLoading enhanced dataset from {config['data_path']}")
    full_dataset = EnhancedCutSelectionDataset(config['data_path'])
    
    # Get video groups for GroupKFold (データリーク防止)
    video_groups = full_dataset.get_video_groups()
    unique_videos = list(set(video_groups))
    logger.info(f"📹 Total unique videos: {len(unique_videos)}")
    logger.info(f"⚠️  Using GroupKFold to prevent data leakage (same video stays in same fold)")
    
    # GroupKFold split (動画単位で分割)
    n_folds = config.get('n_folds', 5)
    group_kfold = GroupKFold(n_splits=n_folds)
    
    # Visualizer
    visualizer = KFoldVisualizer(checkpoint_dir, n_folds)
    
    # Train each fold
    for fold, (train_indices, val_indices) in enumerate(group_kfold.split(range(len(full_dataset)), groups=video_groups)):
        # 各Foldの動画を確認
        train_videos = set([video_groups[i] for i in train_indices])
        val_videos = set([video_groups[i] for i in val_indices])
        
        # データリークチェック
        overlap = train_videos & val_videos
        if overlap:
            logger.error(f"❌ DATA LEAKAGE DETECTED in Fold {fold+1}: {len(overlap)} videos in both train and val!")
            logger.error(f"   Overlapping videos: {list(overlap)[:5]}...")
            raise ValueError("Data leakage detected! Same videos in train and val.")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Fold {fold+1}/{n_folds}")
        logger.info(f"  Train videos: {len(train_videos)}, Val videos: {len(val_videos)}")
        logger.info(f"  ✅ No data leakage (train and val videos are completely separate)")
        logger.info(f"{'='*80}")
        
        history, best_metrics = train_single_fold(
            fold, train_indices, val_indices, full_dataset, config, device, checkpoint_dir, visualizer
        )
        
        visualizer.add_fold_result(fold + 1, history, best_metrics)
    
    # Save summary and plots
    visualizer.plot_fold_comparison()
    visualizer.save_summary()
    
    # Save average inference parameters
    avg_threshold = np.mean(visualizer.summary['optimal_threshold'])
    inference_params = {
        'confidence_threshold': float(avg_threshold),
        'target_duration': 90.0,
        'max_duration': 150.0
    }
    
    inference_params_path = checkpoint_dir / 'inference_params.yaml'
    with open(inference_params_path, 'w', encoding='utf-8') as f:
        yaml.dump(inference_params, f)
    
    logger.info(f"\n✅ K-Fold Cross Validation complete!")
    logger.info(f"Average confidence threshold: {avg_threshold:.3f}")
    logger.info(f"Results saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()
