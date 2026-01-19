"""
Visualization script for pretraining results
Plots train and validation losses over training steps
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def visualize_pretrain_losses():
    """
    Read train_log.csv and val_log.csv
    Create visualization with loss (y-axis) vs training steps (x-axis)
    Save output to outputs/visualization directory
    """
    
    # Define paths
    train_log_path = Path("../outputs/pretrain/train_log.csv")
    val_log_path = Path("../outputs/pretrain/val_log.csv")
    output_dir = Path("../outputs/visualization")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv(train_log_path)
    val_df = pd.read_csv(val_log_path)
    
    # Set visualization style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 8)
    plt.rcParams['font.size'] = 11
    plt.rcParams['lines.linewidth'] = 2.5
    
    # ============================================================
    # Create comprehensive visualization
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Pretraining Results: MLM and RTD Losses Over Training Steps', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # ============================================================
    # 1. MLM Loss (Train vs Val)
    # ============================================================
    ax = axes[0]
    ax.plot(train_df['global_step'], train_df['mlm_loss'], 
            linewidth=2.5, color='#1f77b4', alpha=0.75, label='Train MLM Loss')
    ax.scatter(val_df['global_step'], val_df['mlm_loss'], 
              s=80, color='#d62728', alpha=0.8, marker='o', 
              label='Validation MLM Loss', edgecolors='darkred', linewidth=1.5, zorder=5)
    ax.fill_between(train_df['global_step'], train_df['mlm_loss'], 
                    alpha=0.15, color='#1f77b4')
    ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Generator (MLM) Loss', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    
    # ============================================================
    # 2. RTD Loss (Train vs Val)
    # ============================================================
    ax = axes[1]
    ax.plot(train_df['global_step'], train_df['rtd_loss'], 
            linewidth=2.5, color='#2ca02c', alpha=0.75, label='Train RTD Loss')
    ax.scatter(val_df['global_step'], val_df['rtd_loss'], 
              s=80, color='#ff7f0e', alpha=0.8, marker='s', 
              label='Validation RTD Loss', edgecolors='darkorange', linewidth=1.5, zorder=5)
    ax.fill_between(train_df['global_step'], train_df['rtd_loss'], 
                    alpha=0.15, color='#2ca02c')
    ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Discriminator (RTD) Loss', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    
    # ============================================================
    # 3. All Losses Overlay (Train and Val - MLM and RTD only)
    # ============================================================
    ax = axes[2]
    # Train losses
    ax.plot(train_df['global_step'], train_df['mlm_loss'], 
            linewidth=2, color='#1f77b4', alpha=0.6, linestyle='-', label='Train MLM Loss')
    ax.plot(train_df['global_step'], train_df['rtd_loss'], 
            linewidth=2, color='#2ca02c', alpha=0.6, linestyle='-', label='Train RTD Loss')
    
    # Validation losses
    ax.scatter(val_df['global_step'], val_df['mlm_loss'], 
              s=60, color='#d62728', alpha=0.8, marker='o', label='Val MLM Loss', zorder=5)
    ax.scatter(val_df['global_step'], val_df['rtd_loss'], 
              s=60, color='#ff7f0e', alpha=0.8, marker='s', label='Val RTD Loss', zorder=5)
    
    ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('MLM and RTD Losses Comparison', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10, ncol=2)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = output_dir / "pretrain_loss_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("TRAINING SUMMARY STATISTICS")
    print("="*70)
    print(f"\nTraining Data:")
    print(f"  Total Steps: {train_df['global_step'].iloc[-1]:.0f}")
    print(f"  Final MLM Loss: {train_df['mlm_loss'].iloc[-1]:.4f}")
    print(f"  Final RTD Loss: {train_df['rtd_loss'].iloc[-1]:.4f}")
    print(f"  Final Total Loss: {train_df['total_loss'].iloc[-1]:.4f}")
    print(f"\nValidation Data:")
    print(f"  Total Checkpoints: {len(val_df)}")
    print(f"  Final MLM Loss: {val_df['mlm_loss'].iloc[-1]:.4f}")
    print(f"  Final RTD Loss: {val_df['rtd_loss'].iloc[-1]:.4f}")
    print(f"  Final Total Loss: {val_df['total_loss'].iloc[-1]:.4f}")
    print(f"  Final MLM Accuracy: {val_df['mlm_acc'].iloc[-1]:.4f}")
    print(f"  Final RTD Accuracy: {val_df['rtd_acc'].iloc[-1]:.4f}")
    print("="*70)


if __name__ == "__main__":
    visualize_pretrain_losses()
