"""
Publication-Quality Figure Generator for Fake News Research Paper / Report
Generates publication-ready figures (300 DPI) for:
1. Dataset Distribution and Splits (Tables I & II)
2. Model Performance & Accuracy/Loss Comparison (Table V)
3. Source Credibility Risk Spectrum (Table IV)
4. Hardware & Computational Efficiency Trade-off (Tables III & V)
"""

import os
import sys

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

# Create output folder for figures
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Set global publication styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.color'] = '#e2e8f0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

# ==============================================================================
# FIGURE 1: Dataset Scale and IFND Partition Breakdown (Tables I & II)
# ==============================================================================
def generate_fig1_datasets():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)

    # Subplot 1: Total Dataset Sizes Comparison (Table I)
    datasets = ['LIAR (US)', 'ISOT (US)', 'IFND (India)\n[Ours]']
    sizes = [12836, 44898, 56124]
    colors_ds = ['#94a3b8', '#64748b', '#0284c7']

    bars = ax1.bar(datasets, sizes, color=colors_ds, width=0.55, edgecolor='#1e293b', linewidth=1.2)
    ax1.set_title('(a) Benchmark Dataset Scale Comparison', fontsize=12, fontweight='bold', pad=12)
    ax1.set_ylabel('Number of Samples / Articles', fontsize=11, fontweight='semibold')
    ax1.set_ylim(0, 68000)
    ax1.grid(axis='y', alpha=0.5)

    for bar, val in zip(bars, sizes):
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1200, f'{val:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Subplot 2: IFND Train/Val/Test Splits & Class Distribution (Table II)
    splits = ['Training\n(80%)', 'Validation\n(10%)', 'Testing\n(10%)']
    
    # Class breakdown per split (using exact 67.32% Real, 32.68% Fake stratified ratio)
    real_counts = [30229, 3774, 3784]
    fake_counts = [14675, 1833, 1829]

    bar_width = 0.52
    indices = np.arange(len(splits))

    b_real = ax2.bar(indices, real_counts, bar_width, label='Real News (67.3%)', color='#10b981', edgecolor='#065f46', linewidth=1.2)
    b_fake = ax2.bar(indices, fake_counts, bar_width, bottom=real_counts, label='Fake News (32.7%)', color='#ef4444', edgecolor='#991b1b', linewidth=1.2)

    ax2.set_title('(b) IFND Split Breakdown & Class Balance', fontsize=12, fontweight='bold', pad=12)
    ax2.set_ylabel('Number of Samples', fontsize=11, fontweight='semibold')
    ax2.set_xticks(indices)
    ax2.set_xticklabels(splits, fontsize=10)
    ax2.set_ylim(0, 52000)
    ax2.grid(axis='y', alpha=0.5)
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5, loc='upper right')

    totals = [44904, 5607, 5613]
    for i, total in enumerate(totals):
        ax2.text(i, total + 900, f'Total: {total:,}', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig1_dataset_distribution_and_splits.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Generated: {out_path}")

# ==============================================================================
# FIGURE 2: Model Performance & Accuracy/Loss Comparison (Table V)
# ==============================================================================
def generate_fig2_performance():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), dpi=300)

    # Subplot 1: Accuracy Metrics
    metrics_acc = ['Training Accuracy', 'Validation Accuracy']
    satyacheck_acc = [90.74, 97.77]
    us_baseline_acc = [88.45, 92.20]

    x = np.arange(len(metrics_acc))
    width = 0.32

    rects1 = ax1.bar(x - width/2, satyacheck_acc, width, label='SatyaCheck (XLM-RoBERTa + IFND)', color='#2563eb', edgecolor='#1e40af', linewidth=1.2)
    rects2 = ax1.bar(x + width/2, us_baseline_acc, width, label='US Baseline (RoBERTa + ISOT/LIAR)', color='#94a3b8', edgecolor='#475569', linewidth=1.2)

    ax1.set_title('(a) Accuracy Comparison', fontsize=12, fontweight='bold', pad=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='semibold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_acc, fontsize=10.5, fontweight='semibold')
    ax1.set_ylim(80, 103)
    ax1.grid(axis='y', alpha=0.5)
    ax1.legend(loc='lower right', fontsize=9, frameon=True)

    for rect in rects1:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2.0, h + 0.6, f'{h:.2f}%', ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#1d4ed8')

    for rect in rects2:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2.0, h + 0.6, f'{h:.2f}%', ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#475569')

    # Delta annotation for validation accuracy
    ax1.annotate('+5.57% Gain',
                 xy=(1 - width/2, 97.77), xytext=(1, 101.5),
                 arrowprops=dict(arrowstyle="->", color="#16a34a", lw=1.5),
                 ha='center', fontsize=9.5, fontweight='bold', color='#16a34a')

    # Subplot 2: Loss Metrics (Lower is better)
    metrics_loss = ['Training Loss', 'Validation Loss']
    satyacheck_loss = [0.2264, 0.0960]
    us_baseline_loss = [0.3120, 0.1840]

    rects3 = ax2.bar(x - width/2, satyacheck_loss, width, label='SatyaCheck (Ours)', color='#059669', edgecolor='#047857', linewidth=1.2)
    rects4 = ax2.bar(x + width/2, us_baseline_loss, width, label='US Baseline', color='#cbd5e1', edgecolor='#64748b', linewidth=1.2)

    ax2.set_title('(b) Cross-Entropy Loss Comparison (Lower is Better)', fontsize=12, fontweight='bold', pad=12)
    ax2.set_ylabel('Loss Value', fontsize=11, fontweight='semibold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics_loss, fontsize=10.5, fontweight='semibold')
    ax2.set_ylim(0, 0.38)
    ax2.grid(axis='y', alpha=0.5)
    ax2.legend(loc='upper right', fontsize=9, frameon=True)

    for rect in rects3:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2.0, h + 0.007, f'{h:.4f}', ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#047857')

    for rect in rects4:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2.0, h + 0.007, f'{h:.4f}', ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#64748b')

    # Delta annotation for validation loss - placed higher to prevent collision
    ax2.annotate('47.8% Loss Reduction\n(0.1840 -> 0.0960)',
                 xy=(1 - width/2, 0.108), xytext=(0.85, 0.22),
                 arrowprops=dict(arrowstyle="->", color="#059669", lw=1.6),
                 ha='center', fontsize=9.2, fontweight='bold', color='#059669',
                 bbox=dict(boxstyle="round,pad=0.3", fc="#ecfdf5", ec="#10b981", lw=1.0))

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig2_model_performance_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Generated: {out_path}")

# ==============================================================================
# FIGURE 3: Source Credibility Spectrum / Risk Tier Architecture (Table IV)
# ==============================================================================
def generate_fig3_credibility():
    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=300)

    tiers = [
        "Tier 1: Highly Credible\n(Score: 0.05 – 0.15)",
        "Tier 2: Generally Reliable\n(Score: 0.20 – 0.40)",
        "Tier 3: Partisan / Biased\n(Score: 0.55 – 0.72)",
        "Tier 4: High Misinfo Risk\n(Score: 0.80 – 0.95)"
    ]

    # Representative score ranges
    starts = [0.05, 0.20, 0.55, 0.80]
    widths = [0.10, 0.20, 0.17, 0.15]
    colors = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444']

    publishers = [
        "The Hindu, Indian Express, NDTV, PTI, Reuters, BBC",
        "Times of India, India Today, ABP News, Deccan Herald",
        "Zee News, Republic TV, Sudarshan News",
        "Postcard News, Unverified WhatsApp Forwards, Clickbait"
    ]

    y_pos = np.arange(len(tiers))

    # Horizontal indicator bars
    bars = ax.barh(y_pos, widths, left=starts, height=0.42, color=colors, edgecolor='#1e293b', linewidth=1.2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(tiers, fontsize=10, fontweight='semibold')
    ax.set_ylim(3.5, -0.5)  # Tier 1 at top, Tier 4 at bottom
    ax.set_xlim(0, 1.48)
    ax.set_xlabel('Misinformation Risk Score (0.0 = Verified True  ->  1.0 = High Deception Risk)', fontsize=10.5, fontweight='semibold', labelpad=10)
    ax.set_title('Indian Media Source Credibility Spectrum (Table IV)', fontsize=12, fontweight='bold', pad=14)
    ax.grid(axis='x', alpha=0.5)

    # Shaded boundary for 1.0 risk limit
    ax.axvline(1.0, color='#94a3b8', linestyle='--', linewidth=1.2)
    ax.text(1.01, 3.3, 'Max Score (1.0)', fontsize=8.5, color='#475569', fontweight='bold')

    # Annotate publishers inside / next to bars
    for i, (p, s, w) in enumerate(zip(publishers, starts, widths)):
        ax.text(s + w + 0.02, i, f"Examples: {p}", va='center', fontsize=9.2, color='#1e293b', fontweight='medium')

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig3_source_credibility_tiers.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Generated: {out_path}")

# ==============================================================================
# FIGURE 4: Computational Efficiency & GPU Resource Utilization (Tables III & V)
# ==============================================================================
def generate_fig4_hardware():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.2), dpi=300)

    models = ['US Baseline\n(RoBERTa)', 'SatyaCheck\n(XLM-R Ours)']

    # 1. Trainable Parameters
    params = [85, 170]
    b1 = ax1.bar(models, params, color=['#94a3b8', '#8b5cf6'], width=0.5, edgecolor='#1e293b')
    ax1.set_title('Trainable Parameters (Millions)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Parameters (M)', fontsize=10)
    ax1.set_ylim(0, 210)
    ax1.grid(axis='y', alpha=0.4)
    for b, v in zip(b1, params):
        ax1.text(b.get_x() + b.get_width()/2, v + 4, f'{v} M', ha='center', fontweight='bold', fontsize=9.5)

    # 2. Peak GPU VRAM Usage
    vram = [4.1, 5.2]
    b2 = ax2.bar(models, vram, color=['#94a3b8', '#06b6d4'], width=0.5, edgecolor='#1e293b')
    ax2.set_title('Peak GPU VRAM Usage (RTX 4060)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('VRAM Usage (GB) / 8.0 GB Total', fontsize=10)
    ax2.set_ylim(0, 8.5)
    ax2.axhline(8.0, color='#ef4444', linestyle=':', label='8 GB VRAM Limit')
    ax2.grid(axis='y', alpha=0.4)
    ax2.legend(loc='lower right', fontsize=8.5)
    for b, v in zip(b2, vram):
        ax2.text(b.get_x() + b.get_width()/2, v + 0.18, f'{v} GB', ha='center', fontweight='bold', fontsize=9.5)

    # 3. Training Time per Epoch
    time_min = [12.0, 18.1]
    b3 = ax3.bar(models, time_min, color=['#94a3b8', '#f59e0b'], width=0.5, edgecolor='#1e293b')
    ax3.set_title('Time per Epoch (Minutes)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Duration (Minutes)', fontsize=10)
    ax3.set_ylim(0, 24)
    ax3.grid(axis='y', alpha=0.4)
    for b, v in zip(b3, time_min):
        ax3.text(b.get_x() + b.get_width()/2, v + 0.5, f'{v} min', ha='center', fontweight='bold', fontsize=9.5)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "fig4_hardware_resource_tradeoff.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Generated: {out_path}")

if __name__ == '__main__':
    print("[*] Generating publication figures...")
    generate_fig1_datasets()
    generate_fig2_performance()
    generate_fig3_credibility()
    generate_fig4_hardware()
    print("\n[DONE] All figures successfully generated in 'figures/' directory.")
