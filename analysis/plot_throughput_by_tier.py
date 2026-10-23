#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import os

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

# Data specification
tiers = ["Pi 5 (ARM)", "Ryzen CPU", "Radeon 780M", "RTX 4070"]
quants = ["Q3_K_L", "Q4_K_M", "Q8_0", "F16"]

# Values mapping (None/NaN where data is skipped)
data = {
    "Q3_K_L": [3.80, 12.90, 17.67, 87.39],
    "Q4_K_M": [4.80, 11.67, 16.47, 84.29],
    "Q8_0":   [3.20,  7.20,  9.84, 55.56],
    "F16":    [np.nan, 3.93, np.nan, 32.22]
}

colors = {
    "Q3_K_L": "steelblue",
    "Q4_K_M": "forestgreen",
    "Q8_0": "goldenrod",
    "F16": "tomato"
}

x = np.arange(len(tiers))
total_quants = len(quants)
width = 0.2  # Width of the bars

# Plot bars
rects = []
for i, q in enumerate(quants):
    offset = (i - total_quants / 2 + 0.5) * width
    val_list = data[q]
    
    # Filter out NaNs for plotting to leave slot empty
    valid_x = [x[j] + offset for j in range(len(tiers)) if not np.isnan(val_list[j])]
    valid_y = [val_list[j] for j in range(len(tiers)) if not np.isnan(val_list[j])]
    
    # For positions with NaNs, we can plot with a dummy or skip
    y_plot = [0 if np.isnan(v) else v for v in val_list]
    
    bars = ax.bar(x + offset, y_plot, width, label=q, color=colors[q], alpha=0.9, edgecolor='black', linewidth=0.5)
    
    # Add value labels on top of valid bars
    for j, bar in enumerate(bars):
        val = val_list[j]
        if not np.isnan(val):
            height = bar.get_height()
            ax.annotate(f'{val:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=0)

# Reference horizontal dashed line
ref_y = 84.29
ax.axhline(y=ref_y, color='black', linestyle='--', linewidth=1.2, alpha=0.7, label='Q4_K_M RTX 4070 reference')

# Axes and formatting
ax.set_title("Decode Throughput by Hardware Tier and Quantization Format", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Hardware Tier", fontsize=11, fontweight='bold')
ax.set_ylabel("Decode Speed (Tokens / second)", fontsize=11, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(tiers, fontsize=10, fontweight='bold')
ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)

# Margins and layout
ax.set_ylim(0, 105)
plt.tight_layout()

# Save to exact path
os.makedirs("results", exist_ok=True)
out_path = "results/throughput_by_tier.png"
plt.savefig(out_path)
print(f"SUCCESS: Grouped bar chart saved to {out_path}")
