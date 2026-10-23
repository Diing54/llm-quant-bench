#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
import os

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

# Data structure
data = [
    # Pi 5
    {"tier": "Pi 5 (ARM)", "quant": "Q8_0", "x": 1.257, "y": 99.9},
    {"tier": "Pi 5 (ARM)", "quant": "Q4_K_M", "x": 0.929, "y": 99.2},
    {"tier": "Pi 5 (ARM)", "quant": "Q3_K_L", "x": 1.195, "y": 96.5},
    # Ryzen CPU
    {"tier": "Ryzen CPU", "quant": "F16", "x": 0.800, "y": 100.0},
    {"tier": "Ryzen CPU", "quant": "Q8_0", "x": 0.471, "y": 99.9},
    {"tier": "Ryzen CPU", "quant": "Q4_K_M", "x": 0.306, "y": 99.2},
    {"tier": "Ryzen CPU", "quant": "Q3_K_L", "x": 0.293, "y": 96.5},
    # Radeon 780M
    {"tier": "Radeon 780M", "quant": "Q8_0", "x": 3.061, "y": 99.9},
    {"tier": "Radeon 780M", "quant": "Q4_K_M", "x": 1.882, "y": 99.2},
    {"tier": "Radeon 780M", "quant": "Q3_K_L", "x": 2.123, "y": 96.5},
    # RTX 4070
    {"tier": "RTX 4070", "quant": "F16", "x": 2.173, "y": 100.0},
    {"tier": "RTX 4070", "quant": "Q8_0", "x": 1.266, "y": 99.9},
    {"tier": "RTX 4070", "quant": "Q4_K_M", "x": 0.828, "y": 99.2},
    {"tier": "RTX 4070", "quant": "Q3_K_L", "x": 0.802, "y": 96.5}
]

tier_colors = {
    "Pi 5 (ARM)": "coral",
    "Ryzen CPU": "steelblue",
    "Radeon 780M": "mediumorchid",
    "RTX 4070": "forestgreen"
}

# Map quantization formats to distinct shapes
quant_markers = {
    "F16": "D",      # Diamond
    "Q8_0": "o",     # Circle
    "Q4_K_M": "s",   # Square
    "Q3_K_L": "^"    # Triangle Up
}

# Plot points
for d in data:
    ax.scatter(d["x"], d["y"], s=120, color=tier_colors[d["tier"]], marker=quant_markers[d["quant"]], 
               alpha=0.9, edgecolors='black', linewidth=0.7, zorder=4)
    
    ax.annotate(d["quant"], (d["x"], d["y"]), xytext=(6, 3), textcoords="offset points", 
                fontsize=8, fontweight='bold', zorder=5)

# Reference line for F16 Baseline
ax.axhline(y=100, color='gray', linestyle='--', linewidth=1.5, zorder=2)
ax.text(3.1, 100.1, "F16 Baseline (100%)", color='gray', fontsize=9, ha='right', fontweight='bold')

# Shaded optimal region
rect = patches.Rectangle((0.0, 98.5), 1.0, 2.5, linewidth=0, edgecolor='none', facecolor='lightgreen', alpha=0.08, zorder=1)
ax.add_patch(rect)
ax.text(0.5, 99.75, "High quality, low cost", color='forestgreen', style='italic', fontsize=10, ha='center', va='center', zorder=3)

# Axes scaling & formatting
ax.set_ylim(95, 101)
ax.set_xlim(left=0, right=3.2)

ax.set_title("Quantization Quality-Efficiency Tradeoff Across Hardware Tiers", fontsize=14, fontweight='bold', pad=20)
ax.text(0.5, 1.03, "Accuracy retention: literature estimates from llama.cpp perplexity benchmarks, not independently measured [Gerganov et al., 2023]", 
        transform=ax.transAxes, ha='center', fontsize=9, color='dimgray')

ax.set_xlabel("Energy per Token (J/token) -- lower is better", fontsize=11, fontweight='bold')
ax.set_ylabel("Estimated Accuracy Retention (% of F16 baseline)", fontsize=11, fontweight='bold')

# Dual Legends
tier_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10, label=t) for t, c in tier_colors.items()]
quant_handles = [Line2D([0], [0], marker=m, color='w', markerfacecolor='black', markersize=10, label=q) for q, m in quant_markers.items()]

leg1 = ax.legend(handles=tier_handles, loc='upper right', title="Hardware Tier", frameon=True, facecolor='white', framealpha=0.9)
leg1.get_title().set_fontweight('bold')
ax.add_artist(leg1)

leg2 = ax.legend(handles=quant_handles, loc='lower right', title="Quantization Format", frameon=True, facecolor='white', framealpha=0.9)
leg2.get_title().set_fontweight('bold')

# Bottom disclaimer note
fig.text(0.5, 0.02, "Accuracy estimates sourced from llama.cpp quantization benchmarks. Values are not independently measured in this study.", 
         ha='center', fontsize=8, fontstyle='italic', color='gray')

# Adjust layout to make room for bottom text
plt.tight_layout(rect=[0, 0.04, 1, 1])

# Save to exact output path
os.makedirs("results", exist_ok=True)
out_path = "results/accuracy_vs_energy.png"
plt.savefig(out_path)
print(f"SUCCESS: Accuracy vs. Energy tradeoff chart saved to {out_path}")
