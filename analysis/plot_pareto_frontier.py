#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import os

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(13, 8), dpi=300)

# Data dictionary grouped by hardware tier
tiers_data = {
    "Pi 5 (ARM)": {
        "color": "coral",
        "points": {
            "Q3KL": (1.195, 3.80),
            "Q4KM": (0.929, 4.80),
            "Q80":  (1.257, 3.20)
        },
        "offsets": {
            "Q3KL": (-15, -15),
            "Q4KM": (-15, 12),
            "Q80":  (15, -10)
        }
    },
    "CPU Ryzen": {
        "color": "steelblue",
        "points": {
            "F16":  (0.800, 3.93),
            "Q3KL": (0.293, 12.90),
            "Q4KM": (0.306, 11.67),
            "Q80":  (0.471, 7.20)
        },
        "offsets": {
            "F16":  (-15, 12),
            "Q3KL": (-20, 12),
            "Q4KM": (15, -12),
            "Q80":  (15, 10)
        }
    },
    "Radeon 780M": {
        "color": "mediumorchid",
        "points": {
            "Q3KL": (2.123, 17.67),
            "Q4KM": (1.882, 16.47),
            "Q80":  (3.061, 9.84)
        },
        "offsets": {
            "Q3KL": (-15, 12),
            "Q4KM": (15, -15),
            "Q80":  (15, 10)
        }
    },
    "RTX 4070": {
        "color": "forestgreen",
        "points": {
            "F16":  (2.173, 32.22),
            "Q3KL": (0.802, 87.39),
            "Q4KM": (0.828, 84.29),
            "Q80":  (1.266, 55.56)
        },
        "offsets": {
            "F16":  (18, -5),
            "Q3KL": (-22, 12),
            "Q4KM": (20, -10),
            "Q80":  (15, 10)
        }
    }
}

# Plot points and add annotations
for tier, info in tiers_data.items():
    color = info["color"]
    for q, (x, y) in info["points"].items():
        ax.scatter(x, y, s=150, color=color, alpha=0.9, edgecolors='black', linewidth=0.6, zorder=3)
        
        # Offset labels
        dx, dy = info["offsets"].get(q, (10, 10))
        ax.annotate(f"{q}", (x, y),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=8, fontweight='bold', color='#222222',
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
                    zorder=4)

# Pareto-optimal connecting dashed line
# Points: CPU Q3KL (0.293, 12.90) -> RTX4070 Q3KL (0.802, 87.39)
pareto_x = [0.293, 0.802]
pareto_y = [12.90, 87.39]
ax.plot(pareto_x, pareto_y, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, label='Pareto Frontier', zorder=2)

# Shaded arrow / annotation pointing toward optimal zone (top-left: lower J/token, higher T/s)
ax.annotate('Optimal zone', xy=(0.32, 75.0), xytext=(0.5, 50.0),
            arrowprops=dict(facecolor='gray', shrink=0.05, width=1.5, headwidth=8, alpha=0.5),
            fontsize=11, fontstyle='italic', color='dimgray',
            ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8, linestyle="--"))

# Axes scaling & formatting
ax.set_xscale('log')
ax.set_title('LLM Inference Pareto Frontier: Decode Speed vs Energy Efficiency', fontsize=14, fontweight='bold', pad=15)
ax.text(0.5, 1.02, 'Model: Llama-3.2-3B-Instruct | Workload: 999 tokens | 4 hardware tiers', 
        transform=ax.transAxes, ha='center', fontsize=10, color='dimgray')

ax.set_xlabel('Energy per Token (J/token) -- log scale -- lower is better', fontsize=11, fontweight='bold')
ax.set_ylabel('Decode Speed (Tokens / second) -- higher is better', fontsize=11, fontweight='bold')

# Custom legend for hardware tiers using colored filled circles
legend_handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=info['color'], markersize=9, label=tier)
    for tier, info in tiers_data.items()
]
# Include Pareto frontier line in legend if desired, or keep tier legend clean
ax.legend(handles=legend_handles, loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)

plt.tight_layout()

# Save to exact output path
os.makedirs("results", exist_ok=True)
out_path = "results/pareto_frontier.png"
plt.savefig(out_path)
print(f"SUCCESS: Pareto frontier chart saved to {out_path}")
