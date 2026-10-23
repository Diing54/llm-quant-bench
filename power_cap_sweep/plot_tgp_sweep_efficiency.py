#!/usr/init/env python3
import matplotlib.pyplot as plt
import numpy as np
import os

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# X values (TGP steps)
tgps = [65, 75, 140]
x_labels = ["65W\n(low-power)", "75W\n(balanced)", "140W\n(performance)"]

# Model data specifications (J/token)
models = {
    "Q3_K_L": {
        "values": [0.690, 0.756, 0.904],
        "color": "steelblue",
        "marker": "o",
        "linestyle": "-"
    },
    "Q4_K_M": {
        "values": [0.665, 0.785, 0.844],
        "color": "forestgreen",
        "marker": "s",
        "linestyle": "-"
    },
    "Q8_0": {
        "values": [1.003, 1.215, 1.234],
        "color": "goldenrod",
        "marker": "^",
        "linestyle": "-"
    },
    "F16": {
        "values": [1.880, 2.095, 2.090],
        "color": "tomato",
        "marker": "D",
        "linestyle": "--"
    }
}

# Plot model lines
for model, info in models.items():
    ax.plot(tgps, info["values"], label=model, color=info["color"], 
            marker=info["marker"], linestyle=info["linestyle"], 
            linewidth=2.5, markersize=7, zorder=3)

# Annotation arrow pointing to Q4_K_M and Q3_K_L points at 65W
ax.annotate("Peak efficiency", 
            xy=(65, 0.675), xytext=(78, 0.58),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2, shrinkA=0, shrinkB=5),
            fontsize=9, fontweight='bold', color='black', ha='left', va='center',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8, lw=0.8))

# Invert y-axis (lower values are better, so put them toward the top)
ax.invert_yaxis()

# Axes scaling & formatting
ax.set_title("RTX 4070 Energy Efficiency vs Hardware-Enforced TGP", fontsize=13, fontweight='bold', pad=18)
ax.text(0.5, 1.02, "Lower J/token = more efficient", transform=ax.transAxes, ha='center', fontsize=10, color='dimgray')

ax.set_xlabel("Hardware-Enforced TGP (W)", fontsize=11, fontweight='bold')
ax.set_ylabel("Energy per Token (J/token)", fontsize=11, fontweight='bold')

ax.set_xticks(tgps)
ax.set_xticklabels(x_labels, fontsize=10, fontweight='bold')

# Legend in the lower left
ax.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)

# Note at the bottom of the chart
fig.text(0.5, 0.02, "Note: lower J/token is better. Y-axis is inverted for readability.", 
         ha='center', fontsize=9, fontstyle='italic', color='gray')

plt.tight_layout(rect=[0, 0.03, 1, 1])

# Save to exact output path
os.makedirs("power_cap_sweep/results", exist_ok=True)
out_path = "power_cap_sweep/results/tgp_sweep_efficiency.png"
plt.savefig(out_path)
print(f"SUCCESS: TGP sweep efficiency chart saved to {out_path}")
