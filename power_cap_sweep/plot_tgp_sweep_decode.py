#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import os

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# X values (TGP steps)
tgps = [65, 75, 140]
x_labels = ["65W\n(low-power)", "75W\n(balanced)", "140W\n(performance)"]

# Model data specifications
models = {
    "Q3_K_L": {
        "values": [77.98, 87.13, 94.23],
        "color": "steelblue",
        "marker": "o",
        "linestyle": "-"
    },
    "Q4_K_M": {
        "values": [80.75, 84.10, 88.02],
        "color": "forestgreen",
        "marker": "s",
        "linestyle": "-"
    },
    "Q8_0": {
        "values": [54.36, 55.51, 56.78],
        "color": "goldenrod",
        "marker": "^",
        "linestyle": "-"
    },
    "F16": {
        "values": [32.01, 32.21, 32.22],
        "color": "tomato",
        "marker": "D",
        "linestyle": "--"
    }
}

# Light gray vertical region between x=65 and x=75
ax.axvspan(65, 75, color='lightgray', alpha=0.4, zorder=0)
# Label at the top of the region (using axes coordinates for y=0.92)
ax.text(70, ax.get_ylim()[1] if False else 98.0, "typical server allocation", 
        ha='center', va='top', fontsize=9, fontstyle='italic', color='dimgray', fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

# Plot model lines
for model, info in models.items():
    ax.plot(tgps, info["values"], label=model, color=info["color"], 
            marker=info["marker"], linestyle=info["linestyle"], 
            linewidth=2.5, markersize=7, zorder=3)

# Annotate F16 line at 140W with a small text label
f16_140w_val = models["F16"]["values"][2]
ax.annotate("Memory BW ceiling\n(not compute limited)", 
            xy=(140, f16_140w_val), xytext=(115, f16_140w_val + 7),
            arrowprops=dict(arrowstyle="->", color="tomato", lw=1.2, shrinkA=0, shrinkB=5),
            fontsize=8, fontweight='bold', color='tomato', ha='center', va='bottom',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="tomato", alpha=0.8, lw=0.8))

# Axes scaling & formatting
ax.set_title("RTX 4070 Decode Throughput vs Hardware-Enforced TGP", fontsize=13, fontweight='bold', pad=18)
ax.text(0.5, 1.02, "ACPI Platform Profile Sweep", transform=ax.transAxes, ha='center', fontsize=10, color='dimgray')

ax.set_xlabel("Hardware-Enforced TGP (W)", fontsize=11, fontweight='bold')
ax.set_ylabel("Decode Speed (Tokens / second)", fontsize=11, fontweight='bold')

ax.set_xticks(tgps)
ax.set_xticklabels(x_labels, fontsize=10, fontweight='bold')

# Y-axis limits for clean headroom
ax.set_ylim(20, 105)
ax.set_xlim(58, 147)

# Legend in the upper left
ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)

plt.tight_layout()

# Save to exact output path
os.makedirs("power_cap_sweep/results", exist_ok=True)
out_path = "power_cap_sweep/results/tgp_sweep_decode.png"
plt.savefig(out_path)
print(f"SUCCESS: TGP sweep decode chart saved to {out_path}")
