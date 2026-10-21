#!/usr/bin/env python3
import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="darkgrid")
fig, ax = plt.subplots(figsize=(11, 7))

tier_colors = {
    "Tier 1: Edge Low-Power ARM": "#2ecc71",     # Green
    "Tier 2: Ryzen 8845HS CPU": "#3498db",       # Blue
    "Tier 3: Radeon 780M iGPU": "#9b59b6",       # Purple
    "Tier 4: RTX 4070 Laptop dGPU": "#e74c3c"    # Red
}

tier_markers = {
    "Tier 1: Edge Low-Power ARM": "o",
    "Tier 2: Ryzen 8845HS CPU": "s",
    "Tier 3: Radeon 780M iGPU": "^",
    "Tier 4: RTX 4070 Laptop dGPU": "D"
}

results = []
for f in glob.glob("results/*.json"):
    try:
        data = json.loads(Path(f).read_text())
        if "gen_tps" in data and "avg_power_w" in data:
            results.append(data)
    except Exception:
        pass

for r in results:
    tier = r.get("tier", "Unknown Tier")
    color = tier_colors.get(tier, "#7f8c8d")
    marker = tier_markers.get(tier, "o")
    
    tps = r["gen_tps"]
    watts = r["avg_power_w"]
    quant = r.get("quant", "")

    ax.scatter(watts, tps, color=color, marker=marker, s=120, edgecolors="black", alpha=0.9, label=tier)
    ax.annotate(f"{quant}", (watts, tps), textcoords="offset points", xytext=(5, 5), fontsize=9)

# Clean up duplicate legend entries
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc="upper left", frameon=True)

ax.set_title("Llama 3.2 3B Inference: 4-Tier Hardware Trade-Off (Throughput vs. Power)", fontsize=13, fontweight="bold")
ax.set_xlabel("Average Package/Core Power (Watts)", fontsize=11)
ax.set_ylabel("Generation Throughput (Tokens/sec)", fontsize=11)

plt.tight_layout()
plt.savefig("results/pareto_frontier.png", dpi=300)
print("Updated Pareto chart saved to results/pareto_frontier.png")
