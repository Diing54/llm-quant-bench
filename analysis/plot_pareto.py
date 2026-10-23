#!/usr/bin/env python3
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# IEEE Publication Styling
plt.rcParams.update({
    "font.family": "serif",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.2,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--"
})
sns.set_theme(style="whitegrid")

results = []
for f in glob.glob("results/**/*.json", recursive=True):
    try:
        d = json.loads(Path(f).read_text())
        m = d.get("metrics", {})
        tps = m.get("decode_tokens_per_second")
        jpt = m.get("energy_joules_per_token")
        
        if tps and jpt:
            tier = str(d.get("tier", "")).lower()
            if "tier1" in tier or "pi" in tier: t_label = "Tier 1: Pi 5 (ARM)"
            elif "tier2" in tier or "cpu" in tier: t_label = "Tier 2: Ryzen 8845HS"
            elif "tier3" in tier or "igpu" in tier: t_label = "Tier 3: Radeon 780M"
            elif "tier4" in tier or "dgpu" in tier: t_label = "Tier 4: RTX 4070"
            else: t_label = tier
            
            q_raw = str(d.get("quantization", "")).upper().replace("_", "")
            if "3K" in q_raw or "Q3" in q_raw: q_label = "Q3_K_L"
            elif "4K" in q_raw or "Q4" in q_raw: q_label = "Q4_K_M"
            elif "8" in q_raw: q_label = "Q8_0"
            else: q_label = "F16"

            results.append({
                "Tier": t_label, 
                "Quantization": q_label, 
                "TPS": float(tps), 
                "JPT": float(jpt)
            })
    except Exception:
        pass

df = pd.DataFrame(results)

fig, ax = plt.subplots(figsize=(11, 7), dpi=300)

palette = {
    "Tier 1: Pi 5 (ARM)": "#2ca02c", 
    "Tier 2: Ryzen 8845HS": "#1f77b4", 
    "Tier 3: Radeon 780M": "#9467bd", 
    "Tier 4: RTX 4070": "#d62728"
}
markers = {
    "Tier 1: Pi 5 (ARM)": "o", 
    "Tier 2: Ryzen 8845HS": "s", 
    "Tier 3: Radeon 780M": "^", 
    "Tier 4: RTX 4070": "D"
}

# Plot individual points
for tier, group in df.groupby("Tier"):
    ax.scatter(
        group["JPT"], group["TPS"], 
        color=palette[tier], marker=markers[tier], 
        s=140, edgecolors="black", linewidth=1.2, alpha=0.9, 
        label=tier, zorder=3
    )
    for _, row in group.iterrows():
        ax.annotate(
            f"{row['Quantization']}", 
            (row["JPT"], row["TPS"]), 
            textcoords="offset points", xytext=(6, 6), 
            fontsize=8, fontweight="bold", zorder=4
        )

# Compute Pareto Frontier (Minimizing JPT, Maximizing TPS)
# Sort by JPT ascending
sorted_df = df.sort_values(by="JPT", ascending=True)
pareto_points = []
max_tps = -1
for _, row in sorted_df.iterrows():
    if row["TPS"] > max_tps:
        pareto_points.append((row["JPT"], row["TPS"]))
        max_tps = row["TPS"]

pareto_points = np.array(pareto_points)
if len(pareto_points) > 0:
    ax.plot(
        pareto_points[:, 0], pareto_points[:, 1], 
        color="#333333", linestyle="--", linewidth=2, alpha=0.7, 
        label="Pareto Optimal Frontier", zorder=2
    )

# Highlight Optimal Efficiency Zone
ax.axvspan(0.2, 1.0, color="#e2f0d9", alpha=0.4, zorder=1)
ax.text(
    0.55, 0.92, "Optimal Efficiency Zone\n(< 1.0 Joules/Token)", 
    transform=ax.transAxes, fontsize=10, fontstyle="italic", 
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#2ca02c", alpha=0.9),
    ha="center", va="center", zorder=5
)

ax.set_title("Llama 3.2 3B Inference: Pareto Frontier (Throughput vs. Energy Cost)", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Energy Cost Ranking (Joules per Token) $\\downarrow$ (Lower is Better)", fontsize=11, fontweight="bold")
ax.set_ylabel("Generation Throughput (Tokens/sec) $\\uparrow$ (Higher is Better)", fontsize=11, fontweight="bold")

ax.legend(title="Hardware Architecture", loc="upper left", frameon=True, edgecolor="black", fontsize=10)

plt.tight_layout()
plt.savefig("results/pareto_frontier.png", dpi=300, bbox_inches='tight')
print("Advanced Pareto Frontier chart successfully saved to results/pareto_frontier.png")
