#!/usr/bin/env python3
import json
import glob
import pandas as pd
import numpy as np
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
                "JPT": float(jpt),
                "Config": f"{t_label} ({q_label})"
            })
    except Exception:
        pass

df = pd.DataFrame(results)
palette = {
    "Tier 1: Pi 5 (ARM)": "#2ca02c", 
    "Tier 2: Ryzen 8845HS": "#1f77b4", 
    "Tier 3: Radeon 780M": "#9467bd", 
    "Tier 4: RTX 4070": "#d62728"
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

# -------------------------------------------------------------
# PANEL A: Pareto Frontier (Throughput vs. Energy)
# -------------------------------------------------------------
for tier, group in df.groupby("Tier"):
    ax1.scatter(group["JPT"], group["TPS"], color=palette.get(tier, "gray"), 
                s=140, edgecolors="black", linewidth=1.2, label=tier, zorder=3)
    for _, row in group.iterrows():
        ax1.annotate(row["Quantization"], (row["JPT"], row["TPS"]), 
                     textcoords="offset points", xytext=(6, 6), fontsize=8, fontweight="bold")

# Compute Pareto Front (minimize JPT, maximize TPS)
points = df[["JPT", "TPS", "Config"]].sort_values("JPT").values
pareto_frontier = []
max_tps = -1
for jpt, tps, config in points:
    if tps > max_tps:
        pareto_frontier.append((jpt, tps))
        max_tps = tps

p_jpt, p_tps = zip(*pareto_frontier)
ax1.plot(p_jpt, p_tps, color="black", linestyle="--", linewidth=2, label="Pareto Frontier", zorder=2)

ax1.set_title("Panel A: Pareto Frontier (Speed vs. Energy)", fontsize=13, fontweight="bold", pad=12)
ax1.set_xlabel("Energy Consumption (Joules/Token) $\\downarrow$ Better", fontsize=11, fontweight="bold")
ax1.set_ylabel("Generation Throughput (Tokens/sec) $\\uparrow$ Better", fontsize=11, fontweight="bold")
ax1.legend(loc="upper left", frameon=True, edgecolor="black", fontsize=9)

# -------------------------------------------------------------
# PANEL B: Energy Cost Ranking (Sorted Efficiency)
# -------------------------------------------------------------
df_sorted = df.sort_values("JPT", ascending=True).reset_index(drop=True)
bars = ax2.barh(df_sorted["Config"], df_sorted["JPT"], color=df_sorted["Tier"].map(palette), edgecolor="black", linewidth=1)

# Add exact value labels on bars
for bar in bars:
    width = bar.get_width()
    ax2.text(width + 0.03, bar.get_y() + bar.get_height()/2, f"{width:.3f} J", 
             va='center', ha='left', fontsize=8, fontweight='bold')

ax2.set_title("Panel B: Energy Cost Ranking (Most to Least Efficient)", fontsize=13, fontweight="bold", pad=12)
ax2.set_xlabel("Energy Consumption (Joules/Token) $\\downarrow$ Better", fontsize=11, fontweight="bold")
ax2.set_ylabel("")
ax2.set_xlim(0, df_sorted["JPT"].max() * 1.15)

# Invert Y axis so most efficient is at the top
ax2.invert_yaxis()

plt.suptitle("Llama 3.2 3B Inference: Comprehensive Efficiency & Pareto Analysis", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("results/master_pareto_ranking.png", dpi=300, bbox_inches='tight')
print("Master Pareto & Ranking Graphic successfully saved to results/master_pareto_ranking.png")
