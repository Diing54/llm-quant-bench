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
        if m.get("decode_tokens_per_second") and m.get("energy_joules_per_token"):
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
                "TPS": float(m["decode_tokens_per_second"]), 
                "JPT": float(m["energy_joules_per_token"])
            })
    except Exception:
        pass

df = pd.DataFrame(results)
df["Config"] = df["Tier"] + " (" + df["Quantization"] + ")"

# Create a 2-panel figure: Left = Pareto Frontier, Right = Energy Cost Ranking Bar Chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300, gridspec_kw={'width_ratios': [1.3, 1]})

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

# --- PANEL 1: Pareto Frontier ---
for tier, group in df.groupby("Tier"):
    ax1.scatter(group["JPT"], group["TPS"], 
                color=palette[tier], marker=markers[tier], 
                s=120, edgecolors="black", linewidth=1.2, alpha=0.9, 
                label=tier, zorder=3)
    
    for _, row in group.iterrows():
        ax1.annotate(row["Quantization"], (row["JPT"], row["TPS"]), 
                     textcoords="offset points", xytext=(5, 5), 
                     fontsize=7, fontweight="bold", color="#333333", zorder=4)

# Compute Pareto Frontier
points = df[["JPT", "TPS"]].values
sorted_indices = np.argsort(points[:, 0])
sorted_pts = points[sorted_indices]

pareto_pts = [sorted_pts[0]]
for pt in sorted_pts[1:]:
    if pt[1] > pareto_pts[-1][1]:
        pareto_pts.append(pt)
pareto_pts = np.array(pareto_pts)

ax1.plot(pareto_pts[:, 0], pareto_pts[:, 1], color="#e377c2", linestyle="--", linewidth=2, label="Pareto Frontier", zorder=2)
ax1.axvspan(0.2, 0.9, color="#2ca02c", alpha=0.06, zorder=1)
ax1.text(0.55, ax1.get_ylim()[1]*0.88, "Optimal Zone", 
         horizontalalignment='center', fontsize=9, style='italic', fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2ca02c", alpha=0.8))

ax1.set_title("(a) Efficiency vs. Speed Pareto Trade-Off", fontsize=12, fontweight="bold", pad=10)
ax1.set_xlabel("Energy Consumption (Joules/Token) $\\downarrow$", fontsize=10, fontweight="bold")
ax1.set_ylabel("Throughput (Tokens/sec) $\\uparrow$", fontsize=10, fontweight="bold")

# Move Legend outside to the right of ax1
ax1.legend(title="Hardware Architecture", loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True, edgecolor="black", fontsize=8, title_fontsize=9)


# --- PANEL 2: Horizontal Energy Cost Ranking Bar Chart ---
df_sorted = df.sort_values(by="JPT", ascending=True) # Ascending so lowest energy cost is at the top
bar_colors = [palette[t] for t in df_sorted["Tier"]]

bars = ax2.barh(df_sorted["Config"], df_sorted["JPT"], color=bar_colors, edgecolor="black", linewidth=0.8, alpha=0.85)
ax2.set_title("(b) Energy Cost Ranking (Joules/Token)", fontsize=12, fontweight="bold", pad=10)
ax2.set_xlabel("Joules per Token $\\downarrow$ (Lower is Better)", fontsize=10, fontweight="bold")
ax2.set_ylabel("")
ax2.tick_params(axis='y', labelsize=8)

# Add value annotations on bars
for bar in bars:
    width = bar.get_width()
    ax2.annotate(f'{width:.3f} J',
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0),  # 5 points horizontal offset
                textcoords="offset points",
                ha='left', va='center', fontsize=7, fontweight='bold')

plt.suptitle("Llama 3.2 3B Inference: Comprehensive Energy & Performance Analysis", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("results/creative_pareto_frontier.png", dpi=300, bbox_inches='tight')
print("Creative multi-panel Pareto chart successfully saved to results/creative_pareto_frontier.png")
