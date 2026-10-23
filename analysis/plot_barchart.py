#!/usr/bin/env python3
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.rcParams.update({"font.family": "serif", "axes.edgecolor": "black"})
sns.set_theme(style="whitegrid", rc={"grid.color": "#e0e0e0", "grid.linestyle": "--"})

results = []
for f in glob.glob("results/**/*.json", recursive=True):
    try:
        d = json.loads(Path(f).read_text())
        m = d.get("metrics", {})
        tps = m.get("decode_tokens_per_second")
        jpt = m.get("energy_joules_per_token")
        q = str(d.get("quantization", "")).upper()
        tier = str(d.get("tier", "")).lower()
        
        if tps and jpt:
            # Format tier names for clean Y-axis labels
            if "tier1" in tier or "pi" in tier: t_label = "Tier 1: Pi 5 (ARM)"
            elif "tier2" in tier or "cpu" in tier: t_label = "Tier 2: Ryzen 8845HS"
            elif "tier3" in tier or "igpu" in tier: t_label = "Tier 3: Radeon 780M"
            elif "tier4" in tier or "dgpu" in tier: t_label = "Tier 4: RTX 4070"
            else: t_label = tier
            
            results.append({"Tier": t_label, "Quantization": q, "Throughput (Tokens/s)": float(tps), "Energy (Joules/Token)": float(jpt)})
    except Exception:
        pass

df = pd.DataFrame(results)

# Sort logically
tier_order = ["Tier 1: Pi 5 (ARM)", "Tier 2: Ryzen 8845HS", "Tier 3: Radeon 780M", "Tier 4: RTX 4070"]
quant_order = ["F16", "Q8_0", "Q4_K_M", "Q3_K_L"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

# Plot 1: Throughput
sns.barplot(data=df, y="Tier", x="Throughput (Tokens/s)", hue="Quantization", 
            order=tier_order, hue_order=quant_order, ax=ax1, palette="viridis", edgecolor="black", linewidth=1)
ax1.set_title("Generation Throughput $\\uparrow$ (Higher is Better)", fontweight="bold", pad=10)
ax1.set_ylabel("")
ax1.set_xlabel("Tokens per Second")
ax1.get_legend().remove()

# Plot 2: Energy
sns.barplot(data=df, y="Tier", x="Energy (Joules/Token)", hue="Quantization", 
            order=tier_order, hue_order=quant_order, ax=ax2, palette="magma", edgecolor="black", linewidth=1)
ax2.set_title("Energy Efficiency $\\downarrow$ (Lower is Better)", fontweight="bold", pad=10)
ax2.set_ylabel("")
ax2.set_xlabel("Joules per Token")
ax2.set_yticklabels([]) # Hide Y labels on second chart to save space

# Legend
ax2.legend(title="Quantization", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, edgecolor="black")

plt.suptitle("Llama 3.2 3B Inference: Cross-Architecture Performance & Energy Scaling", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("results/grouped_barchart.png", dpi=300, bbox_inches='tight')
print("Grouped bar chart saved to results/grouped_barchart.png")
