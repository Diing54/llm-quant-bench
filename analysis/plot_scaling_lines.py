#!/usr/bin/env python3
import json
import glob
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
quant_order = ["F16", "Q8_0", "Q4_K_M", "Q3_K_L"]
tier_order = ["Tier 1: Pi 5 (ARM)", "Tier 2: Ryzen 8845HS", "Tier 3: Radeon 780M", "Tier 4: RTX 4070"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

palette = {"Tier 1: Pi 5 (ARM)": "#2ca02c", "Tier 2: Ryzen 8845HS": "#1f77b4", 
           "Tier 3: Radeon 780M": "#9467bd", "Tier 4: RTX 4070": "#d62728"}

# Left Plot: Throughput Scaling across Quantizations
sns.lineplot(data=df, x="Quantization", y="TPS", hue="Tier", style="Tier", 
             hue_order=tier_order, style_order=tier_order, markers=True, dashes=False, 
             linewidth=2.5, markersize=9, palette=palette, ax=ax1)
ax1.set_title("Throughput Scaling by Quantization $\\uparrow$", fontsize=13, fontweight="bold", pad=12)
ax1.set_xlabel("Model Quantization Tier", fontsize=11, fontweight="bold")
ax1.set_ylabel("Generation Throughput (Tokens/sec)", fontsize=11, fontweight="bold")
ax1.legend(title="Hardware Architecture", frameon=True, edgecolor="black")

# Right Plot: Energy Efficiency Scaling
sns.lineplot(data=df, x="Quantization", y="JPT", hue="Tier", style="Tier", 
             hue_order=tier_order, style_order=tier_order, markers=True, dashes=False, 
             linewidth=2.5, markersize=9, palette=palette, ax=ax2)
ax2.set_title("Energy Scaling by Quantization $\\downarrow$", fontsize=13, fontweight="bold", pad=12)
ax2.set_xlabel("Model Quantization Tier", fontsize=11, fontweight="bold")
ax2.set_ylabel("Energy Consumption (Joules/Token)", fontsize=11, fontweight="bold")
ax2.get_legend().remove() # Clean look, shared legend on left

plt.suptitle("Llama 3.2 3B Inference: Architectural Scaling Analysis", fontsize=16, fontweight="bold", y=1.03)
plt.tight_layout()
plt.savefig("results/scaling_lineplot.png", dpi=300, bbox_inches='tight')
print("Scaling line plot successfully saved to results/scaling_lineplot.png")
