#!/usr/bin/env python3
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.lines import Line2D

# Known Relative Accuracy retention for Llama 3 3B GGUF Quantizations
ACCURACY_MAP = {
    "F16": 100.0,
    "Q8_0": 99.9,
    "Q4_K_M": 97.5,
    "Q3_K_L": 92.0
}

RESULTS_DIR = Path("results")
results_list = []

# Parse JSON files
for f in RESULTS_DIR.rglob("*.json"):
    try:
        data = json.loads(f.read_text())
        gen_tps, watts, quant = 0.0, 0.0, ""
        
        if "metrics" in data:
            gen_tps = data["metrics"].get("decode_tokens_per_second", 0)
            watts = data["metrics"].get("mean_power_watts", 0)
            quant = data.get("quantization", "")
        else:
            gen_tps = data.get("gen_tps", 0)
            watts = data.get("avg_power_w", 0)
            quant = data.get("quant", "")
            
        if gen_tps <= 0: continue
        
        fname = str(f).lower()
        if "pi5" in fname: tier = "Tier 1: Raspberry Pi 5"
        elif "ryzen" in fname or "cpu" in fname: tier = "Tier 2: Ryzen 8845HS CPU"
        elif "780m" in fname: tier = "Tier 3: Radeon 780M iGPU"
        elif "4070" in fname or "cuda" in fname: tier = "Tier 4: RTX 4070 dGPU"
        else: tier = "Unknown"
            
        quant = quant.upper().replace("_", "")
        if quant == "Q4KM": quant = "Q4_K_M"
        elif quant == "Q3KL": quant = "Q3_K_L"
        elif quant == "Q80": quant = "Q8_0"
        
        joules_per_token = watts / gen_tps if gen_tps > 0 else 0
        
        results_list.append({
            "Tier": tier,
            "Quantization": quant,
            "Joules/Token": joules_per_token,
            "Relative Accuracy (%)": ACCURACY_MAP.get(quant, 90.0)
        })
    except Exception:
        pass

df = pd.DataFrame(results_list)
df = df.sort_values(by=["Tier", "Relative Accuracy (%)"], ascending=[True, False])

# Print Terminal Table
print("\n" + "="*70)
print(f"{'Hardware Tier':<28} | {'Quant':<6} | {'Accuracy (%)':<12} | {'Joules/Token'}")
print("-" * 70)
for _, r in df.iterrows():
    print(f"{r['Tier']:<28} | {r['Quantization']:<6} | {r['Relative Accuracy (%)']:<12.1f} | {r['Joules/Token']:.3f} J")
print("="*70 + "\n")

# Render Connected Scatter Plot
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(12, 8))

tier_colors = {
    "Tier 1: Raspberry Pi 5": "#2ecc71",
    "Tier 2: Ryzen 8845HS CPU": "#3498db",
    "Tier 3: Radeon 780M iGPU": "#9b59b6",
    "Tier 4: RTX 4070 dGPU": "#e74c3c"
}

quant_markers = {"Q3_K_L": "o", "Q4_K_M": "s", "Q8_0": "^", "F16": "P"}

for tier in df["Tier"].unique():
    subset = df[df["Tier"] == tier].sort_values("Relative Accuracy (%)")
    if not subset.empty:
        ax.plot(subset["Joules/Token"], subset["Relative Accuracy (%)"], 
                color=tier_colors.get(tier, "gray"), linestyle="--", linewidth=2.5, alpha=0.5, zorder=1)
        
        for _, row in subset.iterrows():
            q = row["Quantization"]
            ax.scatter(row["Joules/Token"], row["Relative Accuracy (%)"],
                       color=tier_colors.get(tier, "black"),
                       marker=quant_markers.get(q, "o"),
                       s=300, edgecolor="black", linewidth=1.5, zorder=2)
            
            ax.annotate(q, (row["Joules/Token"], row["Relative Accuracy (%)"]),
                        textcoords="offset points", xytext=(12, -4), fontsize=10, fontweight="bold")

tier_leg = [Line2D([0], [0], color=c, lw=2.5, linestyle="--", marker='o', markersize=10, 
            markerfacecolor=c, markeredgecolor='black', label=t) for t, c in tier_colors.items()]

ax.legend(handles=tier_leg, title="Hardware Tiers", loc="lower right", fontsize=11, title_fontsize=13)

ax.set_title("Quantization Decay: Accuracy vs. Energy Cost", fontsize=16, fontweight="bold")
ax.set_xlabel("Energy Cost (Joules per Token) → [Lower is Better]", fontsize=13, fontweight="bold")
ax.set_ylabel("Estimated Relative Accuracy (%) → [Higher is Better]", fontsize=13, fontweight="bold")

plt.tight_layout()
out_file = "results/accuracy_vs_energy.png"
plt.savefig(out_file, dpi=300)
print(f"Saved Accuracy Trade-off chart to {out_file}")
