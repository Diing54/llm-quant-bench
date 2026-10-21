#!/usr/bin/env python3
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.lines import Line2D

RESULTS_DIR = Path("results")
results_list = []

# 1. Parse JSON files robustly
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
        
        # Map tiers safely based on filename to catch all variations
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
        
        results_list.append({
            "Tier": tier,
            "Quantization": quant,
            "Throughput (t/s)": round(gen_tps, 2),
            "Power (W)": round(watts, 2),
            "Joules/Token": round(watts / gen_tps, 3) if gen_tps > 0 else 0
        })
    except Exception:
        pass

df = pd.DataFrame(results_list)
df = df.sort_values(by=["Tier", "Throughput (t/s)"])

# 2. Print Terminal Table
print("\n" + "="*85)
print(f"{'Hardware Tier':<28} | {'Quant':<6} | {'Gen t/s':<8} | {'Power (W)':<9} | {'Joules/Token'}")
print("-" * 85)
for _, r in df.iterrows():
    print(f"{r['Tier']:<28} | {r['Quantization']:<6} | {r['Throughput (t/s)']:<8.2f} | {r['Power (W)']:<9.2f} | {r['Joules/Token']:.3f} J")
print("="*85 + "\n")

# 3. Render Dual-Panel Visualization
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

tier_colors = {
    "Tier 1: Raspberry Pi 5": "#2ecc71",      # Green
    "Tier 2: Ryzen 8845HS CPU": "#3498db",    # Blue
    "Tier 3: Radeon 780M iGPU": "#9b59b6",    # Purple
    "Tier 4: RTX 4070 dGPU": "#e74c3c"        # Red
}

quant_markers = {
    "Q3_K_L": "o",   # Circle
    "Q4_K_M": "s",   # Square
    "Q8_0": "^",     # Triangle
    "F16": "P"       # Plus
}

# Left Panel: Pareto Scatter Plot
ax1 = axes[0]
for tier in df["Tier"].unique():
    for quant in df["Quantization"].unique():
        subset = df[(df["Tier"] == tier) & (df["Quantization"] == quant)]
        if not subset.empty:
            ax1.scatter(subset["Joules/Token"], subset["Throughput (t/s)"],
                        color=tier_colors.get(tier, "black"), 
                        marker=quant_markers.get(quant, "o"),
                        s=300, edgecolor="black", linewidth=1.5, alpha=0.8)

# Custom Legends to prevent text clutter
tier_leg = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=14, label=t) for t, c in tier_colors.items()]
quant_leg = [Line2D([0], [0], marker=m, color='w', markerfacecolor='gray', markersize=14, label=q) for q, m in quant_markers.items()]

leg1 = ax1.legend(handles=tier_leg, title="Hardware Tiers", loc="upper right", fontsize=10)
ax1.add_artist(leg1)
ax1.legend(handles=quant_leg, title="Quantization Profile", loc="lower right", fontsize=10)

ax1.set_title("Inference Pareto Frontier\n(Optimal Zone: Top-Left)", fontsize=15, fontweight="bold")
ax1.set_ylabel("Generation Throughput (Tokens / sec) →", fontsize=12, fontweight="bold")
ax1.set_xlabel("← Energy Cost (Joules / Token)", fontsize=12, fontweight="bold")

# Right Panel: Ranked Bar Chart
ax2 = axes[1]
df_sorted = df.sort_values(by="Joules/Token", ascending=False)
df_sorted["Label"] = df_sorted["Tier"].str.split(":").str[1].str.strip() + " (" + df_sorted["Quantization"] + ")"

bars = sns.barplot(data=df_sorted, x="Joules/Token", y="Label", hue="Tier", palette=tier_colors, dodge=False, ax=ax2)
ax2.set_title("Energy Cost Ranking (Lower is Better)", fontsize=15, fontweight="bold")
ax2.set_xlabel("Joules per Token", fontsize=12, fontweight="bold")
ax2.set_ylabel("")
ax2.legend_.remove()

# Add numerical labels to bars
for container in ax2.containers:
    ax2.bar_label(container, fmt='%.3f J', padding=5, fontsize=10)

plt.tight_layout(pad=3.0)
plt.savefig("results/pareto_frontier.png", dpi=300)
print("Saved dual-panel dashboard to results/pareto_frontier.png")
