#!/usr/bin/env python3
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

RESULTS_DIR = Path("power_cap_sweep/results")
results_list = []

for f in RESULTS_DIR.glob("sweep_*.json"):
    try:
        data = json.loads(f.read_text())
        metrics = data.get("metrics", {})
        
        profile = data.get("platform_profile", "unknown")
        quant_raw = data.get("quantization", "").upper().replace("_", "")
        
        # Normalize quantization labels
        if "Q4" in quant_raw:
            quant = "Q4_K_M"
        elif "Q8" in quant_raw:
            quant = "Q8_0"
        else:
            quant = quant_raw
        
        decode_tps = metrics.get("decode_tokens_per_second", 0.0)
        mean_power = metrics.get("mean_power_watts", 0.0)
        joules_per_token = metrics.get("energy_joules_per_token", 0.0)
        
        if decode_tps > 0:
            results_list.append({
                "Profile": profile,
                "Quantization": quant,
                "Decode T/s": decode_tps,
                "Power (W)": mean_power,
                "Joules/Token": joules_per_token
            })
    except Exception as e:
        print(f"Error parsing {f}: {e}")

df = pd.DataFrame(results_list)
if df.empty:
    print("Error: DataFrame is empty. Check JSON results.")
    exit(1)

df = df.sort_values(by=["Quantization", "Power (W)"])

# 1. Print Terminal Summary Table
print("\n" + "="*75)
print(f"{'Profile':<14} | {'Quant':<8} | {'Decode T/s':<10} | {'Power (W)':<10} | {'Joules/Token'}")
print("-" * 75)
for _, r in df.iterrows():
    print(f"{r['Profile']:<14} | {r['Quantization']:<8} | {r['Decode T/s']:<10.2f} | {r['Power (W)']:<10.2f} | {r['Joules/Token']:.4f} J")
print("="*75 + "\n")

# 2. Render Two-Panel Visualization with explicit color mapping
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

quant_styles = {
    "Q4_K_M": {"color": "#2980b9", "marker": "s", "label": "Llama 3.2 3B (Q4_K_M)"},
    "Q8_0":   {"color": "#c0392b", "marker": "^", "label": "Llama 3.2 3B (Q8_0)"}
}

# Left Panel: Throughput vs Measured Power
ax1 = axes[0]
for quant, style in quant_styles.items():
    subset = df[df["Quantization"] == quant]
    if not subset.empty:
        ax1.plot(subset["Power (W)"], subset["Decode T/s"], 
                 marker=style["marker"], 
                 color=style["color"],
                 linewidth=2.5, markersize=9, label=style["label"])

ax1.set_title("Inference Throughput vs. Measured Power Draw", fontsize=14, fontweight="bold")
ax1.set_xlabel("Measured GPU Power (Watts) →", fontsize=12, fontweight="bold")
ax1.set_ylabel("Decode Throughput (Tokens / sec) →", fontsize=12, fontweight="bold")
ax1.legend(title="Quantization Profile", fontsize=11)

# Right Panel: Energy Efficiency vs Measured Power
ax2 = axes[1]
for quant, style in quant_styles.items():
    subset = df[df["Quantization"] == quant]
    if not subset.empty:
        ax2.plot(subset["Power (W)"], subset["Joules/Token"], 
                 marker=style["marker"], 
                 color=style["color"],
                 linewidth=2.5, markersize=9, label=style["label"])

ax2.set_title("Energy Efficiency vs. Measured Power Draw", fontsize=14, fontweight="bold")
ax2.set_xlabel("Measured GPU Power (Watts) →", fontsize=12, fontweight="bold")
ax2.set_ylabel("Energy Cost (Joules / Token) → [Lower is Better]", fontsize=12, fontweight="bold")
ax2.legend(title="Quantization Profile", fontsize=11)

plt.tight_layout()
out_file = "power_cap_sweep/results/power_sweep_curve.png"
plt.savefig(out_file, dpi=300)
print(f"Updated Power Sweep efficiency curve saved to {out_file}")
