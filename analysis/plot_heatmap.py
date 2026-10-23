#!/usr/bin/env python3
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.rcParams.update({"font.family": "serif"})

results = []
for f in glob.glob("results/**/*.json", recursive=True):
    try:
        d = json.loads(Path(f).read_text())
        m = d.get("metrics", {})
        if m.get("decode_tokens_per_second") and m.get("energy_joules_per_token"):
            tier = str(d.get("tier", "")).lower()
            if "tier1" in tier or "pi" in tier: t_label = "1. Pi 5 (ARM)"
            elif "tier2" in tier or "cpu" in tier: t_label = "2. Ryzen 8845HS"
            elif "tier3" in tier or "igpu" in tier: t_label = "3. Radeon 780M"
            elif "tier4" in tier or "dgpu" in tier: t_label = "4. RTX 4070"
            else: t_label = tier
            
            # Normalize quantization name (e.g., remove underscores, uppercase)
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
    except Exception as e:
        pass

df = pd.DataFrame(results)

# Safe pivoting with reindex to handle any missing columns gracefully
expected_tiers = ["1. Pi 5 (ARM)", "2. Ryzen 8845HS", "3. Radeon 780M", "4. RTX 4070"]
expected_quants = ["F16", "Q8_0", "Q4_K_M", "Q3_K_L"]

tps_matrix = df.pivot_table(index="Tier", columns="Quantization", values="TPS")
jpt_matrix = df.pivot_table(index="Tier", columns="Quantization", values="JPT")

tps_matrix = tps_matrix.reindex(index=expected_tiers, columns=expected_quants, fill_value=0)
jpt_matrix = jpt_matrix.reindex(index=expected_tiers, columns=expected_quants, fill_value=0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

# Throughput Heatmap (Blues)
sns.heatmap(tps_matrix, annot=True, fmt=".1f", cmap="Blues", ax=ax1, linewidths=1, linecolor='black', cbar_kws={'label': 'Tokens/sec'})
ax1.set_title("Throughput (Tokens/s) $\\uparrow$", fontweight="bold", pad=15)
ax1.set_ylabel("Hardware Tier")
ax1.set_xlabel("Quantization")

# Energy Heatmap (OrRd)
sns.heatmap(jpt_matrix, annot=True, fmt=".3f", cmap="OrRd_r", ax=ax2, linewidths=1, linecolor='black', cbar_kws={'label': 'Joules/Token'})
ax2.set_title("Energy Efficiency (Joules/Token) $\\downarrow$", fontweight="bold", pad=15)
ax2.set_ylabel("")
ax2.set_xlabel("Quantization")

plt.suptitle("Llama 3.2 3B Inference Metrics Matrix", fontsize=16, fontweight="bold", y=1.05)
plt.tight_layout()
plt.savefig("results/heatmap_matrix.png", dpi=300, bbox_inches='tight')
print("Heatmap matrix successfully saved to results/heatmap_matrix.png")
