#!/usr/init/env python3
import json
import glob
import os
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

results_dir = "power_cap_sweep/results"
json_files = glob.glob(os.path.join(results_dir, "*.json"))

data_points = []
print("\n" + "="*80)
print(f"{'PROFILE':<12} | {'TGP (W)':<8} | {'QUANT':<8} | {'DECODE (tok/s)':<16} | {'POWER (W)':<12} | {'JOULES/TOK':<12}")
print("="*80)

for file in sorted(json_files):
    with open(file, 'r') as f:
        d = json.load(f)
        metrics = d.get("metrics", {})
        profile = d.get("acpi_profile", "unknown")
        tgp = d.get("hardware_tgp_watts", 75.0)
        quant = d.get("quantization", d.get("quant", "unknown"))
        tps = metrics.get("decode_tokens_per_second", 0)
        power = metrics.get("mean_power_watts", 0)
        joules = metrics.get("energy_joules_per_token", 0)
        
        data_points.append({
            "profile": profile,
            "tgp": tgp,
            "quant": quant,
            "tps": tps,
            "power": power,
            "joules": joules
        })
        print(f"{profile:<12} | {tgp:<8.1f} | {quant.upper():<8} | {tps:<16.2f} | {power:<12.2f} | {joules:<12.4f}")

print("="*80 + "\n")

quants = sorted(list(set([d["quant"] for d in data_points])))
colors = {'f16': '#e41a1c', 'q80': '#377eb8', 'q4km': '#4daf4a', 'q3kl': '#984ea3'}

# Panel 1: Pareto Frontier
ax1 = axes[0]
for q in quants:
    q_data = sorted([d for d in data_points if d["quant"] == q], key=lambda x: x["tgp"])
    if not q_data:
        continue
    tps_vals = [d["tps"] for d in q_data]
    joule_vals = [d["joules"] for d in q_data]
    
    ax1.plot(joule_vals, tps_vals, marker='o', linestyle='-', linewidth=2, 
             color=colors.get(q, '#333333'), label=f'Model: {q.upper()}')
    
    for d in q_data:
        ax1.annotate(f"{int(d['tgp'])}W", (d["joules"], d["tps"]),
                     textcoords="offset points", xytext=(0,8), ha='center', fontsize=8, fontweight='bold')

ax1.set_title("Study 2: Power-Constrained Pareto Frontier", fontsize=12, fontweight='bold')
ax1.set_xlabel("Energy Efficiency (Joules / Token) [Lower is Better]", fontsize=10)
ax1.set_ylabel("Throughput (Decode Tokens / Sec) [Higher is Better]", fontsize=10)
ax1.legend(frameon=True, facecolor='white', framealpha=0.9)

# Panel 2: Throughput Scaling across Power Envelopes
ax2 = axes[1]
for q in quants:
    q_data = sorted([d for d in data_points if d["quant"] == q], key=lambda x: x["tgp"])
    if not q_data:
        continue
    tgps = [d["tgp"] for d in q_data]
    tps_vals = [d["tps"] for d in q_data]
    
    ax2.plot(tgps, tps_vals, marker='s', linestyle='--', linewidth=2,
             color=colors.get(q, '#333333'), label=f'Model: {q.upper()}')

ax2.set_title("Throughput Scaling vs. Hardware TGP Envelopes", fontsize=12, fontweight='bold')
ax2.set_xlabel("Hardware Power Limit (Watts TGP)", fontsize=10)
ax2.set_ylabel("Throughput (Decode Tokens / Sec)", fontsize=10)
ax2.set_xticks([65, 75, 140])
ax2.legend(frameon=True, facecolor='white', framealpha=0.9)

plt.tight_layout()
out_plot = os.path.join(results_dir, "study2_power_scaling_pareto.png")
plt.savefig(out_plot)
print(f"SUCCESS: Updated visualization saved to {out_plot}")
