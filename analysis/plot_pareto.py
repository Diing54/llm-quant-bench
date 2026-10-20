#!/usr/bin/env python3
import json
import glob
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    results = []
    for path in sorted(glob.glob("results/**/*.json", recursive=True)):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Skipping {path}: {e}")

    if not results:
        print("No result JSON files found under results/ !")
        return

    # Map by 'tier' to correctly catch cpu, tier3, and tier4
    tiers = {
        "cpu": {"name": "CPU (Ryzen 8845HS)", "color": "#1f77b4", "marker": "o"},
        "tier3": {"name": "iGPU (Radeon 780M Vulkan)", "color": "#ff7f0e", "marker": "s"},
        "tier4": {"name": "dGPU (RTX 4070 CUDA)", "color": "#2ca02c", "marker": "^"}
    }

    plt.figure(figsize=(11, 7))

    for r in results:
        tier = r.get("tier")
        quant = r.get("quantization", "").upper()
        metrics = r.get("metrics", {})
        
        decode = metrics.get("decode_tokens_per_second", 0.0)
        energy = metrics.get("energy_joules_per_token", 0.0)

        if tier in tiers and energy > 0 and decode > 0:
            cfg = tiers[tier]
            # Plot with thin black edge for high visual contrast
            plt.scatter(energy, decode, color=cfg["color"], marker=cfg["marker"], s=130, edgecolors='black', linewidths=0.8, zorder=3)
            plt.annotate(f"{tier}-{quant}", (energy, decode), textcoords="offset points", xytext=(0,9), ha='center', fontsize=9, fontweight='semibold', zorder=4)

    plt.title("LLM Inference Pareto Frontier: Throughput vs. Energy Efficiency", fontsize=14, fontweight='bold', pad=20)
    plt.xlabel("Energy Efficiency (Joules / Token) [Lower is Better — Log Scale]", fontsize=11, fontweight='semibold')
    plt.ylabel("Decode Throughput (Tokens / Second) [Higher is Better]", fontsize=11, fontweight='semibold')
    plt.xscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.6, zorder=1)
    
    # Custom legend outside
    handles = [plt.Line2D([0], [0], marker=cfg["marker"], color='w', markerfacecolor=cfg["color"], markersize=11, markeredgecolor='black', label=cfg["name"]) for cfg in tiers.values()]
    plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True, facecolor='#f8f9fa', edgecolor='#cccccc', fontsize=10)

    plt.tight_layout()
    plot_path = Path("results/pareto_frontier.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Creative Pareto frontier plot updated successfully at {plot_path}")

if __name__ == "__main__":
    main()
