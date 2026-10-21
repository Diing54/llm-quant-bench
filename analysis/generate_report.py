#!/usr/bin/env python3
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.lines import Line2D

RESULTS_DIR = Path("results")
results_list = []

# Parse all tier JSON files recursively
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
        if "tier1" in fname or "pi5" in fname: tier = "Tier 1: Raspberry Pi 5"
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

print("\n" + "="*85)
print(f"{'Hardware Tier':<28} | {'Quant':<6} | {'Gen t/s':<8} | {'Power (W)':<9} | {'Joules/Token'}")
print("-" * 85)
for _, r in df.iterrows():
    print(f"{r['Tier']:<28} | {r['Quantization']:<6} | {r['Throughput (t/s)']:<8.2f} | {r['Power (W)']:<9.2f} | {r['Joules/Token']:.3f} J")
print("="*85 + "\n")
