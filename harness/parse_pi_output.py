#!/usr/bin/env python3
import json
import re
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results")
RAW_LOGS_DIR = RESULTS_DIR / "raw_logs"
PI_POWER_DIR = RESULTS_DIR / "raspberry_pi_5"

quants = ["Q3_K_L", "Q4_K_M", "Q8_0"]

for q in quants:
    q_clean = q.lower().replace("_", "")
    log_file = RAW_LOGS_DIR / f"pi5_{q}.log"
    csv_file = PI_POWER_DIR / f"pi5_{q}_power.csv"
    
    gen_tps = 0.0
    prompt_tps = 0.0
    
    # Parse generation metrics from llama-cli output
    if log_file.exists():
        content = log_file.read_text()
        match = re.search(r'\[\s*Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s\s*\]', content)
        if match:
            prompt_tps = float(match.group(1))
            gen_tps = float(match.group(2))
    
    # Parse mean power from PMIC CSV
    avg_power = 0.0
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        if not df.empty and 'power_w' in df.columns:
            # Filter out initial idle padding
            df_active = df[df['power_w'] > 1.0]
            avg_power = df_active['power_w'].mean() if not df_active.empty else df['power_w'].mean()

    data = {
        "device": "Raspberry Pi 5 (8GB, 2.0 GHz)",
        "tier": "Tier 1: Edge Low-Power ARM",
        "quant": q,
        "prompt_tps": round(prompt_tps, 2),
        "gen_tps": round(gen_tps, 2),
        "avg_power_w": round(avg_power, 3),
        "tokens_per_joule": round(gen_tps / avg_power, 3) if avg_power > 0 else 0.0
    }

    out_json = RESULTS_DIR / f"tier1_pi5_{q_clean}.json"
    out_json.write_text(json.dumps(data, indent=2))
    print(f"Generated {out_json}: {gen_tps} t/s @ {avg_power:.2f}W ({data['tokens_per_joule']} t/J)")

