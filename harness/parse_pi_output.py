#!/usr/bin/env python3
import json
import re
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results")
RAW_LOGS_DIR = RESULTS_DIR / "raw_logs"
PI_POWER_DIR = RESULTS_DIR / "raspberry_pi_5"

for log_file in RAW_LOGS_DIR.glob("pi5_*.log"):
    filename_lower = log_file.name.lower()
    if "q3" in filename_lower:
        q_clean = "q3kl"
        csv_name = "pi5_Q3_K_L_power.csv"
    elif "q4" in filename_lower:
        q_clean = "q4km"
        csv_name = "pi5_Q4_K_M_power.csv"
    elif "q8" in filename_lower:
        q_clean = "q80"
        csv_name = "pi5_Q8_0_power.csv"
    else:
        continue

    csv_file = PI_POWER_DIR / csv_name
    
    prefill_tps = 0.0
    decode_tps = 0.0
    
    content = log_file.read_text(encoding="utf-8", errors="ignore")
    
    # Regex search for the old bracketed format: [ Prompt: 17.2 t/s | Generation: 3.2 t/s ]
    match = re.search(r'\[\s*Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s\s*\]', content, re.IGNORECASE)
    if match:
        prefill_tps = float(match.group(1))
        decode_tps = float(match.group(2))

    avg_power = 0.0
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        if not df.empty and 'power_w' in df.columns:
            df_active = df[df['power_w'] > 1.0]
            avg_power = df_active['power_w'].mean() if not df_active.empty else df['power_w'].mean()

    energy_per_token = (avg_power / decode_tps) if decode_tps > 0 else 0.0

    data = {
        "tier": "tier1_arm",
        "device": "pi5",
        "quantization": q_clean,
        "metrics": {
            "prefill_tokens_per_second": round(prefill_tps, 2),
            "decode_tokens_per_second": round(decode_tps, 2),
            "mean_power_watts": round(avg_power, 2),
            "energy_joules_per_token": round(energy_per_token, 4)
        }
    }

    out_json = RESULTS_DIR / "tier1_arm" / f"tier1_pi5_{q_clean}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(data, indent=2))
    print(f"Parsed Pi {q_clean}: {decode_tps} T/s @ {avg_power:.2f}W ({energy_per_token:.4f} J/token)")
