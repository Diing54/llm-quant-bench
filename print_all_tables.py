#!/usr/bin/env python3
import json
import glob
import os

def print_separator():
    print("=" * 105)

# =========================================================================
# TABLE 1: Study 1 - Performance & Power Metrics Across All Hardware Tiers
# =========================================================================
print("\n" + "="*105)
print("TABLE 1: Study 1 - Performance & Power Metrics Across All Hardware Tiers")
print("="*105)
print(f"{'HARDWARE / TIER':<22} | {'QUANT':<8} | {'PREFILL (tok/s)':<16} | {'DECODE (tok/s)':<16} | {'POWER (W)':<12} | {'JOULES/TOK':<12}")
print("-"*105)

tier_files = sorted(glob.glob("results/**/*.json", recursive=True))
for file in tier_files:
    if "power_cap_sweep" in file or ".venv" in file or "CMakeFiles" in file:
        continue
    try:
        with open(file, 'r') as f:
            d = json.load(f)
            metrics = d.get("metrics", d)
            tier = d.get("tier", os.path.basename(os.path.dirname(file)))
            quant = d.get("quantization", d.get("quant", "unknown")).upper()
            prefill = metrics.get("prefill_tokens_per_second", 0)
            decode = metrics.get("decode_tokens_per_second", metrics.get("tokens_per_second", 0))
            power = metrics.get("mean_power_watts", 0)
            joules = metrics.get("energy_joules_per_token", 0)
            
            print(f"{tier:<22} | {quant:<8} | {prefill:<16.2f} | {decode:<16.2f} | {power:<12.2f} | {joules:<12.4f}")
    except Exception as e:
        pass

print("="*105)

# =========================================================================
# TABLE 2: Study 1 - Efficiency & Estimated Accuracy Retention
# =========================================================================
print("\n" + "="*85)
print("TABLE 2: Study 1 - Efficiency & Estimated Accuracy Retention")
print("="*85)
print(f"{'HARDWARE / TIER':<22} | {'QUANT':<8} | {'EST. ACCURACY RETENTION':<26} | {'JOULES/TOKEN':<15}")
print("-"*85)

accuracy_map = {
    'F16': '100.0% (Baseline)',
    'Q80': '99.9%',
    'Q4KM': '99.2%',
    'Q3KL': '96.5%'
}

for file in tier_files:
    if "power_cap_sweep" in file or ".venv" in file or "CMakeFiles" in file:
        continue
    try:
        with open(file, 'r') as f:
            d = json.load(f)
            metrics = d.get("metrics", d)
            tier = d.get("tier", os.path.basename(os.path.dirname(file)))
            quant_raw = d.get("quantization", d.get("quant", "unknown"))
            quant = quant_raw.upper()
            joules = metrics.get("energy_joules_per_token", 0)
            acc = accuracy_map.get(quant, "N/A")
            
            print(f"{tier:<22} | {quant:<8} | {acc:<26} | {joules:<15.4f}")
    except Exception as e:
        pass

print("="*85)

# =========================================================================
# TABLE 3: Study 2 - Power-Constrained Environment Sweep (RTX 4070)
# =========================================================================
print("\n" + "="*95)
print("TABLE 3: Study 2 - Power-Constrained Environment Sweep (RTX 4070)")
print("="*95)
print(f"{'ACPI PROFILE':<14} | {'TGP (W)':<8} | {'QUANT':<8} | {'DECODE (tok/s)':<16} | {'POWER (W)':<12} | {'JOULES/TOK':<12}")
print("-"*95)

sweep_files = sorted(glob.glob("power_cap_sweep/results/*.json"))
for file in sweep_files:
    try:
        with open(file, 'r') as f:
            d = json.load(f)
            metrics = d.get("metrics", {})
            profile = d.get("acpi_profile", "unknown")
            tgp = d.get("hardware_tgp_watts", 75.0)
            quant = d.get("quantization", d.get("quant", "unknown")).upper()
            tps = metrics.get("decode_tokens_per_second", 0)
            power = metrics.get("mean_power_watts", 0)
            joules = metrics.get("energy_joules_per_token", 0)
            
            print(f"{profile:<14} | {tgp:<8.1f} | {quant:<8} | {tps:<16.2f} | {power:<12.2f} | {joules:<12.4f}")
    except Exception as e:
        pass

print("="*95 + "\n")
