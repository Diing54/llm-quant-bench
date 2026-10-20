#!/usr/bin/env python3
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-log", required=True)
    parser.add_argument("--power-log", required=True)
    parser.add_argument("--tier", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--quant", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()

def main():
    args = parse_args()

    prefill_tps_list = []
    decode_tps_list = []
    ttft_list = []

    with open(args.raw_log, "r", encoding="utf-8", errors="ignore") as f:
        log_content = f.read()

    # Bulletproof line-by-line scanning
    for line in log_content.splitlines():
        # Clean up any weird unicode spaces
        clean_line = line.replace("\u00a0", " ")

        # Parse prompt eval speed
        if "prompt eval time" in clean_line and "tokens per second" in clean_line:
            try:
                # Example: ... ( 1.67 ms per token,   600.59 tokens per second)
                part = clean_line.split("tokens per second")[0]
                val_str = part.split(",")[-1].strip()
                prefill_tps_list.append(float(val_str))
            except Exception:
                pass

        # Parse decode / eval speed (ensure it's not the prompt line)
        elif "eval time" in clean_line and "tokens per second" in clean_line:
            try:
                # Example: ... ( 30.92 ms per token,    33.43 tokens per second)
                part = clean_line.split("tokens per second")[0]
                val_str = part.split(",")[-1].strip()
                decode_tps_list.append(float(val_str))
            except Exception:
                pass

        # Parse Time to First Token (Prompt eval total time in ms)
        if "prompt eval time" in clean_line and "ms" in clean_line and "=" in clean_line:
            try:
                # Example: prompt eval time =      88.25 ms
                part = clean_line.split("=")[1].split("ms")[0].strip()
                ttft_list.append(float(part))
            except Exception:
                pass

    mean_prefill_tps = sum(prefill_tps_list) / len(prefill_tps_list) if prefill_tps_list else 0.0
    mean_decode_tps = sum(decode_tps_list) / len(decode_tps_list) if decode_tps_list else 0.0
    mean_ttft = sum(ttft_list) / len(ttft_list) if ttft_list else 0.0

    # Parse power log
    mean_power = 0.0
    power_values = []
    try:
        with open(args.power_log, "r", encoding="utf-8", errors="ignore") as pf:
            header = pf.readline()  # skip header
            for line in pf:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    try:
                        power_values.append(float(parts[1]))
                    except ValueError:
                        continue
        if power_values:
            mean_power = sum(power_values) / len(power_values)
    except Exception as e:
        print(f"Warning: Could not parse power log: {e}")

    energy_per_token = (mean_power / mean_decode_tps) if mean_decode_tps > 0 else 0.0

    output_data = {
        "tier": args.tier,
        "device": args.device,
        "quantization": args.quant,
        "metrics": {
            "prefill_tokens_per_second": round(mean_prefill_tps, 2),
            "decode_tokens_per_second": round(mean_decode_tps, 2),
            "time_to_first_token_ms": round(mean_ttft, 2),
            "mean_power_watts": round(mean_power, 2),
            "energy_joules_per_token": round(energy_per_token, 4)
        }
    }

    with open(args.out, "w", encoding="utf-8") as out_f:
        json.dump(output_data, out_f, indent=2)

    print(f"Successfully parsed and saved to {args.out}")

if __name__ == "__main__":
    main()
