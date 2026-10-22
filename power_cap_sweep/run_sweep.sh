#!/usr/bin/env bash
# ============================================================
# power_cap_sweep/run_sweep.sh
# Platform-profile-based power budget sweep for LLM inference
# ============================================================
set -euo pipefail

MODELS=(
    "q4km:models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    "q80:models/Llama-3.2-3B-Instruct-Q8_0.gguf"
)
PROFILES=("low-power" "balanced" "performance" "max-power")
N_PREDICT=999
STABILISE_SECS=10
SWEEP_RESULTS="power_cap_sweep/results"

mkdir -p "$SWEEP_RESULTS"

# Force Persistence Mode so the GPU sensors stay awake
sudo nvidia-smi -pm 1 > /dev/null 2>&1 || true

# Safety: restore initial profile on any exit
RESTORE_PROFILE=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo "balanced")
trap 'echo "$RESTORE_PROFILE" | sudo tee /sys/firmware/acpi/platform_profile > /dev/null
      echo "[RESTORED] Platform profile reset to: $RESTORE_PROFILE"' EXIT

echo "========================================"
echo "  LLM Power Budget Sweep"
echo "  Profiles: ${PROFILES[*]}"
echo "  Quantizations: ${#MODELS[@]}"
echo "========================================"

for profile in "${PROFILES[@]}"; do
    echo ""
    echo ">>> Setting platform profile: $profile"
    echo "$profile" | sudo tee /sys/firmware/acpi/platform_profile > /dev/null
    echo "    Stabilising for ${STABILISE_SECS}s..."
    sleep "$STABILISE_SECS"

    # Ping the GPU once to wake up the sensor bus, then read
    nvidia-smi > /dev/null 2>&1
    ACTUAL_TGP=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null | head -1 | xargs)
    
    echo "    Confirmed GPU power limit: ${ACTUAL_TGP}W"

    for entry in "${MODELS[@]}"; do
        QUANT="${entry%%:*}"
        MODEL_PATH="${entry##*:}"
        
        echo ""
        echo "    Benchmarking: ${profile} / ${QUANT} @ ${ACTUAL_TGP}W TGP"
        
        export OUT_JSON="${SWEEP_RESULTS}/sweep_${profile}_${QUANT}.json"
        RAW_LOG="${SWEEP_RESULTS}/raw_${profile}_${QUANT}.log"
        POWER_LOG="${SWEEP_RESULTS}/power_${profile}_${QUANT}.csv"
        
        export PROFILE_NAME="$profile"
        export TGP_ANNOTATION="$ACTUAL_TGP"
        
        # Start power telemetry
        python3 power/collector.py --tier tier4 --out "$POWER_LOG" &
        POWER_PID=$!
        
        # Sub-trap to kill collector if interrupted
        trap "kill -9 $POWER_PID 2>/dev/null; echo '$RESTORE_PROFILE' | sudo tee /sys/firmware/acpi/platform_profile > /dev/null" EXIT

        # Warmup
        ./llama.cpp-cuda/build/bin/llama-cli \
            -m "$MODEL_PATH" \
            -p "Warmup." \
            -n 32 -c 2048 -ngl 99 \
            > /dev/null 2>&1

        # Benchmark runs
        > "$RAW_LOG"
        for i in 1 2 3; do
            echo "        Run $i / 3"
            ./llama.cpp-cuda/build/bin/llama-cli \
                -m "$MODEL_PATH" \
                -p "$(cat prompts/benchmark_prompt.txt)" \
                -n "$N_PREDICT" -c 2048 -ngl 99 \
                2>&1 | tee -a "$RAW_LOG"
        done

        # Stop telemetry
        kill -9 "$POWER_PID" 2>/dev/null
        wait "$POWER_PID" 2>/dev/null || true
        
        # Restore main trap
        trap 'echo "$RESTORE_PROFILE" | sudo tee /sys/firmware/acpi/platform_profile > /dev/null
              echo "[RESTORED] Platform profile reset to: $RESTORE_PROFILE"' EXIT

        # Parse metrics
        python3 harness/parse_output.py \
            --raw-log "$RAW_LOG" \
            --power-log "$POWER_LOG" \
            --tier "tier4_${profile}" \
            --device "rtx4070" \
            --quant "$QUANT" \
            --out "$OUT_JSON"

        # Safely annotate JSON using os.environ
        python3 -c '
import json, os
out_file = os.environ.get("OUT_JSON")
profile = os.environ.get("PROFILE_NAME")
tgp = os.environ.get("TGP_ANNOTATION", "").replace("W", "").strip()

try:
    with open(out_file) as f:
        data = json.load(f)
    
    data["platform_profile"] = profile
    
    try:
        data["hardware_tgp_watts"] = float(tgp)
    except ValueError:
        data["hardware_tgp_watts"] = -1.0
        
    with open(out_file, "w") as f:
        json.dump(data, f, indent=2)
        
    print(f"    Annotated: {out_file}")
except Exception as e:
    print(f"    Error annotating JSON: {e}")
'
    done
done

echo ""
echo "========================================"
echo "  Sweep complete. Results in: $SWEEP_RESULTS"
echo "========================================"
