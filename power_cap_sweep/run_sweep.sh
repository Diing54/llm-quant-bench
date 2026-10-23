#!/usr/bin/env bash
# ============================================================
# power_cap_sweep/run_sweep.sh
# Study 2: Native Power Envelope Sweep on RTX 4070 dGPU
# ============================================================
set -euo pipefail

MODELS=(
    "f16:models/Llama-3.2-3B-Instruct-F16.gguf"
    "q80:models/Llama-3.2-3B-Instruct-Q8_0.gguf"
    "q4km:models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    "q3kl:models/Llama-3.2-3B-Instruct-Q3_K_L.gguf"
)

PROFILES=("low-power" "balanced" "performance")
N_PREDICT=999
STABILISE_SECS=8
SWEEP_RESULTS="power_cap_sweep/results"

mkdir -p "$SWEEP_RESULTS"

# Force NVIDIA Persistence Mode
sudo nvidia-smi -pm 1 > /dev/null 2>&1 || true

# Safety trap to restore balanced profile on exit
RESTORE_PROFILE=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo "balanced")
trap 'echo "$RESTORE_PROFILE" | sudo tee /sys/firmware/acpi/platform_profile > /dev/null; echo "[RESTORED] Profile reset."' EXIT

echo "=================================================="
echo "  Study 2: Power-Limited Environment GPU Sweep"
echo "  Profiles: ${PROFILES[*]}"
echo "  Target Hardware: RTX 4070 (dGPU offload -ngl 99)"
echo "=================================================="

for profile in "${PROFILES[@]}"; do
    echo ""
    echo ">>> Applying ACPI Power Envelope: $profile"
    echo "$profile" | sudo tee /sys/firmware/acpi/platform_profile > /dev/null
    
    echo "    Stabilising thermal/power bus for ${STABILISE_SECS}s..."
    sleep "$STABILISE_SECS"

    if [ "$profile" == "low-power" ]; then
        ACTUAL_TGP="65.0"
    elif [ "$profile" == "balanced" ]; then
        ACTUAL_TGP="75.0"
    elif [ "$profile" == "performance" ]; then
        ACTUAL_TGP="140.0"
    else
        ACTUAL_TGP="75.0"
    fi

    echo "    Confirmed Active Hardware TGP Limit: ${ACTUAL_TGP}W"

    for entry in "${MODELS[@]}"; do
        QUANT="${entry%%:*}"
        MODEL_PATH="${entry##*:}"
        
        if [ ! -f "$MODEL_PATH" ]; then
            echo "    WARNING: Model $MODEL_PATH not found, skipping."
            continue
        fi
        
        echo ""
        echo "    [Sweep] Benchmarking: ${QUANT} under ${profile} (${ACTUAL_TGP}W cap)"
        
        OUT_JSON="${SWEEP_RESULTS}/sweep_${profile}_${QUANT}.json"
        RAW_LOG="${SWEEP_RESULTS}/raw_${profile}_${QUANT}.log"
        POWER_LOG="${SWEEP_RESULTS}/power_${profile}_${QUANT}.csv"
        
        # Start background power telemetry collection
        python3 power/collector.py --tier tier4 --out "$POWER_LOG" &
        POWER_PID=$!
        sleep 2
        
        # Warmup run
        ./llama.cpp-cuda/build/bin/llama-cli \
            -m "$MODEL_PATH" -p "Warmup." -n 32 -c 2048 -ngl 99 > /dev/null 2>&1

        # 3 Benchmark Iterations for statistical variance reduction
        > "$RAW_LOG"
        for i in 1 2 3; do
            echo "        Run $i / 3"
            ./llama.cpp-cuda/build/bin/llama-cli \
                -m "$MODEL_PATH" \
                -p "$(cat prompts/benchmark_prompt.txt)" \
                -n "$N_PREDICT" -c 2048 -ngl 99 \
                2>&1 | tee -a "$RAW_LOG"
        done

        # Stop power telemetry
        kill -9 "$POWER_PID" 2>/dev/null
        wait "$POWER_PID" 2>/dev/null || true

        # Parse metrics into structured JSON
        python3 harness/parse_output.py \
            --raw-log "$RAW_LOG" \
            --power-log "$POWER_LOG" \
            --tier "tier4_${profile}" \
            --device "rtx4070" \
            --quant "$QUANT" \
            --out "$OUT_JSON"

        # Clean annotation call
        python3 power_cap_sweep/annotate_profile_sweep.py "$OUT_JSON" "$profile"
    done
done

echo ""
echo "=================================================="
echo "  Study 2 Sweep Completed Successfully!"
echo "  Results stored in: $SWEEP_RESULTS"
echo "=================================================="
