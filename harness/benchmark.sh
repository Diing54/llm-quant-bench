#!/usr/bin/env bash
set -e

TIER="$1"
DEVICE="$2"
QUANT="$3"
MODEL_PATH="$4"
N_PREDICT="${5:-999}"

# Determine executable, offload flags, and output directory based on Tier
if [ "$TIER" = "tier4" ]; then
    OUT_DIR="results/tier4_cuda"
    EXE="./llama.cpp-cuda/build/bin/llama-cli"
    NGL="-ngl 99"
    EXTRA_ARGS=""
    unset VK_ICD_FILENAMES
    unset VK_LOADER_LAYERS_DISABLE
elif [ "$TIER" = "tier3" ]; then
    OUT_DIR="results/tier3_vulkan"
    EXE="./llama.cpp-vulkan/build/bin/llama-cli"
    NGL="-ngl 99"
    EXTRA_ARGS="-sm none --no-mmap"
    export VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/radeon_icd.json"
    export VK_LOADER_LAYERS_DISABLE="~implicit~"
    export GGML_VK_DEVICE=0
elif [ "$TIER" = "cpu" ]; then
    OUT_DIR="results/cpu_baseline"
    EXE="./llama.cpp-cuda/build/bin/llama-cli"
    NGL="-ngl 0"
    EXTRA_ARGS=""
    unset VK_ICD_FILENAMES
    unset VK_LOADER_LAYERS_DISABLE
else
    echo "Unknown tier: $TIER"
    exit 1
fi

RAW_LOG="results/raw_logs/${TIER}_${DEVICE}_${QUANT}_raw.log"
POWER_LOG="power/logs/${TIER}_${DEVICE}_${QUANT}_power.csv"
JSON_OUT="${OUT_DIR}/${TIER}_${DEVICE}_${QUANT}.json"

mkdir -p "$OUT_DIR" results/raw_logs power/logs

echo "Starting power telemetry for $TIER..."
python3 power/collector.py --tier "$TIER" --out "$POWER_LOG" &
POWER_PID=$!
trap "kill -9 $POWER_PID 2>/dev/null" EXIT

echo "========================================"
echo "Warming up (Output visible, discarded)..."
echo "========================================"
$EXE -m "$MODEL_PATH" -p "Write a detailed, comprehensive essay explaining the history of the Linux kernel..." -n 32 -c 2048 $NGL $EXTRA_ARGS --no-warmup > /dev/null 2>&1

echo "========================================"
echo "Running benchmark loop for $TIER $QUANT..."
echo "========================================"
> "$RAW_LOG"

for i in {1..3}; do
    echo "--- Run $i ---"
    $EXE -m "$MODEL_PATH" -p "Write a detailed, comprehensive essay explaining the history of the Linux kernel, its architecture, and why it became the dominant operating system for servers, supercomputers, and cloud infrastructure. Please provide at least three main sections, detailed technical examples, and a conclusion." -n "$N_PREDICT" -c 2048 $NGL $EXTRA_ARGS --no-warmup 2>&1 | tee -a "$RAW_LOG"
done

kill -9 "$POWER_PID" 2>/dev/null
wait "$POWER_PID" 2>/dev/null || true

echo "Parsing results into JSON..."
python3 harness/parse_output.py --raw-log "$RAW_LOG" --power-log "$POWER_LOG" --tier "$TIER" --device "$DEVICE" --quant "$QUANT" --out "$JSON_OUT"
echo "Done! Benchmark saved to $JSON_OUT"
