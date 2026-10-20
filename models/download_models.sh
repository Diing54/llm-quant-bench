#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_URL="https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main"

FILES=(
    "Llama-3.2-3B-Instruct-F16.gguf"
    "Llama-3.2-3B-Instruct-Q8_0.gguf"
    "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    "Llama-3.2-3B-Instruct-Q3_K_S.gguf"
)

# Remove the broken 15-byte pointer files
find "${MODEL_DIR}" -name "*.gguf" -size -1k -delete

for FILE in "${FILES[@]}"; do
    if [ ! -f "${MODEL_DIR}/${FILE}" ]; then
        curl -L "${BASE_URL}/${FILE}?download=true" -o "${MODEL_DIR}/${FILE}"
    fi
done
