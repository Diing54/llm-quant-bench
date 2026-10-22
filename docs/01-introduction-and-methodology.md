# Introduction and Experimental Methodology

## Motivation
Data centre power challenges are straining electricity grids and limiting computational throughput. While much of the research in High-Performance Computing (HPC) focuses on large server-grade accelerators, I built this benchmarking framework to investigate how Large Language Models (LLMs) behave across heterogeneous consumer and edge hardware. 

The primary objective of this project is not merely to measure raw generation speed in tokens per second, but to track real dynamic power consumption and calculate energy efficiency in Joules per token. By profiling models across a four-tier hardware stack, I aimed to discover where the Pareto frontier lies between performance and energy expenditure, and how hardware-enforced power bounds alter that frontier.

## Workload Standardization
To ensure all hardware tiers experienced identical computational pressure, I standardized every benchmark run around the following parameters:
- **Model:** Llama-3.2-3B-Instruct (GGUF format).
- **Context Window:** 2048 tokens.
- **Generation Length:** Exactly 999 generated tokens (`-n 999`).
- **Prompt:** A standardized technical essay prompt detailing the history and architecture of the Linux kernel to ensure consistent autoregressive decode dynamics.

## Hardware Tiers Tested
1. **Tier 1 (Edge ARM):** Raspberry Pi 5 (8GB RAM, Debian). Powered by ARM NEON cores. Power measured via external PMIC VDD_CORE polling at 1Hz.
2. **Tier 2 (x86 CPU):** AMD Ryzen 7 8845HS. Power measured via Linux RAPL (`powercap`) sensor polling.
3. **Tier 3 (iGPU):** AMD Radeon 780M integrated graphics running via Vulkan (RADV driver). Power measured via AMDGPU hwmon sysfs polling at 10Hz.
4. **Tier 4 (dGPU):** NVIDIA GeForce RTX 4070 Laptop GPU (8GB VRAM) running via CUDA (cuBLAS). Power measured via `nvidia-smi` polling at 10Hz.
