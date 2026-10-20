# llm-quant-bench: Heterogeneous LLM Inference Benchmarking

This repository contains my custom benchmarking harness and telemetry
framework for evaluating Large Language Models (LLMs) across different
hardware.

I built this project to test how different quantization formats perform
across my laptop's hardware (CPU, integrated GPU, and discrete GPU) on
Arch Linux. The goal isn't just to measure raw speed (Tokens/second),
but to track real dynamic power consumption to find the sweet spot
between throughput and energy efficiency (Joules/token).

## Hardware Setup & Telemetry

I run all these benchmarks locally on my **Lenovo Legion Slim 5**
running Arch Linux. To keep the power measurements consistent and fair
across all tests, I fixed the generation output to exactly 999 tokens
using the `Llama-3.2-3B-Instruct` model.

| Compute Tier | Execution Backend | My Hardware | How I Measure Power |
| :--- | :--- | :--- | :--- |
| **CPU** | llama.cpp (CPU BLAS) | AMD Ryzen 7 8845HS | Linux RAPL (`powercap`) via `sudo` |
| **Tier 3 (iGPU)** | Vulkan (RADV Driver) | AMD Radeon 780M | AMDGPU hwmon sysfs polling at 10Hz |
| **Tier 4 (dGPU)** | CUDA (cuBLAS) | NVIDIA RTX 4070 (8GB) | `nvidia-smi` polling at 10Hz |

> **Note on Optimus Laptops:** To test the integrated GPU (Tier 3)
> properly, I had to explicitly isolate the Vulkan driver
> (`VK_ICD_FILENAMES` and disabling implicit layers). Otherwise,
> NVIDIA's proprietary drivers on Arch Linux would aggressively hijack
> the workload.

## Key Observations

After running the full matrix, I noticed three interesting behaviors in
how the hardware handles these workloads:

### 1. The iGPU Efficiency Inversion

My AMD 780M integrated GPU is about 40% faster at decoding than the
CPU baseline. However, its energy-per-token is roughly **8.5× worse**
(2.92 J/tok vs. 0.34 J/tok).

Why? To run the workload, the iGPU has to wake up and sustain its full
hardware power state (~42W) for the entire run, whereas the CPU only
uses about ~3.5W for the active cores. For small models, the iGPU is
caught in a dead zone: it's faster than the CPU, but the fixed power
overhead makes it highly inefficient compared to a dedicated GPU.

### 2. CUDA Kernel Switching Penalty (`Q3_K_L` vs. `Q4_K_M`)

Usually, the `Q3_K_L` quantization is faster than `Q4_K_M` (which I
saw on both my CPU and Vulkan runs). But on the discrete RTX 4070,
`Q3_K_L` actually slows down.

This happens because `Q3_K_L` mixes different tensor types (`q3_K`,
`q5_K`, `q6_K`). On the dedicated GPU, this forces the CUDA runtime to
constantly launch and synchronise different dequantization kernels —
each with its own grid configuration and memory access pattern — and
those dispatch stalls bottleneck the already memory-bandwidth-limited
decode phase.

### 3. The 6GB Vulkan Allocation Limit (F16 on Tier 3)

When I tried running the uncompressed F16 model (6.0 GiB) on the
integrated GPU without memory mapping (`--no-mmap`), it crashed with an
Out of Memory error.

Even though the 780M shares the system's massive pool of RAM (UMA), the
Linux RADV driver couldn't allocate a single, physically contiguous 6GB
block of memory. This proves that for strict Linux iGPU deployments,
quantization isn't just about saving disk space — it's a physical
necessity to fit within contiguous memory limits.

## The Results

![Pareto Frontier](results/pareto_frontier.png)
*(This graph maps decode throughput against energy efficiency. The best
setups are closest to the top-left).*

| Tier / Device | Quantization | Prefill (T/s) | Decode (T/s) | TTFT (ms) | Power (W) | Efficiency (J/token) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CPU** (Ryzen 8845HS) | F16 | 56.18 | 3.57 | 943.70 | 3.36 | 0.9412 |
| **CPU** (Ryzen 8845HS) | Q3_K_L | 160.53 | 11.38 | 330.55 | 3.76 | 0.3307 |
| **CPU** (Ryzen 8845HS) | Q4_K_M | 129.82 | 10.41 | 412.47 | 3.58 | 0.3442 |
| **CPU** (Ryzen 8845HS) | Q8_0 | 109.37 | 6.39 | 490.28 | 3.53 | 0.5524 |
| **Tier 3** (AMD 780M) | Q3_K_L | 168.85 | 15.16 | 313.94 | 48.46 | 3.1975 |
| **Tier 3** (AMD 780M) | Q4_K_M | 177.20 | 14.39 | 299.11 | 41.96 | 2.9166 |
| **Tier 3** (AMD 780M) | Q8_0 | 192.06 | 8.50 | 275.96 | 38.85 | 4.5684 |
| **Tier 4** (RTX 4070) | F16 | 612.50 | 32.59 | 86.96 | 51.02 | 1.5657 |
| **Tier 4** (RTX 4070) | Q3_K_L | 755.73 | 77.56 | 70.35 | 52.16 | 0.6724 |
| **Tier 4** (RTX 4070) | Q4_K_M | 783.03 | 81.51 | 68.21 | 50.87 | 0.6241 |
| **Tier 4** (RTX 4070) | Q8_0 | 715.27 | 54.81 | 74.27 | 52.09 | 0.9504 |

*Note: My CPU power metrics capture RAPL package power (excluding DRAM
and SoC uncore overhead). The RTX 4070 power stays locked around
51–52W across all quantizations, meaning the GPU is hitting my
laptop's firmware TDP limit rather than dynamically scaling to the
workload.*

## How to Run This

If you want to reproduce my setup, the entire suite is script-driven.

**1. Run a benchmark** (Specify the tier, device, quantization, and
model path):

```bash
./harness/benchmark.sh "tier4" "rtx4070" "q4km" \
  "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
```

> Note: Running the CPU tier requires `sudo` so the script can read
> the RAPL energy sensors.

**2. Generate the graph:**

```bash
python3 analysis/plot_pareto.py
```

## Repository Structure

```text
.
├── analysis/
│   └── plot_pareto.py       # Python script to plot the Pareto frontier
├── harness/
│   ├── benchmark.sh         # Main script: runs benchmarks, isolates drivers
│   └── parse_output.py      # Combines llama.cpp logs + power CSVs into JSON
├── power/
│   └── collector.py         # Background telemetry: polls power at 10Hz
└── results/                 # JSON metrics, raw logs, and graphs
```
