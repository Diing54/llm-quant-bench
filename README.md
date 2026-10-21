# llm-quant-bench: Heterogeneous LLM Inference Benchmarking

This repository contains my custom benchmarking harness and telemetry
framework for evaluating Large Language Models (LLMs) across
heterogeneous hardware ranging from a low-power edge ARM devices to a
discrete GPU.

I built this project to measure not just raw generation speed
(Tokens/second), but to track real dynamic power consumption and
calculate energy efficiency (Joules/Token) across a 4-tier hardware
stack running Arch Linux and Debian.

## Hardware Setup & Telemetry

To ensure reliable comparisons, I standardized workloads around the
**Llama-3.2-3B-Instruct** model configured with a 2048 token context
window and 999 generated tokens per run.

| Compute Tier | Execution Backend | My Hardware | How I Measure Power |
| :--- | :--- | :--- | :--- |
| **Tier 1 (Edge ARM)** | llama.cpp (ARM NEON) | Raspberry Pi 5 (8GB, Debian) | PMIC VDD_CORE polling at 1Hz (`pi_collector.py`) |
| **Tier 2 (x86 CPU)** | llama.cpp (CPU BLAS) | AMD Ryzen 7 8845HS | Linux RAPL (`powercap`) via `sudo` |
| **Tier 3 (iGPU)** | Vulkan (RADV Driver) | AMD Radeon 780M | AMDGPU hwmon sysfs polling at 10Hz |
| **Tier 4 (dGPU)** | CUDA (cuBLAS) | NVIDIA RTX 4070 (8GB) | `nvidia-smi` polling at 10Hz |

> **Note on Pi 5 Stability:** Running unthrottled 4-core inference on
> the Raspberry Pi 5 caused instantaneous current spikes that tripped
> the PMIC power-good threshold and triggered hard reboots. To maintain
> stable telemetry, I capped the CPU frequency at 2.0 GHz and
> restricted execution to 2 threads (`THREADS=2`), alongside piping
> `/dev/null` to stdin to prevent interactive mode hangs.

> **Note on CPU RAPL Scope:** The Ryzen 8845HS power readings capture
> RAPL package power for the CPU die only. They exclude DRAM, SoC
> uncore, and peripheral power. For a fair node-level comparison, add
> approximately 3–5W to account for these domains.

> **Note on Optimus Laptops:** To benchmark Tier 3 (AMD 780M) in
> isolation, I had to explicitly force the Vulkan ICD to the RADV
> driver using `VK_ICD_FILENAMES`. Without this, NVIDIA's proprietary
> Optimus drivers on Arch Linux silently hijack Vulkan workloads and
> route them to the discrete GPU.


## Key Observations

After profiling the full 4-tier hardware matrix, three architectural
behaviours emerged clearly from the data.

### 1. The Pi 5 Edge Throughput-Efficiency Split

On the Raspberry Pi 5, `Q4_K_M` delivers the highest decode throughput
at **4.10 T/s** — 24% faster than `Q3_K_L` and 46% faster than
`Q8_0`. However, `Q8_0` is unexpectedly the most energy-efficient
format at **0.938 J/token**, despite being slower. This is because
`Q8_0` draws slightly less power (2.98W vs 3.30W) and the Pi's ARM
NEON core dequantizes Q8 weights more efficiently than the
mixed-precision `Q4_K_M` format under a 2-thread ceiling.

The practical takeaway: on a power-constrained edge device, the
*optimal* quantization depends on whether you are optimizing for
response latency or battery life ,they point to different formats.

### 2. The iGPU Efficiency Inversion

The AMD 780M iGPU is roughly 40% faster at decoding than the CPU
baseline, but its energy-per-token is **8.5× worse** (2.92 J/tok vs.
0.34 J/tok at `Q4_K_M`). The explanation is fixed power-state
overhead: to run any GPU workload, the iGPU must hold its full
hardware clock domain ;GPU engine, display fabric, memory controller
,at operating voltage for the entire inference window (~42W). The
CPU at 3.58W only activates the specific compute cores needed.

For small 3B-parameter models on a UMA SoC, the iGPU occupies an
efficiency dead zone: meaningfully faster than the CPU, but with power
costs that a dedicated GPU would spend to achieve 5× higher
throughput. The iGPU advantage only materialises if the workload is
large enough that the fixed power overhead is amortised over many more
tokens per second.

### 3. CUDA Kernel Switching Penalty (`Q3_K_L` vs. `Q4_K_M`)

`Q3_K_L` outperforms `Q4_K_M` on both the CPU and Vulkan tiers. On
the discrete RTX 4070, however, `Q3_K_L` is **5% slower** despite
being a smaller model.

The cause is tensor-type fragmentation. The `Q3_K_L` format uses three
distinct tensor types — `q3_K`, `q5_K`, and `q6_K` ,confirmed in the
`llama.cpp` tensor loader output. On every layer boundary during
autoregressive decode, the CUDA runtime must launch and synchronise a
different dequantization kernel, each with its own grid configuration
and shared memory layout. The inter-kernel dispatch overhead ,which
is negligible on CPU BLAS and low-latency on UMA Vulkan — becomes a
measurable bottleneck in the already memory-bandwidth-limited decode
phase on a device with a discrete command queue. `Q4_K_M` uses only
two tensor types (`q4_K` and `q6_K`), reducing dispatch complexity
and recovering throughput.

This finding generalises: more aggressive quantization is not
monotonically beneficial on CUDA hardware. The compute cost of
dequantization kernel diversity introduces an architecture-specific
floor.


## Visualizations & Detailed Metrics

### 1. Throughput vs. Power (Pareto Frontier)

This chart plots decode throughput against package power draw across
all four tiers and all quantization levels. Points toward the
**top-left** represent the optimal zone, faster generation at lower
power cost.

![Pareto Frontier](results/pareto_frontier.png)

#### Complete Results Matrix

| Hardware Tier | Quantization | Prefill (T/s) | Decode (T/s) | Power (W) | J / token |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Raspberry Pi 5 (ARM)** | Q3_K_L | 5.90 | 3.30 | 3.31 | 0.996 |
| **Raspberry Pi 5 (ARM)** | Q4_K_M | 12.80 | 4.10 | 3.30 | 1.243 |
| **Raspberry Pi 5 (ARM)** | Q8_0 | 15.20 | 2.80 | 2.98 | **0.938** |
| **Ryzen 7 8845HS (CPU)** | F16 | 56.18 | 3.57 | 3.36 | 0.941 |
| **Ryzen 7 8845HS (CPU)** | Q3_K_L | 160.53 | 11.38 | 3.76 | **0.331** |
| **Ryzen 7 8845HS (CPU)** | Q4_K_M | 129.82 | 10.41 | 3.58 | 0.344 |
| **Ryzen 7 8845HS (CPU)** | Q8_0 | 109.37 | 6.39 | 3.53 | 0.552 |
| **Radeon 780M (iGPU)** | Q3_K_L | 168.85 | 15.16 | 48.46 | 3.198 |
| **Radeon 780M (iGPU)** | Q4_K_M | 177.20 | 14.39 | 41.96 | 2.917 |
| **Radeon 780M (iGPU)** | Q8_0 | 192.06 | 8.50 | 38.85 | 4.568 |
| **RTX 4070 (dGPU)** | F16 | 612.50 | 32.59 | 51.02 | 1.566 |
| **RTX 4070 (dGPU)** | Q3_K_L | 755.73 | 77.56 | 52.16 | 0.672 |
| **RTX 4070 (dGPU)** | Q4_K_M | 783.03 | 81.51 | 50.87 | **0.624** |
| **RTX 4070 (dGPU)** | Q8_0 | 715.27 | 54.81 | 52.09 | 0.950 |

*Bold values indicate the most efficient result within each hardware
tier. The RTX 4070 power draw shows less than 2.5% variance across
all quantizations, indicating the GPU is operating at its sustained
firmware TDP ceiling throughout all benchmark windows rather than
dynamically scaling to the workload.*

---

### 2. Quantization Accuracy Retention vs. Energy Cost

The table below cross-references measured energy cost (Joules/Token,
from my benchmarks) against estimated accuracy retention figures drawn
from the `llama.cpp` quantization literature. The accuracy percentages
are **not values I measured**, they are community-established
estimates based on perplexity comparisons documented in the `llama.cpp`
project's quantization type reference [1]. They are included here to
contextualise the efficiency tradeoffs: a lower J/token figure is only
meaningful if the model quality at that quantization level remains
acceptable for the target use case.

![Accuracy vs Energy](results/accuracy_vs_energy.png)

#### Accuracy and Energy Cost Breakdown

| Hardware Tier | Quantization | Est. Accuracy Retention* | Joules/Token |
| :--- | :--- | :--- | :--- |
| **Raspberry Pi 5** | Q8_0 | ~99.9% | 0.938 J |
| **Raspberry Pi 5** | Q4_K_M | ~97.5% | 1.243 J |
| **Raspberry Pi 5** | Q3_K_L | ~92.0% | 0.996 J |
| **Ryzen 8845HS CPU** | F16 | 100% (baseline) | 0.941 J |
| **Ryzen 8845HS CPU** | Q8_0 | ~99.9% | 0.552 J |
| **Ryzen 8845HS CPU** | Q4_K_M | ~97.5% | 0.344 J |
| **Ryzen 8845HS CPU** | Q3_K_L | ~92.0% | 0.330 J |
| **Radeon 780M iGPU** | Q8_0 | ~99.9% | 4.568 J |
| **Radeon 780M iGPU** | Q4_K_M | ~97.5% | 2.917 J |
| **Radeon 780M iGPU** | Q3_K_L | ~92.0% | 3.198 J |
| **RTX 4070 dGPU** | F16 | 100% (baseline) | 1.566 J |
| **RTX 4070 dGPU** | Q8_0 | ~99.9% | 0.950 J |
| **RTX 4070 dGPU** | Q4_K_M | ~97.5% | 0.624 J |
| **RTX 4070 dGPU** | Q3_K_L | ~92.0% | 0.673 J |

*\* Accuracy retention estimates are sourced from `llama.cpp`
quantization type documentation and community perplexity benchmarks
[1], not independently measured in this project. Values represent
approximate retention relative to the F16 baseline on standard
evaluation sets. Actual retention will vary by model and task.*


## How to Run This

The entire suite is script-driven and reproducible.

**1. Run a benchmark on x86 / GPU tiers:**

```bash
./harness/benchmark.sh "tier4" "rtx4070" "q4km" \
  "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
```

> The CPU tier (Tier 2) requires `sudo` for RAPL power sensor access.

**2. Run a benchmark on the Raspberry Pi 5:**

```bash
./harness/benchmark_pi.sh "tier1" "pi5" "q4km" \
  "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
```

**3. Generate the analysis charts:**

```bash
python3 analysis/generate_report.py
python3 analysis/plot_pareto.py
python3 analysis/plot_accuracy_tradeoff.py
```

---

## Repository Structure

```text
.
├── analysis/
│   ├── generate_report.py        # Aggregates multi-tier JSON metrics and renders the Pareto dashboard
│   ├── plot_accuracy_tradeoff.py # Generates the Accuracy vs Energy decay visualization
│   └── plot_pareto.py            # Generates the Throughput vs Power Pareto frontier chart
├── harness/
│   ├── benchmark.sh              # Main execution script for local x86/GPU tiers
│   ├── benchmark_pi.sh           # Automated 2-thread harness for Raspberry Pi 5
│   ├── parse_output.py           # Parses x86/GPU raw logs + power CSVs into structured JSON
│   └── parse_pi_output.py        # Parses ARM edge logs and PMIC telemetry
├── power/
│   ├── collector.py              # Telemetry poller: RAPL, AMDGPU hwmon, nvidia-smi
│   ├── pi_collector.py           # 1Hz VDD_CORE power logging script for Raspberry Pi 5
│   └── logs/                     # Unified power telemetry CSVs across all tiers
├── prompts/
│   └── benchmark_prompt.txt      # Standardised input prompt context
├── README.md
└── results/                      # JSON metrics, raw execution logs, and rendered graphs
```


## References

[1] Gerganov, G. et al. (2023–2024). *Quantization types and k-quants.*
llama.cpp GitHub Repository.
https://github.com/ggerganov/llama.cpp/blob/master/docs/development/token_generation_performance_tips.md
