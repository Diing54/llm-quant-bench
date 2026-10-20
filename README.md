# llm-quant-bench
**Empirical LLM Quantization Benchmarking on Consumer and Integrated GPU Hardware**
A reproducible benchmarking harness for measuring the inference speed, power consumption, and energy efficiency of a locally-run large language model across multiple quantization formats and hardware backends. Built on `llama.cpp`, the harness captures hardware telemetry in parallel with inference and produces structured JSON results suitable for automated analysis and visualisation.

---
## Hardware Testbed
All benchmarks run on a single Lenovo Legion Slim 5 Gen 9 (AMD Ryzen 7 8845HS) running Arch Linux. The machine contains two distinct GPU resources benchmarked independently as separate tiers.

| Tier | Device | Architecture | Backend | VRAM |
|---|---|---|---|---|
| Tier 4 | NVIDIA GeForce RTX 4070 Laptop GPU | Ada Lovelace, sm_89 | llama.cpp CUDA | 8 GB GDDR6X |
| Tier 3 | AMD Radeon 780M (iGPU) | RDNA3, integrated | llama.cpp Vulkan | ~3 GB shared system RAM |
| CPU Baseline | AMD Ryzen 7 8845HS | Zen 4, 8 cores / 16 threads | llama.cpp CPU (AVX512) | — |

Tier 3 and Tier 4 benchmarks are executed in separate, isolated runs. When benchmarking Tier 4 (CUDA), the Vulkan backend is inactive, and vice versa.

---
## Model
**Llama-3.2-3B-Instruct** (Meta AI, 3.21B parameters)
GGUF quantization files sourced from: `bartowski/Llama-3.2-3B-Instruct-GGUF` on Hugging Face.

| Format | BPW | File Size |
|---|---|---|
| F16 | 16.00 | 5.98 GiB |
| Q8_0 | 8.50 | 3.18 GiB |
| Q4_K_M | 5.01 | 1.87 GiB |
| Q3_K_L | 4.50 | 1.68 GiB |

---
## Methodology
### Benchmark Harness
Each benchmark run is orchestrated by `harness/benchmark.sh`, which:
1. Starts the power telemetry daemon (`power/collector.py`) as a background process before any inference begins.
2. Executes one discarded warmup run to ensure GPU clocks are in their steady-state frequency before timed measurements begin.
3. Runs three identical timed inference passes, each generating a fixed number of output tokens from a standardised input prompt.
4. Terminates the power daemon and pipes all llama.cpp stdout to `harness/parse_output.py` for metric extraction.

### Inference Configuration
* **Context window:**  n_ctx = 2048
* **Batch size:**      n_batch = 2048
* **Output tokens:**   n_predict = 999 (all quantisations)
* **GPU offload:**     -ngl 99 (all layers to CUDA/Vulkan)
* **Flash attention:** disabled (default)

**Workload standardisation note:** All quantisation levels use a fixed generation length of 999 tokens. This is critical for power measurement accuracy: shorter workloads (99 tokens) complete in under 5 seconds, causing the power sampling daemon — which polls at 10 Hz — to capture predominantly pre- and post-inference idle states and significantly underestimate active power draw. At 999 tokens, each run lasts 12–30 seconds, providing a statistically stable power sample.

### Metrics
| Metric | Definition | Source |
|---|---|---|
| Prefill TPS | Prompt evaluation throughput (tokens/sec) | `llama_perf_context_print: prompt eval time` |
| Decode TPS | Autoregressive generation throughput (tokens/sec) | `llama_perf_context_print: eval time` |
| TTFT (ms) | Time-to-first-token; total prompt evaluation wall time | `llama_perf_context_print: prompt eval time` (ms field) |
| Power (W) | Mean GPU power draw sampled at 10 Hz during inference | `nvidia-smi --query-gpu=power.draw` / AMD hwmon `power1_input` |
| J/token | Energy cost per generated token = Power (W) / Decode TPS | Computed |

---
## Limitations
**Power measurement duration.** Software-based polling at 10 Hz introduces sampling variance on short workloads. 
**Thermal steady state.** Benchmarks are run on a laptop chassis with active cooling. GPU boost clocks may vary slightly across runs as thermal headroom changes.
**CPU Power Baseline.** CPU RAPL integration is a planned future addition; currently, CPU energy-per-token is not calculated.
