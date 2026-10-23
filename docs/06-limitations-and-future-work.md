# Limitations and Future Work

## Measurement Limitations

### Software Power Measurement Scope

All power measurements in this study use software-level interfaces: RAPL for
the CPU, hwmon sysfs for the AMD iGPU, nvidia-smi for the discrete GPU, and
PMIC polling for the Raspberry Pi. None of these are inline hardware power
meters, which would require physical current measurement equipment between
the power supply and the device under test.

Software power measurement has specific limitations for each tier:

CPU RAPL measures CPU die package power only. DRAM power (approximately
2 to 4W for DDR5 at load), SoC uncore domains, and PCIe bus power are not
captured. For a full system-level comparison, these omissions understate CPU-
tier total power by approximately 3 to 5W.

AMDGPU hwmon measures GPU package power as reported by the hardware monitor.
On an integrated GPU under UMA, the memory power is shared with the CPU's
memory controller and may not be fully reflected in the GPU's hwmon reading.

nvidia-smi measures GPU card power as reported by the internal power sensor.
This captures the GPU die, VRAM, and supporting circuitry but not system-
level costs like PCIe bus power or the CPU overhead of driving the GPU.

PMIC polling on the Raspberry Pi measures VDD_CORE at 1Hz. This is the
lowest sampling rate in the study and captures only core voltage domain power,
not the full board power.

Despite these limitations, the measurements are internally consistent within
each tier and allow valid comparisons within a tier across quantization
formats. Cross-tier energy comparisons should be interpreted as indicative
rather than exact.

### Static Workload

This study uses a single fixed prompt and a fixed 999-token generation length.
Real workloads vary in prompt length, generation length, and content
complexity. Longer contexts increase KV cache memory pressure and may shift
the memory bandwidth bottleneck in ways this study does not capture. Kwon
et al. [13] showed that KV cache management has a significant impact on
serving throughput, which is a dimension this study does not address.

### Single Model, Single Architecture

All results are specific to Llama 3.2-3B-Instruct running on the llama.cpp
inference engine. Different model families (Mistral, Phi, Qwen) may show
different quantization sensitivity depending on their weight distribution
characteristics. Different inference engines (llama.cpp vs vLLM [13] vs
TensorRT-LLM) will show different performance profiles for the same hardware.

### No Quality Measurement

The accuracy retention percentages in this study are sourced from llama.cpp
community benchmarks [1], not independently measured. A complete study would
include perplexity measurements on a standard evaluation dataset like WikiText-2
across all hardware-quantization combinations to verify that the throughput
differences do not coincide with unexpected quality regressions on specific
hardware.

### Platform Profile Granularity

Study 2 uses three discrete ACPI platform profiles rather than a continuous
watt-level sweep. The jump from 75W (balanced) to 140W (performance) is too
large to observe where the throughput curve saturates. A finer-grained sweep
would require custom firmware or external power limiting hardware. Identifying
the exact TGP at which each quantization format's throughput gain becomes
negligible is left as future work.

## Future Work

**Larger model sizes.** Running Llama 3.1-8B or Llama 3.1-70B across the
same tiers would reveal how the iGPU efficiency inversion changes with model
size and whether the fixed power-state overhead becomes more acceptable at
larger models.

**Perplexity measurement.** Adding WikiText-2 perplexity evaluation per
quantization format and hardware tier would complete the quality dimension
of the study and make the accuracy retention figures empirically grounded
rather than literature-estimated.

**Extended hardware stack.** Adding the Intel IdeaPad (Core i5, Intel Iris
Graphics) as a fifth tier would add an Intel integrated GPU data point for
comparison against the AMD 780M under Vulkan and OpenCL backends.

**Serving throughput.** This study measures single-request inference. A
server throughput study using continuous batching (as in vLLM [13]) would
show how these hardware-quantization combinations scale under concurrent
request loads, which is the deployment scenario that matters most for data
centre design.

**CUDA graph optimization.** llama.cpp runs inference with standard CUDA
kernel launches. Capturing repeated computation patterns in CUDA Graphs
would reduce kernel launch overhead and potentially change the Q3_K_L
performance regression described in Finding 3.
