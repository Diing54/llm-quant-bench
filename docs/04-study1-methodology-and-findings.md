# Study 1: Methodology and Findings

## Experimental Design

Study 1 is a cross-hardware benchmarking study. I ran the same inference
workload across four hardware tiers and four quantization formats, measuring
decode throughput (T/s), power draw (W), and energy efficiency (J/token) for
each combination.

Every benchmark run uses the same standardized prompt from
prompts/benchmark_prompt.txt, a fixed context window of 2048 tokens, and
a fixed generation length of 999 tokens. The 999-token generation length
was chosen deliberately. In initial testing with 99 tokens, runs on the RTX
4070 completed in under 5 seconds. At 10Hz polling intervals, a 5-second
window yields only 50 power samples, which is insufficient for a stable mean
especially during the model loading phase. At 999 tokens, RTX 4070 runs last
12 to 30 seconds depending on quantization format, giving 120 to 300 power
samples per run and producing statistically stable averages.

Each configuration was run three times with one discarded warmup run before
timed measurements began. The warmup run ensures GPU clocks and CPU frequency
governors are in their steady-state operating points before I start recording
metrics. Results are averaged across the three timed runs.

## Metric Definitions

**Prefill (T/s):** The rate at which the model processes input prompt tokens
in a single parallel forward pass. This is the speed at which context is
ingested. High prefill speeds indicate efficient batch matrix operations.

**Decode (T/s):** The rate of autoregressive token generation. Each output
token requires a sequential forward pass through the full model. This is the
primary bottleneck in generation tasks and the metric I treat as the main
performance indicator.

**Power (W):** Mean hardware-reported power draw across all samples collected
during the three benchmark runs. See
[docs/02-hardware-and-telemetry.md](02-hardware-and-telemetry.md) for the
per-tier measurement methods.

**J/token:** Energy cost per generated token, computed as Power (W) divided
by Decode (T/s). This is the efficiency metric that integrates both power
consumption and throughput into a single comparable value. Lower is better.

## Why J/token Is the Right Metric

Decode T/s tells you how fast a model generates text. Power (W) tells you
how much energy the hardware draws. Neither alone answers the question that
matters for deployment: how much does each generated token cost in energy?

A GPU that runs at 80 T/s and draws 70W costs 0.875 J/token. A CPU that
runs at 12 T/s and draws 3.5W costs 0.292 J/token. By throughput alone, the
GPU looks like the obvious choice. By energy cost, the CPU is three times
more efficient. Which you choose depends entirely on your constraint. This
is the core insight that motivates using J/token as the primary comparative
metric, and it connects directly to the power-limited environment analysis
of Study 2.

This framing is consistent with how HPC researchers evaluate workloads on
heterogeneous nodes. Williams et al. [5] introduced the Roofline model as a
framework for understanding whether a workload is compute-bound or
memory-bandwidth-bound. LLM autoregressive decoding is definitively
memory-bandwidth-bound: at each step, the model must load the full weight
matrix from memory to compute a single output token. Quantization reduces
the size of that matrix and therefore reduces the bandwidth demand per token.
The J/token metric captures whether that bandwidth reduction translates into
real-world energy savings given the dequantization overhead each format
introduces.

## Finding 1: The Pi 5 Throughput-Efficiency Split

On the Raspberry Pi 5, Q4_K_M achieves the highest decode throughput at
4.80 T/s. This is 26% faster than Q3_K_L (3.80 T/s) and 50% faster than
Q8_0 (3.20 T/s). However, Q4_K_M is not the most energy-efficient format
on the Pi.

Q8_0 at 4.02W and 3.20 T/s produces 1.257 J/token.
Q4_K_M at 4.46W and 4.80 T/s produces 0.929 J/token.
Q3_K_L at 4.54W and 3.80 T/s produces 1.195 J/token.

This means Q4_K_M is both faster and more energy-efficient than both Q8_0
and Q3_K_L on the Pi. The reason Q8_0 is not the clear winner despite drawing
less power is that the ARM NEON instruction set [11] handles 4-bit k-quant
dequantization efficiently under a 2-thread ceiling. The q4_K block structure
maps well to NEON SIMD lanes, while Q8_0's wider 8-bit operations use more
of the limited memory bandwidth per weight without proportionally improving
throughput.

The practical insight here is that optimal quantization on edge hardware is
not a universal rule. On the Pi 5 under these constraints, Q4_K_M is the
better choice for both speed and energy efficiency. This would not hold for
all edge ARM devices or all thread configurations.

## Finding 2: The iGPU Efficiency Inversion

The AMD 780M Vulkan tier shows the most counterintuitive result in the
dataset. At Q4_K_M, the 780M decodes at 16.47 T/s (roughly 41% faster than
the CPU at 11.67 T/s) while drawing 31.01W. The CPU draws 3.57W for the
same quantization.

The resulting J/token values:
CPU Q4_K_M: 0.306 J/token
780M Q4_K_M: 1.882 J/token

The iGPU costs 6.1 times more energy per generated token than the CPU it
shares silicon with.

The explanation is not inefficiency in the GPU itself. It is fixed power-
state overhead. To execute any GPU compute workload, the AMD 780M must hold
its entire hardware clock domain at operating voltage: the GPU shader engines,
the display controller, the memory controller, and the fabric interconnect.
On an integrated GPU, these domains are not independently power-gated during
compute-only workloads. The 31W draw for Q4_K_M is essentially the entire
iGPU complex running at operational voltage, not just the shader cores.

The CPU at 3.57W only activates the specific compute cores handling the
inference. The rest of the CPU complex, including the memory controller and
uncore domains, is in a reduced state.

For this model size and quantization format, the 780M is in an efficiency
dead zone. The throughput gain over the CPU (6.8 T/s improvement) does not
justify the power cost increase (27.4W increase). This is a concrete
illustration of the problem that Dr. Karlin's lab studies at scale: different
hardware architectures have fundamentally different relationships between
power draw and delivered compute throughput for AI workloads [10].

The iGPU would become more efficient relative to the CPU if the model were
large enough that the fixed power overhead is amortised over dramatically
higher throughput. For a 70B parameter model on 128GB of system RAM, the
iGPU might achieve 3 to 5x the CPU throughput while drawing the same 30W.
At that scale, the J/token comparison reverses. For a 3B model, it does not.

## Finding 3: The CUDA Kernel Switching Penalty

The Q3_K_L vs Q4_K_M comparison reveals an architecture-specific effect
that I believe is the most technically interesting finding in Study 1.

On the CPU tier:
Q3_K_L: 12.90 T/s, Q4_K_M: 11.67 T/s. Q3_K_L is 10.5% faster.

On the Vulkan tier:
Q3_K_L: 17.67 T/s, Q4_K_M: 16.47 T/s. Q3_K_L is 7.3% faster.

On the CUDA tier:
Q3_K_L: 87.39 T/s, Q4_K_M: 84.29 T/s. Q3_K_L is 3.7% faster.

Wait, Q3_K_L is still faster on CUDA in Study 1. But its J/token is very
close: Q3_K_L at 0.802 J/tok vs Q4_K_M at 0.828 J/tok. The performance
regression appears in Study 2 under the balanced profile, where Q3_K_L at
87.13 T/s vs Q4_K_M at 84.10 T/s but the power costs diverge:
Q3_K_L at 65.87W and Q4_K_M at 65.99W produces almost identical efficiency.

The key observation is that on CUDA, the throughput advantage of Q3_K_L over
Q4_K_M is substantially narrower than on CPU or Vulkan, despite Q3_K_L being
a smaller model. On CPU and Vulkan, a smaller model transfers fewer bytes per
decode step and improves throughput proportionally. On CUDA, this is partially
negated by dequantization kernel switching overhead.

The llama.cpp tensor loader confirms the mechanism. For Q3_K_L:
112 tensors use q3_K type, 84 tensors use q5_K type, 1 tensor uses q6_K type.
For Q4_K_M: 168 tensors use q4_K type, 29 tensors use q6_K type.

Q3_K_L requires the CUDA runtime to dispatch three distinct dequantization
kernels during each decode step. Each kernel has a different grid
configuration, shared memory layout, and warp occupancy pattern. On the
CUDA discrete command queue, transitioning between kernels requires a
synchronization barrier and a new kernel launch. This inter-kernel dispatch
overhead is present but negligible on CPU BLAS (where the matrix-vector
operations are sequential anyway) and on UMA Vulkan (where the command
latency between dispatches is very low). On CUDA, where the discrete command
queue between the CPU host and the GPU device adds latency to each kernel
launch, this overhead is measurable.

The generalization from this finding is: more aggressive quantization is not
monotonically beneficial on CUDA hardware. The number of distinct tensor
types in a format determines its CUDA kernel dispatch complexity, and this
complexity has a throughput cost that grows with GPU memory bandwidth. This
is consistent with the broader HPC observation that workload irregularity
hurts GPU efficiency [5].

Dettmers et al. [12] noted similar effects when designing LLM.int8(), where
outlier-aware quantization requires per-channel scaling operations that
introduce dispatch irregularity on GPU. The k-quant mixing approach in
Q3_K_L is a different mechanism but produces a related effect.
