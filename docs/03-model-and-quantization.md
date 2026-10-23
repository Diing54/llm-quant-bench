# Model Selection and Quantization Formats

## Why Llama 3.2-3B-Instruct

I chose Llama 3.2-3B-Instruct for this benchmark for several specific reasons,
not by default.

The first reason is size. At 3.21 billion parameters, this model is small
enough to run on all four hardware tiers in at least one quantization format.
A larger model like Llama 3.1-8B would not fit in 8GB of VRAM in F16 format,
which would have prevented a full cross-tier comparison at the highest
precision level. The 3B model gives me a complete matrix.

The second reason is practical relevance. 3B parameter models represent the
frontier of what is genuinely deployable on edge and consumer hardware. They
are not toy models. Llama 3.2-3B-Instruct performs competitively on many
instruction-following benchmarks [2] and is used in production deployments
on resource-constrained devices. Benchmarking at this size produces findings
that generalize to real deployment scenarios, not artificial laboratory
conditions.

The third reason is the GGUF format availability. The bartowski GGUF
conversion of Llama 3.2-3B-Instruct on Hugging Face provides all four
quantization levels I tested in a single repository with consistent
imatrix-calibrated weights. Using the same source weights across all
quantization levels ensures the performance differences I observe are due
to quantization format, not variation in the base model.

I used the instruction-tuned variant rather than the base model because the
instruction format affects the tokenization of the prompt. Using a consistent,
structured prompt with a known token count was important for ensuring
reproducible prefill measurements across all hardware tiers.

## The GGUF K-Quant Formats

llama.cpp uses a family of quantization formats called k-quants [1]. These
are block-quantized formats where groups of weights are quantized together
using a shared scaling factor. The k-quant family uses superblocks and
sub-blocks to improve accuracy by adapting the quantization scale at a finer
granularity than naive per-tensor quantization.

Understanding why I chose these specific four formats requires understanding
what each one actually does at the tensor level.

### F16 (16-bit Floating Point, Baseline)

F16 stores each weight as a standard 16-bit half-precision floating point
value. There is no quantization. This is the baseline measurement of what
the hardware can achieve without any compression overhead or accuracy loss.
For the CPU and GPU tiers, F16 establishes the upper bound on model quality
and the lower bound on throughput and memory efficiency.

F16 at 3.21B parameters requires 5.98 GiB of memory. This is the format
that reveals the memory bandwidth ceiling of each hardware tier most clearly,
because there is no dequantization work. Every cycle spent reading weights
from memory is a cycle spent doing useful inference.

### Q8_0 (8-bit, near-lossless)

Q8_0 quantizes each weight to 8 bits using a per-block scale factor. It
retains approximately 99.9% of the original model accuracy [1]. The model
size reduces to 3.18 GiB, roughly half of F16. The dequantization overhead
is minimal because 8-bit to float16 conversion is a simple bit manipulation
that modern SIMD instruction sets (AVX-512 on x86, NEON on ARM) handle
efficiently.

I included Q8_0 as the near-lossless anchor point. If the energy cost of
Q8_0 is substantially worse than Q4_K_M while the accuracy difference is
negligible, that is a strong practical argument for preferring Q4_K_M in
deployment.

### Q4_K_M (4-bit, k-quant medium)

Q4_K_M is a mixed-precision 4-bit format using two tensor types: q4_K for
the bulk of the weight matrices and q6_K for the embedding and output layers.
The 6-bit precision for critical layers is what distinguishes K_M from K_S,
and it provides a meaningful accuracy improvement. At approximately 99.2%
accuracy retention [1] and 1.87 GiB model size, Q4_K_M occupies what the
community widely considers the optimal operating point for k-quant formats.

The choice to include Q4_K_M as the primary comparison point rather than
a simpler Q4_0 format was deliberate. Q4_0 uses uniform 4-bit quantization
and tends to show more accuracy degradation on instruction-following tasks.
Q4_K_M's mixed-precision design is specifically intended to preserve the
quality of the most accuracy-sensitive weight matrices.

### Q3_K_L (3-bit, k-quant large)

Q3_K_L is the most aggressive quantization format I tested. It uses three
tensor types: q3_K for attention layers, q5_K for feed-forward layers, and
q6_K for the output layer. The model reduces to 1.68 GiB, a 72% reduction
from F16. Accuracy retention is approximately 96.5% [1], which is a
noticeable but not catastrophic quality reduction.

I included Q3_K_L specifically because it demonstrates the architecture-
dependent performance regression described in Finding 3. The fact that Q3_K_L
uses three distinct tensor types is not incidental. It is the structural
property that creates the CUDA kernel switching overhead I measured on the
RTX 4070, and understanding this required including the format in the study.

## Why F16 Was Excluded from Tier 1 and Tier 3

### Tier 1: Raspberry Pi 5

The F16 model is 5.98 GiB. The Pi 5 has 8 GB of LPDDR4X. At first glance,
this seems sufficient. In practice, it is not.

By the time the Debian OS, kernel, and active processes consume their share
of RAM, and the llama.cpp process allocates its inference buffers and KV cache
on top of the model weights, the total memory requirement for F16 inference
exceeds what is available in physical RAM. The result is either an OOM kill
during model loading or severe swap thrashing that renders power telemetry
meaningless.

Even if F16 could be loaded, the practical throughput at 2 threads and
2.0 GHz would be below 1 T/s. That data point is not informative for the
purposes of this study, which is concerned with deployable performance ranges.
F16 on the Pi 5 is a confirmed architectural constraint, not an oversight.

### Tier 3: AMD Radeon 780M

The 780M uses Unified Memory Architecture. In principle, it shares the
system's RAM pool and should be able to load a 5.98 GiB model. In practice,
the Linux RADV Vulkan driver allocates device memory as a single contiguous
block when running with the --no-mmap flag.

I used --no-mmap across all tiers to ensure consistent telemetry. Memory-
mapped file access introduces OS page-fault overhead during inference that
varies with system memory pressure and is not representative of controlled
deployment. Without mmap, the driver must preallocate the entire model as
one contiguous physical memory region. The Linux memory allocator cannot
guarantee a physically contiguous 6 GiB block at runtime, even when total
free RAM is sufficient. The RADV driver returns ERROR_OUT_OF_DEVICE_MEMORY
and the process aborts during initialization.

Using mmap as a workaround would have made Tier 3 results non-comparable
to the other tiers. I therefore excluded F16 from Tier 3 and documented this
as an architectural constraint of integrated GPU deployment on Linux. This
finding itself is useful: it means that for strict iGPU deployments using
RADV on Linux, quantized formats are not optional for models above
approximately 4 GB.

## Quantization Format Summary

| Format | Size (GiB) | BPW | Tensor Types | Est. Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| F16 | 5.98 | 16.00 | f16 | 100% (baseline) |
| Q8_0 | 3.18 | 8.50 | q8_0 | ~99.9% |
| Q4_K_M | 1.87 | 5.01 | q4_K, q6_K | ~99.2% |
| Q3_K_L | 1.68 | 4.50 | q3_K, q5_K, q6_K | ~96.5% |

Accuracy retention estimates are from llama.cpp community perplexity
benchmarks [1] and are not independently measured in this project.
