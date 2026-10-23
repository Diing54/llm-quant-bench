# Motivation and Background

## Why I Built This

I built llm-quant-bench to answer a question that matters both practically
and academically: how does the choice of quantization format interact with
hardware architecture to determine the real energy cost of running a large
language model?

Most public LLM benchmarks focus on throughput in Tokens/second. That
metric tells you how fast a model generates text, but it tells you nothing
about how much energy it consumes doing so. In the context of modern data
centre design, energy efficiency is often the binding constraint. Data centre
power challenges are both straining electricity grids and limiting
computational throughput [10]. A model that runs at 80 T/s and costs 1.0
J/token is strictly worse than one running at 75 T/s at 0.3 J/token if your
constraint is the power draw of a single compute node.

This distinction is the starting point for the research program of Dr. Ian
Karlin's Beyond Moore's Lab at Queen's University, which investigates how
code optimization, hardware setting changes, and different architectures
impact the energy efficiency and performance of systems in power-limited
environments [10]. I designed this project to produce the kind of empirical,
hardware-level data that feeds directly into that research direction.

## The Research Gap

As of 2024, post-training quantization has become the dominant method for
deploying large language models on consumer and edge hardware. Formats like
GPTQ [3] and AWQ [4] have demonstrated that significant model compression
is achievable with minimal accuracy loss. The llama.cpp k-quant formats [1]
extend this with mixed-precision approaches that optimize for ARM and x86
SIMD instruction sets.

What the literature does not comprehensively cover is how different
quantization formats interact with the specific microarchitecture of the
device running inference. A format like Q3_K_L uses a mixed tensor type
layout that is optimal on some architectures and actively harmful on others.
Understanding which formats are appropriate for which hardware is a practical
systems engineering problem, not just an accuracy question.

The Roofline model [5] provides the theoretical framework for understanding
why these differences exist. Autoregressive LLM decoding is a
memory-bandwidth-bound operation. At each generation step, the model must
transfer the entire weight matrix from VRAM or DRAM to compute a single
output token. The binding constraint is not compute throughput (FLOPS) but
memory bandwidth (GB/s). Quantization reduces model size and therefore
reduces the bandwidth demand per token. But quantization also introduces
dequantization overhead. Whether the bandwidth savings outweigh the compute
cost of dequantization depends entirely on the architecture executing the
kernels.

I designed this study to measure that tradeoff empirically, across four
genuinely different hardware tiers.

## Connection to HPC Research

Modern heterogeneous computing systems, including both data centre nodes and
consumer laptops, increasingly contain multiple types of compute: dedicated
CPUs, integrated GPUs, and discrete GPUs with different memory subsystems,
bandwidth characteristics, and power management mechanisms [10]. The number
of processor types available for data centre designs is increasing as Moore's
Law ends, and how to design, program and use systems that take advantage of
these options is becoming a complex challenge [10].

My hardware stack is a miniature version of exactly this problem. A Lenovo
Legion Slim 5 contains an AMD Ryzen CPU, an AMD Radeon integrated GPU sharing
unified memory with the CPU, and an NVIDIA RTX 4070 with dedicated GDDR6X
VRAM. These three compute resources have fundamentally different memory
subsystems and power envelopes. Running the same workload across all three
and measuring both throughput and energy cost produces empirical data on how
the same inference task behaves across heterogeneous node components.

Study 2 extends this into power-constrained scenarios. Real-world data
centres operate under rack-level or node-level power caps enforced by
infrastructure management systems. I simulate these constraints using
hardware-enforced ACPI platform profiles on the Lenovo Legion, which engage
the laptop's Embedded Controller to enforce distinct GPU Total Graphics Power
(TGP) budgets. The result is a Pareto frontier for how GPU inference
throughput scales with a progressively tightening power envelope, which is a
direct empirical contribution to the question Dr. Karlin's lab is studying
at scale [10].
