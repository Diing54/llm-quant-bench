# Quantization Formats and CUDA Kernel Fragmentation

## Why Llama 3.2 3B and GGUF
I selected Llama-3.2-3B-Instruct because it represents an optimal scale for edge and desktop deployment, balancing conversational capability with memory footprint. I chose the GGUF format because `llama.cpp` provides transparent, highly optimized backends across CPU, Vulkan, and CUDA, allowing me to isolate hardware execution characteristics without framework overhead from PyTorch or Hugging Face.

## The Quantization Matrix
I evaluated four quantization profiles:
- **F16:** Uncompressed baseline (100% relative accuracy, highest memory bandwidth pressure).
- **Q8_0:** 8-bit quantization (~99.9% accuracy retention).
- **Q4_K_M:** 4-bit medium k-quant (~97.5% accuracy retention, balancing precision and memory footprint).
- **Q3_K_L:** 3-bit large k-quant (~92.0% accuracy retention).

## The CUDA Kernel Fragmentation Penalty (`Q3_K_L` vs `Q4_K_M`)
A counterintuitive finding emerged during testing. While `Q3_K_L` outperformed `Q4_K_M` in generation throughput on both the CPU and Vulkan tiers, it suffered a performance degradation on the discrete RTX 4070. Despite containing fewer parameters, `Q3_K_L` ran approximately 5% slower than `Q4_K_M`.

The root cause is tensor-type fragmentation. The `Q3_K_L` format mixes three distinct tensor types (`q3_K`, `q5_K`, and `q6_K`). On every layer boundary during autoregressive decode, the CUDA runtime must launch and synchronize a different dequantization kernel, each with its own grid configuration and memory layout. On a discrete GPU with an independent command queue, this inter-kernel dispatch overhead introduces a measurable bottleneck during the memory-bandwidth-limited decode phase. By contrast, `Q4_K_M` uses only two tensor types (`q4_K` and `q6_K`), reducing dispatch complexity and recovering throughput.
