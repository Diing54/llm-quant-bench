# Hardware Architectures, Telemetry, and Failure Modes

## Tier 1: Raspberry Pi 5 Power Stability
Running unthrottled four-core inference on the Raspberry Pi 5 caused instantaneous current spikes ($dI/dt$) that tripped the PMIC power-good threshold, triggering hard system reboots. To maintain stable telemetry, I had to cap the CPU frequency at 2.0 GHz and restrict execution to two threads (`THREADS=2`), alongside piping `/dev/null` to stdin to prevent interactive mode hangs. 

Furthermore, executing uncompressed F16 models on the Pi 5 is infeasible due to memory bandwidth saturation and capacity constraints, necessitating quantization for edge viability.

## Tier 2: Ryzen 8845HS RAPL Scope
The AMD Ryzen 7 8845HS power readings capture RAPL package power for the CPU die only. They explicitly exclude DRAM, SoC uncore, and peripheral power domains. For a rigorous node-level comparison against discrete accelerators, approximately 3W to 5W should be added to account for these external domains.

## Tier 3: Radeon 780M iGPU and the F16 Memory Allocation Failure
When attempting to run the uncompressed F16 model (6.0 GiB) on the AMD Radeon 780M integrated GPU without memory mapping (`--no-mmap`), the execution crashed with an Out of Memory error. 

Even though the 780M shares the system system RAM pool (Unified Memory Architecture or UMA), the Linux RADV Vulkan driver could not allocate a single, physically contiguous 6GB block of memory. This proved that for strict Linux iGPU deployments, quantization is not merely a tool for saving disk space or accelerating inference, it is a physical necessity to fit within contiguous memory allocation limits imposed by the driver layer.

## Tier 4: RTX 4070 Laptop GPU and TDP Ceilings
The NVIDIA RTX 4070 Laptop GPU operates within a firmware-enforced Total Graphics Power (TGP) envelope. Across all standard quantization runs, the power draw remained locked between 51W and 52W. This indicated that the GPU was hitting its sustained thermal design power ceiling rather than dynamically scaling power to match the workload. This realization directly motivated the second phase of research: the power-cap sweep study.
