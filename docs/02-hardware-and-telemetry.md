# Hardware Setup and Telemetry Design

## Hardware Overview

I ran all benchmarks on two machines: a Lenovo Legion Slim 5 Gen 9 running
Arch Linux (the primary machine), and a Raspberry Pi 5 running Debian Bookworm
(the edge tier).

### Lenovo Legion Slim 5 Gen 9 (Primary)

This machine contains three distinct compute resources that I benchmarked
independently:

**AMD Ryzen 7 8845HS (Tier 2, CPU)**
The 8845HS is a Zen 4 architecture CPU with 8 cores and 16 threads. It
includes AVX-512 instruction set support, which llama.cpp uses for
accelerated CPU matrix-vector operations during decode. In CPU inference
mode, the model weights are stored in system DRAM and transferred to the CPU
at DDR5 bandwidth rates during each decode step. On this machine, DDR5
bandwidth is approximately 85 GB/s, which is the memory bandwidth ceiling
for CPU inference.

**AMD Radeon 780M (Tier 3, iGPU)**
The 780M is an RDNA3 integrated GPU with 12 compute units, sharing the
system RAM pool with the CPU under a Unified Memory Architecture (UMA). It
has no dedicated video memory. The Vulkan RADV driver handles compute
dispatch. Because it uses UMA, it accesses model weights through the same
DDR5 bandwidth pool as the CPU, but under a different memory allocator and
with different cache topology.

**NVIDIA RTX 4070 Laptop GPU (Tier 4, dGPU)**
The RTX 4070 Laptop is an Ada Lovelace architecture GPU with 8GB of dedicated
GDDR6X VRAM and approximately 250 GB/s of GPU-local memory bandwidth. All
28 model layers were offloaded to CUDA VRAM using -ngl 99 in llama.cpp.
For Study 2, I enabled nvidia-powerd to allow NVIDIA's Dynamic Boost
subsystem to negotiate the Total Graphics Power with the Embedded Controller.
Without nvidia-powerd, the GPU remains at a conservative default TGP
regardless of the ACPI platform profile.

### Raspberry Pi 5 (Tier 1, Edge ARM)

The Pi 5 uses a Broadcom BCM2712 SoC with 4 ARM Cortex-A76 cores at 2.4 GHz
and 8 GB of LPDDR4X memory. For benchmarking, I capped the maximum CPU
frequency at 2.0 GHz and restricted llama.cpp to 2 threads.

The frequency cap and thread restriction were not optional. In initial
experiments, running 4-core unthrottled inference caused instantaneous
current spikes (dI/dt transients) that exceeded the PMIC power-good
threshold. The Pi would hard-reboot mid-benchmark, making stable power
telemetry impossible. At 2 threads and 2.0 GHz, the current draw remained
within safe bounds and benchmarks ran to completion consistently.

I also piped /dev/null to stdin for all Pi benchmarks. Without this, llama.cpp
enters an interactive waiting mode after generating the initial output tokens,
which causes the process to hang indefinitely rather than exit cleanly.

## Power Telemetry Design

Choosing the right power measurement method for each hardware tier was one
of the more careful design decisions in this project. I did not want software
estimates. I wanted hardware-reported values wherever possible.

### Tier 2 (CPU): Linux RAPL via powercap

Intel and AMD processors expose energy consumption counters through the RAPL
(Running Average Power Limit) interface [8]. On Linux, these are accessible
via the powercap sysfs subsystem at
/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj. I read this counter
before and after each inference run, compute the delta in microjoules, and
divide by elapsed time in seconds to get average watts.

RAPL on the Ryzen 8845HS measures CPU die package power. It does not include
DRAM power, SoC uncore power, or PCIe bus power. For a full node-level power
comparison, these domains add approximately 3 to 5W. I document this scope
limitation explicitly in my findings. Reading RAPL sensors requires root
access on most Linux configurations, which is why the CPU tier benchmarks
require sudo.

### Tier 3 (iGPU): AMDGPU hwmon sysfs

The AMDGPU kernel driver exposes GPU power consumption through the Linux
hwmon sysfs interface. The collector.py script dynamically discovers the
amdgpu hwmon node at runtime by scanning /sys/class/hwmon/hwmon*/name for
the value "amdgpu" and reading power1_input, which returns instantaneous
GPU package power in microwatts. I poll this at 10Hz throughout each
benchmark run and average the readings.

I chose power1_input over power1_average because the instantaneous reading
captures dynamic power variation during token generation, while the rolling
average tends to smooth out the peaks that are most important for
understanding thermal and electrical stress.

### Tier 4 (dGPU): nvidia-smi NVML

For the RTX 4070, I poll nvidia-smi --query-gpu=power.draw at 100ms
intervals using the NVML API through the collector.py subprocess. This
returns the GPU card's total power draw as measured by the internal power
sensor.

In Study 1, I discovered that the RTX 4070 power readings showed less than
0.9% variance across all quantization formats (ranging from 69.80W to
70.34W). This indicates the GPU was operating at its firmware TDP ceiling
throughout every benchmark window and was not dynamically scaling to the
workload. This observation becomes a key motivating finding for Study 2.

### Tier 1 (Pi 5): PMIC VDD_CORE Polling

The Raspberry Pi 5 uses an RP1 I/O controller chip and exposes power domain
readings via its PMIC (Power Management Integrated Circuit). The
pi_collector.py script reads the VDD_CORE voltage and current at 1Hz to
compute instantaneous core power draw. This is a lower sampling rate than
the other tiers because the Pi's PMIC interface does not support higher
frequency reads without risking I2C bus contention with other PMIC consumers.
At 1Hz and a benchmark duration of 300+ seconds, this provides a statistically
stable average.

## Isolating Hardware Backends on an Optimus Laptop

The Lenovo Legion uses NVIDIA Optimus, which means the NVIDIA GPU manages
display output and the AMD GPU handles compute by default in some
configurations. When I attempted to run Vulkan workloads for Tier 3 without
explicit driver isolation, nvidia-smi confirmed that all Vulkan dispatches
were being routed silently to the RTX 4070 rather than the 780M.

The fix was to explicitly set the Vulkan ICD (Installable Client Driver) to
the RADV AMD driver before launching any Tier 3 benchmark:

```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json
```

After setting this, I verified the correct GPU was active by checking the
Vulkan device name reported in the llama.cpp initialization output before
every Tier 3 run.
