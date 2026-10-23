# Study 2: Power-Constrained Inference Sweep

## Motivation

Study 1 revealed that the RTX 4070's power draw across all quantization
formats varied by less than 0.9% (69.80W to 70.34W). This means the GPU
was hitting its firmware TDP ceiling regardless of workload intensity. The
implication is that from Study 1 alone, I cannot determine how the RTX 4070's
throughput and efficiency behave when a real power constraint is applied from
outside. The GPU was never actually power-constrained in Study 1; it was
always at its ceiling.

This matters because real data centre environments operate under node-level
or rack-level power budgets. Data centre power challenges are both straining
electricity grids and limiting computational throughput [10]. Understanding
how inference throughput scales within a progressively tightening power
envelope is a direct empirical question for this research area.

I designed Study 2 to answer: given a fixed hardware-enforced power budget
on the GPU node, how does decode throughput and energy efficiency change as
that budget increases from constrained to permissive?

## Why nvidia-smi -pl Failed and What I Used Instead

My first attempt was to sweep the RTX 4070's power limit directly using
nvidia-smi -pl at values of 35W, 45W, 55W, 65W, and so on up to 115W. The
output was:

    Changing power management limit is not supported for GPU: 00000000:01:00.0.

The Lenovo Legion's Embedded Controller (EC) holds the GPU Total Graphics
Power (TGP). The ACPI DSDT table on Lenovo Legion laptops contains a
Notify(NPCF, 0xC0) call that should transfer TGP authority from the EC to
the NVIDIA driver, but NVIDIA's Linux driver stack does not respond to this
ACPI notification the same way the Windows driver stack does. The result is
that nvidia-smi -pl sends a command to a driver layer that has no authority
over the hardware TGP. The EC ignores it.

The correct solution on Lenovo Legion hardware is to use the ACPI Platform
Profile interface introduced in Linux 5.12 [9], combined with the nvidia-
powerd daemon that allows NVIDIA Dynamic Boost to negotiate TGP with the EC.

I enabled nvidia-powerd:

```bash
sudo systemctl enable --now nvidia-powerd.service
```

With nvidia-powerd running, switching the ACPI platform profile causes the
EC to renegotiate the GPU TGP budget, and nvidia-powerd communicates this
budget to the NVIDIA driver stack. The actual TGP is then reported by
nvidia-smi and is hardware-enforced.

## Why Platform Profiles Are More Rigorous Than nvidia-smi -pl

Platform profile TGP enforcement is a physical hardware constraint applied
by the Embedded Controller. The GPU cannot exceed the allocated TGP budget
regardless of how hard the workload pushes. When I set a power limit via
nvidia-smi -pl on a desktop GPU, I am relying on the driver to honour a
software request. On this laptop, the EC enforces the limit at the silicon
level.

This means my power readings in Study 2 are not approximations of a target.
They are measured outcomes against a physically imposed ceiling. This is
actually a more rigorous form of power-constrained benchmarking than
arbitrary software cap setting, because it accurately represents the kind
of external enforcement that a data centre PDU or rack controller applies.

## Profiles and TGP Values

I tested three platform profiles on the Lenovo Legion Slim 5. I verified
the actual TGP by reading nvidia-smi after a 10-second stabilization window
following each profile switch:

| ACPI Profile | Hardware TGP Enforced | Analogous Deployment Scenario |
| :--- | :--- | :--- |
| low-power | 65.0W | Power-capped shared rack slot |
| balanced | 75.0W | Standard data centre GPU allocation |
| performance | 140.0W | Dedicated HPC compute node with Dynamic Boost |

I switched profiles programmatically in the run_sweep.sh script:

```bash
echo "$profile" | sudo tee /sys/firmware/acpi/platform_profile
sleep 10  # stabilization window
```

The 10-second stabilization window ensures GPU clocks, fan speeds, and
the EC's power negotiation have settled before benchmarks begin.

## Findings

### F16 Throughput Is Insensitive to TGP

F16 decode throughput across all three profiles was 32.01, 32.21, and
32.22 T/s for low-power, balanced, and performance respectively. This is
effectively flat. Expanding the TGP budget from 65W to 140W produced a
0.6% throughput improvement.

The reason connects directly to the Roofline model [5]. F16 decode is
bounded by GPU memory bandwidth. The RTX 4070's GDDR6X memory bus runs
at approximately 250 GB/s. Transferring 5.98 GiB of F16 weights per token
generation step at this bandwidth sets a hard throughput ceiling regardless
of how much compute power is available. More TGP does not increase memory
bandwidth on this GPU. F16 is at the memory bandwidth roof, and more watts
does not move that roof.

### Q3_K_L and Q4_K_M Benefit Substantially from Higher TGP

Q3_K_L throughput: 77.98 T/s (65W), 87.13 T/s (75W), 94.23 T/s (140W).
This is a 21% improvement from low-power to performance.

Q4_K_M throughput: 80.75 T/s (65W), 84.10 T/s (75W), 88.02 T/s (140W).
This is a 9% improvement from low-power to performance.

The reason these formats respond to TGP while F16 does not is that they are
no longer purely memory-bandwidth-bound. The dequantization step introduces
a compute component. With more TGP, the GPU can sustain higher shader
frequencies and execute the dequantization kernels faster, which shifts the
throughput ceiling upward.

### The Low-Power Profile Is Most Energy-Efficient

Despite lower throughput, the 65W low-power profile achieves the best J/token
for Q3_K_L (0.690 J/tok) and Q4_K_M (0.665 J/tok). The performance profile
at 140W delivers higher throughput but at higher power draw, producing worse
J/token values.

This counterintuitive result is an important finding for data centre
workload scheduling. If the objective is to minimize energy cost per query
(for example, to maximize queries per kilowatt-hour), operating at a
lower TGP is more efficient even though absolute throughput is lower. The
optimal TGP depends on whether the system is optimizing for throughput
(performance profile), energy cost per query (low-power profile), or a
balance between the two (balanced profile).

This maps directly to the kind of operational decision that data centre
operators make when managing heterogeneous GPU nodes under power budgets [10].
The experimental result here provides a quantitative basis for that decision
on consumer-class Ada Lovelace GPU hardware.

### Q8_0 Efficiency Improves Modestly with TGP

Q8_0 J/token: 1.003 (65W), 1.215 (75W), 1.234 (140W). Unexpectedly, Q8_0
is most efficient at the lowest TGP setting. At 65W, it achieves 54.36 T/s.
At 140W, 56.78 T/s. The throughput gain is only 4.5% while power increases
substantially. Q8_0 is closer to the memory bandwidth ceiling than the
quantized formats because its dequantization overhead is minimal, making
it behave more like F16 in terms of TGP sensitivity.
