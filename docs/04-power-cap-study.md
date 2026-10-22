# Study 2: Power Budget Scaling Under Hardware-Enforced TGP Constraints

## Motivation and Simulation of Data Centre Constraints
Data centres increasingly operate under strict node-level power envelopes enforced at the rack or PDU controller level. To study how LLM inference behaves in such environments, I designed an empirical sub-study using the Lenovo Legion hardware Embedded Controller (EC). 

Unlike software-level power caps (such as `nvidia-smi -pl`, which are often blocked by laptop VBIOS limits), the Linux kernel ACPI platform profile interface interacts directly with `nvidia-powerd` to negotiate hardware-enforced Total Graphics Power (TGP) budgets. The GPU cannot physically exceed these budgets regardless of workload intensity.

## Experimental Design
I swept across four hardware-negotiated platform profiles:
- **low-power:** ~54.7W to 56.3W TGP (simulating power-capped edge nodes).

- **balanced:** ~68.8W to 70.8W TGP (simulating standard data centre allocations).

- **performance:** ~77.4W to 71.8W TGP.

- **max-power:** ~78.0W to 72.8W TGP (unconstrained peak workload).


Each profile executed the `Q4_K_M` and `Q8_0` models across three consecutive benchmark runs, recording telemetry via background polling.

## Findings and Implications
The resulting efficiency curves revealed a sharp diminishing-returns wall. When tightening the power envelope from ~78W down to ~55W:
- Decode throughput dropped by a mere **4.7%** (from 85.24 down to 81.23 T/s).

- Energy efficiency improved by **26.4%** (from 0.915 down to 0.674 Joules per token).


This proves that unconstrained max-power operation on consumer hardware is highly inefficient for LLM inference. Operating within a constrained power profile achieves near-identical throughput while drastically reducing thermal and electrical expenditure.
