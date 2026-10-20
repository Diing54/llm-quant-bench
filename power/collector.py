#!/usr/bin/env python3
import argparse
import subprocess
import time
from pathlib import Path

def get_nvidia_watts():
    cmd = ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return float(res.stdout.strip()) if res.returncode == 0 else 0.0

def get_amd_watts():
    for hwmon in Path("/sys/class/hwmon").glob("hwmon*"):
        name_file = hwmon / "name"
        if name_file.exists() and name_file.read_text().strip() == "amdgpu":
            for pwr_file in ["power1_input", "power1_average", "power2_average"]:
                p_path = hwmon / pwr_file
                if p_path.exists():
                    try:
                        val = float(p_path.read_text().strip())
                        if val > 0:
                            return val / 1_000_000.0
                    except ValueError:
                        pass
    return 0.0

def get_cpu_watts():
    # Read AMD/Intel RAPL (Running Average Power Limit) energy counters if available
    # Typically located at /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj
    # Or AMD family equivalent energy counters in hwmon (k10temp)
    for hwmon in Path("/sys/class/hwmon").glob("hwmon*"):
        name_file = hwmon / "name"
        if name_file.exists() and name_file.read_text().strip() == "k10temp":
            for pwr_file in ["power1_input", "power2_input"]:
                p_path = hwmon / pwr_file
                if p_path.exists():
                    try:
                        val = float(p_path.read_text().strip())
                        if val > 0:
                            return val / 1_000_000.0
                    except ValueError:
                        pass
    
    # Fallback to powercap RAPL if available
    for rapl in Path("/sys/class/powercap").glob("intel-rapl*"):
        energy_file = rapl / "energy_uj"
        if energy_file.exists():
            try:
                # Read energy delta over time or use instantaneous if available
                e1 = int(energy_file.read_text().strip())
                time.sleep(0.1)
                e2 = int(energy_file.read_text().strip())
                return (e2 - e1) / 1_000_000.0 * 10.0 # converted to watts (joules per 0.1s * 10)
            except Exception:
                pass
    return 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        f.write("timestamp,power_watts\n")
        try:
            while True:
                if args.tier == "tier4":
                    w = get_nvidia_watts()
                elif args.tier == "tier3":
                    w = get_amd_watts()
                elif args.tier == "cpu":
                    w = get_cpu_watts()
                else:
                    w = 0.0
                
                f.write(f"{time.time():.3f},{w:.2f}\n")
                f.flush()
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    main()
