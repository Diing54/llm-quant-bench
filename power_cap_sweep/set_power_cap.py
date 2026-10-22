#!/usr/bin/env python3
import sys
import subprocess

def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip()

def main():
    if len(sys.argv) != 2:
        print("Usage: sudo python3 set_power_cap.py <limit_in_watts>")
        print("Example: sudo python3 set_power_cap.py 45")
        sys.exit(1)

    target_w = sys.argv[1]

    # 1. Fetch hardware constraints
    try:
        min_limit = run_cmd("nvidia-smi --query-gpu=power.min_limit --format=csv,noheader,nounits")
        max_limit = run_cmd("nvidia-smi --query-gpu=power.max_limit --format=csv,noheader,nounits")
        default_limit = run_cmd("nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits")
        
        print(f"Hardware Constraints: Min = {min_limit}W | Max = {max_limit}W | Default = {default_limit}W")
        
        if float(target_w) < float(min_limit) or float(target_w) > float(max_limit):
            print(f"Error: Target {target_w}W is outside hardware bounds ({min_limit}W - {max_limit}W).")
            sys.exit(1)
    except Exception as e:
        print(f"Failed to read constraints: {e}")
        sys.exit(1)

    # 2. Apply the cap
    print(f"Applying {target_w}W power limit...")
    set_result = subprocess.run(f"nvidia-smi -pl {target_w}", shell=True, capture_output=True, text=True)
    
    if set_result.returncode != 0:
        print("Error setting power limit. Did you run with sudo?")
        print(set_result.stderr)
        sys.exit(1)

    # 3. Verify
    current = run_cmd("nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits")
    print(f"Success: GPU power limit currently locked at {current}W")

if __name__ == "__main__":
    main()
