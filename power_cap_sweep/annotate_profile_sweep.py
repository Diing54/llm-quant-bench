#!/usr/bin/env python3
import json
import sys
import os

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 annotate_profile_sweep.py <json_file> <profile_name>")
        sys.exit(1)

    out_file = sys.argv[1]
    profile = sys.argv[2]
    
    tgp_map = {
        "low-power": 65.0,
        "balanced": 75.0,
        "performance": 140.0
    }
    tgp = tgp_map.get(profile, 75.0)

    try:
        if os.path.exists(out_file):
            with open(out_file, 'r') as f:
                data = json.load(f)
            
            data['acpi_profile'] = profile
            data['hardware_tgp_watts'] = tgp
            data['study'] = 'power_constrained_sweep'
            
            with open(out_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"    SUCCESS: Annotated {out_file} with profile {profile} ({tgp}W)")
        else:
            print(f"    ERROR: Output JSON missing for {out_file}")
    except Exception as e:
        print(f"    ERROR annotating JSON: {e}")

if __name__ == "__main__":
    main()
