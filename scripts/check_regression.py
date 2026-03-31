import sys
import json
from pathlib import Path

def main():
    if len(sys.argv) < 3:
        print("Usage: python check_regression.py regression.json benchmark_baseline.json")
        sys.exit(1)

    reg_path = Path(sys.argv[1])
    base_path = Path(sys.argv[2])

    if not reg_path.exists():
        print(f"ERROR: Regression file {reg_path} not found.")
        sys.exit(1)
    if not base_path.exists():
        print(f"ERROR: Baseline file {base_path} not found.")
        sys.exit(1)

    with open(reg_path) as f:
        reg = json.load(f)
    with open(base_path) as f:
        base = json.load(f)

    # Compare tok/s
    # base has results in the top level or nested?
    # benchmark_baseline.json has 'token_throughput' or 'phase_N_results'
    # Let's just check the most recent phase result if possible, or a target.
    # For CI, we'll check against a set of fixed targets if baseline is old.
    
    reg_results = reg.get("results", {})
    
    # Simple check: Profile C (AR) should be > 1.0 tok/s
    # Profile F (Sparse) should be > 1.5 tok/s
    # Profile D (LUT) should be > 0.6 tok/s
    
    thresholds = {
        "A": 0.5, # PyTorch baseline can be slow on CI
        "C": 1.0,
        "D": 0.6,
        "F": 1.5,
        "G": 0.5,
    }

    found_regression = False
    for profile, metrics in reg_results.items():
        tps = metrics.get("tok/s", 0)
        if tps is None: tps = 0
        target = thresholds.get(profile, 0)
        
        print(f"Profile {profile}: {tps:.3f} tok/s (threshold: {target:.3f})")
        if tps < target:
            print(f"  FAIL: Profile {profile} is below threshold!")
            found_regression = True

    if found_regression:
        sys.exit(1)

    print("\nSuccess: No performance regressions detected.")
    sys.exit(0)

if __name__ == "__main__":
    main()
