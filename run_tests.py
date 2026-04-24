#!/usr/bin/env python3
"""
Master test runner for Greek Exposure Engine.

Usage:
  python run_tests.py              # Run all tests
  python run_tests.py -k models    # Run only model tests
  python run_tests.py -v           # Verbose output
  python run_tests.py --quick      # Skip slow performance tests
"""

import sys
import subprocess


def main():
    args = ["python", "-m", "pytest", "tests/", "-v", "--tb=short", "-x"]

    # Pass through any CLI args
    if len(sys.argv) > 1:
        if "--quick" in sys.argv:
            sys.argv.remove("--quick")
            args.extend(["-m", "not slow"])
        args.extend(sys.argv[1:])

    result = subprocess.run(args, cwd=".")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
