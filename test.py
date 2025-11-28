#!/usr/bin/env python3
"""Simple wrapper to run the environment import check.

This script performs the import checks (same as devtools/check_env.py) and is intended
for use with `python test.py` after activating a virtual environment.

Usage:
  python test.py
  python test.py --verbose
"""
import sys, argparse, importlib

parser = argparse.ArgumentParser()
parser.add_argument('--verbose', '-v', action='store_true')
args = parser.parse_args()

REQUIRED_PACKAGES = ['pandas', 'numpy', 'nptdms', 'scipy', 'plotly', 'streamlit']

failed = []
for pkg in REQUIRED_PACKAGES:
    try:
        importlib.import_module(pkg)
    except Exception as e:
        failed.append((pkg, str(e)))

if failed:
    print('Failed imports:')
    for name, err in failed:
        print(f"- {name}: {err}" if args.verbose else f"- {name}")
    sys.exit(1)

print('All imports OK')