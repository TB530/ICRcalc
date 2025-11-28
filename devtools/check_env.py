#!/usr/bin/env python3
"""Check Python environment and imports for required packages.

Usage examples:
    python devtools/check_env.py        # just run import checks
    python devtools/check_env.py --verbose
"""
import sys
import importlib
import argparse

parser = argparse.ArgumentParser(description='Check environment and imports')
parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed errors for failed imports')
args = parser.parse_args()

REQUIRED_PACKAGES = ['pandas', 'numpy', 'nptdms', 'scipy', 'plotly', 'streamlit']

failed = []
for pkg in REQUIRED_PACKAGES:
    try:
        importlib.import_module(pkg)
    except Exception as e:
        failed.append((pkg, str(e)))

if failed:
    print('FAILED IMPORTS:')
    for name, err in failed:
        print(f"- {name}: {err}" if args.verbose else f"- {name}")
    print('\nSuggested actions:')
    print('- Check that `requirements.txt` contains the correct package names')
    print('- Ensure your Python version is compatible (recommended: 3.10 or 3.11)')
    print('- If using a self-hosted environment, install build deps for SciPy:')
    print('  sudo apt-get update && sudo apt-get install -y build-essential gfortran libopenblas-dev liblapack-dev')
    sys.exit(1)

print('All imports OK')
sys.exit(0)
