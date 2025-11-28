#!/usr/bin/env python3
"""Download and extract pure-Python packages into the repo `vendor/` folder.

This script downloads the wheel files for the desired packages using `pip download` and
extracts their Python packages into `vendor/` so `ICR_analysis_tool` can import them
without requiring pip installation at runtime (useful for a static or restricted environment).

Limitations:
- This is designed for pure-Python packages (no compiled extensions) like `nptdms` or `plotly`.
- Compiled packages (e.g., numpy, scipy, pandas) should still be installed via pip or provided
  as wheels for the target architecture; the script won't extract compiled `.so` files correctly
  for cross-platform usage.

Usage:
  python scripts/vendorize.py nptdms plotly
  python scripts/vendorize.py --packages nptdms plotly

This will populate the `vendor/` directory with the actual package directories.
"""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
VENDOR_DIR = os.path.join(REPO_ROOT, 'vendor')


def _download_wheels(packages: list[str], dest_dir: str) -> None:
    args = [sys.executable, '-m', 'pip', 'download', '--no-deps', '--only-binary=:all:', '-d', dest_dir]
    args.extend(packages)
    print('Running:', ' '.join(args))
    subprocess.check_call(args)


def _extract_wheel(wheel_path: str, dest_dir: str) -> None:
    # A wheel is a zip file; extract only Python packages to dest_dir top-level
    with zipfile.ZipFile(wheel_path, 'r') as zf:
        for member in zf.namelist():
            # We only want top-level package directories and module files
            if member.startswith('__pycache__'):
                continue
            if '.dist-info/' in member or '.data/' in member:
                continue
            parts = member.split('/')
            if len(parts) == 1 and parts[0].endswith('.py'):
                # module file
                zf.extract(member, dest_dir)
            elif len(parts) >= 2:
                # likely a package/directory
                zf.extract(member, dest_dir)


def vendorize(packages: list[str]) -> None:
    os.makedirs(VENDOR_DIR, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        _download_wheels(packages, td)
        for fname in os.listdir(td):
            wheel_path = os.path.join(td, fname)
            print('Extracting', wheel_path)
            _extract_wheel(wheel_path, VENDOR_DIR)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('packages', nargs='*', help='Packages to vendorize')
    parser.add_argument('--packages', '-p', nargs='*', help='Packages to vendorize')
    args = parser.parse_args()
    pkgs = []
    if args.packages:
        pkgs.extend(args.packages)
    if args.packages is None and args.packages is None:
        # nothing specified
        parser.error('No packages provided')
    # merge and dedupe
    pkgs = list(dict.fromkeys(pkgs))
    if not pkgs:
        parser.error('No packages provided')
    print('Vendorizing:', pkgs)
    vendorize(pkgs)
    print('Done. Please add and commit the `vendor/` directory if desired.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
