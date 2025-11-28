#!/usr/bin/env bash
set -euo pipefail

# Creates a wheelhouse directory and downloads wheels for the current requirements.txt
# Usage: ./scripts/build_wheelhouse.sh [--only-pure] [--python-version 3.11]

WHEELHOUSE_DIR="wheelhouse"
PYTHON_VERSION="3.11"
ONLY_PURE=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --python-version)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --only-pure)
      ONLY_PURE=1
      shift
      ;;
    *)
      echo "Unknown arg $1"
      exit 1
      ;;
  esac
done

mkdir -p "$WHEELHOUSE_DIR"
python -m pip download -r requirements.txt -d "$WHEELHOUSE_DIR"

if [[ "$ONLY_PURE" -eq 1 ]]; then
  echo "Filtering wheelhouse to pure-Python packages only"
  # We can attempt to remove compiled wheels, though this is best effort.
  find "$WHEELHOUSE_DIR" -type f ! -name "*none-any.whl" -delete
fi

echo "Wheels downloaded into $WHEELHOUSE_DIR. To install from it, use:\n
pip install --no-index --find-links $WHEELHOUSE_DIR -r requirements.txt\n"
