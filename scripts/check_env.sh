#!/usr/bin/env bash
set -euo pipefail

# Create Python virtualenv (if not already created)
if [ ! -d "venv" ]; then
  python -m venv venv
fi

# Activate venv and install requirements
# ps: you can run the next lines manually if you prefer to stay in your shell
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Run the Python import check
python devtools/check_env.py "$@"
