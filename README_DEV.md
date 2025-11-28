Development checks
===================

This file contains instructions to verify and reproduce the environment locally.

1) Create a venv and install dependencies
```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

2) Run the Python import test
```bash
python test.py  # or python devtools/check_env.py
```

3) Run the app
```bash
streamlit run ICR_analysis_tool.py
```

4) If SciPy fails to install, review the pip log and install system packages. See `README.md`.

