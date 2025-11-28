# ICRcalc

This repository contains a Streamlit app to load TDMS files and calculate contact resistance (ICR) vs pressure.

Key dependencies are listed in `requirements.txt`. If you get import errors for packages such as `scipy` or `nptdms`, make sure `requirements.txt` includes them (it should already), and install them locally or re-deploy to Streamlit.

To run locally (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run ICR_analysis_tool.py
```

If deploying on Streamlit Community Cloud or other containerized service, make sure `requirements.txt` is present at the repo root (Streamlit uses it during environment setup).

If you still see missing imports, double-check that the package names are spelled correctly in `requirements.txt` (e.g., `nptdms`, not `nptmds`) and that there are no private/private dependency constraints preventing installation.

If you need help, open an issue in the repo with the error message and environment details (Python version, where you're running the app, and relevant logs).

Troubleshooting common failing packages
--------------------------------------

SciPy installation fails
- SciPy requires binary wheels for the target platform, otherwise pip will attempt to build SciPy from source and may fail if system libraries and compilers are missing.
- If you're using Streamlit Community Cloud, it uses Ubuntu images and should install SciPy via wheels — but if you've constrained Python to an odd version or used a custom Docker image (e.g., Alpine), wheel files may not be available.
- Fixes:
	- Use a compatible Python version (3.10 or 3.11 are recommended).
	- Pin a SciPy version that has prebuilt wheels for the target Python version (e.g., `scipy>=1.11.0`).
	- If using a self-hosted container/VM that is missing build tools, install system packages first (Ubuntu example):

```bash
sudo apt-get update && sudo apt-get install -y build-essential gfortran libopenblas-dev liblapack-dev
python -m venv venv
source venv/bin/activate
pip install -U pip wheel setuptools
pip install -r requirements.txt
```

Plotly installation fails
- Plotly is pure-Python and usually installs cleanly; problems usually originate from pip cache/network issues or old pip.
- Fixes:
	- Upgrade pip: `pip install --upgrade pip`
	- Re-run `pip install -r requirements.txt`

`nptdms` installation fails
- `nptdms` is a pure-Python package and should install via pip (no extra system dependencies). If it fails:
	- Check the pip log for underlying errors (e.g., network/DNS or permission issues).
	- Verify the package spelled correctly: `nptdms`.

Checking deploy logs
- Streamlit Cloud/other hosts provide logs that show why pip install failed; look for errors referencing `gcc`, `gfortran`, or missing `libopenblas` or `lapack` if SciPy compile occurs.

Fallback behavior
- If SciPy can't be installed in your environment, the app will try to use a fallback linear interpolation (`numpy.interp`) when `scipy.interpolate.interp1d` is not available. This is less flexible than SciPy's interface but avoids a hard crash.

If you're still stuck, open an issue with the exact pip/OS logs and I'll help debug further.

Docker & wheelhouse options
---------------------------

If you cannot rely on the host environment to install compiled packages (like SciPy), you have a few options:

1) Use the provided Dockerfile (recommended for reproducible behavior):

```bash
# Build (from repo root):
docker build -t icrcalc:latest .

# Run interactively on port 8501:
docker run --rm -p 8501:8501 icrcalc:latest
```

2) Build a wheelhouse for offline installs or pinning: this script downloads wheels into `wheelhouse/` and allows installing directly from there:

```bash
./scripts/build_wheelhouse.sh
# Then install from wheelhouse:
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install --no-index --find-links wheelhouse -r requirements.txt
```

Note: If you commit `wheelhouse/` to your repo it will make the repo large (not recommended), but CI artifacts are a better approach.

Vendorizing packages (optional)
------------------------------

If you want to embed small pure-Python packages directly into the repo so they are always available at runtime (useful for restricted hosts), you can use the `scripts/vendorize.py` script to extract them into the `vendor/` folder. This method works for pure-python modules like `nptdms` and `plotly`, but not for compiled packages like `numpy`/`scipy`.

```bash
# Example: extract nptdms and plotly packages to vendor/ and include them in your repo
python scripts/vendorize.py nptdms plotly
git add vendor/ && git commit -m "Vendorize nptdms and plotly"
```

After vendorizing, `ICR_analysis_tool.py` will prefer the `vendor/` directory at runtime and import those local copies.

Caveat:
- Vendorizing is not a recommended long-term approach for large libraries or compiled dependencies — prefer wheelhouse or Docker for deterministic and reproducible environment setup.

Deployment with Docker (recommended)
-----------------------------------

We added a Docker image and a GitHub Actions workflow to build and publish the image to GHCR (GitHub Container Registry) and optionally Docker Hub.

Action summary:
- `.github/workflows/build-and-push-docker.yml` will build the Docker image, test imports inside the image, and push artifacts.
- It pushes to `ghcr.io/${{ github.repository_owner }}/icrcalc:latest` and `ghcr.io/${{ github.repository_owner }}/icrcalc:${{ github.sha }}`.
- If you want the Action to also push to Docker Hub, set these repository secrets in your GitHub repo:
	- `DOCKERHUB_USERNAME`
	- `DOCKERHUB_TOKEN` (or password)

How to use the image:
```bash
# Build the image locally (or pull from GHCR if published)
docker build -t icrcalc:latest .

# Run the container
docker run --rm -p 8501:8501 icrcalc:latest
```

If your deployment target cannot run Docker images, you should still use a wheelhouse or the `requirements.txt` approach with a compatible Python runtime to avoid SciPy compilation issues.
