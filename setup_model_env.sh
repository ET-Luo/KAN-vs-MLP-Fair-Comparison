#!/bin/sh
# Setup Conda environment and install dependencies for CUDA 12.4

set -eu

printf "[INFO] Sourcing conda.sh...\n"
# shellcheck disable=SC1091
. "$HOME/miniconda3/etc/profile.d/conda.sh"

printf "[INFO] Creating conda environment 'model' (Python 3.12)...\n"
conda create -y -n model python=3.12

printf "[INFO] Activating conda environment 'model'...\n"
conda activate model

printf "[INFO] Upgrading pip...\n"
python -m pip install --upgrade pip

printf "[INFO] Installing PyTorch (CUDA 12.4)...\n"
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

printf "[INFO] Installing Python dependencies...\n"
python -m pip install \
  pykan scikit-learn \
  pandas numpy pyyaml \
  matplotlib seaborn tqdm \
  gensim wandb

printf "[INFO] Verifying CUDA and PyTorch...\n"
python - <<'PY'
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("GPU name:", "N/A")
PY

printf "[INFO] Done.\n"
