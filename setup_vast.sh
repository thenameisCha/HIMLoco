# !/usr/bin/env bash
set -euo pipefail


# --- Find conda and load its shell hook (works even if .bashrc wasn't modified) ---
source /opt/miniforge3/etc/profile.d/conda.sh
# ---- 1) Install Python 3.8 via deadsnakes PPA (system Python) ----
sudo apt-get update -y
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -y
sudo apt-get install -y python3.8 python3.8-distutils python3.8-venv

# ---- 2) Create conda env with Python 3.8 ----
if conda env list | awk '{print $1}' | grep -qx "isaac_env"; then
  echo "Conda env 'isaac_env' already exists. Skipping creation."
else
  conda create -n isaac_env python=3.8 -y
fi

# ---- 3) Create ~/.bash_aliases with requested alias ----
# (exact string you provided)
mkdir -p "$HOME"
echo "alias iv='conda activate isaac_env && cd workspace/HIMLoco'" > "$HOME/.bash_aliases"
echo "alias train='conda activate isaac_env && cd workspace/HIMLoco/legged_gym/legged_gym/scripts && python train.py'" >> "$HOME/.bash_aliases"
conda activate isaac_env
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# ---- 4) Install PyTorch CUDA 11.3 wheels + other Python deps into the conda env ----
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 tensorboard==2.8.0 pybullet==3.2.1 opencv-python==4.5.5.64 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# ---- 5) Install Isaac Gym Python package in editable mode ----
# expects a folder named 'isaacgym' in the current working directory
# mv workspace/HIMLoco/IsaacGym_Preview_4_Package.tar.gz workspace/ 
cd /workspace && tar -xzf /workspace/IsaacGym_Preview_4_Package.tar.gz
cd /workspace/isaacgym/python && pip install -e .
cd /workspace/HIMLoco/rsl_rl && pip install -e .
cd /workspace/HIMLoco/legged_gym && pip install -e .

# ---- 6) Install wandb ----
pip install wandb

# ---- 7) Version updates ----
python -m pip install -U "setuptools==59.5.0" "wheel<0.41"
pip install lxml
pip install joblib
pip install --extra-index-url https://download.pytorch.org/whl/cu118 \
  torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118
pip install "numpy==1.23.5"
python -m pip install --upgrade "protobuf==3.19.6"
echo "All done."
echo "DONT FORGET TO CHANGE DIRECTORIES AND ON_POLICY_RUNNER WANDB.RUN_NAME!!!"
echo "WHEN USING GIT IN VAST, ex) env -u LD_LIBRARY_PATH git pull origin main"
echo "Run export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-} when lippython is not found"
# WHEN USING GIT IN VAST
# env -u LD_LIBRARY_PATH git pull origin main