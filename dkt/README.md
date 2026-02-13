````md
## GPU Training Environment (MindSpore + Docker + JupyterLab) — Ubuntu 22.04 (RTX 2060)

This project is designed to run **MindSpore on GPU inside Docker**.  
All training and notebooks should be executed **inside the container**, not on the host Python environment.

---

### 1) Verify NVIDIA GPU Driver (host)
Check that your NVIDIA driver is installed and the GPU is visible:

```bash
nvidia-smi
````

You should see your GPU (e.g. **GeForce RTX 2060**) and total VRAM (e.g. **6144 MiB**).

---

### 2) Install Docker (host)

If Docker is not installed:

```bash
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io
```

Verify:

```bash
docker --version
```

---

### 3) Install NVIDIA Container Toolkit (host)

This enables Docker containers to access the GPU.

```bash
sudo apt-get update
sudo apt-get install -y curl gnupg

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
 | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
 | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
 | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

Configure Docker runtime and restart Docker:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

### 4) Confirm GPU works inside Docker (host)

Run a CUDA container test:

```bash
docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi
```

If it prints your GPU information, Docker ↔ GPU is working.

---

## Running MindSpore GPU Container

This repo is mounted into the container at `/workspace`.

### 5) Start MindSpore GPU container (interactive shell)

From the repo root:

```bash
docker run --rm -it --gpus all \
  -v "$PWD":/workspace \
  -w /workspace \
  mindspore/mindspore-gpu-cuda11.1:1.8.0 bash
```

Inside the container, verify MindSpore sees GPU:

```bash
python - << 'PY'
import mindspore as ms
ms.set_context(device_target="GPU")
print("Device target:", ms.get_context("device_target"))
PY
```

Expected:

```text
Device target: GPU
```

---

## Running Training Code (inside container)

> IMPORTANT: Run training inside Docker. Host Python may not have dependencies.

Example:

```bash
cd /workspace/dkt
python train.py
```

If your script needs a CSV:

```bash
python train.py --data_path /workspace/dkt/assistments_2009_2010.csv --epochs 10
```

### One-command run (no interactive shell)

This still runs **inside the container**:

```bash
docker run --rm --gpus all \
  -v "$PWD":/workspace \
  -w /workspace/dkt \
  mindspore/mindspore-gpu-cuda11.1:1.8.0 \
  python train.py --data_path /workspace/dkt/assistments_2009_2010.csv --epochs 10
```

---

## Installing Python Dependencies (inside container)

If you need extra libs (numpy/pandas/etc.), install them **inside the container**:

```bash
pip install --upgrade pip
pip install numpy pandas scikit-learn tqdm
```

Best practice: keep a `requirements.txt` and install:

```bash
pip install -r requirements.txt
```

> Note: if you use `--rm`, the container is deleted when you exit, so you may need to reinstall deps unless you build a custom image.

---

## JupyterLab (Notebook) on GPU inside Docker

### 1) Start container with Jupyter port exposed

From repo root:

```bash
docker run --rm -it --gpus all \
  -p 8888:8888 \
  -v "$PWD":/workspace \
  -w /workspace \
  mindspore/mindspore-gpu-cuda11.1:1.8.0 bash
```

### 2) Install and run JupyterLab (inside container)

```bash
pip install --upgrade pip
pip install -U jupyterlab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Open in your browser:

* `http://localhost:8888` (use the token URL printed in terminal)

### 3) Use correct paths in notebooks

Inside the container:

* Repo root is `/workspace`
* DKT folder is `/workspace/dkt`

If notebook and data are both in `dkt/`, in the notebook:

```python
import os
os.chdir("/workspace/dkt")

from mindspore import context
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
```

Then load data with a relative path:

```python
import pandas as pd
df = pd.read_csv("assistments_2009_2010.csv")
```

---

## Stopping / Exiting

* Stop a running script: `Ctrl + C`
* Exit container: `exit` or `Ctrl + D`
* If detached containers are used: `docker ps` then `docker stop <container_id>`

---

## Common Issues

### `E: Unable to locate package nvidia-container-toolkit`

Cause: NVIDIA repo not added.
Fix: run the repo + keyring setup in “Install NVIDIA Container Toolkit”.

### `ModuleNotFoundError: No module named 'numpy'`

Cause: you ran Python on the host, or deps not installed inside container.
Fix: run training inside Docker and install deps inside the container.

### `pip install jupterlab` fails

Cause: typo.
Fix: package name is `jupyterlab`.

```
::contentReference[oaicite:0]{index=0}
```
