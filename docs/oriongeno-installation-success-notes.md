# OrionGeno Installation Success Notes

This document keeps only the validated success path. It avoids hostnames, IP
addresses, private user paths, and trial-and-error notes.

The commands below assume you are already inside the OrionGeno `predicting`
repository directory.

## Copy-Paste Setup

Create workspace-local directories for the environment, package caches, temporary
files, Torch extension builds, Hugging Face cache, and downloaded wheels:

```bash
mkdir -p \
  .conda-envs \
  .conda-pkgs \
  .cache/oriongeno/pip \
  .cache/oriongeno/xdg \
  .cache/oriongeno/torch_extensions \
  .cache/oriongeno/huggingface \
  .tmp/oriongeno \
  wheels
```

Initialize Conda and keep all caches inside this repository directory:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"

export CONDA_PKGS_DIRS="$PWD/.conda-pkgs"
export PIP_CACHE_DIR="$PWD/.cache/oriongeno/pip"
export XDG_CACHE_HOME="$PWD/.cache/oriongeno/xdg"
export TMPDIR="$PWD/.tmp/oriongeno"
export TORCH_EXTENSIONS_DIR="$PWD/.cache/oriongeno/torch_extensions"
export HF_HOME="$PWD/.cache/oriongeno/huggingface"
```

If the `oriongeno` environment already exists, activate it:

```bash
conda activate ./.conda-envs/oriongeno
```

If it does not exist yet, create a minimal Python 3.10 environment first:

```bash
conda create -y -p ./.conda-envs/oriongeno python=3.10.18 pip biopython=1.85
conda activate ./.conda-envs/oriongeno
```

Check that you are using the correct Python:

```bash
which python
python --version
```

Expected output:

- `which python` should point somewhere under `./.conda-envs/oriongeno`.
- `python --version` should be `Python 3.10.18`.

## Install Python Dependencies

Install the regular Python dependencies first. `requirements.min.txt` does not
include `causal-conv1d` or `mamba-ssm`; those native extension packages are
installed from matching wheels in the next step.

```bash
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.min.txt
```

Seeing many `nvidia-*` packages during the `torch` installation is normal for
the CUDA-enabled PyTorch wheel.

## Install the Native Extension Wheels

On machines without `nvcc`, use matching official release wheels for
`causal-conv1d` and `mamba-ssm` instead of source builds.

Download the validated wheels from the official GitHub release assets:

- `causal-conv1d` 1.5.2: [causal_conv1d-1.5.2+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl](https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.2/causal_conv1d-1.5.2%2Bcu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl)
- `mamba-ssm` 2.2.5: [mamba_ssm-2.2.5+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl](https://github.com/state-spaces/mamba/releases/download/v2.2.5/mamba_ssm-2.2.5%2Bcu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl)

The native requirements file installs these validated wheel files directly from
their official release URLs:

- `causal_conv1d-1.5.2+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl`
- `mamba_ssm-2.2.5+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl`

Install them with:

```bash
python -m pip install --no-cache-dir --no-deps -r requirements.native-cu126.txt
```

## Verify the Installation

Run these checks from the same activated environment:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
python -c "import transformers, mamba_ssm, causal_conv1d, fastapi; print('imports_ok')"
```

Expected result:

- `torch = 2.7.0+cu126`
- `torch.cuda.is_available() = True`
- `torch._C._GLIBCXX_USE_CXX11_ABI = True`
- `imports_ok`

## If Something Goes Wrong

- If `which python` points to `base`, activate `./.conda-envs/oriongeno` again.
- If `torch.cuda.is_available()` is `False`, check the GPU driver and CUDA
  runtime compatibility.
- If `causal-conv1d` or `mamba-ssm` tries to compile from source, stop and use
  the matching wheel files instead.
- If source builds are unavoidable, retry with `--no-build-isolation --no-deps`,
  but the validated success path uses wheels.
