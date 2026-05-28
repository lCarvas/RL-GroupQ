# Reinforcement Learning Masters Project

## Authors

- Diogo Carvalho - 20221935
- Ricardo Pereira - 20250343
- Yehor Malakhov - 20221691

## Installation

If you're using pip, create a virtual environment with Python 3.14 before
installing the packages to avoid conflicts with other versions of packages. If
you're using [uv](https://docs.astral.sh/uv/), this is automatically handled for
you.

The code has not been tested on other Python versions, the versions of the
packages in the `requirements*.txt`/`pyproject.toml` files might only be
compatible with Python 3.14. Running the code with other package versions or
Python versions could lead to unexpected errors.

### NVIDIA GPUs

#### Using pip

```bash
pip install -r "requirements-nvidia.txt"
```

#### Using uv

```bash
uv sync --extra nvidia
```

### AMD GPUs

Running on AMD GPUs is only supported on Linux. You will need to have ROCm
installed on your system. You can find the installation instructions for ROCm on
the official AMD website:
<https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html>

If you're running on Arch Linux, you can install ROCm by following these steps:

1. Update your system

   ```bash
   sudo pacman -Syu
   ```

2. Install ROCm

   ```bash
   sudo pacman -S rocm-hip-sdk rocm-opencl-sdk
   ```

3. Add current user to render and video groups

   ```bash
   sudo usermod -aG render,video $USER
   ```

#### Using pip

```bash
pip install -r "requirements-amd.txt"
```

#### Using uv

```bash
uv sync --extra amd
```
