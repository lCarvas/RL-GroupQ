# Reinforcement Learning Masters Project

## Authors

- Diogo Carvalho - 20221935
- Ricardo Pereira - 20250343
- Yehor Malakhov - 20221691

## Repository Structure

- src/config_a.ipynb: Notebook running config a
- src/config_b.ipynb: Notebook running config b
- src/plots.ipynb: Notebook containing the code for generating the plots used in
  the report
- src/config_a: Folder containing functions related to the algorithms used in
  config a
- src/config_b: Folder containing functions related to the algorithms used in
  config b
- src/utils: Folder containing functions used in more than one algorithm in
  either config a or b

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

Choose depending on your installed CUDA version.

#### Using pip

```bash
pip install -r "requirements-nvidia-cu12.txt"
```

```bash
pip install -r "requirements-nvidia-cu13.txt"
```

#### Using uv

```bash
uv sync --extra nvidia-cu12
```

```bash
uv sync --extra nvidia-cu13
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

### CPU

#### Using pip

```bash
pip install -r "requirements-cpu.txt"
```

#### Using uv

```bash
uv sync --extra cpu
```
