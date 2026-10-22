# GPU setup (stage 3 embeddings)

Stage 3 (`03_embed.py`) is the only GPU stage. It auto-detects CUDA and falls
back to CPU; CPU works but is far slower, so you want the NVIDIA GPU live first.

## Diagnosis (as found 2026-10-22, Ubuntu 24.04.4, kernel 7.0.0-38-generic)

`nvidia-smi` failed with *"couldn't communicate with the NVIDIA driver"*. Root causes:

1. **Driver version mismatch** — userspace libs at `615.71.09`, but the kernel
   module package `nvidia-dkms` at `610.57.04`. They must match.
2. **Kernel module never built** — `dkms status` is empty, so no `nvidia.ko`
   exists for the running kernel. Nothing to load → `nvidia-smi` can't connect.
3. **Secure Boot is enabled** — even once the module builds, it must be signed
   with an enrolled MOK key or Secure Boot refuses to load it.

The GPU itself is present: `lspci` reports
`NVIDIA AD107M [GeForce RTX 4060 Max-Q / Mobile]` on `01:00.0`. The display runs
on the Intel UHD iGPU (`00:02.0`), so reinstalling the NVIDIA driver will **not**
risk your desktop.

> **VRAM: 8 GB** (confirmed by the user — RTX 4060 Laptop). The repo
> (`README.md`, `requirements.txt`, `03_embed.py`) has been corrected from the
> earlier "4050 / 6 GB" assumption. Re-confirm with `nvidia-smi` (step 4) once
> the driver loads.

## Fix (run these yourself — they need sudo + a reboot)

In this Claude Code session you can prefix a command with `!` to run it here.

### 1. Install one consistent driver (matches the existing 615 userspace)

```bash
sudo apt update
sudo apt install --reinstall nvidia-driver-615-open
```

Or let Ubuntu pick the recommended one automatically:

```bash
sudo ubuntu-drivers install
```

This pulls the matching DKMS package and builds the module for your kernel.

### 2. Handle Secure Boot (MOK enrollment)

During the install, DKMS prompts you to set a one-time **MOK password**. Then:

```bash
sudo reboot
```

At reboot you get a blue **MOK Manager** screen → **Enroll MOK** → **Continue** →
enter that password. Skipping this = the module silently won't load.

*Alternative:* disable Secure Boot in the BIOS/UEFI instead of enrolling a key.

### 3. (If the module still won't load after reboot) rebuild DKMS

```bash
sudo dkms autoinstall
sudo modprobe nvidia
```

### 4. Verify the driver + read the real VRAM

```bash
nvidia-smi
```

You should see the GPU, driver version, and total memory. **Confirm it reads
~8 GB** (expected for the 4060 Laptop).

### 5. Install the CUDA build of PyTorch

The default `pip install torch` may give a CPU-only wheel. Force CUDA 12.x
(works on Ada / sm_89):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expect `True <your GPU name>`.

## Running stage 3 on the GPU

```bash
python 03_embed.py           # auto-detects CUDA, fp16
```

- `EMB_BATCH` in `config.py` defaults to `128`. With 8 GB you can raise it
  (e.g. 192–256) for more throughput — watch VRAM in `nvidia-smi` while it runs.
- Force CPU for debugging: `python 03_embed.py --device cpu` (slow).

### Estimate the time before committing to the full run

1.2M sentences on this class of GPU is roughly **8–20 min** of inference (plus a
one-time ~1.1 GB model download), but mobile thermal/power limits vary — so
**measure instead of guessing**:

```bash
python 02_clean_split.py --target-sentences 50000 --force   # ~50k-sentence slice
python 03_embed.py --force                                  # read "sent/s" from the log
```

Multiply the logged `sent/s` into 1.2M (≈ ×24 vs the 50k slice) for a real ETA.
