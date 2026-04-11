# rdna35-toolbox — AMD Radeon 860M ML Distrobox

This repository provides a containerized **ML development environment** for **AMD Radeon 860M** (gfx1152 / RDNA3.5) based on **Ubuntu 24.04** with **ROCm nightly SDK**. It includes:

- **llama.cpp** compiled with HIP backend (gfx1151 + gfx1152 native support)
- **PyTorch** from TheRock nightlies with ROCm support
- **vLLM** inference engine for fast LLM serving
- **Unsloth** for efficient fine-tuning
- **bitsandbytes** (ROCm-enabled) for quantized operations
- **Jupyter, transformers, datasets**, and common ML/dev tools
- Optimized for **Unified Memory Architecture (UMA)** APUs with full system RAM exposure to GPU kernels
- Multi-stage Docker build with pre-compiled binaries (~6-7GB final image)

---

## Repository Structure

```
.
├── .github/
│   └── workflows/
│       └── build.yml          # GitHub Actions workflow
├── .dockerignore              # Files to exclude from build context
├── Dockerfile                 # Container build instructions (multi-stage)
└── README.md                  # This file
```

**Note:** llama.cpp, vLLM, and bitsandbytes are cloned fresh during the Docker build (not from git submodules). The `LLAMACPP_REF`, `VLLM_REF`, and other build args control which branches/tags are fetched.

---

## Prerequisites

- **Podman** or **Docker** with BuildKit support.
- **Git** (to clone this repository).
- AMD GPU with ROCm‑compatible driver installed on the host.
- (Optional) Access to a container registry if you wish to push images.

---

## Configuration

### Build Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `THEROCK_RELEASE_ID` | `20260405-latest` | TheRock (ROCm nightly) release ID from [rocm.nightlies.amd.com/deb/](https://rocm.nightlies.amd.com/deb/). Format: `YYYYMMDD-COMMITSHA` |
| `ROCM_GFX_ARCH` | `gfx1151` | GPU arch for system packages. Set to `gfx1152` once [TheRock#2310](https://github.com/ROCm/TheRock/issues/2310) is resolved |
| `HIP_GPU_TARGETS` | `gfx1151;gfx1152` | Semicolon-separated GPU architectures to compile for (llama.cpp, vLLM, bitsandbytes) |
| `PYTHON_VERSION` | `3.12` | Python version to install (3.13 not supported by Unsloth) |
| `VLLM_REF` | `main` | vLLM git branch/tag to build |
| `LLAMACPP_REF` | `b8755` | llama.cpp stable release tag. Use stable tags for better ROCm compatibility ([releases](https://github.com/ggml-org/llama.cpp/releases)) |
| `AMDGPU_TOP_VERSION` | `0.11.3` | amdgpu_top release version for GPU monitoring |

### Build Environment

The image is optimized for **Radeon 860M (gfx1152 / RDNA3.5)** as a UMA APU:

- ✅ **llama.cpp** compiled with stable release tag (b8755) for better ROCm compatibility; native gfx1151+gfx1152 support, rocWMMA FlashAttention
- ✅ **vLLM** with gfx1151 nightly wheels (gfx1152 fallback via `HSA_OVERRIDE_GFX_VERSION=11.5.1`)
- ✅ **PyTorch, Unsloth, bitsandbytes** from TheRock gfx1151 nightlies
- ✅ **ML tools**: Jupyter, transformers, datasets, accelerate, peft, trl, huggingface_hub
- ✅ **GPU monitoring**: amdgpu_top, rocm-smi, nvtop
- ✅ **Distrobox/Toolbx compatible**: bash, sudo, build-essential, git, vim, etc.

**Note:** gfx1152 native packages are not yet published by TheRock. The image uses gfx1151 packages which work on gfx1152 hardware. If issues occur, set `HSA_OVERRIDE_GFX_VERSION=11.5.1` at runtime (see [Troubleshooting](#troubleshooting)).

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/fine2006/llama-rocm-build-docker.git
cd llama-rocm-build-docker
```

### 2. Build the Container Image

```bash
podman build \
  --build-arg THEROCK_RELEASE_ID=20260405-latest \
  --build-arg ROCM_GFX_ARCH=gfx1151 \
  --build-arg HIP_GPU_TARGETS="gfx1151;gfx1152" \
  -t rdna35-toolbox:latest .
```

Or for **Docker** with BuildKit enabled:

```bash
docker build \
  --build-arg THEROCK_RELEASE_ID=20260405-latest \
  --build-arg ROCM_GFX_ARCH=gfx1151 \
  -t rdna35-toolbox:latest .
```

### 3. Enter the Container (Distrobox/Toolbx)

**Distrobox:**
```bash
distrobox create --image localhost/rdna35-toolbox:latest --name rdna35-dev
distrobox enter rdna35-dev
```

**Toolbx:**
```bash
toolbox create --image localhost/rdna35-toolbox:latest rdna35-dev
toolbox run -c rdna35-dev bash -l
```

Or just run it directly:
```bash
podman run -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --ipc=host \
  -v /path/to/models:/models:Z \
  localhost/rdna35-toolbox:latest
```

---

## Customizing for Your GPU

### 1. Find TheRock Nightly Release ID

Visit [rocm.nightlies.amd.com/deb/](https://rocm.nightlies.amd.com/deb/) and look for directories named like `YYYYMMDD-COMMITSHA` (e.g., `20260405-1a2b3c4d`). Use the latest or a specific date.

### 2. Determine Your GPU Architecture

```bash
# On a host with ROCm installed:
rocminfo | grep gfx | head -1

# AMD GPU compatibility table:
# https://rocm.docs.amd.com/projects/rocm-compute-profiles/en/latest/gpu.html
```

Common architectures:
- **Radeon RX 7900 series** → `gfx1100`
- **Radeon RX 7600** → `gfx1102`
- **Radeon RX 780M (iGPU)** → `gfx1103`
- **Radeon RX 8600M** → `gfx1152`
- **Radeon 860M (APU)** → `gfx1152`
- **MI300 series** → `gfx942`

### 3. Rebuild with Custom Arguments

```bash
podman build \
  --build-arg THEROCK_RELEASE_ID=20260405-1a2b3c4d \
  --build-arg ROCM_GFX_ARCH=gfx1152 \
  --build-arg HIP_GPU_TARGETS="gfx1151;gfx1152" \
  --build-arg PYTHON_VERSION=3.12 \
  --build-arg VLLM_REF=main \
  --build-arg LLAMACPP_REF=b8755 \
  -t rdna35-toolbox:latest .
```

⏱️ **Build time:** ~45–90 minutes depending on CPU cores and network speed. Use `--build-arg MAX_JOBS=2` if you hit memory limits during vLLM compilation.

---

## Available Commands & Tools

The image provides a complete ML development environment:

### Inference & LLM Serving

| Binary | Description |
|--------|-------------|
| `llama-server` | HTTP server for running GGUF models |
| `llama-cli` | Command-line interface for inference |
| `python -m vllm.entrypoints.openai.api_server` | vLLM OpenAI-compatible API |

### Model Tools

| Tool | Description |
|------|-------------|
| `llama-quantize` | Quantize GGUF models to lower precisions |
| `llama-bench` | Benchmark llama.cpp performance |
| `llama-perplexity` | Calculate model perplexity on text |

### Python ML Stack

- **PyTorch** — Deep learning framework
- **Transformers** — Hugging Face model hub + tools
- **Unsloth** — Fast fine-tuning via gradient checkpointing
- **vLLM** — Fast LLM inference engine
- **bitsandbytes** — Quantization & parameter-efficient training
- **Accelerate** — Distributed training utilities
- **PEFT** — Parameter-Efficient Fine-Tuning (LoRA, QLoRA, etc.)
- **TRL** — Transformer Reinforcement Learning
- **Jupyter Lab** — Interactive notebooks

### System Tools

| Tool | Description |
|------|-------------|
| `amdgpu_top` | Real-time AMD GPU utilisation & power |
| `rocm-smi` | AMD GPU device info & clock control |
| `nvtop` | GPU monitoring (ROCm-enabled) |
| `htop` | CPU & memory monitoring |

### Usage Examples

```bash
# Inside the container:

# Run llama-server for HTTP inference
llama-server -m /models/model.gguf --port 8080

# Run vLLM inference server
python -m vllm.entrypoints.openai.api_server \
  --model /models/model.gguf \
  --port 8000

# Quantize a model
llama-quantize /models/model-f16.gguf /models/model-q4_0.gguf q4_0

# Benchmark performance
llama-bench -m /models/model.gguf -n 512

# Interactive Python with GPU access
python3
>>> import torch
>>> torch.cuda.is_available()  # Should return True
>>> import vllm
>>> import unsloth
```

---

## Environment Variables

Inside the container, the following ROCm + GPU memory management variables are pre-configured:

| Variable | Value | Description |
|----------|-------|-------------|
| `GPU_MAX_HEAP_SIZE` | `100` | Expose 100% of unified memory heap to ROCm |
| `GPU_MAX_ALLOC_PERCENT` | `100` | Allow single allocation up to 100% of heap |
| `GPU_SINGLE_ALLOC_PERCENT` | `100` | Remove cap on contiguous allocations |
| `HSA_ENABLE_SDMA` | `0` | Disable SDMA DMA engine (causes corruption on UMA APUs) |
| `GGML_HIP_UMA` | `1` | Instruct llama.cpp to use unified memory path |
| `GGML_CUDA_ENABLE_UNIFIED_MEMORY` | `1` | Allow llama.cpp to spill into system RAM |
| `HIP_VISIBLE_DEVICES` | `0` | Select first GPU (override for multi-GPU) |

### Fallback: gfx1152 Hardware on gfx1151 Packages

If PyTorch, vLLM, or other tools fail to detect your GPU natively, set:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.5.1
```

This makes gfx1152 hardware masquerade as gfx1151, allowing gfx1151 kernel bitcode to run on your hardware. Expect <5% performance variance. Remove this override once TheRock publishes gfx1152 packages (track [TheRock#2310](https://github.com/ROCm/TheRock/issues/2310)).

---

## Troubleshooting

### Issue: "Cannot find kernel for gfx1152"

**Solution:** Use the gfx1151 fallback:
```bash
export HSA_OVERRIDE_GFX_VERSION=11.5.1
```

Then re-run your workload (PyTorch, vLLM, llama.cpp, etc.).

### Issue: "HIP device not found" or GPU not detected

**Possible causes:**
1. **AMD GPU driver not installed** on the host. Install ROCm stack or amdgpu driver.
2. **Device not exposed to container.** Ensure you're passing `--device=/dev/kfd --device=/dev/dri` when running.
3. **User permissions.** Add your user to the `render` and `video` groups:
   ```bash
   sudo usermod -a -G render,video $USER
   newgrp render
   ```

**Verify:** Inside the container, run:
```bash
rocm-smi          # Should list your GPU
amdgpu_top        # Should show GPU info
python3 -c "import torch; print(torch.cuda.is_available())"  # Should be True
```

### Issue: Out-of-memory errors during inference

The image is optimized for unified memory (UMA) on Radeon 860M. If you still hit OOM:

1. **Verify GPU_MAX_* variables are set:**
   ```bash
   echo $GPU_MAX_HEAP_SIZE $GPU_MAX_ALLOC_PERCENT
   ```

2. **Reduce model precision.** Use `q4_0` quantization instead of `f16`:
   ```bash
   llama-quantize /models/model-f16.gguf /models/model-q4_0.gguf q4_0
   ```

3. **Reduce context length:**
   ```bash
   llama-server -m /models/model-q4_0.gguf -n 512 --port 8080
   ```

### Issue: Distrobox/Toolbx integration

**For Distrobox:**
```bash
distrobox create --image localhost/rdna35-toolbox:latest --name rdna35-dev
# Pass GPU devices:
distrobox-init --root -- distrobox create -i localhost/rdna35-toolbox:latest -n rdna35-dev
```

Then inside distrobox, GPU access should work normally.

**For Toolbx:** Similar setup; toolbx automatically forwards `/dev/kfd` and `/dev/dri`.

### Issue: WebUI not accessible on llama-server

**Solution:** Ensure you're exposing the port correctly:
```bash
podman run -p 8080:8080 \
  --device=/dev/kfd --device=/dev/dri \
  --ipc=host \
  localhost/rdna35-toolbox:latest \
  llama-server -m /models/model.gguf --port 8080
```

Then open `http://localhost:8080` in your browser.

---

## Image Size Optimization

Final image size: **~6–7 GB** (unified memory runtime; no discrete VRAM model).

**Optimization strategy:**
1. **Multi-stage Docker build** — Separate builder and runtime stages; discard ~20GB of build artifacts
2. **Ubuntu 24.04 base** — Lean system packages
3. **Compile for HIP backend only** — All other GPU backends disabled
4. **Pre-compiled binaries** — llama.cpp, vLLM, bitsandbytes wheels copied directly (no runtime recompilation)
5. **Aggressive cleanup** — Remove pip cache, apt lists, build artifacts

**What's included:**
- Full Python 3.12 environment
- PyTorch, vLLM, Unsloth, bitsandbytes pre-installed
- llama.cpp with HIP backend
- Jupyter, transformers, datasets, and common ML packages
- System tools: git, vim, build-essential, GPU monitoring tools

---

## GitHub Actions Workflow

The repository includes a GitHub Actions workflow (`.github/workflows/build.yml`) that automatically builds and optionally pushes the container image.

### Workflow Features

- ✅ **Ubuntu 24.04 runner** for ROCm SDK compatibility
- ✅ **Automatic THEROCK_RELEASE_ID detection** — Fetches the latest nightly release from rocm.nightlies.amd.com
- ✅ **Multi-platform support** — linux/amd64 only (ROCm is x86_64-specific)
- ✅ **Docker BuildKit caching** — Speeds up incremental rebuilds
- ✅ **Conditional registry push** — Only pushes on merges to main branch

### Manual Trigger via GitHub UI

1. Go to **Repositories → Actions → Build Container Image**
2. Click **Run workflow**
3. Select branch (default: `main`)
4. Optionally set `THEROCK_RELEASE_ID` or other build args

### Customization

Edit `.github/workflows/build.yml` to:
- Change registry (currently disabled; uncomment to push to Docker Hub, GitHub Container Registry, etc.)
- Modify build arguments (PYTHON_VERSION, VLLM_REF, LLAMACPP_REF, etc.)
- Adjust caching strategy

---

## Using uv for Project Management

The distrobox includes **uv** (fast Python package manager) pre-installed, and comes with **system-wide pre-built wheels** for PyTorch, vLLM, Unsloth, bitsandbytes, and other ML libraries. This allows for efficient project management without rebuilding heavy packages.

### Quick Start with uv

```bash
# Inside distrobox
uv init my-ml-project
cd my-ml-project

# Edit pyproject.toml to add custom dependencies
uv sync  # Installs dependencies, inherits system packages
```

### How It Works

By default, `uv venv` creates virtual environments with `--system-site-packages`, which means:

1. **System packages are inherited:** PyTorch, Unsloth, vLLM, transformers, etc. are available globally
2. **Custom deps go into the venv:** Your project's `uv sync` only installs packages not already available system-wide
3. **No wheel rebuilding:** Heavy packages like PyTorch are already compiled; your projects reuse them
4. **Isolated dependencies:** Custom packages are isolated per project, preventing conflicts

### Example pyproject.toml

```toml
[project]
name = "my-fine-tune"
version = "0.1.0"
description = "Fine-tuning with Unsloth"
dependencies = [
    # Pre-installed system-wide (no reinstall):
    "torch",
    "unsloth",
    "transformers",
    
    # Custom dependencies (installed into venv):
    "wandb",
    "peft",
    "datasets",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

Then inside the project:
```bash
uv sync        # Installs only wandb + peft + datasets
python -c "import torch; import unsloth"  # Works (system package)
```

### Fallback Methods if System Packages Aren't Inherited

If uv doesn't automatically inherit system packages, try these methods:

**1. Explicit `--system-site-packages` flag:**
```bash
uv venv --system-site-packages .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

**2. Manual venv creation with pip:**
```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**3. Check what's available system-wide:**
```bash
python3 -c "import sys; print('\n'.join(sys.path))"  # Shows site-packages paths
python3 -c "import torch; print(torch.__version__)"  # Verify torch is available
```

**4. List installed system packages:**
```bash
pip list  # Shows both system and virtual env packages
```

**5. Pin versions to match system installs:**
```toml
[project]
dependencies = [
    "torch>=2.0.0",      # System torch is already installed
    "unsloth>=2024.12",  # System unsloth is already installed
    "custom-lib==1.2.3", # New package, installed into venv
]
```

### Distrobox + uv Workflow

The distrobox acts as a **pre-configured ML environment**:

- **Global tools:** PyTorch, Jupyter, vLLM, Unsloth are always available
- **Project isolation:** Each `uv` project has isolated dependencies
- **No bloat:** System packages aren't duplicated in every venv

Example session:
```bash
# Start distrobox
distrobox enter rdna35-dev

# Quick experimentation (uses system PyTorch)
python3 -c "import torch; print(torch.cuda.is_available())"

# Create a project
uv init ml-training
cd ml-training
uv sync

# Run training script (uses both system + project packages)
python train.py
```

---

## Security Considerations

### Container Privileges

The container requires elevated privileges to access GPU devices:

```bash
--device=/dev/kfd \
--device=/dev/dri \
--ipc=host \
--cap-add=SYS_PTRACE
```

### Mitigation

- Run container in user namespace if supported
- Use read-only filesystem where possible
- Mount models with appropriate permissions

---

## License

This guide and the accompanying configuration files are provided under the MIT License. `llama.cpp` is distributed under its own license (MIT). ROCm components are subject to AMD's licenses.

---

## Contributing

Contributions are welcome! Please ensure:

1. Follow the existing code style
2. Update documentation for any changes
3. Test builds locally before submitting
4. Include clear commit messages

---

## References

- [llama.cpp Documentation](https://github.com/ggml-org/llama.cpp)
- [ROCm Installation Guide](https://rocm.docs.amd.com/)
- [ROCm Nightly Releases](https://rocm.nightlies.amd.com/deb/)
- [AMD GPU Compatibility](https://rocm.docs.amd.com/projects/rocm-compute-profiles/en/latest/gpu.html)
- [Docker Multi-stage Builds](https://docs.docker.com/develop/develop-images/multistage-build/)

---

## Support

For issues or questions:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review [llama.cpp documentation](https://github.com/ggml-org/llama.cpp)
3. Open an issue on this repository
