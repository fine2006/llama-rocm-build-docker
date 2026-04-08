# llama.cpp ROCm Container

This repository provides a containerized build of [`llama.cpp`](https://github.com/ggml-org/llama.cpp) with HIP/ROCm support for AMD GPUs. The build process:

- Uses **Ubuntu 24.04** base image with **ROCm nightly SDK** for a specific GPU architecture.
- Relies on a **Git submodule** for `llama.cpp` and a **Makefile** for build configuration.
- Produces a container image that runs `llama-server` with all ROCm devices available.
- Includes **multi-stage Docker build** for optimized image size (~6-7GB).

---

## Repository Structure

```
.
├── .github/
│   └── workflows/
│       └── build.yml          # GitHub Actions workflow
├── .dockerignore              # Files to exclude from build context
├── .gitmodules                # Defines llama.cpp submodule
├── Dockerfile                 # Container build instructions (multi-stage)
├── Makefile                   # Builds llama.cpp with ROCm
├── llama.cpp/                 # Git submodule (ggml-org/llama.cpp)
└── README.md                  # This file
```

---

## Prerequisites

- **Podman** or **Docker** with BuildKit support.
- **Git** (to clone the repository and submodules).
- AMD GPU with ROCm‑compatible driver installed on the host.
- (Optional) Access to a container registry if you wish to push images.

---

## Configuration

### Build Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `RELEASE_ID` | `20260310-12345678` | ROCm nightly release ID from [rocm.nightlies.amd.com/deb/](https://rocm.nightlies.amd.com/deb/) |
| `GFX_ARCH` | `gfx1152` | GPU target architecture (e.g., `gfx1152`, `gfx1100`, `gfx942`) |
| `ENABLE_ROCWMMA_FATTN` | `ON` | Enable rocWMMA for RDNA3+/CDNA FlashAttention |
| `BUILD_WEBUI` | `ON` | Build WebUI executable (`llama-server-webui`) |
| `BUILD_LLAMA_SERVER` | `ON` | Build llama-server binary |

### Build Options

The following options are **hardcoded** to optimize the build:

- ✅ Only **HIP backend** enabled (all other GPU backends disabled)
- ✅ **Static libraries** (`BUILD_SHARED_LIBS=OFF`) for self-contained deployment
- ✅ **CPU variants disabled** (`GGML_CPU_ALL_VARIANTS=OFF`)
- ✅ **Tests, tools, examples disabled** for smaller binary

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
  --build-arg RELEASE_ID=20260310-12345678 \
  --build-arg GFX_ARCH=gfx1152 \
  --build-arg ENABLE_ROCWMMA_FATTN=ON \
  --build-arg BUILD_WEBUI=ON \
  --build-arg BUILD_LLAMA_SERVER=ON \
  -t llama-cpp-rocm:latest .
```

### 3. Run the Container

```bash
podman run \
  --device=/dev/kfd \
  --device=/dev/dri \
  --ipc=host \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.0 \
  -v /path/to/your/models:/models:Z \
  localhost/llama-cpp-rocm:latest \
  -m /models/your-model.gguf \
  --port 8080 \
  -ngl 999 \
  --temp 0.6 \
  --top-p 0.95 \
  --top-k 20
```

---

## Customizing for Your GPU

### 1. Find ROCm Nightly Release ID

Visit [rocm.nightlies.amd.com/deb/](https://rocm.nightlies.amd.com/deb/) and look for a directory named like `YYYYMMDD-12345678`.

### 2. Determine Your GPU Architecture

```bash
# On a host with ROCm installed
rocminfo | grep gfx | head -1 | awk '{print $2}'

# Or check AMD GPU compatibility table
# https://rocm.docs.amd.com/projects/rocm-compute-profiles/en/latest/gpu.html
```

Common examples:
- **Radeon RX 7900 series** → `gfx1100`
- **Radeon RX 7600** → `gfx1102`
- **Radeon RX 780M (iGPU)** → `gfx1103`
- **Radeon RX 8600M** → `gfx1152`
- **MI300 series** → `gfx942`

### 3. Update Build Arguments

```bash
podman build \
  --build-arg RELEASE_ID=YOUR_RELEASE_ID \
  --build-arg GFX_ARCH=YOUR_GPU_ARCH \
  --build-arg ENABLE_ROCWMMA_FATTN=ON \
  -t llama-cpp-rocm:latest .
```

---

## Available Commands

The container includes the following binaries:

| Binary | Description |
|--------|-------------|
| `llama-server` | HTTP server for running models |
| `llama-quantize` | Quantize models to different precisions |
| `llama-bench` | Benchmark model performance |
| `llama-perplexity` | Calculate perplexity of text |
| `llama-convert-pth` | Convert PyTorch models to GGUF |
| `llama-convert-hf-to-gguf` | Convert HuggingFace models to GGUF |
| `llama-convert-lora-to-gguf` | Convert LoRA adapters to GGUF |

### Usage Examples

```bash
# Run server
podman run localhost/llama-cpp-rocm:latest -m /models/model.gguf --port 8080

# Quantize model
podman run -v /models:/models:Z \
  localhost/llama-cpp-rocm:latest \
  -q /models/model-f16.gguf /models/model-q4_0.gguf 4

# Benchmark
podman run -v /models:/models:Z \
  localhost/llama-cpp-rocm:latest \
  -b /models/model.gguf -n 512

# Calculate perplexity
podman run -v /models:/models:Z \
  localhost/llama-cpp-rocm:latest \
  -p /models/model.gguf -f /path/to/text.txt
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HSA_OVERRIDE_GFX_VERSION` | Override GPU architecture (e.g., `11.5.0` for RDNA3) |
| `HIP_VISIBLE_DEVICES` | Specify which GPU(s) to use (e.g., `0,1` for multi-GPU) |
| `GGML_CUDA_ENABLE_UNIFIED_MEMORY` | Enable unified memory (works for HIP backend too) |
| `LLAMA_LOG_LEVEL` | Set logging level (DEBUG, INFO, WARNING, ERROR) |

---

## Troubleshooting

### Issue: "Cannot find valid GPU for '-arch=native'"

**Solution:** Override GPU architecture:
```bash
podman run -e HSA_OVERRIDE_GFX_VERSION=11.5.0 ...
```

### Issue: "HIP device library not found"

**Solution:** Ensure ROCm runtime is installed on the host, or rebuild with specific architecture.

### Issue: "Permission denied" for /dev/kfd or /dev/dri

**Solution:** Verify device permissions on the host:
```bash
ls -la /dev/kfd /dev/dri
```

### Issue: WebUI not accessible

**Solution:** Ensure `BUILD_WEBUI=ON` when building, and use correct port:
```bash
podman run -p 8080:8080 ...
```

---

## Image Size Optimization

This build uses several optimizations to minimize image size:

1. **Multi-stage Docker build** - Separate builder and runtime stages
2. **Ubuntu 24.04 base** - Smaller than Fedora alternatives
3. **Only HIP backend** - All other GPU backends disabled
4. **Static libraries** - Self-contained, no runtime dependencies
5. **Aggressive cleanup** - Removes all build artifacts and cache

**Estimated size:** ~6-7GB (compared to ~10GB for unoptimized builds)

---

## GitHub Actions Workflow

The repository includes a GitHub Actions workflow that automatically builds the container image on every push to the `main` branch.

### Workflow Features

- ✅ **Ubuntu 24.04 runner** for ROCm compatibility
- ✅ **Build caching** via GitHub Actions cache
- ✅ **Multi-platform support** (linux/amd64 only, ROCm is x86_64-specific)
- ✅ **Automatic tagging** based on branch and commit
- ✅ **Registry push** only on merges to main

### Manual Trigger

You can manually trigger a build via the GitHub UI:
1. Go to repository → Actions → Build Container Image
2. Click "Run workflow"
3. Select branch (default: main)

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
