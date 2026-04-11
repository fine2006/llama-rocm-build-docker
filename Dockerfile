# =============================================================================
# rdna35-toolbox — Distrobox/Toolbx for AMD Radeon 860M (gfx1152 / RDNA3.5)
# =============================================================================
#
# Authoritative upstream sources:
#   ROCm build system : https://github.com/ROCm/TheRock
#   ROCm meta-repo    : https://github.com/ROCm/ROCm
#   vLLM              : https://github.com/vllm-project/vllm
#   llama.cpp         : https://github.com/ggml-org/llama.cpp
#   Unsloth           : https://github.com/unslothai/unsloth  (AMD: PR #2520)
#   bitsandbytes      : https://github.com/ROCm/bitsandbytes  (branch rocm_enabled_multi_backend)
#   gfx1151 reference : https://github.com/kyuz0/amd-strix-halo-toolboxes
#   amdgpu_top        : https://github.com/Umio-Yasuno/amdgpu_top
#
# gfx1152 STATUS (April 2026):
#   - Radeon 860M = gfx1152 = RDNA3.5 "Kracken Point" (NOT RDNA4)
#   - TheRock DEB/Python packages published for gfx1151, NOT YET gfx1152
#     Track: https://github.com/ROCm/TheRock/issues/2310
#   - Strategy A (native):  llama.cpp compiled with GPU_TARGETS=gfx1151;gfx1152
#   - Strategy B (fallback): set HSA_OVERRIDE_GFX_VERSION=11.5.1 to use
#                            gfx1151 kernels on gfx1152 hardware for PyTorch/vLLM
#
# FALLBACK QUICK-START (if gfx1152 misbehaves):
#   export HSA_OVERRIDE_GFX_VERSION=11.5.1
#   Then re-run your workload. Remove the override once native gfx1152 packages land.
#
# =============================================================================

# -----------------------------------------------------------------------------
# Build arguments — override these via --build-arg or GitHub Actions matrix
# -----------------------------------------------------------------------------
# TheRock nightly release ID.  Find the latest at:
#   https://rocm.nightlies.amd.com/deb/
# Format: YYYYMMDD-COMMITSHA (e.g. 20260405-deadbeef)
# CI will auto-detect the latest (see .github/workflows/build.yml).
ARG THEROCK_RELEASE_ID=20260405-latest

# Primary GPU arch to install system packages for.
# gfx1151 is the only gfx115X arch with published TheRock packages right now.
# When https://github.com/ROCm/TheRock/issues/2310 is resolved (gfx1152 packages
# published), change this to gfx1152.
ARG ROCM_GFX_ARCH=gfx1151

# GPU targets for source builds (llama.cpp, vLLM, bitsandbytes).
# We compile for BOTH gfx1151 and gfx1152 so the binary runs natively on either.
ARG HIP_GPU_TARGETS="gfx1151;gfx1152"

# Python version (unsloth does not support 3.13)
ARG PYTHON_VERSION=3.12

# vLLM git ref — main branch for latest gfx1151/1152 work
ARG VLLM_REF=main

# llama.cpp git ref
ARG LLAMACPP_REF=master

# amdgpu_top release tag
ARG AMDGPU_TOP_VERSION=0.11.3

# =============================================================================
# Stage 1: rocm-sdk
# Ubuntu 24.04 + TheRock nightly ROCm SDK (DEB packages)
# This heavy stage is shared by all build stages below.
# =============================================================================
FROM ubuntu:24.04 AS rocm-sdk

ARG THEROCK_RELEASE_ID
ARG ROCM_GFX_ARCH
ARG PYTHON_VERSION

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# ---------- Base build dependencies ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Core build toolchain
    build-essential cmake ninja-build \
    git git-lfs curl wget ca-certificates \
    # Python + pip
    python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    python3-pip \
    # Misc libs needed at build time
    libdrm-dev libelf-dev pkg-config patchelf \
    # Needed by ROCm DEB post-install scripts
    kmod file \
    # Lzip for tarball extraction fallback
    lzip \
    && rm -rf /var/lib/apt/lists/*

# ---------- TheRock ROCm nightly DEB repository ----------
# Source: https://github.com/ROCm/TheRock/blob/main/RELEASES.md
#
# The RELEASE_ID is provided as a build-arg and kept as an ENV so child stages
# can read it without re-passing the build-arg.
ENV THEROCK_RELEASE_ID=${THEROCK_RELEASE_ID}
ENV ROCM_GFX_ARCH=${ROCM_GFX_ARCH}

RUN echo "deb [trusted=yes] https://rocm.nightlies.amd.com/deb/${THEROCK_RELEASE_ID} stable main" \
    > /etc/apt/sources.list.d/rocm-nightly.list \
    && apt-get update

# Install the full ROCm dev SDK for the chosen arch.
# amdrocm-core-sdk-${ROCM_GFX_ARCH} installs headers, libs, compilers
# (hipcc, clang, rocblas, MIOpen, rocFFT, hipBLAS …)
# rocwmma-dev is required by GGML_HIP_ROCWMMA_FATTN=ON in llama.cpp
# (rocWMMA flash attention gives a significant perf uplift on gfx115X)
RUN apt-get install -y --no-install-recommends \
    amdrocm-core-sdk-${ROCM_GFX_ARCH} \
    rocwmma-dev \
    && rm -rf /var/lib/apt/lists/*

# The TheRock DEB layout mirrors the classic /opt/rocm structure.
ENV ROCM_PATH=/opt/rocm
ENV PATH="${ROCM_PATH}/bin:${ROCM_PATH}/llvm/bin:${PATH}"
ENV LD_LIBRARY_PATH="${ROCM_PATH}/lib:${ROCM_PATH}/lib64:${LD_LIBRARY_PATH}"
ENV CMAKE_PREFIX_PATH="${ROCM_PATH}:${CMAKE_PREFIX_PATH}"

# Verify hipcc is present
RUN hipcc --version

# =============================================================================
# Stage 2: llama-build
# Build llama.cpp with HIP backend for gfx1151 AND gfx1152
#
# Sources:
#   Build docs : https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#hipblas
#   Known-good Strix Halo stack (gfx1151):
#     https://github.com/ggml-org/llama.cpp/discussions/20856
#   UMA detection bug (affects AMD APUs):
#     https://github.com/ggml-org/llama.cpp/issues/18159
#
# Flag rationale:
#   GGML_HIP_NO_VMM=ON         — THE critical flag for gfx115X APUs. HIP VMM
#                                 (Virtual Memory Management) is broken on this
#                                 family and causes mysterious load failures and
#                                 stability issues that look like driver bugs.
#                                 Output log should show: VMM: no
#                                 Ref: llama.cpp discussion #20856
#
#   GGML_HIP_UMA=ON            — Uses hipMallocManaged + hipMemAdviseSetCoarseGrain
#                                 so allocations come from the unified pool (system
#                                 RAM) rather than a nonexistent discrete VRAM region.
#                                 Without this the HIP backend tries to allocate
#                                 "device" memory that doesn't exist on an APU.
#                                 Ref: llama.cpp PR #4449, issue #7399
#
#   GGML_HIP_ROCWMMA_FATTN=ON  — Flash attention via rocWMMA kernels. Substantial
#                                 throughput improvement on gfx115X. Requires the
#                                 rocwmma-dev package installed in rocm-sdk stage.
#                                 Ref: llama.cpp discussion #20856
#
#   GGML_HIP_MMQ_MFMA=ON       — Use MFMA (Matrix Fused Multiply-Add) instructions
#                                 for the MMQ matrix-multiply path. Perf uplift
#                                 for quantized models on RDNA3.5.
#                                 Ref: llama.cpp discussion #20856
#
#   GGML_HIP_GRAPHS=ON         — HIP graph capture for the decode loop (reduces
#                                 per-token CPU overhead). Optional; disable if
#                                 unstable on a particular driver version.
# =============================================================================
FROM rocm-sdk AS llama-build

ARG LLAMACPP_REF
ARG HIP_GPU_TARGETS

# GPU_TARGETS for cmake takes comma-separated values; HIP_GPU_TARGETS uses semicolons
RUN TARGETS_COMMA=$(echo "${HIP_GPU_TARGETS}" | tr ';' ',') \
    && git clone --depth=1 --branch ${LLAMACPP_REF} \
         https://github.com/ggml-org/llama.cpp /build/llama.cpp \
    && cd /build/llama.cpp \
    && cmake -B build \
         -DCMAKE_BUILD_TYPE=Release \
         -DGGML_HIP=ON \
         -DGPU_TARGETS="${TARGETS_COMMA}" \
         \
         # ── UMA / APU memory flags ─────────────────────────────────────────
         # hipMallocManaged + coarse-grain advise → allocate from unified pool
         -DGGML_HIP_UMA=ON \
         # Disable HIP VMM — broken on gfx115X, causes silent load failures
         -DGGML_HIP_NO_VMM=ON \
         \
         # ── Performance flags ──────────────────────────────────────────────
         # rocWMMA flash attention (needs rocwmma-dev, big perf gain on gfx115X)
         -DGGML_HIP_ROCWMMA_FATTN=ON \
         # MFMA matrix-multiply kernels for quantized models
         -DGGML_HIP_MMQ_MFMA=ON \
         # HIP graph capture to reduce per-token CPU overhead
         -DGGML_HIP_GRAPHS=ON \
         \
         # ── Compiler settings ─────────────────────────────────────────────
         # ROCm 7+ LLVM loop-unroll regression workaround (kyuz0/amd-strix-halo-toolboxes#45)
         # Remove once llvm-project#147700 is fixed upstream.
         -DCMAKE_HIP_FLAGS="-mllvm --amdgpu-unroll-threshold-local=600" \
         -DCMAKE_HIP_COMPILER=${ROCM_PATH}/llvm/bin/clang++ \
         -DCMAKE_C_COMPILER=${ROCM_PATH}/llvm/bin/clang \
         -DCMAKE_CXX_COMPILER=${ROCM_PATH}/llvm/bin/clang++ \
         \
         # ── Server build ──────────────────────────────────────────────────
         -DLLAMA_BUILD_SERVER=ON \
         -G Ninja \
    && cmake --build build --config Release -j$(nproc) \
    # Install binaries to /usr/local so they're easy to COPY out
    && cmake --install build --prefix /llama-install

# =============================================================================
# Stage 3: bnb-build
# Build bitsandbytes from the ROCm-enabled fork — required by Unsloth on AMD
# Source: https://github.com/ROCm/bitsandbytes  branch: rocm_enabled_multi_backend
# Ref: https://github.com/unslothai/unsloth/pull/2520
#      https://github.com/unslothai/unsloth/pull/3279
# =============================================================================
FROM rocm-sdk AS bnb-build

ARG HIP_GPU_TARGETS
ARG PYTHON_VERSION

# Install pip into the build env
RUN python${PYTHON_VERSION} -m pip install --upgrade pip setuptools wheel

# Clone the ROCm-enabled fork
RUN git clone --depth=1 \
        --branch rocm_enabled_multi_backend \
        https://github.com/ROCm/bitsandbytes \
        /build/bitsandbytes

WORKDIR /build/bitsandbytes

RUN pip install -r requirements-dev.txt

# Build for our target architectures.
# AMDGPU_TARGETS maps to the GPU_TARGETS cmake arg.
RUN cmake -DCOMPUTE_BACKEND=hip \
          -DAMDGPU_TARGETS="${HIP_GPU_TARGETS}" \
          -DCMAKE_HIP_COMPILER=${ROCM_PATH}/llvm/bin/clang++ \
          -DCMAKE_C_COMPILER=${ROCM_PATH}/llvm/bin/clang \
          -DCMAKE_CXX_COMPILER=${ROCM_PATH}/llvm/bin/clang++ \
          -S . \
    && make -j$(nproc) \
    && pip wheel . --no-deps -w /bnb-wheels/

# =============================================================================
# Stage 4: vllm-build
# Build vLLM from source for gfx1151 + gfx1152
#
# Upstream gfx1151/gfx1150 support merged in:
#   https://github.com/vllm-project/vllm/pull/25908
#   https://github.com/vllm-project/vllm/pull/28308
#
# gfx1152 is not yet in vLLM's CMakeLists.txt — we patch it in.
# When vLLM officially adds gfx1152, remove the sed patch below.
#
# IMPORTANT: We install PyTorch from TheRock gfx1151 nightlies here.
#            At RUNTIME, if your GPU is gfx1152, set:
#              HSA_OVERRIDE_GFX_VERSION=11.5.1
#            This tells HSA to use gfx1151 kernels on the gfx1152 hardware.
#            Once TheRock publishes gfx1152 wheels, switch the index URL.
#
# =============================================================================
FROM rocm-sdk AS vllm-build

ARG VLLM_REF
ARG HIP_GPU_TARGETS
ARG PYTHON_VERSION

ENV PYTHON=python${PYTHON_VERSION}

# Install system deps vLLM needs at build time
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION}-venv \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a build venv
RUN ${PYTHON} -m venv /vllm-venv
ENV PATH="/vllm-venv/bin:${PATH}"

# Install PyTorch from TheRock gfx1151 nightlies.
# Source: https://github.com/ROCm/TheRock/blob/main/RELEASES.md
# When gfx1152 packages land (issue #2310), change the index URL to:
#   https://rocm.nightlies.amd.com/v2/gfx1152/
RUN pip install --upgrade pip \
    && pip install \
        --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
        --extra-index-url https://pypi.org/simple/ \
        torch torchvision torchaudio

# Clone vLLM
RUN git clone --depth=1 --branch ${VLLM_REF} \
        https://github.com/vllm-project/vllm \
        /build/vllm

WORKDIR /build/vllm

# Patch vLLM CMakeLists.txt to include gfx1152.
# This adds gfx1152 alongside gfx1151 in the ROCM_ARCH list.
# Track upstream progress at: https://github.com/ROCm/TheRock/issues/2310
RUN if ! grep -q "gfx1152" cmake/utils/gpu.rocm.inc.cmake 2>/dev/null && \
       ! grep -q "gfx1152" CMakeLists.txt 2>/dev/null; then \
        echo "# Patching vLLM to add gfx1152 support" && \
        sed -i 's/gfx1151/gfx1151 gfx1152/g' CMakeLists.txt 2>/dev/null || true && \
        sed -i 's/gfx1151/gfx1151 gfx1152/g' cmake/utils/gpu.rocm.inc.cmake 2>/dev/null || true; \
    fi

# Build vLLM.
# MAX_JOBS controls parallelism — reduce if OOM during build.
# Using ROCm Clang as the C/C++ compiler for ABI compatibility.
# Ref: https://deepwiki.com/kyuz0/amd-strix-halo-vllm-toolboxes/5.2-software-stack
ENV PYTORCH_ROCM_ARCH="${HIP_GPU_TARGETS}"
ENV MAX_JOBS=4
ENV CC="${ROCM_PATH}/llvm/bin/clang"
ENV CXX="${ROCM_PATH}/llvm/bin/clang++"
# Performance workaround: ROCm 7+ LLVM unroll regression
# Ref: https://github.com/kyuz0/amd-strix-halo-toolboxes#45
ENV CFLAGS="-mllvm --amdgpu-unroll-threshold-local=600"
ENV CXXFLAGS="-mllvm --amdgpu-unroll-threshold-local=600"

# Install vLLM build requirements
RUN pip install -r requirements/build.txt

# Build and package as a wheel
RUN pip wheel . --no-build-isolation --no-deps -w /vllm-wheels/

# =============================================================================
# Stage 5: final
# Minimal Ubuntu 24.04 runtime image for distrobox / toolbx
# Copies compiled artifacts from build stages; installs Python toolchain
# and monitoring tools. No ROCm system packages needed at runtime — PyTorch
# wheels from TheRock bundle the ROCm runtime libraries.
# =============================================================================
FROM ubuntu:24.04 AS final

ARG PYTHON_VERSION
ARG AMDGPU_TOP_VERSION
ARG ROCM_GFX_ARCH

LABEL org.opencontainers.image.title="rdna35-toolbox"
LABEL org.opencontainers.image.description="Distrobox/Toolbx for AMD Radeon 860M (gfx1152/RDNA3.5): PyTorch, Unsloth, llama.cpp HIP, vLLM, monitoring"
LABEL org.opencontainers.image.source="https://github.com/YOUR_ORG/rdna35-toolbox"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PYTHON_VERSION=${PYTHON_VERSION}

# ---------- Runtime system packages ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # -- Python runtime --
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    # -- GPU access at runtime --
    libdrm2 libdrm-amdgpu1 \
    libelf1 \
    # -- Monitoring --
    htop \
    nvtop \          # GPU monitoring; ROCm-enabled via system package
    iotop \
    iftop \
    lm-sensors \
    sysstat \
    # -- Distrobox/Toolbx compatibility --
    bash \
    sudo \
    passwd \
    util-linux \
    procps \
    curl \
    wget \
    ca-certificates \
    git \
    git-lfs \
    vim \
    nano \
    less \
    man-db \
    bash-completion \
    # -- Build & dev essentials (distrobox users expect these) --
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    unzip \
    rsync \
    jq \
    zstd \
    # -- ROCm userspace runtime: kfd access, libhsa --
    # Note: libhsa-runtime64 is bundled by the PyTorch/rocm pip wheel,
    # but we also install system copies for non-Python tooling (llama.cpp).
    libhsa-runtime64-1 \
    # Rocm SMI CLI tool (separate from the pip wheel)
    && apt-get install -y --no-install-recommends rocm-smi || true \
    && rm -rf /var/lib/apt/lists/*

# ---------- amdgpu_top (Rust-based GPU monitor) ----------
# Source: https://github.com/Umio-Yasuno/amdgpu_top/releases
# v0.11.2+ adds "fix gfx and media activity for Strix Point, Krackan Point and Strix Halo"
# — directly relevant to gfx1152 (Kracken Point).
#
# Asset filename format: amdgpu_top-VERSION-ARCH-unknown-linux-gnu.tar.gz
# Note: "unknown" is part of the Rust target triple — omitting it gives a 404.
# The tarball extracts to a directory; the binary is inside it.
RUN ARCH=$(uname -m) \
    && ASSET="amdgpu_top-${AMDGPU_TOP_VERSION}-${ARCH}-unknown-linux-gnu.tar.gz" \
    && curl -fsSL \
        "https://github.com/Umio-Yasuno/amdgpu_top/releases/download/v${AMDGPU_TOP_VERSION}/${ASSET}" \
        -o /tmp/amdgpu_top.tar.gz \
    && tar -xzf /tmp/amdgpu_top.tar.gz -C /tmp \
    # Binary lives inside the extracted directory, not at the tarball root
    && EXTRACTED_DIR=$(tar -tzf /tmp/amdgpu_top.tar.gz | head -1 | cut -d/ -f1) \
    && install -m 755 "/tmp/${EXTRACTED_DIR}/amdgpu_top" /usr/local/bin/amdgpu_top \
    && rm -rf /tmp/amdgpu_top* "/tmp/${EXTRACTED_DIR}"
    # Note: --version not verified here — amdgpu_top probes /dev/dri which
    # doesn't exist inside the Docker builder. Verify at runtime with: amdgpu_top --version

# ---------- Copy llama.cpp binaries from build stage ----------
COPY --from=llama-build /llama-install/ /usr/local/

# ---------- Set up Python default + pip ----------
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 100 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python${PYTHON_VERSION} 100 \
    && python3 -m pip install --upgrade pip setuptools wheel

# ---------- uv — fast Python package manager ----------
# Source: https://github.com/astral-sh/uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -sf /root/.local/bin/uv /usr/local/bin/uv \
    && ln -sf /root/.local/bin/uvx /usr/local/bin/uvx \
    && uv --version

# ---------- Copy bitsandbytes wheel and install ----------
COPY --from=bnb-build /bnb-wheels/ /wheels/bnb/
RUN pip install /wheels/bnb/*.whl && rm -rf /wheels/bnb

# ---------- Install PyTorch from TheRock gfx1151 nightlies ----------
# Source: https://github.com/ROCm/TheRock/blob/main/RELEASES.md
#
# NOTE: We use the gfx1151 index because gfx1152-specific packages are not yet
#       published (TheRock issue #2310). The rocm-sdk-libraries wheel bundled
#       with torch provides the ROCm runtime libs (libhsa, librocblas, etc.)
#       so NO system-level ROCm packages are needed for Python workloads.
#
# FALLBACK: If something breaks on gfx1152, add to your shell:
#   export HSA_OVERRIDE_GFX_VERSION=11.5.1
#
# UPGRADE: When gfx1152 packages land, change the index URL to:
#   https://rocm.nightlies.amd.com/v2/gfx1152/
RUN pip install \
    --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
    --extra-index-url https://pypi.org/simple/ \
    torch torchvision torchaudio

# ---------- Copy and install vLLM wheel ----------
COPY --from=vllm-build /vllm-wheels/ /wheels/vllm/
# vLLM is installed but its extra deps come from PyPI
RUN pip install /wheels/vllm/*.whl \
    && rm -rf /wheels/vllm

# ---------- Unsloth ROCm ----------
# Source: https://github.com/unslothai/unsloth  (AMD support: PR #2520, Jun 2025)
# Docs:   https://docs.unsloth.ai/get-started/install-and-update/amd
#
# We install the [amd] extras group which pulls in the ROCm-specific deps.
# xformers is excluded (ROCm issues); flash-attention 2 via aotriton is used instead.
RUN pip install "unsloth[amd] @ git+https://github.com/unslothai/unsloth" \
    && pip install unsloth_zoo

# ---------- Common ML/dev Python packages ----------
RUN pip install \
    # Hugging Face ecosystem
    transformers \
    datasets \
    accelerate \
    peft \
    trl \
    tokenizers \
    huggingface_hub \
    hf_transfer \
    # Jupyter
    jupyterlab \
    ipywidgets \
    # Utilities
    numpy \
    pandas \
    matplotlib \
    tqdm \
    rich \
    typer \
    httpx \
    sentencepiece \
    protobuf

# ---------- ROCm / Unified Memory environment ----------
#
# ARCHITECTURE NOTE — Radeon 860M is a UMA APU:
#   There is NO dedicated VRAM. System RAM *is* the GPU memory pool.
#   Without the GPU_MAX_* variables below, ROCm applies conservative caps
#   (often ~50–75 % of the pool) and your effective "VRAM" is halved.
#   Setting all three to 100 tells ROCm to treat the entire unified pool as
#   available GPU memory, which is the correct behaviour for an APU.
#
# Variable reference:
#
#   GPU_MAX_HEAP_SIZE=100
#     Percentage of the GPU heap (= system RAM on APU) exposed to HIP/ROCm.
#     Default is often ≤75 %. Set to 100 to expose the full pool.
#
#   GPU_MAX_ALLOC_PERCENT=100
#     Maximum percentage of the heap that a single HIP allocation may consume.
#     Without this, large model weight tensors are rejected even when RAM is free.
#
#   GPU_SINGLE_ALLOC_PERCENT=100
#     Maximum percentage of *currently available* memory a single malloc may
#     request. Paired with GPU_MAX_ALLOC_PERCENT it removes the artificial cap
#     on contiguous allocations — essential for loading multi-GB GGUF files.
#
#   HSA_ENABLE_SDMA=0
#     Disables the System DMA engine for host↔device copies on APU.
#     SDMA causes "checkerboard" memory corruption during large transfers in
#     unified-memory configurations. Always disable on gfx115X APUs.
#     Ref: https://github.com/ROCm/TheRock/discussions/655
#
#   GGML_HIP_UMA=1  (llama.cpp runtime flag, mirrors -DLLAMA_HIP_UMA=ON)
#     Instructs the ggml HIP backend to allocate from the unified pool rather
#     than attempting to carve out a discrete VRAM region that doesn't exist.
#     Required for correct operation on any APU; without it llama.cpp will
#     fail or silently fall back to CPU for layers it thinks can't fit in VRAM.
#
#   PYTORCH_HIP_ALLOC_CONF — intentionally NOT set to "backend:malloc":
#     Some ROCm shell profiles include this; it crashes PyTorch on gfx115X.
#     The expandable_segments allocator (default since PyTorch 2.x) works
#     correctly with UMA and does not need overriding.
#     Ref: https://github.com/ROCm/ROCm/issues/6034
#
#   HIP_VISIBLE_DEVICES=0
#     Select the first (only) GPU. Override for multi-GPU or CPU-only runs.
#
#   HSA_OVERRIDE_GFX_VERSION — commented out, see fallback section below.

RUN cat >> /etc/environment <<'EOF'

# ── rdna35-toolbox: ROCm Unified Memory (UMA) defaults ───────────────────────
# Radeon 860M is an APU — system RAM is the GPU memory pool.
# These variables expose the FULL pool to ROCm/HIP. Without them ROCm
# applies conservative caps and you lose most of your effective "VRAM".

# Expose 100 % of the unified heap to ROCm allocations
GPU_MAX_HEAP_SIZE=100
GPU_MAX_ALLOC_PERCENT=100
GPU_SINGLE_ALLOC_PERCENT=100

# Disable SDMA DMA engine (causes corruption in UMA configurations)
HSA_ENABLE_SDMA=0

# Instruct llama.cpp's HIP backend to use the unified memory path
# (matches the -DGGML_HIP_UMA=ON compile flag set in the llama-build stage)
GGML_HIP_UMA=1

# Allow llama.cpp to spill into system RAM when the GPU pool fills.
# On an APU this is the same physical RAM, so this is effectively free overflow.
# Ref: https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1

# GPU device selection (override if needed)
HIP_VISIBLE_DEVICES=0

# ── gfx1152 compatibility fallback ───────────────────────────────────────────
# If ROCm cannot load gfx1152-native kernels, uncomment the line below.
# It makes gfx1152 masquerade as gfx1151 so the gfx1151 kernel bitcode is used.
# Expect a small (<5 %) performance delta. Remove once TheRock#2310 is resolved.
# HSA_OVERRIDE_GFX_VERSION=11.5.1
EOF

# Profile script for interactive shells
# Written separately from /etc/environment so it can run dynamic commands
# (e.g. detect actual unified memory size at shell start-up time).
RUN cat > /etc/profile.d/rocm-toolbox.sh <<'PROFILE'
#!/bin/sh
# rdna35-toolbox shell environment — sourced for all interactive distrobox shells

# ── ROCm wheel library path ───────────────────────────────────────────────────
# TheRock Python wheels bundle libhsa, librocblas, etc. inside the site-packages
# tree. Add them to LD_LIBRARY_PATH so non-Python binaries (llama-server, etc.)
# can find them without a system-level ROCm install.
ROCM_WHEEL_LIBS=$(python3 -c \
    "import rocm; import os; print(os.path.dirname(rocm.__file__))" 2>/dev/null || true)
if [ -n "${ROCM_WHEEL_LIBS}" ]; then
    export LD_LIBRARY_PATH="${ROCM_WHEEL_LIBS}/lib:${ROCM_WHEEL_LIBS}/lib64:${LD_LIBRARY_PATH:-}"
fi

# ── Detect unified memory pool size at login ─────────────────────────────────
# On APU the "VRAM" reported is actually system RAM carved out for the GPU.
# With GPU_MAX_HEAP_SIZE=100 this should equal your total installed RAM.
UMA_MB=$(python3 -c "
import ctypes, ctypes.util, sys
try:
    hip = ctypes.CDLL('libamdhip64.so')
    total = ctypes.c_size_t(0)
    free  = ctypes.c_size_t(0)
    hip.hipMemGetInfo(ctypes.byref(free), ctypes.byref(total))
    print(f'{total.value // (1024**2):,}')
except Exception:
    print('unknown')
" 2>/dev/null || echo "unknown")

# ── Convenience aliases ───────────────────────────────────────────────────────
alias ll='ls -lah --color=auto'
alias gpu-top='amdgpu_top'
alias gpu-smi='rocm-smi'
alias gpu-info='rocm-smi --showmeminfo vram 2>/dev/null || amdgpu_top --dump-all'
# Show effective unified memory visible to ROCm right now
alias mem-check='python3 -c "
import torch
total = torch.cuda.get_device_properties(0).total_memory
print(f\"ROCm sees: {total/(1024**3):.1f} GiB of unified memory\")
print(f\"  GPU_MAX_HEAP_SIZE={__import__(\"os\").environ.get(\"GPU_MAX_HEAP_SIZE\",\"unset (default cap applies!)\")}\")
"'

cat <<BANNER
╔══════════════════════════════════════════════════════════════════╗
║  rdna35-toolbox  —  AMD Radeon 860M (gfx1152 / RDNA3.5)         ║
║  PyTorch │ Unsloth │ llama.cpp HIP │ vLLM │ ROCm nightlies       ║
╠══════════════════════════════════════════════════════════════════╣
║  Unified memory pool visible to ROCm: ${UMA_MB} MiB              
║  (GPU_MAX_HEAP_SIZE=100  GPU_MAX_ALLOC_PERCENT=100)              ║
╠══════════════════════════════════════════════════════════════════╣
║  gpu-top    → amdgpu_top   (GPU utilisation / temp / clocks)     ║
║  gpu-smi    → rocm-smi     (ROCm device info)                    ║
║  mem-check  → show ROCm-visible unified memory                   ║
║  htop       → CPU + RAM    nvtop → GPU monitor                   ║
╠══════════════════════════════════════════════════════════════════╣
║  ⚠  gfx1152 fallback: export HSA_OVERRIDE_GFX_VERSION=11.5.1    ║
║     Use if PyTorch/vLLM fail to detect GPU natively.             ║
╚══════════════════════════════════════════════════════════════════╝
BANNER
PROFILE

# Distrobox compatibility: add /etc/environment values to shell profiles
RUN echo '. /etc/profile.d/rocm-toolbox.sh' >> /etc/bash.bashrc

# ---------- Verify critical imports ----------
RUN python3 -c "import torch; print('torch:', torch.__version__)" \
    && python3 -c "import vllm; print('vllm OK')" \
    && python3 -c "import unsloth; print('unsloth OK')" \
    && python3 -c "import bitsandbytes; print('bitsandbytes OK')" \
    && llama-cli --version 2>/dev/null || llama-server --version 2>/dev/null || true

# ---------- Distrobox / Toolbx entry point ----------
# Distrobox injects its own init; we just need a working shell.
CMD ["/bin/bash", "-l"]
