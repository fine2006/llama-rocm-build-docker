# =============================================================================
# llama.cpp — ROCm-only build, Wolfi runtime, self-contained
#
# Build args:
#   RELEASE_ID            ROCm nightly build ID (default: 20260408-24115666439)
#   GFX_ARCH              GPU target, e.g. gfx1152, gfx1100, gfx906
#   ENABLE_ROCWMMA_FATTN  rocWMMA flash attention (ON | OFF)
#   BUILD_ALL             OFF = llama-server only
#                         ON  = + llama-{quantize,bench,perplexity}
#                         NOTE: llama-convert-* are Python scripts in modern
#                               llama.cpp builds — add them separately if needed
#   AMDGPU_TOP_VERSION    amdgpu_top release to embed (static musl binary)
#
# Examples:
#   # Minimal — server only
#   podman build \
#     --build-arg GFX_ARCH=gfx1152 \
#     -t llama-rocm:server .
#
#   # Full toolkit
#   podman build \
#     --build-arg GFX_ARCH=gfx1152 \
#     --build-arg BUILD_ALL=ON \
#     -t llama-rocm:full .
# =============================================================================


# ─── Stage 1: llama.cpp builder (Ubuntu 24.04 + ROCm nightly) ────────────────
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

ARG RELEASE_ID=20260408-24115666439
ARG GFX_ARCH=gfx1152
ARG ENABLE_ROCWMMA_FATTN=ON
ARG BUILD_ALL=OFF

# Build toolchain — keep minimal, no doc/dev extras
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      build-essential \
      cmake \
      ninja-build \
      python3 \
      python3-pip \
      libssl-dev \
      git \
    && rm -rf /var/lib/apt/lists/*

# ROCm nightly apt repo — pinned to the exact build ID
# trusted=yes is intentional here: nightly builds don't ship signed Release files
RUN echo "deb [trusted=yes] https://rocm.nightlies.amd.com/deb/${RELEASE_ID} stable main" \
    > /etc/apt/sources.list.d/rocm-nightly.list \
    && apt-get update && apt-get install -y --no-install-recommends \
         amdrocm-core-sdk-${GFX_ARCH} \
         rocm-device-libs \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH \
    HIP_PATH=/opt/rocm \
    ROCM_PATH=/opt/rocm

# ── Validate ROCm install before trying to use it ────────────────────────────
# Runs as a separate layer so failures here produce a clear, targeted error
# rather than a cryptic "cmake exit code 1".
RUN echo "=== hipconfig output ===" \
    # hipconfig is the canonical way to query ROCm paths. If this fails, the SDK
    # package likely only installed runtime libs, not the full build toolchain.
    && hipconfig --full \
    && echo "=== LLVM/clang compiler ===" \
    # nightly builds put clang at /opt/rocm/lib/llvm/bin/; stable at /opt/rocm/llvm/bin/
    && find /opt/rocm -name 'clang' -path '*/llvm/bin/clang' 2>/dev/null \
    && (clang --version || /opt/rocm/lib/llvm/bin/clang --version) \
    && echo "=== HIP CMake integration ===" \
    # CMake's FindHIP needs these .cmake files; their absence means GGML_HIP=ON fails.
    && find /opt/rocm/lib/cmake /opt/rocm/share \
         \( -name 'hip-config.cmake' -o -name 'HIPConfig.cmake' \) 2>/dev/null \
       | head -5 \
    && echo "=== ROCm device libs (OCML bitcode) ===" \
    && ls /opt/rocm/lib/bitcode/*.bc 2>/dev/null | head -5 \
       || echo "WARN: no bitcode files found — rocm-device-libs may be missing" \
    && echo "=== /opt/rocm/bin ===" \
    && ls /opt/rocm/bin/ | head -30

WORKDIR /build
COPY . .

# ── Confirm the llama.cpp submodule was actually checked out ─────────────────
# An empty submodule is the most common silent failure — cmake exits 1 with
# "CMakeLists.txt not found", which is easy to miss in truncated CI logs.
RUN test -f llama.cpp/CMakeLists.txt \
    || { printf '\nERROR: llama.cpp/CMakeLists.txt not found.\n'; \
         printf 'The llama.cpp submodule is missing or empty.\n'; \
         printf 'Ensure submodules: true is set in the checkout step and that\n'; \
         printf 'the submodule is initialised: git submodule update --init --recursive\n\n'; \
         exit 1; }

# ── Resolve the HIP clang path robustly ──────────────────────────────────────
# hipconfig --hipclangpath returns the directory containing clang, e.g.:
#   /opt/rocm/lib/llvm/bin   (ROCm nightly, observed)
#   /opt/rocm/llvm/bin       (some stable releases)
# We probe hipconfig first, then try both known hardcoded locations.
RUN HIPCXX="$(hipconfig --hipclangpath 2>/dev/null)/clang" \
    && if [ ! -x "${HIPCXX}" ]; then HIPCXX=/opt/rocm/lib/llvm/bin/clang; fi \
    && if [ ! -x "${HIPCXX}" ]; then HIPCXX=/opt/rocm/llvm/bin/clang;     fi \
    && if [ ! -x "${HIPCXX}" ]; then \
         echo "ERROR: cannot locate HIP clang — searched:"; \
         echo "  hipconfig --hipclangpath -> $(hipconfig --hipclangpath 2>/dev/null)"; \
         echo "  /opt/rocm/lib/llvm/bin/clang"; \
         echo "  /opt/rocm/llvm/bin/clang"; \
         exit 1; \
       fi \
    && echo "Resolved HIPCXX=${HIPCXX}"

# Configure and build — CPU variants explicitly disabled; HIP only
# Both CMAKE_C_COMPILER and CMAKE_CXX_COMPILER must point to ROCm clang.
# If only CXX is set, CMake falls back to system gcc for .c files — gcc does
# not know -Wunreachable-code-break/-return (clang-only flags) and every C
# translation unit fails immediately.
RUN HIPCXX="$(hipconfig --hipclangpath 2>/dev/null)/clang" \
    && if [ ! -x "${HIPCXX}" ]; then HIPCXX=/opt/rocm/lib/llvm/bin/clang; fi \
    && if [ ! -x "${HIPCXX}" ]; then HIPCXX=/opt/rocm/llvm/bin/clang;     fi \
    && HIP_PATH="$(hipconfig -R)" \
    cmake -S llama.cpp -B llama.cpp/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER="${HIPCXX}" \
      -DCMAKE_CXX_COMPILER="${HIPCXX}" \
      -DHIP_CXX_COMPILER="${HIPCXX}" \
      -DGGML_CPU_ALL_VARIANTS=OFF \
      -DGGML_NATIVE=OFF \
      -DGGML_CUDA=OFF \
      -DGGML_METAL=OFF \
      -DGGML_VULKAN=OFF \
      -DGGML_HIP=ON \
      -DGGML_HIP_ROCWMMA_FATTN="${ENABLE_ROCWMMA_FATTN}" \
      -DGPU_TARGETS="${GFX_ARCH}" \
      -DCMAKE_HIP_FLAGS="--rocm-path=/opt/rocm" \
      --log-level=STATUS \
    && cmake --build llama.cpp/build -j$(nproc) --verbose

# Stage binaries: llama-server always; rest behind BUILD_ALL
RUN mkdir -p /staging/bin \
    && install -Dm755 llama.cpp/build/bin/llama-server /staging/bin/ \
    && if [ "${BUILD_ALL}" = "ON" ]; then \
         for bin in llama-quantize llama-bench llama-perplexity; do \
           if [ -f "llama.cpp/build/bin/${bin}" ]; then \
             install -Dm755 "llama.cpp/build/bin/${bin}" /staging/bin/; \
             echo "Staged: ${bin}"; \
           else \
             echo "WARN: ${bin} not found — skipping"; \
           fi; \
         done; \
       fi

# ── Collect ROCm runtime libraries ──────────────────────────────────────────
# Strategy: copy the entire /opt/rocm/lib tree (preserving symlinks and
# subdirectory layout), then strip files that are only needed at build time:
#   - *.a     static archives (~60% of space)
#   - cmake/  CMake integration files
#   - pkgconfig/ pkg-config metadata
# This is more reliable than ldd-chasing because ROCm libs dlopen() other
# libs at runtime (e.g. device-specific kernels) that don't appear in ldd.
RUN cp -a /opt/rocm/lib /staging/rocm-lib \
    && find /staging/rocm-lib \
         \( -name '*.a' \
         -o -path '*/cmake/*' \
         -o -path '*/pkgconfig/*' \
         \) -delete

# ── Collect rocm-smi (Python script + its Python bindings module) ────────────
# The script lives in /opt/rocm/bin; its Python module is version-dependent:
#   ROCm ≥ 5.7  → /opt/rocm/libexec/rocm_smi/
#   ROCm < 5.7  → /opt/rocm/share/rocm_smi/  or  /opt/rocm/lib/python*/
RUN mkdir -p /staging/rocm-smi/bin /staging/rocm-smi/module \
    && cp /opt/rocm/bin/rocm-smi /staging/rocm-smi/bin/ 2>/dev/null \
       || echo "WARN: rocm-smi binary not found" \
    && for dir in \
         /opt/rocm/libexec/rocm_smi \
         /opt/rocm/share/rocm_smi \
         $(find /opt/rocm/lib -maxdepth 3 -type d -name 'rocm_smi' 2>/dev/null | head -1); \
       do \
         if [ -d "$dir" ]; then \
           cp -r "$dir/." /staging/rocm-smi/module/; \
           echo "Staged rocm-smi module from: $dir"; \
           break; \
         fi; \
       done


# ─── Stage 2: GPU monitoring tools ───────────────────────────────────────────
FROM ubuntu:24.04 AS monitoring

ARG AMDGPU_TOP_VERSION=0.10.3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      radeontop \
    && rm -rf /var/lib/apt/lists/*

# amdgpu_top: statically linked musl binary — works on Wolfi without extra libs.
# Falls back gracefully; radeontop is always available via apt above.
RUN mkdir -p /staging/bin \
    && TARBALL="amdgpu_top-v${AMDGPU_TOP_VERSION}-x86_64-unknown-linux-musl.tar.gz" \
    && URL="https://github.com/Umio-Yasuno/amdgpu_top/releases/download/v${AMDGPU_TOP_VERSION}/${TARBALL}" \
    && echo "Fetching amdgpu_top ${AMDGPU_TOP_VERSION} ..." \
    && if curl -fsSL --retry 3 --retry-delay 2 -o /tmp/at.tar.gz "${URL}"; then \
         tar -xzf /tmp/at.tar.gz -C /tmp/ \
         && find /tmp -name 'amdgpu_top' -type f -not -path '*/\.*' | head -1 \
              | xargs -I{} install -Dm755 {} /staging/bin/amdgpu_top \
         && echo "amdgpu_top installed"; \
       else \
         echo "WARN: amdgpu_top download failed — radeontop will be used as fallback"; \
       fi

# Bundle radeontop: copy binary and the non-glibc shared libs it needs
# (libpciaccess, libxcb-dri2, libdrm) so they work on Wolfi's glibc
RUN install -Dm755 /usr/bin/radeontop /staging/bin/radeontop \
    && mkdir -p /staging/radeontop-libs \
    && ldd /usr/bin/radeontop \
       | awk '/=>/ && $3 !~ /^(\/lib\/x86_64|\/lib64)/ { print $3 }' \
       | xargs -r -I{} cp -L {} /staging/radeontop-libs/ \
    && echo "Bundled radeontop libs:" && ls /staging/radeontop-libs/


# ─── Stage 3: Wolfi runtime ──────────────────────────────────────────────────
FROM cgr.dev/chainguard/wolfi-base AS runtime

# Runtime APK deps:
#   libgomp   — OpenMP, required by llama.cpp
#   python3   — rocm-smi is a Python script
#   bash      — rocm-smi shebang + convenience
RUN apk add --no-cache \
      libgomp \
      python3 \
      bash

# ── ROCm runtime libraries ────────────────────────────────────────────────────
COPY --from=builder /staging/rocm-lib/ /opt/rocm/lib/
ENV LD_LIBRARY_PATH=/opt/rocm/lib

# ── llama.cpp binaries ────────────────────────────────────────────────────────
COPY --from=builder /staging/bin/ /usr/local/bin/

# ── rocm-smi ──────────────────────────────────────────────────────────────────
COPY --from=builder /staging/rocm-smi/bin/   /opt/rocm/bin/
COPY --from=builder /staging/rocm-smi/module/ /opt/rocm/libexec/rocm_smi/
# Make rocm-smi importable by the bundled script
ENV PYTHONPATH=/opt/rocm/libexec
RUN ln -sf /opt/rocm/bin/rocm-smi /usr/local/bin/rocm-smi 2>/dev/null || true

# ── amdgpu_top + radeontop ────────────────────────────────────────────────────
COPY --from=monitoring /staging/bin/           /usr/local/bin/
COPY --from=monitoring /staging/radeontop-libs/ /usr/local/lib/radeontop-libs/

# Wrapper: prefer amdgpu_top, fall through to radeontop
RUN printf '#!/usr/bin/env bash\n\
if command -v amdgpu_top &>/dev/null; then\n\
  exec amdgpu_top "$@"\n\
else\n\
  LD_LIBRARY_PATH=/usr/local/lib/radeontop-libs:${LD_LIBRARY_PATH} \\\n\
    exec radeontop "$@"\n\
fi\n' > /usr/local/bin/gpu-monitor \
    && chmod +x /usr/local/bin/gpu-monitor

# ── Sanity check ──────────────────────────────────────────────────────────────
RUN echo "=== llama.cpp binaries ===" \
    && ls -1 /usr/local/bin/llama-* \
    && echo "=== monitoring tools ===" \
    && ls -1 /usr/local/bin/rocm-smi /usr/local/bin/gpu-monitor \
                /usr/local/bin/amdgpu_top /usr/local/bin/radeontop 2>/dev/null \
    || true

WORKDIR /models

EXPOSE 8080

ENTRYPOINT ["llama-server"]
CMD []
