# =============================================================================
# llama.cpp — ROCm-only build, Wolfi runtime, self-contained
#
# Build args:
#   ROCM_VERSION      ROCm version from nightly builds (e.g., 7.11.0a20260408)
#   ROCM_RELEASE_TYPE Release type: nightlies, prereleases, devreleases, stable
#                      (default: nightlies)
#   GFX_ARCH          GPU target, e.g. gfx1152, gfx1100, gfx906
#   ENABLE_ROCWMMA_FATTN  rocWMMA flash attention (ON | OFF)
#   BUILD_ALL         OFF = llama-server only
#                     ON  = + llama-{quantize,bench,perplexity}
#
# Examples:
#   # Minimal — server only, nightly ROCm
#   podman build \
#     --build-arg ROCM_VERSION=7.11.0a20260408 \
#     --build-arg GFX_ARCH=gfx1152 \
#     -t llama-rocm:server .
#
#   # Full toolkit, stable ROCm
#   podman build \
#     --build-arg ROCM_VERSION=7.10.0 \
#     --build-arg ROCM_RELEASE_TYPE=stable \
#     --build-arg GFX_ARCH=gfx1152 \
#     --build-arg BUILD_ALL=ON \
#     -t llama-rocm:full .
# =============================================================================


# ─── Stage 1: llama.cpp builder (Fedora + ROCm from tarball) ────────────────
FROM fedora:latest AS builder

ARG ROCM_VERSION=7.13.0a20260408
ARG ROCM_RELEASE_TYPE=nightlies
ARG GFX_ARCH=gfx1152
ARG ENABLE_ROCWMMA_FATTN=ON
ARG BUILD_ALL=OFF

# Build toolchain
RUN dnf install -y --setopt=install_weak_deps=False \
      ca-certificates \
      curl \
      gcc \
      gcc-c++ \
      make \
      cmake \
      ninja-build \
      python3 \
      python3-pip \
      python3-devel \
      openssl-devel \
      git \
      tar \
      gzip \
      elfutils-libelf \
      numactl-libs \
      libunwind-devel \
      ncurses-libs \
      perl \
      file \
      kmod \
    && dnf clean all

# Download and extract ROCm tarball from TheRock distribution
# TheRock provides unified tarballs with all components (LLVM, HIP, device libs, etc.)
# URL format: https://rocm.{RELEASE_TYPE}.amd.com/tarball/therock-dist-linux-{AMDGPU_FAMILY}-{VERSION}.tar.gz
RUN mkdir -p /tmp/rocm-install && cd /tmp/rocm-install \
    && echo "Downloading ROCm ${ROCM_VERSION} from ${ROCM_RELEASE_TYPE}..." \
    && VERSION_ENCODED="${ROCM_VERSION//+/%2B}" \
    && if [ "${ROCM_RELEASE_TYPE}" = "stable" ]; then \
         ROCM_TARBALL_URL="https://repo.amd.com/rocm/tarball/therock-dist-linux-${GFX_ARCH}-${VERSION_ENCODED}.tar.gz"; \
       else \
         ROCM_TARBALL_URL="https://rocm.${ROCM_RELEASE_TYPE}.amd.com/tarball/therock-dist-linux-${GFX_ARCH}-${VERSION_ENCODED}.tar.gz"; \
       fi \
    && echo "URL: ${ROCM_TARBALL_URL}" \
    && curl -fsSL --retry 3 --retry-delay 2 -o rocm.tar.gz "${ROCM_TARBALL_URL}" \
    && tar -xzf rocm.tar.gz \
    && rm rocm.tar.gz \
    && mkdir -p /opt/rocm-${ROCM_VERSION} \
    && mv * /opt/rocm-${ROCM_VERSION}/ 2>/dev/null || true \
    && ln -sfn /opt/rocm-${ROCM_VERSION} /opt/rocm \
    && echo "ROCm ${ROCM_VERSION} installed at /opt/rocm"

ENV ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH \
    LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# ── Validate ROCm install ──────────────────────────────────────────────────
RUN echo "=== ROCm Installation Summary ===" \
    && echo "ROCM_PATH: $ROCM_PATH" \
    && hipconfig --full \
    && echo "=== Checking for HIP compiler ===" \
    && test -x "$ROCM_PATH/bin/hipcc" && echo "hipcc found at $ROCM_PATH/bin/hipcc" || echo "hipcc not found" \
    && echo "=== Checking for device libraries ===" \
    && if [ -d "$ROCM_PATH/lib/bitcode" ]; then \
         BITCODE_COUNT=$(find $ROCM_PATH/lib/bitcode -name '*.bc' 2>/dev/null | wc -l); \
         echo "Device libraries found: $BITCODE_COUNT bitcode files"; \
         ls $ROCM_PATH/lib/bitcode/ 2>/dev/null | head -10; \
       else \
         echo "WARNING: Device libraries not found at $ROCM_PATH/lib/bitcode"; \
       fi \
    && echo "=== HIP CMake files ===" \
    && find $ROCM_PATH/lib/cmake -name "hip-config.cmake" -o -name "HIPConfig.cmake" 2>/dev/null | head -3 \
    && echo "=== ROCm installation validated ===" 

WORKDIR /build
COPY . .

# ── Confirm the llama.cpp submodule was actually checked out ─────────────────
RUN test -f llama.cpp/CMakeLists.txt \
    || { printf '\nERROR: llama.cpp/CMakeLists.txt not found.\n'; \
         printf 'The llama.cpp submodule is missing or empty.\n'; \
         printf 'Ensure submodules: true is set in the checkout step and that\n'; \
         printf 'the submodule is initialised: git submodule update --init --recursive\n\n'; \
         exit 1; }

# ── Configure and build ────────────────────────────────────────────────────
RUN cmake -S llama.cpp -B llama.cpp/build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=hipcc \
      -DCMAKE_CXX_COMPILER=hipcc \
      -DHIP_CXX_COMPILER=hipcc \
      -DCMAKE_PREFIX_PATH=$ROCM_PATH \
      -DCMAKE_CXX_FLAGS="-D__HIP_PLATFORM_AMD__" \
      -DCMAKE_HIP_FLAGS="-D__HIP_PLATFORM_AMD__ --offload-arch=${GFX_ARCH}" \
      -DGGML_CPU_ALL_VARIANTS=OFF \
      -DGGML_NATIVE=OFF \
      -DGGML_CUDA=OFF \
      -DGGML_METAL=OFF \
      -DGGML_VULKAN=OFF \
      -DGGML_HIP=ON \
      -DGGML_HIP_ROCWMMA_FATTN="${ENABLE_ROCWMMA_FATTN}" \
      -DGPU_TARGETS="${GFX_ARCH}" \
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

# ── Collect llama.cpp build libraries (needed by binaries due to RPATH) ──────
# The llama-server binary has RPATH=/build/llama.cpp/build/bin, so we need
# to include any .so files from there
RUN mkdir -p /staging/llama-build-lib \
    && find llama.cpp/build/bin -name "*.so*" -exec cp -L {} /staging/llama-build-lib/ \; 2>/dev/null || true \
    && find llama.cpp/build -name "*.so*" -path "*/lib/*" -exec cp -L {} /staging/llama-build-lib/ \; 2>/dev/null || true \
    && echo "Collected $(find /staging/llama-build-lib -name '*.so*' | wc -l) libraries from llama.cpp build"

# ── Collect ROCm runtime libraries ─────────────────────────────────────────
# Copy the entire /opt/rocm/lib tree (preserving symlinks), then remove
# build-time-only files: *.a (static archives), cmake/, pkgconfig/
RUN cp -a $ROCM_PATH/lib /staging/rocm-lib \
    && find /staging/rocm-lib \
         \( -name '*.a' \
         -o -path '*/cmake/*' \
         -o -path '*/pkgconfig/*' \
         \) -delete

# ── Collect ROCm bin directory (contains runtime binaries and libs) ──────────
RUN mkdir -p /staging/rocm-bin && cp -a $ROCM_PATH/bin/* /staging/rocm-bin/ 2>/dev/null || true

# ── Collect ROCm include directory (needed for some runtime operations) ─────
RUN mkdir -p /staging/rocm-include && cp -a $ROCM_PATH/include /staging/rocm-include/rocm-include 2>/dev/null || true

# ── Collect rocm-smi (Python script + bindings) ────────────────────────────
RUN mkdir -p /staging/rocm-smi/bin /staging/rocm-smi/module \
    && cp $ROCM_PATH/bin/rocm-smi /staging/rocm-smi/bin/ 2>/dev/null \
       || echo "WARN: rocm-smi binary not found" \
    && for dir in \
         $ROCM_PATH/libexec/rocm_smi \
         $ROCM_PATH/share/rocm_smi \
         $(find $ROCM_PATH/lib -maxdepth 3 -type d -name 'rocm_smi' 2>/dev/null | head -1); \
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


# ─── Stage 3: Ubuntu runtime ─────────────────────────────────────────────────
FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Runtime APK deps:
#   libomp-dev       — LLVM OpenMP library (needed for llama.cpp binaries)
#   libatomic1       — Atomic operations library
#   libgomp1         — GNU OpenMP, required by llama.cpp
#   python3          — rocm-smi is a Python script
#   bash             — rocm-smi shebang + convenience
RUN apt-get update && apt-get install -y --no-install-recommends \
      libomp-dev \
      libatomic1 \
      libgomp1 \
      python3 \
      bash \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── ROCm runtime libraries ────────────────────────────────────────────────────
COPY --from=builder /staging/rocm-lib/ /opt/rocm/lib/
COPY --from=builder /staging/rocm-bin/ /opt/rocm/bin/
COPY --from=builder /staging/rocm-include/ /opt/rocm/include/
# ── llama.cpp build libraries (for RPATH resolution) ────────────────────────
COPY --from=builder /staging/llama-build-lib/ /build/llama.cpp/build/bin/
ENV LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/bin:/opt/rocm/lib/llvm/lib:/build/llama.cpp/build/bin:$LD_LIBRARY_PATH \
    PATH=/opt/rocm/bin:/opt/rocm/lib/llvm/bin:$PATH

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
