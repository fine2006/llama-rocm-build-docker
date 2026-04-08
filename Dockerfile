# =============================================================================
# llama.cpp ROCm Container - Optimized Build
# =============================================================================
# Build with:
#   podman build --build-arg RELEASE_ID=20260408-24115666439 \
#                --build-arg GFX_ARCH=gfx1152 \
#                --build-arg ENABLE_ROCWMMA_FATTN=ON \
#                --build-arg BUILD_WEBUI=ON \
#                --build-arg BUILD_LLAMA_SERVER=ON \
#                -t llama-cpp-rocm:latest .
# =============================================================================

# Stage 1: Build (Ubuntu + ROCm nightly)
FROM ubuntu:24.04 AS builder

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Build arguments
ARG RELEASE_ID=20260408-24115666439
ARG GFX_ARCH=gfx1152
ARG ENABLE_ROCWMMA_FATTN=ON
ARG BUILD_WEBUI=ON
ARG BUILD_ALL_TOOLS=ON

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    build-essential \
    cmake \
    gcc \
    g++ \
    python3 \
    python3-pip \
    python3-dev \
    libssl-dev \
    git \
    && apt-get purge -y --auto-remove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/*

# Add ROCm nightly repository
RUN tee /etc/apt/sources.list.d/rocm-nightly.list <<EOF
deb [trusted=yes] https://rocm.nightlies.amd.com/deb/${RELEASE_ID} stable main
EOF

# Install ROCm SDK for target GPU architecture
RUN apt-get update && apt-get install -y --no-install-recommends \
    amdrocm-core-sdk-${GFX_ARCH} \
    && apt-get purge -y --auto-remove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/*

# Set ROCm environment variables
ENV PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH \
    HIP_PATH=/opt/rocm \
    ROCM_PATH=/opt/rocm

WORKDIR /build

# Copy project files (excluding git data)
COPY . .

# Build llama.cpp with optimized CMake configuration
RUN HIPCXX="$(hipconfig -l)/clang" \
    HIP_PATH="$(hipconfig -R)" \
    cmake -S llama.cpp -B llama.cpp/build \
        -DGGML_HIP=ON \
        -DGGML_HIP_ROCWMMA_FATTN="${ENABLE_ROCWMMA_FATTN}" \
        -DGPU_TARGETS="${GFX_ARCH}" \
        -DGGML_CUDA=OFF \
        -DGGML_VULKAN=OFF \
        -DGGML_METAL=OFF \
        -DGGML_OPENCL=OFF \
        -DGGML_MUSA=OFF \
        -DGGML_CANN=OFF \
        -DGGML_ZENDNN=OFF \
        -DGGML_OPENVINO=OFF \
        -DGGML_WEBGPU=OFF \
        -DGGML_BACKEND_DL=OFF \
        -DGGML_CPU_ALL_VARIANTS=OFF \
        -DBUILD_SHARED_LIBS=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_TOOLS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_SERVER="ON" \
        -DLLAMA_BUILD_WEBUI="${BUILD_WEBUI}" \
    && if [ "${BUILD_ALL_TOOLS}" = "ON" ]; then \
            cmake --build llama.cpp/build --target llama-server llama-quantize llama-bench llama-perplexity llama-convert-pth llama-convert-hf-to-gguf llama-convert-lora-to-gguf -j$(nproc); \
        else \
            cmake --build llama.cpp/build --target llama-server -j$(nproc); \
        fi

# Copy built binaries
RUN if [ "${BUILD_ALL_TOOLS}" = "ON" ]; then \
        cp llama.cpp/build/bin/llama-server /usr/local/bin/ && \
        cp llama.cpp/build/bin/llama-quantize /usr/local/bin/ && \
        cp llama.cpp/build/bin/llama-bench /usr/local/bin/ && \
        cp llama.cpp/build/bin/llama-perplexity /usr/local/bin/ && \
        cp llama.cpp/build/bin/llama-convert-pth /usr/local/bin/ && \
        cp llama.cpp/build/bin/llama-convert-hf-to-gguf /usr/local/bin/ && \
        cp llama.cpp/build/bin/llama-convert-lora-to-gguf /usr/local/bin/ && \
        chmod +x /usr/local/bin/llama-*; \
    else \
        cp llama.cpp/build/bin/llama-server /usr/local/bin/ && \
        chmod +x /usr/local/bin/llama-server; \
    fi

# Clean up build artifacts
RUN rm -rf /build/llama.cpp/build

# Stage 2: Runtime (minimal Ubuntu + ROCm runtime)
FROM ubuntu:24.04 AS runtime

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install ROCm runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    curl \
    && apt-get purge -y --auto-remove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/*

# Set ROCm environment
ENV PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH \
    HIP_PATH=/opt/rocm \
    ROCM_PATH=/opt/rocm \
    LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

# Copy binaries from builder stage
COPY --from=builder /usr/local/bin/llama-server /usr/local/bin/
COPY --from=builder /usr/local/bin/llama-quantize /usr/local/bin/
COPY --from=builder /usr/local/bin/llama-bench /usr/local/bin/
COPY --from=builder /usr/local/bin/llama-perplexity /usr/local/bin/
COPY --from=builder /usr/local/bin/llama-convert-pth /usr/local/bin/
COPY --from=builder /usr/local/bin/llama-convert-hf-to-gguf /usr/local/bin/
COPY --from=builder /usr/local/bin/llama-convert-lora-to-gguf /usr/local/bin/

# Create working directory
WORKDIR /models

# Default command
ENTRYPOINT ["llama-server"]
CMD []
