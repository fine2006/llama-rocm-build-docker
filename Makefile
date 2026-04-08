# Makefile for building llama.cpp with HIP/ROCm support

LLAMA_DIR = llama.cpp
BUILD_DIR = $(LLAMA_DIR)/build

# GPU architecture – override at command line (e.g. make GFX_ARCH=gfx1030)
GFX_ARCH ?= gfx1152

# Enable rocWMMA for RDNA3/CDNA (ON/OFF)
ENABLE_ROCWMMA_FATTN ?= ON

# Build llama-server only (ON/OFF)
BUILD_LLAMA_SERVER ?= ON

# Build WebUI (ON/OFF)
BUILD_WEBUI ?= ON

.PHONY: all clean

all: $(BUILD_DIR)/bin/llama-server

$(BUILD_DIR)/bin/llama-server:
	@echo "Building llama.cpp for $(GFX_ARCH) with HIP support..."
	@mkdir -p $(BUILD_DIR)
	@# Locate the ROCm device library path (required for some ROCm setups)
	@HIP_DEVICE_LIB_PATH=$$(find /opt/rocm -name 'oclc_abi_version_400.bc' -printf '%h' -quit 2>/dev/null); \
	if [ -n "$$HIP_DEVICE_LIB_PATH" ]; then \
		export HIP_DEVICE_LIB_PATH; \
	fi; \
	HIPCXX="$$(hipconfig -l)/clang" \
	HIP_PATH="$$(hipconfig -R)" \
	cmake -S $(LLAMA_DIR) -B $(BUILD_DIR) \
		-DGGML_HIP=ON \
		-DGGML_HIP_ROCWMMA_FATTN=$(ENABLE_ROCWMMA_FATTN) \
		-DGPU_TARGETS=$(GFX_ARCH) \
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
		-DLLAMA_BUILD_SERVER=$(BUILD_LLAMA_SERVER) \
		-DLLAMA_BUILD_WEBUI=$(BUILD_WEBUI)
	cmake --build $(BUILD_DIR) --config Release -- -j$(nproc)

clean:
	rm -rf $(BUILD_DIR)
