/**
 * Device Runner Implementation
 *
 * This file implements the device execution utilities for launching and managing
 * AICPU and AICore kernels on Ascend devices.
 */

#include "devicerunner.h"
#include "../graph/graph.h"
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

/**
 * Read a file into a byte buffer
 *
 * @param path  Path to the file
 * @param buf   Buffer to store file contents
 * @return 0 on success, -1 on failure
 */
static int ReadFile(const std::string &path, std::vector<uint8_t> &buf) {
    std::ifstream fs(path, std::ios::binary | std::ios::ate);
    if (!fs.is_open()) {
        std::cerr << "无法打开内核文件: " << path << '\n';
        return -1;
    }
    std::streamsize size = fs.tellg();
    fs.seekg(0, std::ios::beg);
    buf.resize(static_cast<size_t>(size));
    if (!fs.read(reinterpret_cast<char *>(buf.data()), size)) {
        std::cerr << "读取内核文件失败: " << path << '\n';
        return -1;
    }
    return 0;
}

// =============================================================================
// KernelArgsHelper Implementation
// =============================================================================

int KernelArgsHelper::InitDeviceArgs(const DeviceArgs &hostDeviceArgs, MemoryAllocator& allocator) {
    allocator_ = &allocator;

    // Allocate device memory for deviceArgs
    if (args.deviceArgs == nullptr) {
        uint64_t deviceArgsSize = sizeof(DeviceArgs);
        void* deviceArgsDev = allocator_->Alloc(deviceArgsSize);
        if (deviceArgsDev == nullptr) {
            std::cerr << "Error: Alloc for deviceArgs failed\n";
            return -1;
        }
        args.deviceArgs = reinterpret_cast<int64_t *>(deviceArgsDev);
    }
    // Copy hostDeviceArgs to device memory via deviceArgs
    int rc =
        rtMemcpy(args.deviceArgs, sizeof(DeviceArgs), &hostDeviceArgs, sizeof(DeviceArgs), RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        std::cerr << "Error: rtMemcpy failed: " << rc << '\n';
        allocator_->Free(args.deviceArgs);
        args.deviceArgs = nullptr;
        return rc;
    }
    return 0;
}

int KernelArgsHelper::FinalizeDeviceArgs() {
    if (args.deviceArgs != nullptr && allocator_ != nullptr) {
        int rc = allocator_->Free(args.deviceArgs);
        args.deviceArgs = nullptr;
        return rc;
    }
    return 0;
}

int KernelArgsHelper::InitGraphArgs(const Graph& hostGraph, MemoryAllocator& allocator) {
    allocator_ = &allocator;

    if (args.graphArgs == nullptr) {
        uint64_t graphSize = sizeof(Graph);
        void* graphDev = allocator_->Alloc(graphSize);
        if (graphDev == nullptr) {
            std::cerr << "Error: Alloc for graphArgs failed\n";
            return -1;
        }
        args.graphArgs = reinterpret_cast<Graph*>(graphDev);
    }
    int rc = rtMemcpy(args.graphArgs, sizeof(Graph), &hostGraph, sizeof(Graph), RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        std::cerr << "Error: rtMemcpy for graph failed: " << rc << '\n';
        allocator_->Free(args.graphArgs);
        args.graphArgs = nullptr;
        return rc;
    }
    return 0;
}

int KernelArgsHelper::FinalizeGraphArgs() {
    if (args.graphArgs != nullptr && allocator_ != nullptr) {
        int rc = allocator_->Free(args.graphArgs);
        args.graphArgs = nullptr;
        return rc;
    }
    return 0;
}

// =============================================================================
// AicpuSoInfo Implementation
// =============================================================================

int AicpuSoInfo::Init(const std::string &soPath, MemoryAllocator& allocator) {
    allocator_ = &allocator;

    std::ifstream file(soPath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open " << soPath << '\n';
        return -1;
    }

    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(fileSize);
    file.read(buffer.data(), fileSize);

    void *dAicpuData = allocator_->Alloc(fileSize);
    if (dAicpuData == nullptr) {
        std::cerr << "Error: Alloc failed for AICPU SO\n";
        return -1;
    }

    int rc = rtMemcpy(dAicpuData, fileSize, buffer.data(), fileSize, RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        std::cerr << "Error: rtMemcpy failed: " << rc << '\n';
        allocator_->Free(dAicpuData);
        dAicpuData = nullptr;
        return rc;
    }

    aicpuSoBin = reinterpret_cast<uint64_t>(dAicpuData);
    aicpuSoLen = fileSize;
    return 0;
}

int AicpuSoInfo::Finalize() {
    if (aicpuSoBin != 0 && allocator_ != nullptr) {
        int rc = allocator_->Free(reinterpret_cast<void *>(aicpuSoBin));
        aicpuSoBin = 0;
        return rc;
    }
    return 0;
}

// =============================================================================
// DeviceRunner Implementation
// =============================================================================

DeviceRunner &DeviceRunner::Get() {
    static DeviceRunner runner;
    return runner;
}

int DeviceRunner::Init(int deviceId, int numCores, const std::string &aicpuSoPath,
                       const std::string &aicoreKernelPath) {
    if (initialized_) {
        std::cerr << "Error: DeviceRunner already initialized\n";
        return -1;
    }

    deviceId_ = deviceId;
    numCores_ = numCores;
    aicoreKernelPath_ = aicoreKernelPath;

    // Set device
    int rc = rtSetDevice(deviceId);
    if (rc != 0) {
        std::cerr << "Error: rtSetDevice(" << deviceId << ") failed: " << rc << '\n';
        return rc;
    }

    // Initialize memory allocator
    rc = memAlloc_.Init(deviceId);
    if (rc != 0) {
        std::cerr << "Error: MemoryAllocator::Init failed: " << rc << '\n';
        return rc;
    }

    // Create streams
    rc = rtStreamCreate(&streamAicpu_, 0);
    if (rc != 0) {
        std::cerr << "Error: rtStreamCreate (AICPU) failed: " << rc << '\n';
        return rc;
    }

    rc = rtStreamCreate(&streamAicore_, 0);
    if (rc != 0) {
        std::cerr << "Error: rtStreamCreate (AICore) failed: " << rc << '\n';
        rtStreamDestroy(streamAicpu_);
        streamAicpu_ = nullptr;
        return rc;
    }

    // Load AICPU SO
    rc = soInfo_.Init(aicpuSoPath, memAlloc_);
    if (rc != 0) {
        std::cerr << "Error: AicpuSoInfo::Init failed: " << rc << '\n';
        rtStreamDestroy(streamAicpu_);
        rtStreamDestroy(streamAicore_);
        streamAicpu_ = nullptr;
        streamAicore_ = nullptr;
        return rc;
    }

    // Initialize device args
    deviceArgs_.aicpuSoBin = soInfo_.aicpuSoBin;
    deviceArgs_.aicpuSoLen = soInfo_.aicpuSoLen;
    rc = kernelArgs_.InitDeviceArgs(deviceArgs_, memAlloc_);
    if (rc != 0) {
        std::cerr << "Error: InitDeviceArgs failed: " << rc << '\n';
        soInfo_.Finalize();
        rtStreamDestroy(streamAicpu_);
        rtStreamDestroy(streamAicore_);
        streamAicpu_ = nullptr;
        streamAicore_ = nullptr;
        return rc;
    }

    // Initialize handshake buffers
    hankArgs_.resize(numCores);
    for (int i = 0; i < numCores; i++) {
        hankArgs_[i].aicpu_ready = 0;
        hankArgs_[i].aicore_done = 0;
        hankArgs_[i].control = 0;
        hankArgs_[i].task = 0;
        hankArgs_[i].task_status = 0;
    }

    // Allocate and copy handshake to device
    size_t total_size = sizeof(Handshake) * numCores;
    void *hankDev = memAlloc_.Alloc(total_size);
    if (hankDev == nullptr) {
        std::cerr << "Error: Alloc for handshake failed\n";
        kernelArgs_.FinalizeDeviceArgs();
        soInfo_.Finalize();
        rtStreamDestroy(streamAicpu_);
        rtStreamDestroy(streamAicore_);
        streamAicpu_ = nullptr;
        streamAicore_ = nullptr;
        return -1;
    }

    rc = rtMemcpy(hankDev, total_size, hankArgs_.data(), total_size, RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        std::cerr << "Error: rtMemcpy for handshake failed: " << rc << '\n';
        memAlloc_.Free(hankDev);
        kernelArgs_.FinalizeDeviceArgs();
        soInfo_.Finalize();
        rtStreamDestroy(streamAicpu_);
        rtStreamDestroy(streamAicore_);
        streamAicpu_ = nullptr;
        streamAicore_ = nullptr;
        return rc;
    }

    kernelArgs_.args.hankArgs = reinterpret_cast<int64_t *>(hankDev);
    kernelArgs_.args.core_num = numCores;

    initialized_ = true;
    std::cout << "DeviceRunner initialized: device=" << deviceId << ", cores=" << numCores << '\n';
    return 0;
}

void* DeviceRunner::AllocateTensor(size_t bytes) {
    if (!initialized_) {
        std::cerr << "Error: DeviceRunner not initialized\n";
        return nullptr;
    }

    return memAlloc_.Alloc(bytes);
}

void DeviceRunner::FreeTensor(void* devPtr) {
    if (devPtr != nullptr) {
        memAlloc_.Free(devPtr);
    }
}

int DeviceRunner::CopyToDevice(void* devPtr, const void* hostPtr, size_t bytes) {
    if (!initialized_) {
        std::cerr << "Error: DeviceRunner not initialized\n";
        return -1;
    }
    return rtMemcpy(devPtr, bytes, hostPtr, bytes, RT_MEMCPY_HOST_TO_DEVICE);
}

int DeviceRunner::CopyFromDevice(void* hostPtr, const void* devPtr, size_t bytes) {
    if (!initialized_) {
        std::cerr << "Error: DeviceRunner not initialized\n";
        return -1;
    }
    return rtMemcpy(hostPtr, bytes, devPtr, bytes, RT_MEMCPY_DEVICE_TO_HOST);
}

int DeviceRunner::Run(const Graph& graph, int launchAicpuNum) {
    if (!initialized_) {
        std::cerr << "Error: DeviceRunner not initialized\n";
        return -1;
    }

    // Initialize graph args
    int rc = kernelArgs_.InitGraphArgs(graph, memAlloc_);
    if (rc != 0) {
        std::cerr << "Error: InitGraphArgs failed: " << rc << '\n';
        return rc;
    }

    // Launch AICPU init kernel
    rc = LaunchAiCpuKernel(streamAicpu_, &kernelArgs_.args, "DynTileFwkKernelServerInit", 1);
    if (rc != 0) {
        std::cerr << "Error: LaunchAiCpuKernel (init) failed: " << rc << '\n';
        kernelArgs_.FinalizeGraphArgs();
        return rc;
    }

    // Launch AICPU main kernel
    rc = LaunchAiCpuKernel(streamAicpu_, &kernelArgs_.args, "DynTileFwkKernelServer", launchAicpuNum);
    if (rc != 0) {
        std::cerr << "Error: LaunchAiCpuKernel (main) failed: " << rc << '\n';
        kernelArgs_.FinalizeGraphArgs();
        return rc;
    }

    // Launch AICore kernel
    rc = LauncherAicoreKernel(streamAicore_, &kernelArgs_.args);
    if (rc != 0) {
        std::cerr << "Error: LauncherAicoreKernel failed: " << rc << '\n';
        kernelArgs_.FinalizeGraphArgs();
        return rc;
    }

    // Synchronize streams
    rc = rtStreamSynchronize(streamAicpu_);
    if (rc != 0) {
        std::cerr << "Error: rtStreamSynchronize (AICPU) failed: " << rc << '\n';
        kernelArgs_.FinalizeGraphArgs();
        return rc;
    }

    rc = rtStreamSynchronize(streamAicore_);
    if (rc != 0) {
        std::cerr << "Error: rtStreamSynchronize (AICore) failed: " << rc << '\n';
        kernelArgs_.FinalizeGraphArgs();
        return rc;
    }

    // Cleanup graph args
    kernelArgs_.FinalizeGraphArgs();

    return 0;
}

void DeviceRunner::PrintHandshakeResults() {
    if (!initialized_ || hankArgs_.empty()) {
        return;
    }

    size_t total_size = sizeof(Handshake) * numCores_;
    rtMemcpy(hankArgs_.data(), total_size, kernelArgs_.args.hankArgs, total_size, RT_MEMCPY_DEVICE_TO_HOST);

    std::cout << "Handshake results for " << numCores_ << " cores:" << std::endl;
    for (int i = 0; i < numCores_; i++) {
        std::cout << "  Core " << i << ": aicore_done=" << hankArgs_[i].aicore_done
                  << " aicpu_ready=" << hankArgs_[i].aicpu_ready
                  << " control=" << hankArgs_[i].control
                  << " task=" << hankArgs_[i].task << std::endl;
    }
}

int DeviceRunner::Finalize() {
    if (!initialized_) {
        return 0;
    }

    // Cleanup kernel args (deviceArgs, graphArgs if any)
    kernelArgs_.FinalizeDeviceArgs();

    // Cleanup AICPU SO
    soInfo_.Finalize();

    // Destroy streams
    if (streamAicpu_ != nullptr) {
        rtStreamDestroy(streamAicpu_);
        streamAicpu_ = nullptr;
    }
    if (streamAicore_ != nullptr) {
        rtStreamDestroy(streamAicore_);
        streamAicore_ = nullptr;
    }

    // Free all remaining allocations (including handshake buffer)
    memAlloc_.Finalize();

    initialized_ = false;
    deviceId_ = -1;
    numCores_ = 0;
    aicoreKernelPath_.clear();
    hankArgs_.clear();

    std::cout << "DeviceRunner finalized\n";
    return 0;
}

int DeviceRunner::LaunchAiCpuKernel(rtStream_t stream, KernelArgs *kArgs, const char *kernelName, int aicpuNum) {
    struct Args {
        KernelArgs kArgs;
        char kernelName[32];
        const char soName[32] = {"libaicpu_extend_kernels.so"};
        const char opName[32] = {""};
    } args;

    args.kArgs = *kArgs;
    std::strncpy(args.kernelName, kernelName, sizeof(args.kernelName) - 1);
    args.kernelName[sizeof(args.kernelName) - 1] = '\0';

    rtAicpuArgsEx_t rtArgs;
    std::memset(&rtArgs, 0, sizeof(rtArgs));
    rtArgs.args = &args;
    rtArgs.argsSize = sizeof(args);
    rtArgs.kernelNameAddrOffset = offsetof(struct Args, kernelName);
    rtArgs.soNameAddrOffset = offsetof(struct Args, soName);

    return rtAicpuKernelLaunchExWithArgs(rtKernelType_t::KERNEL_TYPE_AICPU_KFC, "AST_DYN_AICPU", aicpuNum, &rtArgs,
                                         nullptr, stream, 0);
}

int DeviceRunner::LauncherAicoreKernel(rtStream_t stream, KernelArgs *kernelArgs) {
    const std::string binPath = aicoreKernelPath_;
    std::vector<uint8_t> bin;
    if (ReadFile(binPath, bin) != 0) {
        return -1;
    }

    size_t binSize = bin.size();
    const void *binData = bin.data();

    rtDevBinary_t binary;
    std::memset(&binary, 0, sizeof(binary));
    binary.magic = RT_DEV_BINARY_MAGIC_ELF;
    binary.version = 0;
    binary.data = binData;
    binary.length = binSize;
    void *binHandle = nullptr;
    int rc = rtRegisterAllKernel(&binary, &binHandle);
    if (rc != RT_ERROR_NONE) {
        std::cerr << "rtRegisterAllKernel失败: " << rc << '\n';
        return rc;
    }

    struct Args {
        int64_t *hankArgs;
    };
    Args args = {kernelArgs->hankArgs};
    rtArgsEx_t rtArgs;
    std::memset(&rtArgs, 0, sizeof(rtArgs));
    rtArgs.args = &args;
    rtArgs.argsSize = sizeof(args);

    rtTaskCfgInfo_t cfg = {};
    cfg.schemMode = RT_SCHEM_MODE_BATCH;

    rc = rtKernelLaunchWithHandleV2(binHandle, 0, 1, &rtArgs, nullptr, stream, &cfg);
    if (rc != RT_ERROR_NONE) {
        std::cerr << "rtKernelLaunchWithHandleV2失败: " << rc << '\n';
        return rc;
    }

    return rc;
}

