/**
 * Device Runner Implementation
 *
 * This file implements the device execution utilities for launching and managing
 * AICPU and AICore kernels on Ascend devices.
 */

#include "devicerunner.h"
#include "binary_loader.h"
#include "kernel_compiler.h"
#include "graph.h"
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

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

int AicpuSoInfo::Init(const std::vector<uint8_t>& aicpuSoBinary, MemoryAllocator& allocator) {
    allocator_ = &allocator;

    if (aicpuSoBinary.empty()) {
        std::cerr << "Error: AICPU binary is empty\n";
        return -1;
    }

    size_t fileSize = aicpuSoBinary.size();
    void *dAicpuData = allocator_->Alloc(fileSize);
    if (dAicpuData == nullptr) {
        std::cerr << "Error: Alloc failed for AICPU SO\n";
        return -1;
    }

    int rc = rtMemcpy(dAicpuData, fileSize, aicpuSoBinary.data(), fileSize, RT_MEMCPY_HOST_TO_DEVICE);
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

int DeviceRunner::Init(int deviceId, const std::vector<uint8_t>& aicpuSoBinary,
                       const std::vector<uint8_t>& aicoreKernelBinary, const std::string& ptoIsaRoot) {
    if (initialized_) {
        std::cerr << "Error: DeviceRunner already initialized\n";
        return -1;
    }

    deviceId_ = deviceId;
    aicoreKernelBinary_ = aicoreKernelBinary;
    ptoIsaRoot_ = ptoIsaRoot;

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
    rc = soInfo_.Init(aicpuSoBinary, memAlloc_);
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
    // NOTE: Moved to Run() to initialize per Graph instance with numCores parameter

    // NOTE: Kernel registration and loading moved to runtime compilation
    // Users should call Init() with ptoIsaRoot, then compile kernels:
    // Example:
    //   runner.Init(0, aicpuBinary, aicoreBinary, "/path/to/pto-isa");
    //   runner.CompileAndLoadKernel(0, "./aicore/kernels/kernel_add.cpp", 1);
    //   runner.CompileAndLoadKernel(1, "./aicore/kernels/kernel_add_scalar.cpp", 1);
    //   runner.CompileAndLoadKernel(2, "./aicore/kernels/kernel_mul.cpp", 1);

    initialized_ = true;
    std::cout << "DeviceRunner initialized: device=" << deviceId << '\n';
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

int DeviceRunner::Run(Graph& graph, int numCores, int launchAicpuNum) {
    if (!initialized_) {
        std::cerr << "Error: DeviceRunner not initialized\n";
        return -1;
    }

    // Initialize handshake buffers in graph
    if (numCores > GRAPH_MAX_WORKER) {
        std::cerr << "Error: numCores (" << numCores << ") exceeds GRAPH_MAX_WORKER (" << GRAPH_MAX_WORKER << ")\n";
        return -1;
    }

    graph.worker_count = numCores;

    // Calculate number of AIC cores (1/3 of total)
    int numAic = (numCores + 2) / 3;  // Round up for 1/3

    for (int i = 0; i < numCores; i++) {
        graph.workers[i].aicpu_ready = 0;
        graph.workers[i].aicore_done = 0;
        graph.workers[i].control = 0;
        graph.workers[i].task = 0;
        graph.workers[i].task_status = 0;
        // Set core type: first 1/3 are AIC (0), remaining 2/3 are AIV (1)
        graph.workers[i].core_type = (i < numAic) ? 0 : 1;
    }

    // Allocate and copy handshake to device
    size_t total_size = sizeof(Handshake) * numCores;
    void *hankDev = memAlloc_.Alloc(total_size);
    if (hankDev == nullptr) {
        std::cerr << "Error: Alloc for handshake failed\n";
        return -1;
    }

    int rc = rtMemcpy(hankDev, total_size, graph.workers, total_size, RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        std::cerr << "Error: rtMemcpy for handshake failed: " << rc << '\n';
        memAlloc_.Free(hankDev);
        return rc;
    }

    kernelArgs_.args.hankArgs = reinterpret_cast<int64_t *>(hankDev);
    kernelArgs_.args.core_num = numCores;

    // Create a mutable reference to the graph for setting functionBinAddr
    // (no longer need to create a copy since graph is already non-const)

    // Set functionBinAddr for all tasks (NEW - Runtime function pointer dispatch)
    std::cout << "\n=== Setting functionBinAddr for Tasks ===" << '\n';
    for (int i = 0; i < graph.get_task_count(); i++) {
        Task* task = graph.get_task(i);
        if (task != nullptr) {
            uint64_t addr = GetFunctionBinAddr(task->func_id);
            task->functionBinAddr = addr;
            std::cout << "  Task " << i << " (func_id=" << task->func_id
                      << ") -> functionBinAddr=0x" << std::hex << addr << std::dec << '\n';
        }
    }
    std::cout << '\n';

    // Initialize graph args
    rc = kernelArgs_.InitGraphArgs(graph, memAlloc_);
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

void DeviceRunner::PrintHandshakeResults(Graph& graph) {
    if (!initialized_ || graph.worker_count == 0) {
        return;
    }

    size_t total_size = sizeof(Handshake) * graph.worker_count;
    rtMemcpy(graph.workers, total_size, kernelArgs_.args.hankArgs, total_size, RT_MEMCPY_DEVICE_TO_HOST);

    std::cout << "Handshake results for " << graph.worker_count << " cores:" << std::endl;
    for (int i = 0; i < graph.worker_count; i++) {
        std::cout << "  Core " << i << ": aicore_done=" << graph.workers[i].aicore_done
                  << " aicpu_ready=" << graph.workers[i].aicpu_ready
                  << " control=" << graph.workers[i].control
                  << " task=" << graph.workers[i].task << std::endl;
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

    // Cleanup kernel binary cache (NEW)
    if (binCache_ != nullptr) {
        delete[] reinterpret_cast<uint8_t*>(binCache_);
        binCache_ = nullptr;
    }
    funcIdToAddr_.clear();
    funcIdToBinPath_.clear();
    binGmAddr_ = nullptr;  // Will be freed by memAlloc_.Finalize()

    // Destroy streams
    if (streamAicpu_ != nullptr) {
        rtStreamDestroy(streamAicpu_);
        streamAicpu_ = nullptr;
    }
    if (streamAicore_ != nullptr) {
        rtStreamDestroy(streamAicore_);
        streamAicore_ = nullptr;
    }

    // Free all remaining allocations (including handshake buffer and binGmAddr)
    memAlloc_.Finalize();

    initialized_ = false;
    deviceId_ = -1;
    aicoreKernelBinary_.clear();

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
    if (aicoreKernelBinary_.empty()) {
        std::cerr << "Error: AICore kernel binary is empty\n";
        return -1;
    }

    size_t binSize = aicoreKernelBinary_.size();
    const void *binData = aicoreKernelBinary_.data();

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

// =============================================================================
// Kernel Binary Loading Implementation (NEW - Runtime Function Pointer Dispatch)
// =============================================================================

void DeviceRunner::RegisterKernel(int funcId, const std::string& binPath) {
    funcIdToBinPath_[funcId] = binPath;
    std::cout << "Registered kernel: func_id=" << funcId << " -> " << binPath << '\n';
}

int DeviceRunner::LoadKernelsToDevice() {
    if (funcIdToBinPath_.empty()) {
        std::cerr << "Error: No kernels registered. Call RegisterKernel() first.\n";
        return -1;
    }

    std::cout << "\n=== Loading Kernels to Device ===" << '\n';
    std::cout << "Number of kernels: " << funcIdToBinPath_.size() << '\n';

    // Step 1: Load all kernel binaries (extract .text sections)
    std::map<int, std::vector<uint8_t>> binaries;
    for (const auto& pair : funcIdToBinPath_) {
        int funcId = pair.first;
        const std::string& binPath = pair.second;

        std::vector<uint8_t> binData = LoadBinData(binPath);
        if (binData.empty()) {
            std::cerr << "Error: Failed to load binary for func_id=" << funcId << '\n';
            return -1;
        }

        binaries[funcId] = std::move(binData);
        std::cout << "  func_id=" << funcId << " -> " << binaries[funcId].size() << " bytes\n";
    }

    // Step 2: Calculate cache size
    uint64_t numKernels = binaries.size();
    uint64_t headerSize = sizeof(CoreFunctionBinCache) + numKernels * sizeof(uint64_t);
    uint64_t binaryDataSize = 0;

    for (const auto& pair : binaries) {
        binaryDataSize += sizeof(uint64_t) + pair.second.size();  // size field + data
    }

    uint64_t totalSize = headerSize + binaryDataSize;
    std::cout << "Cache size: " << totalSize << " bytes (header: " << headerSize
              << ", data: " << binaryDataSize << ")\n";

    // Step 3: Build cache structure in host memory
    uint8_t* hostBuf = new uint8_t[totalSize];
    std::memset(hostBuf, 0, totalSize);

    binCache_ = reinterpret_cast<CoreFunctionBinCache*>(hostBuf);
    binCache_->dataSize = binaryDataSize;
    binCache_->numKernels = numKernels;

    // Fill offset array and copy binaries
    uint64_t* offsets = binCache_->GetOffsets();
    uint8_t* dataPtr = binCache_->GetBinaryData();
    uint64_t currentOffset = 0;

    size_t index = 0;
    for (const auto& pair : binaries) {
        // Store offset
        offsets[index] = currentOffset;

        // Write CoreFunctionBin at this offset
        CoreFunctionBin* funcBin = reinterpret_cast<CoreFunctionBin*>(dataPtr + currentOffset);
        funcBin->size = pair.second.size();
        std::memcpy(funcBin->data, pair.second.data(), funcBin->size);

        std::cout << "  Kernel " << index << " (func_id=" << pair.first
                  << "): offset=" << currentOffset << ", size=" << funcBin->size << '\n';

        currentOffset += sizeof(uint64_t) + funcBin->size;
        index++;
    }

    // Step 4: Allocate device GM memory
    void* gmAddr = memAlloc_.Alloc(totalSize);
    if (gmAddr == nullptr) {
        std::cerr << "Error: Failed to allocate device GM memory for kernel cache\n";
        delete[] hostBuf;
        binCache_ = nullptr;
        return -1;
    }

    binGmAddr_ = gmAddr;
    std::cout << "Allocated device GM memory: " << gmAddr << " (" << totalSize << " bytes)\n";

    // Step 5: Copy cache to device
    int rc = rtMemcpy(binGmAddr_, totalSize, hostBuf, totalSize, RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        std::cerr << "Error: rtMemcpy to device failed: " << rc << '\n';
        memAlloc_.Free(binGmAddr_);
        binGmAddr_ = nullptr;
        delete[] hostBuf;
        binCache_ = nullptr;
        return rc;
    }

    // Step 6: Calculate functionBinAddr for each kernel
    uint64_t gmBase = reinterpret_cast<uint64_t>(binGmAddr_);
    uint64_t dataOffset = headerSize;  // Offset to start of binary data

    index = 0;
    for (const auto& pair : binaries) {
        int funcId = pair.first;
        uint64_t offset = offsets[index];

        // functionBinAddr = GM base + header + offset + sizeof(size field)
        uint64_t functionBinAddr = gmBase + dataOffset + offset + sizeof(uint64_t);

        funcIdToAddr_[funcId] = functionBinAddr;

        std::cout << "  func_id=" << funcId << " -> functionBinAddr=0x"
                  << std::hex << functionBinAddr << std::dec << '\n';

        index++;
    }

    std::cout << "=== Kernel Loading Complete ===\n\n";

    // Keep hostBuf for now (will be freed in Finalize)
    return 0;
}

uint64_t DeviceRunner::GetFunctionBinAddr(int funcId) {
    auto it = funcIdToAddr_.find(funcId);
    if (it == funcIdToAddr_.end()) {
        std::cerr << "Warning: functionBinAddr not found for func_id=" << funcId << '\n';
        return 0;
    }
    return it->second;
}

// =============================================================================
// Runtime Kernel Compilation Implementation
// =============================================================================

int DeviceRunner::CompileAndLoadKernel(int funcId,
                                       const std::string& sourcePath,
                                       int coreType) {
    if (!initialized_) {
        std::cerr << "Error: DeviceRunner not initialized. Call Init() first.\n";
        return -1;
    }

    const char* coreTypeName = (coreType == 1) ? "AIV" : "AIC";
    std::cout << "\n=== Compiling and Loading Kernel (Runtime, " << coreTypeName << ") ===" << '\n';
    std::cout << "func_id=" << funcId << ", source=" << sourcePath << '\n';

    // Step 1: Compile the kernel source
    std::string outputPath;
    std::string errorMsg;
    int rc = KernelCompiler::CompileKernel(sourcePath, ptoIsaRoot_, coreType, outputPath, errorMsg);
    if (rc != 0) {
        std::cerr << "Error: Kernel compilation failed: " << errorMsg << '\n';
        return -1;
    }

    std::cout << "Compiled to: " << outputPath << '\n';

    // Step 2: Register the kernel
    RegisterKernel(funcId, outputPath);

    // Step 3: Load the kernel to device
    rc = LoadSingleKernelToDevice(funcId, outputPath);
    if (rc != 0) {
        std::cerr << "Error: Failed to load kernel to device\n";
        return -1;
    }

    std::cout << "=== Kernel Compilation and Loading Complete ===" << '\n';
    std::cout << "  func_id=" << funcId << " -> functionBinAddr=0x"
              << std::hex << GetFunctionBinAddr(funcId) << std::dec << '\n' << '\n';

    return 0;
}

int DeviceRunner::LoadSingleKernelToDevice(int funcId, const std::string& binPath) {
    if (!initialized_) {
        std::cerr << "Error: DeviceRunner not initialized\n";
        return -1;
    }

    std::cout << "Loading kernel to device: func_id=" << funcId << ", path=" << binPath << '\n';

    // Step 1: Load binary data (extract .text section)
    std::vector<uint8_t> binData = LoadBinData(binPath);
    if (binData.empty()) {
        std::cerr << "Error: Failed to load binary data from " << binPath << '\n';
        return -1;
    }

    std::cout << "Loaded binary: " << binData.size() << " bytes\n";

    // Step 2: Allocate device GM memory for this kernel
    uint64_t binSize = binData.size();
    uint64_t allocSize = sizeof(uint64_t) + binSize;  // size field + data

    void* gmAddr = memAlloc_.Alloc(allocSize);
    if (gmAddr == nullptr) {
        std::cerr << "Error: Failed to allocate device GM memory\n";
        return -1;
    }

    std::cout << "Allocated device GM: " << gmAddr << " (" << allocSize << " bytes)\n";

    // Step 3: Build host buffer with CoreFunctionBin structure
    std::vector<uint8_t> hostBuf(allocSize);
    uint64_t* sizePtr = reinterpret_cast<uint64_t*>(hostBuf.data());
    *sizePtr = binSize;
    std::memcpy(hostBuf.data() + sizeof(uint64_t), binData.data(), binSize);

    // Step 4: Copy to device
    int rc = rtMemcpy(gmAddr, allocSize, hostBuf.data(), allocSize, RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        std::cerr << "Error: rtMemcpy to device failed: " << rc << '\n';
        memAlloc_.Free(gmAddr);
        return rc;
    }

    // Step 5: Calculate functionBinAddr (skip size field)
    uint64_t functionBinAddr = reinterpret_cast<uint64_t>(gmAddr) + sizeof(uint64_t);
    funcIdToAddr_[funcId] = functionBinAddr;

    std::cout << "  func_id=" << funcId << " -> functionBinAddr=0x"
              << std::hex << functionBinAddr << std::dec << '\n';

    return 0;
}


