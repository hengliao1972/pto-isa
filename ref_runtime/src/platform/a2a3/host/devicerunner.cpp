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
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

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
    numCores_ = 0;
    totalCores_ = 0;

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

    // Allocate and initialize the runtime mailbox (DeviceArgs.opaque).
    void *rtArgsDevRaw = memAlloc_.Alloc(sizeof(PtoRuntimeArgs));
    if (rtArgsDevRaw == nullptr) {
        std::cerr << "Error: Alloc for PtoRuntimeArgs failed\n";
        soInfo_.Finalize();
        rtStreamDestroy(streamAicpu_);
        rtStreamDestroy(streamAicore_);
        streamAicpu_ = nullptr;
        streamAicore_ = nullptr;
        return -1;
    }
    runtimeArgsDev_ = reinterpret_cast<PtoRuntimeArgs *>(rtArgsDevRaw);
    runtimeArgsHost_ = PtoRuntimeArgs{};
    runtimeArgsHost_.hankArgs = nullptr;
    runtimeArgsHost_.graphArgs = nullptr;
    runtimeArgsHost_.core_num = 0;
    rc = rtMemcpy(runtimeArgsDev_, sizeof(PtoRuntimeArgs), &runtimeArgsHost_, sizeof(PtoRuntimeArgs),
                  RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        std::cerr << "Error: rtMemcpy for PtoRuntimeArgs failed: " << rc << '\n';
        soInfo_.Finalize();
        rtStreamDestroy(streamAicpu_);
        rtStreamDestroy(streamAicore_);
        streamAicpu_ = nullptr;
        streamAicore_ = nullptr;
        return rc;
    }

    // Initialize cfgdata (DeviceArgs) consumed by DynTileFwkKernelServer*.
    deviceArgs_ = DeviceArgs{};
    deviceArgs_.nrAicpu = 1;
    deviceArgs_.opaque = reinterpret_cast<uint64_t>(runtimeArgsDev_);
    deviceArgs_.aicpuSoBin = soInfo_.aicpuSoBin;
    deviceArgs_.aicpuSoLen = soInfo_.aicpuSoLen;
    deviceArgs_.deviceId = static_cast<uint64_t>(deviceId_);
    deviceArgs_.taskType = 0;
    deviceArgs_.machineConfig = 0;
    deviceArgs_.taskId = 0;
    deviceArgs_.enableCtrl = 0;
    deviceArgs_.validGetPgMask = 0;
    deviceArgs_.disableSync = 0;

    deviceArgsDev_ = memAlloc_.Alloc(sizeof(DeviceArgs));
    if (deviceArgsDev_ == nullptr) {
        std::cerr << "Error: Alloc for cfgdata(DeviceArgs) failed\n";
        soInfo_.Finalize();
        rtStreamDestroy(streamAicpu_);
        rtStreamDestroy(streamAicore_);
        streamAicpu_ = nullptr;
        streamAicore_ = nullptr;
        return -1;
    }
    rc = rtMemcpy(deviceArgsDev_, sizeof(DeviceArgs), &deviceArgs_, sizeof(DeviceArgs), RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        std::cerr << "Error: rtMemcpy for cfgdata(DeviceArgs) failed: " << rc << '\n';
        soInfo_.Finalize();
        rtStreamDestroy(streamAicpu_);
        rtStreamDestroy(streamAicore_);
        streamAicpu_ = nullptr;
        streamAicore_ = nullptr;
        return rc;
    }

    kernelArgs_ = DeviceKernelArgs{};
    kernelArgs_.cfgdata = reinterpret_cast<int64_t *>(deviceArgsDev_);

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
    if (numCores <= 0) {
        std::cerr << "Error: numCores must be > 0\n";
        return -1;
    }
    if (numCores > GRAPH_MAX_WORKER) {
        std::cerr << "Error: numCores (" << numCores << ") exceeds GRAPH_MAX_WORKER (" << GRAPH_MAX_WORKER << ")\n";
        return -1;
    }

    const int prevTotalCores = totalCores_;
    totalCores_ = numCores;
    // A2/A3 mix kernels launch 1 cube block + 2 vector subblocks per cube block.
    // Derive cube blocks from total workers (1/3 AIC, 2/3 AIV).
    numCores_ = (totalCores_ + 2) / 3;
    if (numCores_ > totalCores_) {
        numCores_ = totalCores_;
    }

    // Reset (or reallocate) handshake buffers for this run.
    hankArgs_.resize(static_cast<size_t>(totalCores_));
    for (int i = 0; i < totalCores_; i++) {
        hankArgs_[i].aicpu_ready = 0;
        hankArgs_[i].aicore_done = 0;
        hankArgs_[i].control = 0;
        hankArgs_[i].task = 0;
        hankArgs_[i].task_status = 0;
        hankArgs_[i].core_type = (i < numCores_) ? 0 : 1;
        hankArgs_[i].profile_enable = enableProfile_ ? 1U : 0U;
        hankArgs_[i].reserved = 0;
    }

    const size_t hankBytes = sizeof(Handshake) * static_cast<size_t>(totalCores_);
    if (hankDev_ == nullptr || prevTotalCores != totalCores_) {
        if (hankDev_ != nullptr) {
            memAlloc_.Free(hankDev_);
        }
        void* hankDevRaw = memAlloc_.Alloc(hankBytes);
        if (hankDevRaw == nullptr) {
            std::cerr << "Error: Alloc for handshake failed\n";
            return -1;
        }
        hankDev_ = reinterpret_cast<Handshake*>(hankDevRaw);
    }

    int rc = rtMemcpy(hankDev_, hankBytes, hankArgs_.data(), hankBytes, RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        std::cerr << "Error: rtMemcpy for handshake reset failed: " << rc << '\n';
        return rc;
    }

    // Set functionBinAddr for all tasks (runtime function pointer dispatch).
    Graph mutableGraph = graph;
    std::cout << "\n=== Setting functionBinAddr for Tasks ===" << std::endl;
    for (int i = 0; i < mutableGraph.get_task_count(); i++) {
        Task* task = mutableGraph.get_task(i);
        if (task == nullptr) {
            continue;
        }
        uint64_t addr = GetFunctionBinAddr(task->func_id);
        task->functionBinAddr = addr;
        std::cout << "  Task " << i << " (func_id=" << task->func_id
                  << ") -> functionBinAddr=0x" << std::hex << addr << std::dec << std::endl;
    }
    std::cout << std::endl;

    // Copy graph to device memory.
    void* graphDevRaw = memAlloc_.Alloc(sizeof(Graph));
    if (graphDevRaw == nullptr) {
        std::cerr << "Error: Alloc for graphArgs failed\n";
        return -1;
    }
    auto* graphDev = reinterpret_cast<Graph*>(graphDevRaw);
    rc = rtMemcpy(graphDev, sizeof(Graph), &mutableGraph, sizeof(Graph), RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        std::cerr << "Error: rtMemcpy for graph failed: " << rc << '\n';
        memAlloc_.Free(graphDev);
        return rc;
    }

    // Update the runtime mailbox in device memory.
    runtimeArgsHost_.hankArgs = hankDev_;
    runtimeArgsHost_.graphArgs = graphDev;
    runtimeArgsHost_.core_num = totalCores_;
    rc = rtMemcpy(runtimeArgsDev_, sizeof(PtoRuntimeArgs), &runtimeArgsHost_, sizeof(PtoRuntimeArgs),
                  RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        std::cerr << "Error: rtMemcpy for PtoRuntimeArgs failed: " << rc << '\n';
        memAlloc_.Free(graphDev);
        return rc;
    }

    // Update cfgdata (DeviceArgs) in device memory.
    deviceArgs_.nrAic = static_cast<uint32_t>(numCores_);
    deviceArgs_.nrAiv = static_cast<uint32_t>(totalCores_ - numCores_);
    deviceArgs_.nrValidAic = static_cast<uint32_t>(numCores_);
    rc = rtMemcpy(deviceArgsDev_, sizeof(DeviceArgs), &deviceArgs_, sizeof(DeviceArgs), RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        std::cerr << "Error: rtMemcpy for cfgdata(DeviceArgs) failed: " << rc << '\n';
        memAlloc_.Free(graphDev);
        return rc;
    }

    // Launch AICPU init kernel
    rc = LaunchAiCpuKernel(streamAicpu_, &kernelArgs_, "DynTileFwkKernelServerInit", 1);
    if (rc != 0) {
        std::cerr << "Error: LaunchAiCpuKernel (init) failed: " << rc << '\n';
        memAlloc_.Free(graphDev);
        return rc;
    }

    // Launch AICPU main kernel
    rc = LaunchAiCpuKernel(streamAicpu_, &kernelArgs_, "DynTileFwkKernelServer", launchAicpuNum);
    if (rc != 0) {
        std::cerr << "Error: LaunchAiCpuKernel (main) failed: " << rc << '\n';
        memAlloc_.Free(graphDev);
        return rc;
    }

    // Launch AICore kernel
    rc = LauncherAicoreKernel(streamAicore_, hankDev_);
    if (rc != 0) {
        std::cerr << "Error: LauncherAicoreKernel failed: " << rc << '\n';
        memAlloc_.Free(graphDev);
        return rc;
    }

    // Synchronize streams
    std::cout << "Synchronizing AICPU stream..." << std::endl;
    rc = rtStreamSynchronize(streamAicpu_);
    if (rc != 0) {
        std::cerr << "Error: rtStreamSynchronize (AICPU) failed: " << rc << '\n';
        memAlloc_.Free(graphDev);
        return rc;
    }
    std::cout << "AICPU stream done" << std::endl;

    std::cout << "Synchronizing AICore stream..." << std::endl;
    rc = rtStreamSynchronize(streamAicore_);
    if (rc != 0) {
        std::cerr << "Error: rtStreamSynchronize (AICore) failed: " << rc << '\n';
        memAlloc_.Free(graphDev);
        return rc;
    }
    std::cout << "AICore stream done" << std::endl;

    // Copy graph back for profiling (task start/end, core id, etc).
    hasLastGraph_ = false;
    if (enableProfile_) {
        rc = rtMemcpy(&lastGraph_, sizeof(Graph), graphDev, sizeof(Graph), RT_MEMCPY_DEVICE_TO_HOST);
        if (rc != 0) {
            std::cerr << "Warning: rtMemcpy for graph (device->host) failed: " << rc << '\n';
        } else {
            hasLastGraph_ = true;
        }
    }

    // Cleanup graph args
    memAlloc_.Free(graphDev);

    return 0;
}

void DeviceRunner::PrintHandshakeResults(Graph& graph) {
    if (!initialized_ || hankDev_ == nullptr || totalCores_ <= 0) {
        return;
    }

    if (totalCores_ > GRAPH_MAX_WORKER) {
        std::cerr << "Warning: totalCores (" << totalCores_ << ") exceeds GRAPH_MAX_WORKER (" << GRAPH_MAX_WORKER
                  << "); truncating handshake print.\n";
    }
    graph.worker_count = std::min(totalCores_, GRAPH_MAX_WORKER);

    size_t total_size = sizeof(Handshake) * static_cast<size_t>(graph.worker_count);
    int rc = rtMemcpy(graph.workers, total_size, hankDev_, total_size, RT_MEMCPY_DEVICE_TO_HOST);
    if (rc != 0) {
        std::cerr << "Warning: rtMemcpy for handshake (device->host) failed: " << rc << '\n';
        return;
    }

    std::cout << "Handshake results for " << graph.worker_count << " cores:" << std::endl;
    for (int i = 0; i < graph.worker_count; i++) {
        std::cout << "  Core " << i << ": aicore_done=" << graph.workers[i].aicore_done
                  << " aicpu_ready=" << graph.workers[i].aicpu_ready
                  << " control=" << graph.workers[i].control
                  << " task=" << graph.workers[i].task << std::endl;
    }
}

void DeviceRunner::SetProfileEnabled(bool enabled) {
    enableProfile_ = enabled;
    if (!enableProfile_) {
        hasLastGraph_ = false;
    }
}

bool DeviceRunner::ProfileEnabled() const {
    return enableProfile_;
}

bool DeviceRunner::HasLastProfile() const {
    return hasLastGraph_;
}

std::vector<TaskProfileRecord> DeviceRunner::GetLastProfile() const {
    std::vector<TaskProfileRecord> out;
    if (!hasLastGraph_) {
        return out;
    }
    const int taskCount = lastGraph_.get_task_count();
    if (taskCount <= 0) {
        return out;
    }
    out.reserve(static_cast<size_t>(taskCount));
    for (int i = 0; i < taskCount; i++) {
        const Task* t = lastGraph_.get_task(i);
        if (t == nullptr) {
            continue;
        }
        TaskProfileRecord rec;
        rec.task_id = t->task_id;
        rec.func_id = t->func_id;
        rec.core_type = t->core_type;
        rec.exec_core_id = t->profile.exec_core_id;
        rec.exec_core_type = t->profile.exec_core_type;
        rec.exec_phys_core_id = t->profile.exec_phys_core_id;
        rec.start_time = t->profile.start_time;
        rec.end_time = t->profile.end_time;
        for (size_t j = 0; j < rec.pmu_cnt.size(); j++) {
            rec.pmu_cnt[j] = t->profile.pmu_cnt[j];
        }
        out.push_back(rec);
    }
    return out;
}

int DeviceRunner::RunTask(const std::vector<uint64_t>& args, int funcId, int launchAicpuNum) {
    Graph graph;
    std::vector<uint64_t> argsCopy = args;
    int taskId = graph.add_task(argsCopy.data(), static_cast<int>(argsCopy.size()), funcId);
    if (taskId < 0) {
        std::cerr << "Error: Graph::add_task failed\n";
        return -1;
    }
    const int cores = (totalCores_ > 0) ? totalCores_ : 3;
    return Run(graph, cores, launchAicpuNum);
}

int DeviceRunner::Finalize() {
    if (!initialized_) {
        return 0;
    }

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
    numCores_ = 0;
    totalCores_ = 0;
    ptoIsaRoot_.clear();
    hankArgs_.clear();
    deviceArgsDev_ = nullptr;
    runtimeArgsDev_ = nullptr;
    hankDev_ = nullptr;
    hasLastGraph_ = false;
    enableProfile_ = false;

    std::cout << "DeviceRunner finalized\n";
    return 0;
}

int DeviceRunner::LaunchAiCpuKernel(rtStream_t stream, DeviceKernelArgs *kArgs, const char *kernelName, int aicpuNum) {
    struct Args {
        DeviceKernelArgs kArgs;
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

int DeviceRunner::LauncherAicoreKernel(rtStream_t stream, Handshake* hankArgs) {
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
        Handshake* hankArgs;
    };
    Args args = {hankArgs};
    rtArgsEx_t rtArgs;
    std::memset(&rtArgs, 0, sizeof(rtArgs));
    rtArgs.args = &args;
    rtArgs.argsSize = sizeof(args);

    rtTaskCfgInfo_t cfg = {};
    cfg.schemMode = RT_SCHEM_MODE_BATCH;

    uint32_t blockDim = static_cast<uint32_t>(numCores_);
    if (blockDim == 0) {
        blockDim = 1;
    }
    rc = rtKernelLaunchWithHandleV2(binHandle, 0, blockDim, &rtArgs, nullptr, stream, &cfg);
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
