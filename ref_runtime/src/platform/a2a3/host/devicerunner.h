/**
 * Device Runner - Ascend Device Execution Utilities
 *
 * This module provides utilities for launching and managing AICPU and AICore
 * kernels on Ascend devices using CANN runtime APIs.
 *
 * Key Components:
 * - DeviceArgs: AICPU device argument structure
 * - KernelArgsHelper: Helper for managing kernel arguments with device memory
 * - AicpuSoInfo: AICPU shared object (.so) file management
 * - DeviceRunner: Singleton for kernel launching and execution
 */

#ifndef RUNTIME_DEVICERUNNER_H
#define RUNTIME_DEVICERUNNER_H

#include <cstdint>
#include <array>
#include <map>
#include <string>
#include <vector>
#include <runtime/rt.h>
#include "graph.h"
#include "kernel_args.h"
#include "memoryallocator.h"
#include "function_cache.h"

struct TaskProfileRecord {
    int task_id{0};
    int func_id{0};
    int core_type{0};  // requested core type (0=any, 1=AIC, 2=AIV)
    uint32_t exec_core_id{0};
    uint32_t exec_core_type{0};  // executing core type (1=AIC, 2=AIV)
    uint32_t exec_phys_core_id{0};
    uint64_t start_time{0};
    uint64_t end_time{0};
    std::array<uint32_t, 8> pmu_cnt{};
};

/**
 * AICPU shared object information and management
 *
 * This class manages loading and device memory allocation for AICPU
 * shared object (.so) files.
 */
struct AicpuSoInfo {
    uint64_t aicpuSoBin{0};
    uint64_t aicpuSoLen{0};
    MemoryAllocator* allocator_{nullptr};

    /**
     * Load shared object binary data and copy to device memory
     *
     * @param aicpuSoBinary  Binary data of the AICPU shared object
     * @param allocator      Memory allocator to use
     * @return 0 on success, error code on failure
     */
    int Init(const std::vector<uint8_t>& aicpuSoBinary, MemoryAllocator& allocator);

    /**
     * Free device memory allocated for shared object
     *
     * @return 0 on success, error code on failure
     */
    int Finalize();
};

/**
 * Device runner singleton for kernel execution
 *
 * This class provides a unified interface for launching AICPU and AICore
 * kernels on Ascend devices. It handles:
 * - Device initialization and resource management
 * - Tensor memory allocation and data transfer
 * - AICPU kernel launching with dynamic arguments
 * - AICore kernel registration and launching
 * - Coordinated execution of both kernel types
 * - Graph execution workflow
 */
class DeviceRunner {
public:
    /**
     * Get singleton instance
     *
     * @return Reference to the singleton DeviceRunner instance
     */
    static DeviceRunner &Get();

    /**
     * Initialize device and runtime resources
     *
     * Must be called before any other operations.
     *
     * @param deviceId            Device ID (0-15)
     * @param aicpuSoBinary       Binary data of AICPU shared object
     * @param aicoreKernelBinary  Binary data of AICore kernel
     * @param ptoIsaRoot          Path to PTO-ISA root directory (headers location)
     * @return 0 on success, error code on failure
     */
    int Init(int deviceId, const std::vector<uint8_t>& aicpuSoBinary,
             const std::vector<uint8_t>& aicoreKernelBinary, const std::string& ptoIsaRoot);

    /**
     * Allocate device tensor memory
     *
     * @param bytes  Size of tensor in bytes
     * @return Device pointer on success, nullptr on failure
     */
    void* AllocateTensor(size_t bytes);

    /**
     * Free device tensor memory
     *
     * @param devPtr  Device pointer to free
     */
    void FreeTensor(void* devPtr);

    /**
     * Copy data from host to device
     *
     * @param devPtr   Device pointer
     * @param hostPtr  Host pointer
     * @param bytes    Number of bytes to copy
     * @return 0 on success, error code on failure
     */
    int CopyToDevice(void* devPtr, const void* hostPtr, size_t bytes);

    /**
     * Copy data from device to host
     *
     * @param hostPtr  Host pointer
     * @param devPtr   Device pointer
     * @param bytes    Number of bytes to copy
     * @return 0 on success, error code on failure
     */
    int CopyFromDevice(void* hostPtr, const void* devPtr, size_t bytes);

    /**
     * Execute a graph
     *
     * This method:
     * 1. Initializes worker handshake buffers in the graph based on numCores
     * 2. Transfers graph to device memory
     * 3. Launches AICPU init kernel
     * 4. Launches AICPU main kernel
     * 5. Launches AICore kernel
     * 6. Synchronizes streams
     * 7. Cleans up graph memory
     *
     * @param graph          Graph to execute (will be modified to initialize workers)
     * @param numCores       Number of cores for handshake (e.g., 3 for 1c2v)
     * @param launchAicpuNum Number of AICPU instances (default: 1)
     * @return 0 on success, error code on failure
     */
    int Run(Graph& graph, int numCores, int launchAicpuNum = 1);

    void SetProfileEnabled(bool enabled);
    bool ProfileEnabled() const;
    bool HasLastProfile() const;
    std::vector<TaskProfileRecord> GetLastProfile() const;

    /**
     * Execute a single task (convenience wrapper)
     *
     * Builds a 1-task Graph and calls Run(...).
     *
     * @param args           Kernel argument list (device pointers/scalars encoded as uint64_t)
     * @param funcId         Function identifier
     * @param launchAicpuNum Number of AICPU instances (default: 1)
     * @return 0 on success, error code on failure
     */
    int RunTask(const std::vector<uint64_t>& args, int funcId, int launchAicpuNum = 1);

    /**
     * Print handshake results from device
     *
     * Copies handshake buffers from device and prints their status.
     * Must be called after Run() with the same graph.
     *
     * @param graph  The graph whose handshake results should be printed
     */
    void PrintHandshakeResults(Graph& graph);

    /**
     * Cleanup all resources
     *
     * Frees all device memory, destroys streams, and resets state.
     *
     * @return 0 on success, error code on failure
     */
    int Finalize();

    /**
     * Launch an AICPU kernel
     *
     * Internal method used by Run(). Can be called directly for custom workflows.
     *
     * @param stream      AICPU stream
     * @param kArgs       Kernel arguments
     * @param kernelName  Name of the kernel to launch
     * @param aicpuNum    Number of AICPU instances to launch
     * @return 0 on success, error code on failure
     */
    int LaunchAiCpuKernel(rtStream_t stream, DeviceKernelArgs *kArgs,
                          const char *kernelName, int aicpuNum);

    /**
     * Launch an AICore kernel
     *
     * Internal method used by Run(). Can be called directly for custom workflows.
     *
     * @param stream       AICore stream
     * @param kernelArgs   Kernel arguments
     * @return 0 on success, error code on failure
     */
    int LauncherAicoreKernel(rtStream_t stream, Handshake* hankArgs);

    /**
     * Register a kernel binary path for a func_id
     *
     * Should be called during Init() for each kernel before LoadKernelsToDevice().
     * Maps a function ID (used in tasks) to the path of its compiled .o file.
     *
     * @param funcId   Function identifier (0, 1, 2, ...)
     * @param binPath  Path to the kernel .o file
     */
    void RegisterKernel(int funcId, const std::string& binPath);

    /**
     * Load all registered kernels, build cache, copy to device
     *
     * Called once after all RegisterKernel() calls during initialization.
     * This method:
     * 1. Loads each .o file using LoadBinData() (extracts .text section)
     * 2. Builds CoreFunctionBinCache with all kernel binaries
     * 3. Allocates device GM memory for the cache
     * 4. Copies cache to device
     * 5. Calculates functionBinAddr[i] = gmBaseAddr + offset[i]
     * 6. Stores addresses for later retrieval via GetFunctionBinAddr()
     *
     * @return 0 on success, error code on failure
     */
    int LoadKernelsToDevice();

    /**
     * Get functionBinAddr for a given func_id
     *
     * Returns the device GM address where the kernel binary resides.
     * This address can be cast to a function pointer and called.
     *
     * @param funcId  Function identifier
     * @return Device GM address of kernel, or 0 if not found
     */
    uint64_t GetFunctionBinAddr(int funcId);

    /**
     * Compile and load a kernel at runtime
     *
     * This function combines compilation, registration, and loading:
     * 1. Compiles the kernel source file using KernelCompiler
     * 2. Registers the compiled binary with the given func_id
     * 3. Loads the kernel binary to device GM memory
     * 4. Updates funcIdToAddr_ mapping
     *
     * Requirements:
     * - ASCEND_HOME_PATH must be set (for ccec compiler)
     * - PTO-ISA headers must be configured during Init()
     * - DeviceRunner must be initialized before calling this
     *
     * @param funcId      Function identifier for this kernel
     * @param sourcePath  Path to kernel source file (.cpp)
     * @param coreType    Core type: 0=AIC, 1=AIV (determines compilation flags)
     * @return 0 on success, -1 on error
     *
     * Example:
     *   runner.Init(0, 3, "./aicpu/lib.so", "./aicore/kernel.o", "/path/to/pto-isa");
     *   runner.CompileAndLoadKernel(0, "./aicore/kernels/aiv/kernel_add.cpp", 1);
     */
    int CompileAndLoadKernel(int funcId,
                            const std::string& sourcePath,
                            int coreType);

    /**
     * Load a single kernel binary to device GM memory
     *
     * This is a helper function for incremental kernel loading.
     * It loads a single .o file and extends the device GM cache.
     *
     * @param funcId   Function identifier
     * @param binPath  Path to compiled .o file
     * @return 0 on success, -1 on error
     */
    int LoadSingleKernelToDevice(int funcId, const std::string& binPath);

private:
    DeviceRunner() = default;

    // Internal state
    bool initialized_{false};
    int deviceId_{-1};
    int numCores_{0};     // cube (AIC) blocks launched on device
    int totalCores_{0};   // total workers (AIC + AIV subblocks)
    std::vector<uint8_t> aicoreKernelBinary_;
    std::string ptoIsaRoot_;  // PTO-ISA root directory for kernel compilation

    // Memory management
    MemoryAllocator memAlloc_;

    // Device resources
    rtStream_t streamAicpu_{nullptr};
    rtStream_t streamAicore_{nullptr};
    AicpuSoInfo soInfo_;
    DeviceKernelArgs kernelArgs_{};
    DeviceArgs deviceArgs_{};
    void* deviceArgsDev_{nullptr};      // DeviceArgs stored at kernelArgs_.cfgdata
    PtoRuntimeArgs runtimeArgsHost_{};  // Host shadow copy (written to runtimeArgsDev_)
    PtoRuntimeArgs* runtimeArgsDev_{nullptr};
    Handshake* hankDev_{nullptr};
    std::vector<Handshake> hankArgs_;

    // Kernel binary management (NEW - for runtime function pointer dispatch)
    CoreFunctionBinCache* binCache_{nullptr};         // Host-side cache structure
    void* binGmAddr_{nullptr};                        // Device GM base address
    std::map<int, uint64_t> funcIdToAddr_;           // func_id -> functionBinAddr
    std::map<int, std::string> funcIdToBinPath_;     // func_id -> .o file path

    // Profiling: last executed graph snapshot (copied back from device).
    bool enableProfile_{false};
    bool hasLastGraph_{false};
    Graph lastGraph_{};
};

#endif  // RUNTIME_DEVICERUNNER_H
