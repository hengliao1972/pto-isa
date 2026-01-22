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
#include <string>
#include <vector>
#include <runtime/rt.h>
#include "../graph/kernel_args.h"
#include "../graph/handshake.h"
#include "memoryallocator.h"

// Forward declarations
class Graph;

/**
 * DeviceArgs structure for AICPU device arguments
 *
 * This structure contains pointers to device memory for the AICPU shared object.
 * The layout is hardcoded in libaicpu_extend_kernels.so, which expects specific
 * offsets for aicpuSoBin and aicpuSoLen fields.
 */
struct DeviceArgs {
    uint64_t unused[12] = {0};
    uint64_t aicpuSoBin{0};
    uint64_t aicpuSoLen{0};
};

/**
 * Helper class for managing KernelArgs with device memory
 *
 * This class wraps KernelArgs and provides host-side initialization methods
 * for allocating device memory and copying data to the device. It separates
 * the concerns of device memory management (host-only) from the structure
 * layout (shared with kernels).
 *
 * The helper provides implicit conversion to KernelArgs* for seamless use
 * with runtime APIs.
 */
struct KernelArgsHelper {
    KernelArgs args;
    MemoryAllocator* allocator_{nullptr};

    /**
     * Initialize device arguments by allocating device memory and copying data
     *
     * @param hostDeviceArgs  Host-side device arguments to copy
     * @param allocator       Memory allocator to use
     * @return 0 on success, error code on failure
     */
    int InitDeviceArgs(const DeviceArgs &hostDeviceArgs, MemoryAllocator& allocator);

    /**
     * Free device memory allocated for device arguments
     *
     * @return 0 on success, error code on failure
     */
    int FinalizeDeviceArgs();

    /**
     * Initialize graph arguments by allocating device memory and copying data
     *
     * @param hostGraph  Host-side graph to copy to device
     * @param allocator  Memory allocator to use
     * @return 0 on success, error code on failure
     */
    int InitGraphArgs(const Graph& hostGraph, MemoryAllocator& allocator);

    /**
     * Free device memory allocated for graph arguments
     *
     * @return 0 on success, error code on failure
     */
    int FinalizeGraphArgs();

    /**
     * Implicit conversion operators for seamless use with runtime APIs
     *
     * These operators allow KernelArgsHelper to be used wherever KernelArgs*
     * is expected, enabling transparent device memory management while
     * maintaining API compatibility.
     */
    operator KernelArgs*() { return &args; }
    KernelArgs* operator&() { return &args; }
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
     * Load shared object file and copy to device memory
     *
     * @param soPath     Path to the .so file
     * @param allocator  Memory allocator to use
     * @return 0 on success, error code on failure
     */
    int Init(const std::string &soPath, MemoryAllocator& allocator);

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
     * @param deviceId       Device ID (0-15)
     * @param numCores       Number of cores for handshake (e.g., 3 for 1c2v)
     * @param aicpuSoPath    Path to AICPU shared object
     * @param aicoreKernelPath Path to AICore kernel binary (default: "./aicore/kernel.o")
     * @return 0 on success, error code on failure
     */
    int Init(int deviceId, int numCores, const std::string &aicpuSoPath,
             const std::string &aicoreKernelPath = "./aicore/kernel.o");

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
     * 1. Transfers graph to device memory
     * 2. Launches AICPU init kernel
     * 3. Launches AICPU main kernel
     * 4. Launches AICore kernel
     * 5. Synchronizes streams
     * 6. Cleans up graph memory
     *
     * @param graph          Graph to execute
     * @param launchAicpuNum Number of AICPU instances (default: 1)
     * @return 0 on success, error code on failure
     */
    int Run(const Graph& graph, int launchAicpuNum = 1);

    /**
     * Print handshake results from device
     *
     * Copies handshake buffers from device and prints their status.
     */
    void PrintHandshakeResults();

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
    int LaunchAiCpuKernel(rtStream_t stream, KernelArgs *kArgs,
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
    int LauncherAicoreKernel(rtStream_t stream, KernelArgs *kernelArgs);

private:
    DeviceRunner() = default;

    // Internal state
    bool initialized_{false};
    int deviceId_{-1};
    int numCores_{0};
    std::string aicoreKernelPath_;

    // Memory management
    MemoryAllocator memAlloc_;

    // Device resources
    rtStream_t streamAicpu_{nullptr};
    rtStream_t streamAicore_{nullptr};
    AicpuSoInfo soInfo_;
    KernelArgsHelper kernelArgs_;
    DeviceArgs deviceArgs_;

    // Handshake buffers
    std::vector<Handshake> hankArgs_;
};

#endif  // RUNTIME_DEVICERUNNER_H
