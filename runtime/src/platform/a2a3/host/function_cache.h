/**
 * Function Cache Structures
 *
 * Defines data structures for caching compiled kernel binaries and managing
 * their addresses in device GM memory.
 *
 * These structures follow the production system design from:
 * - src/interface/cache/core_func_data.h
 * - src/interface/cache/function_cache.h
 *
 * Memory Layout:
 * ┌────────────────────────────────────────────────┐
 * │ CoreFunctionBinCache                            │
 * │ ┌────────────────────────────────────────────┐ │
 * │ │ dataSize                                   │ │
 * │ ├────────────────────────────────────────────┤ │
 * │ │ offset[0]                                  │ │
 * │ │ offset[1]                                  │ │
 * │ │ ...                                        │ │
 * │ ├────────────────────────────────────────────┤ │
 * │ │ CoreFunctionBin[0]                         │ │
 * │ │   size                                     │ │
 * │ │   data[...binary...]                       │ │
 * │ ├────────────────────────────────────────────┤ │
 * │ │ CoreFunctionBin[1]                         │ │
 * │ │   size                                     │ │
 * │ │   data[...binary...]                       │ │
 * │ └────────────────────────────────────────────┘ │
 * └────────────────────────────────────────────────┘
 */

#ifndef RUNTIME_FUNCTION_CACHE_H
#define RUNTIME_FUNCTION_CACHE_H

#include <cstdint>

/**
 * Single kernel binary container
 *
 * Contains the size and binary data for one compiled kernel.
 * The data field is a flexible array member that extends beyond
 * the struct boundary.
 */
#pragma pack(1)
struct CoreFunctionBin {
    uint64_t size;      // Size of binary data in bytes
    uint8_t data[0];    // Flexible array member for kernel binary
};
#pragma pack()

/**
 * Binary cache structure for all kernels
 *
 * This structure packs multiple kernel binaries into a single contiguous
 * memory block for efficient device memory allocation and copying.
 *
 * Memory Layout:
 * [dataSize][numKernels][offset0][offset1]...[offsetN][CoreFunctionBin0][CoreFunctionBin1]...
 *
 * Each offset points to the start of a CoreFunctionBin structure relative
 * to the beginning of the cache.
 */
struct CoreFunctionBinCache {
    uint64_t dataSize;      // Total size of all data (excluding this header)
    uint64_t numKernels;    // Number of kernels in this cache

    /**
     * Get offset array pointer
     * @return Pointer to array of offsets
     */
    uint64_t* GetOffsets() {
        return reinterpret_cast<uint64_t*>(
            reinterpret_cast<uint8_t*>(this) + sizeof(CoreFunctionBinCache));
    }

    /**
     * Get pointer to binary data region
     * @return Pointer to start of binary data
     */
    uint8_t* GetBinaryData() {
        return reinterpret_cast<uint8_t*>(GetOffsets()) +
               numKernels * sizeof(uint64_t);
    }

    /**
     * Get CoreFunctionBin by index
     * @param index  Kernel index
     * @return Pointer to CoreFunctionBin structure
     */
    CoreFunctionBin* GetKernel(uint64_t index) {
        if (index >= numKernels) {
            return nullptr;
        }
        uint64_t offset = GetOffsets()[index];
        return reinterpret_cast<CoreFunctionBin*>(GetBinaryData() + offset);
    }

    /**
     * Calculate total cache size including header
     * @return Total size in bytes
     */
    uint64_t GetTotalSize() const {
        return sizeof(CoreFunctionBinCache) +
               numKernels * sizeof(uint64_t) +
               dataSize;
    }
};

/**
 * Workspace address information for each kernel
 *
 * Contains the address of the kernel binary in device GM memory along
 * with other execution metadata.
 *
 * This structure is populated on the host and copied to device memory
 * where it's accessed by AICore kernels during task execution.
 */
struct CoreFunctionWsAddr {
    uint64_t functionBinAddr;       // *** THE KEY FIELD *** - Address in device GM
    uint64_t invokeEntryAddr;       // Parameter entry address (unused in simple runtime)
    uint64_t psgId;                 // Program subgraph ID (func_id in this runtime)
    uint64_t topoAddr;              // Topology address (unused in simple runtime)
    uint64_t invokeEntryInfo;       // Tensor info address (unused in simple runtime)
    uint64_t invokeEntryNum;        // Number of entries (unused in simple runtime)
    uint64_t invokeEntryOriAddr;    // Original address backup (unused in simple runtime)

    CoreFunctionWsAddr()
        : functionBinAddr(0), invokeEntryAddr(0), psgId(0), topoAddr(0),
          invokeEntryInfo(0), invokeEntryNum(0), invokeEntryOriAddr(0) {}

    CoreFunctionWsAddr(uint64_t bin, uint64_t psg)
        : functionBinAddr(bin), invokeEntryAddr(0), psgId(psg), topoAddr(0),
          invokeEntryInfo(0), invokeEntryNum(0), invokeEntryOriAddr(0) {}
};

#endif  // RUNTIME_FUNCTION_CACHE_H
