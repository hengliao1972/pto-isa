/**
 * KernelArgs Structure - Shared between Host, AICPU, and AICore
 *
 * This structure is used to pass arguments to both AICPU and AICore kernels.
 * It contains pointers to device memory for arguments and graph data.
 *
 * Memory Layout:
 * This structure's layout is hardcoded in libaicpu_extend_kernels.so, which
 * expects specific offsets for deviceArgs fields. The unused[5] array provides
 * the required offset alignment for compatibility with the CANN runtime.
 */

#ifndef RUNTIME_COMMON_KERNEL_ARGS_H
#define RUNTIME_COMMON_KERNEL_ARGS_H

#include <cstdint>

// Forward declaration
class Graph;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Kernel arguments structure
 *
 * This structure is passed to both AICPU and AICore kernels by the host.
 *
 * Field Access Patterns:
 * - unused[5]: Padding for alignment with CANN runtime expectations
 * - deviceArgs: Written by host, read by AICPU (contains aicpuSoBin/aicpuSoLen)
 * - hankArgs: Written by host, read/written by AICPU and AICore (handshake array)
 * - core_num: Written by host, read by AICPU (number of AICore instances)
 * - graphArgs: Written by host, read by AICPU (task graph structure)
 *
 * Note: AICore kernels receive handshake buffers directly, not this structure.
 */
struct KernelArgs {
    uint64_t unused[5] = {0};        // Alignment padding (required by CANN runtime offset)
    int64_t *deviceArgs{nullptr};    // Device arguments (AICPU reads, contains SO info)
    int64_t *hankArgs{nullptr};      // Handshake buffer array (shared AICPU/AICore)
    int64_t core_num;                // Number of AICore instances
    Graph *graphArgs{nullptr};       // Task graph in device memory (AICPU reads)
};

#ifdef __cplusplus
}
#endif

#endif  // RUNTIME_COMMON_KERNEL_ARGS_H
