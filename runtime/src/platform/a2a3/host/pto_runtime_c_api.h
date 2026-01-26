/**
 * PTO Runtime C API
 *
 * Pure C interface for Python ctypes bindings. Wraps C++ classes (Graph, DeviceRunner)
 * as opaque pointers and provides C functions to manipulate them.
 *
 * Key design:
 * - All functions use C linkage (extern "C")
 * - Opaque pointers hide C++ implementation details
 * - Error codes: 0 = success, negative = error
 * - Memory management: C++ owns all Graph and device memory
 */

#ifndef PTO_RUNTIME_C_API_H
#define PTO_RUNTIME_C_API_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque pointer types for C interface.
 * These hide the C++ class implementations.
 */
typedef void* GraphHandle;

/* =========================================================================== */
/* Graph API */
/* =========================================================================== */

/**
 * Initialize a graph for the basic example.
 *
 * Takes a graph handle and initializes it with the example graph structure
 * (4 tasks with dependencies). C++ allocates device tensors, builds the graph,
 * and initializes data.
 *
 * @param graph  Graph handle to initialize (will be filled by C++)
 * @return 0 on success, -1 on failure
 */
int InitGraph(GraphHandle graph);

/**
 * Validate results and cleanup resources.
 *
 * Copies results from device, validates correctness, frees device tensors,
 * and deletes the graph structure.
 *
 * @param graph  Graph handle to validate and cleanup (will be deleted)
 * @return 0 on success, -1 on failure
 */
int ValidateGraph(GraphHandle graph);

/* =========================================================================== */
/* DeviceRunner API */
/* =========================================================================== */

/**
 * Initialize the device runner.
 *
 * Must be called before any device operations.
 * Uses the DeviceRunner singleton internally.
 *
 * @param device_id              Device ID (0-15)
 * @param aicpu_binary           Binary data of AICPU shared object
 * @param aicpu_size             Size of AICPU binary in bytes
 * @param aicore_binary          Binary data of AICore kernel
 * @param aicore_size            Size of AICore binary in bytes
 * @param pto_isa_root           Path to PTO-ISA root directory (headers location)
 * @return 0 on success, error code on failure
 */
int DeviceRunner_Init(int device_id,
                      const uint8_t* aicpu_binary, size_t aicpu_size,
                      const uint8_t* aicore_binary, size_t aicore_size,
                      const char* pto_isa_root);

/**
 * Execute a graph on the device.
 *
 * Uses the DeviceRunner singleton internally.
 *
 * @param graph            Graph handle to execute
 * @param num_cores        Number of cores for handshake (e.g., 3 for 1c2v)
 * @param launch_aicpu_num Number of AICPU instances to launch (default 1)
 * @return 0 on success, error code on failure
 */
int DeviceRunner_Run(GraphHandle graph, int num_cores, int launch_aicpu_num);

/**
 * Print handshake results from device.
 *
 * Uses the DeviceRunner singleton internally.
 *
 * @param graph  Graph handle whose handshake results should be printed
 */
void DeviceRunner_PrintHandshakeResults(GraphHandle graph);

/**
 * Cleanup all resources and finalize the device runner.
 *
 * Uses the DeviceRunner singleton internally.
 *
 * @return 0 on success, error code on failure
 */
int DeviceRunner_Finalize(void);

/**
 * Compile and load a kernel at runtime.
 *
 * Uses the DeviceRunner singleton internally.
 *
 * @param func_id       Function identifier for this kernel
 * @param kernel_path   Path to kernel source file (.cpp)
 * @param core_type     Core type: 0=AIC, 1=AIV (default 1)
 * @return 0 on success, error code on failure
 */
int DeviceRunner_CompileAndLoadKernel(int func_id,
                                      const char* kernel_path,
                                      int core_type);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* PTO_RUNTIME_C_API_H */

