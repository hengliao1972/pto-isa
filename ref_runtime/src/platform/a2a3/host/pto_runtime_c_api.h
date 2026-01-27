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
 * Create an empty task graph.
 *
 * Caller must destroy it via Graph_Destroy().
 *
 * @return Graph handle on success, NULL on failure
 */
GraphHandle Graph_Create(void);

/**
 * Destroy a graph created by Graph_Create().
 *
 * @param graph Graph handle
 * @return 0 on success, -1 on failure
 */
int Graph_Destroy(GraphHandle graph);

/**
 * Add a task to the graph.
 *
 * @param graph     Graph handle
 * @param args      Argument array (uint64_t)
 * @param num_args  Number of arguments
 * @param func_id   Kernel function identifier
 * @param core_type Requested core type (0=any, 1=AIC(cube), 2=AIV(vector))
 * @return Task ID (>=0) on success, -1 on failure
 */
int Graph_AddTask(GraphHandle graph,
                  const uint64_t* args,
                  int num_args,
                  int func_id,
                  int core_type);

/**
 * Add a dependency edge from from_task -> to_task.
 *
 * @param graph      Graph handle
 * @param from_task  Producer task ID
 * @param to_task    Consumer task ID
 * @return 0 on success, -1 on failure
 */
int Graph_AddSuccessor(GraphHandle graph, int from_task, int to_task);

/**
 * Get total number of tasks in the graph.
 *
 * @param graph Graph handle
 * @return Task count (>=0) on success, -1 on failure
 */
int Graph_GetTaskCount(GraphHandle graph);

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
 * Allocate device tensor memory.
 *
 * @param bytes Size of tensor in bytes
 * @return Device pointer on success, NULL on failure
 */
void* DeviceRunner_AllocateTensor(size_t bytes);

/**
 * Free device tensor memory.
 *
 * @param dev_ptr Device pointer
 */
void DeviceRunner_FreeTensor(void* dev_ptr);

/**
 * Copy raw bytes from host to device.
 *
 * @param dev_ptr  Device pointer
 * @param host_ptr Host pointer
 * @param bytes    Number of bytes
 * @return 0 on success, error code on failure
 */
int DeviceRunner_CopyToDevice(void* dev_ptr, const void* host_ptr, size_t bytes);

/**
 * Copy raw bytes from device to host.
 *
 * @param host_ptr Host pointer
 * @param dev_ptr  Device pointer
 * @param bytes    Number of bytes
 * @return 0 on success, error code on failure
 */
int DeviceRunner_CopyFromDevice(void* host_ptr, const void* dev_ptr, size_t bytes);

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
 * Enable/disable per-task profiling (graph copy-back).
 *
 * @param enabled 0=disable, non-zero=enable
 * @return 0 on success, -1 on failure
 */
int DeviceRunner_SetProfileEnabled(int enabled);

/**
 * Return 1 if profiling is enabled, else 0.
 */
int DeviceRunner_ProfileEnabled(void);

/**
 * Return 1 if profiling data from the last run is available, else 0.
 */
int DeviceRunner_HasLastProfile(void);

#ifndef PTO_PROFILE_PMU_CNT
#define PTO_PROFILE_PMU_CNT 8
#endif

typedef struct {
    int task_id;
    int func_id;
    int core_type;  // requested core type (0=any, 1=AIC, 2=AIV)
    uint32_t exec_core_id;
    uint32_t exec_core_type;  // executing core type (1=AIC, 2=AIV)
    uint32_t exec_phys_core_id;
    uint64_t start_time;
    uint64_t end_time;
    uint32_t pmu_cnt[PTO_PROFILE_PMU_CNT];
} PtoTaskProfileRecord;

/**
 * Get profiling records from the last profiled run.
 *
 * If out_records is NULL or max_records <= 0, returns the required count.
 *
 * @param out_records Output buffer for records
 * @param max_records Maximum records that fit in out_records
 * @return Number of records written/required (>=0) on success, -1 on failure
 */
int DeviceRunner_GetLastProfile(PtoTaskProfileRecord* out_records, int max_records);

/**
 * Compile and load a kernel at runtime.
 *
 * Uses the DeviceRunner singleton internally.
 *
 * @param func_id       Function identifier for this kernel
 * @param kernel_path   Path to kernel source file (.cpp)
 * @param core_type     Core type: 0=AIC, 1=AIV (default 0)
 * @return 0 on success, error code on failure
 */
int DeviceRunner_CompileAndLoadKernel(int func_id,
                                      const char* kernel_path,
                                      int core_type);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* PTO_RUNTIME_C_API_H */
