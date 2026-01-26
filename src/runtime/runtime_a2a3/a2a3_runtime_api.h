/**
 * PTO Runtime - Ascend A2/A3 Runtime API
 * 
 * This header provides the public API for initializing and running
 * the A2A3 runtime with dynamic .so loading support.
 * 
 * Thread Configuration:
 * - 1 Orchestration AICPU thread: Executes the orchestration function
 * - 3 Dependency AICPU threads: Handle task dependency resolution
 * - 48 AIV workers: Execute Vector InCore functions
 * - 24 AIC workers: Execute Cube InCore functions
 */

#ifndef A2A3_RUNTIME_API_H
#define A2A3_RUNTIME_API_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Default Configuration
// =============================================================================

#define A2A3_DEFAULT_ORCH_THREADS    1
#define A2A3_DEFAULT_DEP_THREADS     3
#define A2A3_DEFAULT_AIV_WORKERS     48
#define A2A3_DEFAULT_AIC_WORKERS     24

// Maximum number of InCore functions that can be loaded
#define A2A3_MAX_INCORE_FUNCS        256

// =============================================================================
// Runtime Configuration
// =============================================================================

/**
 * Runtime initialization configuration.
 * 
 * All paths are relative to the current working directory unless
 * specified as absolute paths.
 */
typedef struct {
    // Orchestration function .so path
    const char* orchestration_so_path;
    
    // Orchestration function name (NULL for default: "orchestration_entry")
    const char* orchestration_func_name;
    
    // InCore function directories
    const char* incore_aiv_dir;    // Directory containing AIV .so files
    const char* incore_aic_dir;    // Directory containing AIC .so files
    
    // Thread configuration
    int num_orch_threads;          // Orchestration threads (default: 1)
    int num_dep_threads;           // Dependency resolution threads (default: 3)
    int num_aiv_workers;           // AIV worker threads (default: 48)
    int num_aic_workers;           // AIC worker threads (default: 24)
    
    // Optional: User data passed to orchestration function
    void* user_data;
    
    // Optional: Enable debug output
    bool debug_enabled;
    
    // DEBUG_ORCHESTRATION mode:
    // When enabled, only runs the orchestration function to build task graph,
    // then returns immediately without executing tasks.
    // The number of submitted tasks can be retrieved via a2a3_runtime_get_stats().
    bool debug_orchestration_only;
} A2A3RuntimeConfig;

/**
 * Initialize configuration with default values.
 */
static inline void a2a3_config_init_defaults(A2A3RuntimeConfig* config) {
    if (!config) return;
    config->orchestration_so_path = NULL;
    config->orchestration_func_name = NULL;  // Default: "orchestration_entry"
    config->incore_aiv_dir = NULL;
    config->incore_aic_dir = NULL;
    config->num_orch_threads = A2A3_DEFAULT_ORCH_THREADS;
    config->num_dep_threads = A2A3_DEFAULT_DEP_THREADS;
    config->num_aiv_workers = A2A3_DEFAULT_AIV_WORKERS;
    config->num_aic_workers = A2A3_DEFAULT_AIC_WORKERS;
    config->user_data = NULL;
    config->debug_enabled = false;
    config->debug_orchestration_only = false;
}

// =============================================================================
// Runtime Lifecycle API
// =============================================================================

/**
 * Initialize the A2A3 runtime.
 * 
 * This function:
 * 1. Loads the orchestration .so file
 * 2. Scans and loads InCore functions from AIV/AIC directories
 * 3. Initializes the thread pools
 * 4. Sets up memory management
 * 
 * @param config  Runtime configuration
 * @return 0 on success, negative error code on failure
 */
int a2a3_runtime_init(A2A3RuntimeConfig* config);

/**
 * Execute the orchestration function and run all tasks.
 * 
 * This function:
 * 1. Starts the orchestration thread to build the task graph
 * 2. Starts dependency resolution threads
 * 3. Dispatches tasks to AIV/AIC workers
 * 4. Waits for all tasks to complete
 * 
 * @param user_data  User data passed to orchestration function
 * @return 0 on success, negative error code on failure
 */
int a2a3_runtime_execute(void* user_data);

/**
 * Finalize the runtime and release all resources.
 * 
 * This function:
 * 1. Shuts down all worker threads
 * 2. Unloads all .so files
 * 3. Frees all allocated memory
 */
void a2a3_runtime_finalize(void);

// =============================================================================
// Memory Management API
// =============================================================================

/**
 * Allocate memory on the device.
 * 
 * @param size_bytes  Number of bytes to allocate
 * @return Pointer to allocated memory, or NULL on failure
 */
void* a2a3_runtime_malloc(size_t size_bytes);

/**
 * Free device memory.
 * 
 * @param ptr  Pointer to memory to free
 */
void a2a3_runtime_free(void* ptr);

/**
 * Copy data from host to device.
 * 
 * @param dst_device  Destination pointer on device
 * @param src_host    Source pointer on host
 * @param size_bytes  Number of bytes to copy
 * @return 0 on success, negative error code on failure
 */
int a2a3_runtime_copy_to_device(void* dst_device, const void* src_host, size_t size_bytes);

/**
 * Copy data from device to host.
 * 
 * @param dst_host    Destination pointer on host
 * @param src_device  Source pointer on device
 * @param size_bytes  Number of bytes to copy
 * @return 0 on success, negative error code on failure
 */
int a2a3_runtime_copy_from_device(void* dst_host, const void* src_device, size_t size_bytes);

// =============================================================================
// InCore Function Registry
// =============================================================================

/**
 * InCore function pointer type.
 * All InCore functions must have this signature.
 */
typedef void (*A2A3InCoreFunc)(void** args, int32_t num_args);

/**
 * Register an InCore function manually.
 * 
 * Normally, InCore functions are loaded automatically from the
 * incore_aiv_dir and incore_aic_dir directories. This function
 * allows manual registration for testing or special cases.
 * 
 * @param func_name  Function name (used for lookup)
 * @param func_ptr   Function pointer
 * @param is_cube    True if this is a Cube function (AIC), false for Vector (AIV)
 * @return 0 on success, negative error code on failure
 */
int a2a3_runtime_register_incore(const char* func_name, A2A3InCoreFunc func_ptr, bool is_cube);

/**
 * Lookup an InCore function by name.
 * 
 * @param func_name  Function name to look up
 * @return Function pointer, or NULL if not found
 */
A2A3InCoreFunc a2a3_runtime_lookup_incore(const char* func_name);

// =============================================================================
// Status and Statistics
// =============================================================================

/**
 * Runtime statistics.
 */
typedef struct {
    int64_t total_tasks_scheduled;
    int64_t total_tasks_completed;
    int64_t aiv_tasks_executed;
    int64_t aic_tasks_executed;
    double  total_execution_time_ms;
    int     num_incore_funcs_loaded;
} A2A3RuntimeStats;

/**
 * Get runtime statistics.
 * 
 * @param stats  Pointer to stats structure to fill
 */
void a2a3_runtime_get_stats(A2A3RuntimeStats* stats);

/**
 * Print runtime statistics to stdout.
 */
void a2a3_runtime_print_stats(void);

/**
 * Check if the runtime is initialized.
 * 
 * @return true if initialized, false otherwise
 */
bool a2a3_runtime_is_initialized(void);

// =============================================================================
// Error Codes
// =============================================================================

#define A2A3_SUCCESS                 0
#define A2A3_ERROR_INVALID_CONFIG   -1
#define A2A3_ERROR_SO_LOAD_FAILED   -2
#define A2A3_ERROR_FUNC_NOT_FOUND   -3
#define A2A3_ERROR_MEMORY_ALLOC     -4
#define A2A3_ERROR_THREAD_CREATE    -5
#define A2A3_ERROR_NOT_INITIALIZED  -6
#define A2A3_ERROR_ALREADY_INIT     -7

/**
 * Get error message for error code.
 * 
 * @param error_code  Error code
 * @return Human-readable error message
 */
const char* a2a3_runtime_error_string(int error_code);

#ifdef __cplusplus
}
#endif

#endif // A2A3_RUNTIME_API_H
