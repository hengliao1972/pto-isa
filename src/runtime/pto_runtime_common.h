/**
 * PTO Runtime System - Common Header (Platform Independent)
 * 
 * This header contains platform-independent data structures and API declarations
 * for the PTO runtime system.
 * 
 * Platform-independent components:
 * - Task table and task management (sliding window)
 * - TensorMap for dependency tracking
 * - Input/output argument data structures
 * - Cycle tracing infrastructure
 * 
 * Platform-dependent components (in separate files):
 * - Task completion and dependency propagation
 * - Ready queue management
 * - Worker thread implementation
 * 
 * NOTE: Record & Replay feature has been archived to pto_record_replay_archived.h
 * due to conflicts with the sliding window task management scheme.
 */

#ifndef PTO_RUNTIME_COMMON_H
#define PTO_RUNTIME_COMMON_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>

// =============================================================================
// Configuration Constants (Platform Independent)
// =============================================================================

// Sliding window for task management - limits memory usage
#define PTO_TASK_WINDOW_SIZE   8192    // Sliding window size (8K tasks in flight)
#define PTO_MAX_TASKS          PTO_TASK_WINDOW_SIZE  // Alias for backward compatibility
#define PTO_TASK_SLOT(task_id) ((task_id) & (PTO_TASK_WINDOW_SIZE - 1))  // Fast modulo (window must be power of 2)

#define PTO_MAX_ARGS           16      // Maximum arguments per task
#define PTO_TENSORMAP_SIZE     PTO_TASK_WINDOW_SIZE       // Hash bucket count (power of 2)
#define PTO_TENSORMAP_POOL_SIZE (PTO_TASK_WINDOW_SIZE * 4) // Ring pool entries (headroom for multiple outputs)
#define PTO_DEP_LIST_POOL_SIZE (PTO_TASK_WINDOW_SIZE * 16) // Dep list nodes (fanin+fanout headroom)
#define PTO_HEAP_SIZE_BYTES    (64 * 1024 * 1024)          // Packed output heap (host/sim default)
#define PTO_MAX_READY_QUEUE    65536   // Ready queue size (64K, 2x window for safety)
#define PTO_MAX_WORKERS        128     // Maximum worker threads (A2A3: 48 vector + 24 cube = 72)

// Debug output control
#ifndef PTO_DEBUG
#define PTO_DEBUG 0
#endif

#if PTO_DEBUG
#define DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
#define DEBUG_PRINT(...) ((void)0)
#endif

// =============================================================================
// Forward Declarations
// =============================================================================

struct PTORuntime;

// =============================================================================
// Core Data Structures (Platform Independent)
// =============================================================================

/**
 * Tensor region identifier
 * Uniquely identifies a tensor region by base pointer, offset, and shape
 */
typedef struct {
    void*    raw_tensor;     // Base pointer to tensor data
    int64_t  row_offset;     // Row offset within tensor
    int64_t  col_offset;     // Column offset within tensor
    int64_t  rows;           // Number of rows in this region
    int64_t  cols;           // Number of columns in this region
} TensorRegion;

/**
 * Task argument - either input or output tensor
 */
typedef struct {
    TensorRegion region;     // Tensor region
    bool         is_output;  // True if this is an output argument
} TaskArg;

// =============================================================================
// Dependency List Pool (Ring Buffer of Linked-List Nodes)
// =============================================================================

typedef struct {
    int32_t task_id;      // Dependent/dependency task id
    int32_t next_offset;  // Next node offset (0 = end)
} DepListEntry;

// =============================================================================
// TensorMap (Ring Buffer + Hash Chains with Lazy Invalidation)
// =============================================================================

typedef struct {
    TensorRegion region;        // Tensor region key
    int32_t producer_task_id;   // Producing task (absolute id)
    int32_t next_in_bucket;     // Next in hash chain (-1 = end)
    int32_t next_in_task;       // Next for same task (-1 = end)
    bool in_bucket;             // True if linked in a bucket chain
} TensorMapEntry;

typedef struct {
    int32_t buckets[PTO_TENSORMAP_SIZE];                 // Offsets into entry_pool (-1 empty)
    TensorMapEntry entry_pool[PTO_TENSORMAP_POOL_SIZE];  // Ring pool
    int32_t pool_head;                                  // Next allocation slot (wraps)
    int32_t task_entry_head[PTO_TASK_WINDOW_SIZE];       // Per-task entry head (task_id % window)
    int32_t last_task_alive;                             // Cached validity threshold
} TensorMap;

/**
 * Cycle cost function pointer type
 * Returns estimated cycle count for the InCore function
 */
typedef int64_t (*CycleCostFunc)(void** args, int32_t num_args);

/**
 * Pending task entry - Full task representation
 * Contains all information needed for task execution and dependency tracking
 */
typedef struct {
    int32_t      task_id;                    // Unique task identifier
    const char*  func_name;                  // InCore function to call
    void*        func_ptr;                   // Function pointer
    CycleCostFunc cycle_func;                // Cycle cost function (for simulation mode)
    
    // Arguments
    TaskArg      args[PTO_MAX_ARGS];         // Input/output arguments
    int32_t      num_args;                   // Number of arguments

    // Outputs (subset of args[] where is_output==true)
    int32_t      output_arg_index[PTO_MAX_ARGS];  // Indices into args[]
    int32_t      num_outputs;

    // Packed output buffer (runtime-managed, optional)
    void*        packed_buffer_base;
    void*        packed_buffer_end;
    int32_t      output_offsets[PTO_MAX_ARGS];    // Byte offsets within packed buffer (-1 for external)
    int32_t      dep_pool_end;                    // DepList pool top snapshot after submit
    
    // Buffer size estimation
    int32_t      buffer_size_bytes;          // Estimated InCore tile buffer size
    int32_t      buffer_size_with_reuse;     // Buffer size with reuse optimization
    
    // Dependency tracking (static; written by orchestrator)
    int32_t      fanin_head;                 // DepListEntry offset (0 = none)
    int32_t      fanin_count;                // Total dependencies (immutable after submit)
    int32_t      fanout_head;                // DepListEntry offset (consumers only)
    int32_t      fanout_consumer_count;      // Consumers count (fanout list length)
    int32_t      fanout_count;               // Total refs = scope_depth + consumers
    int32_t      scope_depth;                // Depth at creation time
    volatile int32_t fanout_lock;            // Protect fanout_head/fanout_count atomicity
    
    // Status
    bool         is_active;                  // Task slot is in use
    bool         is_submitted;               // Task has been submitted to scheduler
    bool         is_complete;                // Task has finished execution
    bool         is_consumed;                // Output fully consumed (eligible for reclamation)
    
    // Worker type hint (for heterogeneous backends like a2a3)
    bool         is_cube;                    // True if requires cube unit (matmul)
    
    // Timing (for dependency-aware simulation)
    int64_t      earliest_start_cycle;       // Earliest time this task can start (after deps)
    int64_t      end_cycle;                  // Time when this task finished
} PendingTask;

// =============================================================================
// Cycle Trace Data Structures
// =============================================================================

#define PTO_MAX_TRACE_ENTRIES 65536    // 64K trace entries (limited by window)
#define PTO_MAX_FUNC_NAME_LEN 64

/**
 * Single trace entry recording one task execution
 */
typedef struct {
    char func_name[PTO_MAX_FUNC_NAME_LEN];
    int32_t worker_id;
    int64_t start_cycle;
    int64_t end_cycle;
} CycleTraceEntry;

/**
 * Cycle trace buffer for recording task execution timing
 */
typedef struct {
    CycleTraceEntry entries[PTO_MAX_TRACE_ENTRIES];
    int32_t count;
    int32_t num_workers;
    int32_t num_vector_workers;  // Number of vector workers (for A2A3)
    int32_t num_cube_workers;    // Number of cube workers (for A2A3)
    int64_t per_worker_cycle[PTO_MAX_WORKERS];  // Current cycle per worker
    bool enabled;
} CycleTrace;

// =============================================================================
// Runtime Context Structure (Platform Independent Core)
// =============================================================================

/**
 * Runtime execution mode - controls window overflow behavior
 */
typedef enum {
    PTO_MODE_BENCHMARK_ONLY = 0,  // No window check, no execution (measure orchestration)
    PTO_MODE_DUMP_GRAPH,          // Abort when window full (limit dump/graph size)
    PTO_MODE_EXECUTE,             // Stall when window full (actual execution)
    PTO_MODE_SIMULATE             // Stall when window full (cycle-accurate simulation)
} PTORuntimeMode;

typedef enum {
    PTO_TASK_PENDING = 0,
    PTO_TASK_READY = 1,
    PTO_TASK_RUNNING = 2,
    PTO_TASK_COMPLETED = 3,
    PTO_TASK_CONSUMED = 4,
} PTOTaskState;

/**
 * PTO Runtime context
 * 
 * Contains both platform-independent state and platform-specific extensions.
 * Platform-specific code should access only the relevant fields.
 */
typedef struct PTORuntime {
    // =========================================================================
    // Platform-Independent: Task Management with Sliding Window
    // =========================================================================
    
    PendingTask  pend_task[PTO_TASK_WINDOW_SIZE];    // Sliding window task table
    int32_t      next_task_id;                // Next task ID to allocate (absolute, not wrapped)
    int32_t      active_task_count;           // Number of active tasks in window
    
    // Sliding window tracking (oldest task not yet CONSUMED)
    int32_t      last_task_alive;             // Oldest task not yet consumed (absolute ID)
    bool         window_aborted;              // True if orchestration was aborted due to full window
    PTORuntimeMode runtime_mode;              // Current execution mode
    
    // Dependency list pool (ring buffer)
    DepListEntry dep_list_pool[PTO_DEP_LIST_POOL_SIZE];
    int32_t      dep_list_top;            // Next allocation offset (0 reserved for NULL)
    int32_t      dep_list_tail;           // Oldest live offset (tracks last_task_alive)

    // TensorMap for dependency tracking (ring buffer + lazy invalidation)
    TensorMap     tensor_map;

    // Packed output heap ring (host/sim)
    uint8_t*      heap_base;
    int32_t       heap_size;
    int32_t       heap_top;               // Allocation pointer (bytes, wraps)
    int32_t       heap_tail;              // Free pointer (bytes, tracks last_task_alive)

    // Scheduler-owned dynamic state arrays (indexed by PTO_TASK_SLOT(task_id))
    int32_t       fanin_refcount[PTO_TASK_WINDOW_SIZE];
    int32_t       fanout_refcount[PTO_TASK_WINDOW_SIZE];
    uint8_t       task_state[PTO_TASK_WINDOW_SIZE];
    
    // Statistics (absolute counts, not affected by window wrap)
    int64_t      total_tasks_scheduled;
    int64_t      total_tasks_completed;
    
    // =========================================================================
    // Platform-Specific: Ready Queues
    // =========================================================================
    
    // Single ready queue (for single-queue platforms like ARM64)
    int32_t      ready_queue[PTO_MAX_READY_QUEUE];
    int32_t      ready_head;
    int32_t      ready_tail;
    int32_t      ready_count;
    
    // Dual ready queues (for A2A3: separate vector and cube queues)
    int32_t      vector_ready_queue[PTO_MAX_READY_QUEUE];  // is_cube=0 tasks
    int32_t      vector_ready_head;
    int32_t      vector_ready_tail;
    int32_t      vector_ready_count;
    
    int32_t      cube_ready_queue[PTO_MAX_READY_QUEUE];    // is_cube=1 tasks
    int32_t      cube_ready_head;
    int32_t      cube_ready_tail;
    int32_t      cube_ready_count;
    
    // =========================================================================
    // Platform-Specific: Thread Synchronization
    // =========================================================================
    
    pthread_mutex_t   queue_mutex;           // Protects ready_queue access
    pthread_mutex_t   task_mutex;            // Protects task state updates
    pthread_cond_t    queue_not_empty;       // Signaled when task added to queue
    pthread_cond_t    all_done;              // Signaled when all tasks complete
    pthread_cond_t    vector_queue_not_empty;// For A2A3 vector queue
    pthread_cond_t    cube_queue_not_empty;  // For A2A3 cube queue
    pthread_cond_t    window_not_full;       // Signaled when window has space
    
    // Worker threads
    pthread_t         workers[PTO_MAX_WORKERS];
    int32_t           num_workers;           // Total number of worker threads
    int32_t           num_vector_workers;    // Number of vector workers (A2A3)
    int32_t           num_cube_workers;      // Number of cube workers (A2A3)
    volatile bool     shutdown_requested;    // Signal workers to exit
    volatile bool     execution_started;     // Orchestration has submitted all tasks
    volatile bool     orchestration_complete; // Orchestration function has finished
    int32_t           execution_task_threshold;  // Start workers when task_count > threshold
    
    // Dependency resolver threads (A2A3)
    pthread_t         dep_resolvers[8];      // Up to 8 dependency resolver threads
    int32_t           num_dep_resolvers;     // Number of dep resolver threads (default: 3)
    
    // Orchestration thread (A2A3)
    pthread_t         orch_thread;           // Orchestration thread handle

    // Scope stack (orchestrator only)
    int32_t           scope_stack[1024];
    int32_t           scope_stack_top;
    
    // =========================================================================
    // Platform-Specific: Mode Configuration
    // =========================================================================
    
    bool              simulation_mode;       // If true, call cycle_func and record traces
    bool              dual_queue_mode;       // If true, use separate cube/vector queues
    
    // InCore function registry (maps func_name to actual function pointer)
    void*             func_registry[1024];  // Function pointer cache (limited number of unique funcs)
} PTORuntime;

// =============================================================================
// Function Pointer Types
// =============================================================================

/**
 * InCore function signature
 * All InCore functions must match this signature
 */
typedef void (*PTOInCoreFunc)(void** args, int32_t num_args);

/**
 * Orchestration function signature
 * Called to build the task graph
 */
typedef void (*PTOOrchFunc)(PTORuntime* rt, void* user_data);

// =============================================================================
// Platform-Independent API: Runtime Lifecycle
// =============================================================================

/**
 * Initialize the PTO runtime (platform-independent parts)
 */
void pto_runtime_init(PTORuntime* rt);

/**
 * Shutdown the PTO runtime and free resources
 */
void pto_runtime_shutdown(PTORuntime* rt);

/**
 * Set runtime execution mode
 * 
 * @param rt   Runtime context
 * @param mode Execution mode:
 *             - PTO_MODE_BENCHMARK_ONLY: No window check, measure orchestration speed
 *             - PTO_MODE_DUMP_GRAPH: Abort when window full (limit dump size)
 *             - PTO_MODE_EXECUTE: Stall when window full (actual execution)
 *             - PTO_MODE_SIMULATE: Stall when window full (simulation)
 */
void pto_runtime_set_mode(PTORuntime* rt, PTORuntimeMode mode);

/**
 * Check if orchestration was aborted due to window full
 */
bool pto_runtime_was_aborted(PTORuntime* rt);

/**
 * Reset runtime state for next run (keeps mode, clears tasks)
 */
void pto_runtime_reset(PTORuntime* rt);

/**
 * Print runtime statistics
 */
void pto_runtime_stats(PTORuntime* rt);

// =============================================================================
// Orchestrator Scope API (buffer lifetime management)
// =============================================================================

void pto_scope_begin(PTORuntime* rt);
void pto_scope_end(PTORuntime* rt);
int32_t pto_get_scope_depth(PTORuntime* rt);

// =============================================================================
// Platform-Independent API: Task Allocation and Arguments
// =============================================================================

/**
 * Allocate a new task ID and initialize task entry
 * 
 * @param rt            Runtime context
 * @param func_name     InCore function name
 * @param func_ptr      Function pointer (can be NULL)
 * @param buffer_bytes  Estimated tile buffer size in bytes (without reuse)
 * @param reuse_bytes   Estimated tile buffer size with reuse optimization
 * @param is_cube       If true, task requires cube unit (scheduled on cube workers)
 * @return task_id or -1 on failure
 */
int32_t pto_task_alloc_impl(PTORuntime* rt, const char* func_name, void* func_ptr,
                            int32_t buffer_bytes, int32_t reuse_bytes, bool is_cube);

/**
 * Backward compatible task alloc variants
 */
static inline int32_t pto_task_alloc_5(PTORuntime* rt, const char* func_name, 
                                       void* func_ptr, int32_t buffer_bytes, 
                                       int32_t reuse_bytes) {
    return pto_task_alloc_impl(rt, func_name, func_ptr, buffer_bytes, reuse_bytes, false);
}

static inline int32_t pto_task_alloc_6(PTORuntime* rt, const char* func_name, 
                                       void* func_ptr, int32_t buffer_bytes, 
                                       int32_t reuse_bytes, bool is_cube) {
    return pto_task_alloc_impl(rt, func_name, func_ptr, buffer_bytes, reuse_bytes, is_cube);
}

// Macro to select correct overload based on argument count
#define _PTO_TASK_ALLOC_NARG(...) _PTO_TASK_ALLOC_NARG_(__VA_ARGS__, 6, 5, 4, 3, 2, 1, 0)
#define _PTO_TASK_ALLOC_NARG_(_1, _2, _3, _4, _5, _6, N, ...) N
#define _PTO_TASK_ALLOC_DISPATCH(N) _PTO_TASK_ALLOC_DISPATCH_(N)
#define _PTO_TASK_ALLOC_DISPATCH_(N) pto_task_alloc_##N
#define pto_task_alloc(...) _PTO_TASK_ALLOC_DISPATCH(_PTO_TASK_ALLOC_NARG(__VA_ARGS__))(__VA_ARGS__)

/**
 * Set the cycle cost function for a task (for simulation mode)
 */
void pto_task_set_cycle_func(PTORuntime* rt, int32_t task_id, CycleCostFunc cycle_func);

/**
 * Add an input argument to a task
 * Looks up producer in TensorMap and updates dependencies
 */
void pto_task_add_input(PTORuntime* rt, int32_t task_id,
                        void* tensor, int64_t row_off, int64_t col_off,
                        int64_t rows, int64_t cols);

/**
 * Add an output argument to a task
 * Output regions are registered in TensorMap on task submit.
 */
void pto_task_add_output(PTORuntime* rt, int32_t task_id,
                         void* tensor, int64_t row_off, int64_t col_off,
                         int64_t rows, int64_t cols);

/**
 * Finalize a task for submission:
 * - Allocates packed outputs for output args with NULL base pointer
 * - Registers outputs in TensorMap
 * - Marks task as submitted
 * @return true if task is ready (all deps satisfied)
 */
bool pto_task_prepare_submit(PTORuntime* rt, int32_t task_id);

// =============================================================================
// Internal Helpers (callers must hold rt->task_mutex)
// =============================================================================

bool pto_try_mark_consumed_locked(PTORuntime* rt, int32_t task_id);
bool pto_advance_last_task_alive_locked(PTORuntime* rt);

// =============================================================================
// Platform-Independent API: TensorMap
// =============================================================================

/**
 * Compute hash for tensor region
 */
uint32_t pto_tensormap_hash(TensorRegion* region);

/**
 * Check if two tensor regions match
 */
bool pto_region_match(TensorRegion* a, TensorRegion* b);

/**
 * Lookup producer task for a tensor region
 * @return task_id or -1 if not found
 */
int32_t pto_tensormap_lookup(PTORuntime* rt, TensorRegion* region);

/**
 * Insert/update tensor region -> task mapping
 */
void pto_tensormap_insert(PTORuntime* rt, TensorRegion* region, int32_t task_id);

/**
 * Clear the tensor map
 */
void pto_tensormap_clear(PTORuntime* rt);

/**
 * Garbage collect stale entries from the tensor map
 * No-op in current design: stale tails are truncated during lookup and
 * entries are unlinked from buckets on pool reuse.
 */
void pto_tensormap_gc(PTORuntime* rt);

// =============================================================================
// Platform-Independent API: Cycle Tracing
// =============================================================================

/**
 * Global cycle trace (for single-trace use case)
 */
extern CycleTrace* pto_global_trace;

/**
 * Initialize cycle tracing
 */
void pto_trace_init(int32_t num_workers);

/**
 * Initialize cycle tracing for dual-queue mode (A2A3)
 * Vector workers: 0 to num_vector_workers-1
 * Cube workers: num_vector_workers to num_vector_workers+num_cube_workers-1
 */
void pto_trace_init_dual(int32_t num_vector_workers, int32_t num_cube_workers);

/**
 * Record a task execution (simple version - no dependency tracking)
 */
void pto_trace_record(int32_t worker_id, const char* func_name, int64_t cycle_cost);

/**
 * Record a task execution with explicit timing (dependency-aware)
 */
void pto_trace_record_with_time(int32_t worker_id, const char* func_name, 
                                 int64_t start_cycle, int64_t end_cycle);

/**
 * Get the current cycle for a worker
 */
int64_t pto_trace_get_cycle(int32_t worker_id);

/**
 * Cleanup trace resources
 */
void pto_trace_cleanup(void);

/**
 * Generate Chrome Tracing JSON format
 * @return newly allocated string that caller must free
 */
char* pto_trace_to_chrome_json(void);

/**
 * Write trace to file in Chrome Tracing JSON format
 */
void pto_trace_write_json(const char* filename);

/**
 * Print trace summary statistics
 */
void pto_trace_print_summary(void);

// =============================================================================
// Platform-Independent API: Debug and Dump
// =============================================================================

/**
 * Dump runtime state to a text file
 */
int pto_runtime_dump(PTORuntime* rt, const char* filename);

/**
 * Dump runtime state to stdout (condensed format)
 */
int pto_runtime_dump_stdout(PTORuntime* rt);

// =============================================================================
// Platform-Dependent API (to be implemented by platform-specific files)
// =============================================================================

/**
 * Submit a task to the runtime
 * Platform-specific: determines which ready queue to use
 */
void pto_task_submit(PTORuntime* rt, int32_t task_id);

/**
 * Mark a task as complete and update dependencies
 * Platform-specific: handles dependency propagation
 */
void pto_task_complete(PTORuntime* rt, int32_t task_id);

/**
 * Thread-safe version of pto_task_complete
 */
void pto_task_complete_threadsafe(PTORuntime* rt, int32_t task_id);

/**
 * Get next ready task from queue (non-blocking)
 * @return task_id or -1 if no tasks ready
 */
int32_t pto_get_ready_task(PTORuntime* rt);

/**
 * Get next ready task from queue (blocking)
 * @return task_id or -1 if shutdown
 */
int32_t pto_get_ready_task_blocking(PTORuntime* rt);

/**
 * Execute all pending tasks until completion (single-threaded, mainly for debug)
 */
void pto_execute_all(PTORuntime* rt);

/**
 * Execute a single task with specified worker ID
 */
void pto_execute_task_with_worker(PTORuntime* rt, int32_t task_id, int32_t worker_id);

/**
 * Register an InCore function with the runtime
 */
void pto_register_incore_func(PTORuntime* rt, const char* func_name, PTOInCoreFunc func_ptr);

// =============================================================================
// Cycle-Accurate Simulation API
// =============================================================================

/**
 * Estimate cycle cost based on function name pattern.
 * Uses A2A3 Core Simulator when available (compile with -DA2A3_CORE_SIM_AVAILABLE
 * and link with liba2a3_core.a), otherwise uses heuristic estimation.
 */
int64_t pto_estimate_cycle_cost(const char* func_name);

/**
 * Cleanup the core simulator (called automatically by pto_runtime_shutdown).
 * Safe to call multiple times.
 */
void pto_cleanup_core_sim(void);

/**
 * Simulate all pending tasks with cycle-accurate timing.
 * 
 * This function:
 * 1. Processes tasks in dependency order
 * 2. Estimates cycle costs based on function names
 * 3. Records trace entries for visualization
 * 4. Simulates multi-worker parallel execution
 * 
 * @param rt The runtime instance (must have simulation_mode enabled)
 */
void pto_simulate_all(PTORuntime* rt);

#endif // PTO_RUNTIME_COMMON_H
