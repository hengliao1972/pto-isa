/**
 * PTO Runtime2 - Core Type Definitions
 * 
 * This header defines all fundamental types used by the PTO Runtime2 system:
 * - Configuration constants
 * - Worker types and task states
 * - Tensor regions and task parameters
 * - Task descriptors with fanin/fanout tracking
 * - Dependency list entries
 * 
 * Based on: docs/runtime_buffer_manager_methods.md
 */

#ifndef PTO_RUNTIME2_TYPES_H
#define PTO_RUNTIME2_TYPES_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

// =============================================================================
// Configuration Constants
// =============================================================================

// Task management
// NOTE: PTO2_TASK_WINDOW_SIZE is now the DEFAULT value only.
// Actual window size is passed at runtime to pto2_runtime_create_threaded_custom().
// Use pto2_task_slot(sched, task_id) instead of PTO2_TASK_SLOT macro.
#define PTO2_TASK_WINDOW_SIZE     16384   // Default task window size (power of 2)

// DEPRECATED: Use pto2_task_slot(sched, task_id) instead
// This macro only works when runtime window == PTO2_TASK_WINDOW_SIZE
#define PTO2_TASK_SLOT(task_id)   ((task_id) & (PTO2_TASK_WINDOW_SIZE - 1))

// Memory pools
#define PTO2_HEAP_SIZE            (64 * 1024 * 1024)  // 64MB default heap
#define PTO2_DEP_LIST_POOL_SIZE   65536   // Dependency list pool entries
#define PTO2_TENSORMAP_POOL_SIZE  32768   // TensorMap entry pool
#define PTO2_TENSORMAP_NUM_BUCKETS 4096   // Power of 2 for fast hash

// Task parameters
#define PTO2_MAX_OUTPUTS          16      // Maximum outputs per task
#define PTO2_MAX_INPUTS           16      // Maximum inputs per task
#define PTO2_MAX_INOUTS           8       // Maximum in-out params per task

// Scope management
#define PTO2_MAX_SCOPE_DEPTH      64      // Maximum nesting depth

// Ready queue
#define PTO2_READY_QUEUE_SIZE     65536   // Per-worker-type queue size (16x larger to avoid queue full)

// Memory alignment
#define PTO2_ALIGN_SIZE           64      // Cache line alignment
#define PTO2_ALIGN_UP(x, align)   (((x) + (align) - 1) & ~((align) - 1))

// TensorMap cleanup interval
#define PTO2_TENSORMAP_CLEANUP_INTERVAL 64  // Cleanup every N retired tasks

// =============================================================================
// Worker Types
// =============================================================================

/**
 * Worker type enumeration
 * Each worker type has its own ready queue for load balancing
 */
typedef enum {
    PTO2_WORKER_CUBE = 0,       // AICore CUBE unit (matrix ops)
    PTO2_WORKER_VECTOR = 1,     // AICore VECTOR unit (element-wise ops)
    PTO2_WORKER_AI_CPU = 2,     // AI_CPU (scalar ops, control flow)
    PTO2_WORKER_ACCELERATOR = 3,// Fixed-function accelerators (DMA, etc.)
    PTO2_NUM_WORKER_TYPES = 4
} PTO2WorkerType;

// =============================================================================
// Task States
// =============================================================================

/**
 * Task state enumeration
 * 
 * State transitions:
 *   PENDING -> READY -> RUNNING -> COMPLETED -> CONSUMED
 * 
 * Conditions:
 *   PENDING->READY:     fanin_refcount == fanin_count
 *   COMPLETED->CONSUMED: fanout_refcount == fanout_count && state == COMPLETED
 */
typedef enum {
    PTO2_TASK_PENDING = 0,    // Waiting for dependencies (fanin_refcount < fanin_count)
    PTO2_TASK_READY = 1,      // All dependencies satisfied, waiting in ready queue
    PTO2_TASK_RUNNING = 2,    // Currently executing on a worker
    PTO2_TASK_COMPLETED = 3,  // Execution finished, output may still be in use
    PTO2_TASK_CONSUMED = 4    // Output fully consumed, buffers can be released
} PTO2TaskState;

// =============================================================================
// Tensor Region (Legacy, for simple 1D regions)
// =============================================================================

/**
 * Tensor region identifier
 * Uniquely identifies a region within a tensor buffer
 */
typedef struct {
    void*    base_ptr;        // Buffer base pointer
    int32_t  tile_index;      // Tile index within buffer
    int32_t  offset;          // Byte offset within tile
    int32_t  size;            // Region size in bytes
} PTO2TensorRegion;

// =============================================================================
// Logical Tensor (for view/reshape/transpose operations)
// =============================================================================

/**
 * Maximum dimensions supported for logical tensors
 */
#define PTO2_MAX_TENSOR_DIM   8

/**
 * Tensor extraction type (for tracking how tensor was created)
 */
typedef enum {
    PTO2_TENSOR_RAW = 0,           // Original raw tensor (owns storage)
    PTO2_TENSOR_VIEW = 1,          // view() - subset selection, shared storage
    PTO2_TENSOR_RESHAPE = 2,       // reshape() - shape change, shared storage
    PTO2_TENSOR_TRANSPOSE = 3,     // transpose() - dimension permute, shared storage
    PTO2_TENSOR_DEEP_VIEW = 4,     // deep_view() - copied subset, new storage
    PTO2_TENSOR_DEEP_RESHAPE = 5,  // deep_reshape() - copied reshape, new storage
    PTO2_TENSOR_DEEP_TRANSPOSE = 6 // deep_transpose() - copied transpose, new storage
} PTO2TensorExtractionType;

/**
 * Raw tensor (storage provider)
 * 
 * The raw tensor owns the actual memory allocation.
 * Multiple logical tensors can share the same raw tensor (aliasing).
 */
typedef struct {
    void*    base_ptr;        // Base pointer of allocated memory
    int64_t  total_size;      // Total size in bytes
    int32_t  refcount;        // Number of logical tensors referencing this storage
                              // (for memory management, 0 = can be freed)
} PTO2RawTensor;

/**
 * Logical tensor structure
 * 
 * A "view" into raw tensor storage with specific layout.
 * Supports multi-dimensional tensors with strides (for view/reshape/transpose).
 * 
 * Memory footprint is determined by:
 *   - storage_offset: byte offset from raw_base to first element
 *   - shape[d]: number of elements in dimension d
 *   - strides[d]: byte offset between consecutive elements in dimension d
 * 
 * For element at indices [i0, i1, ..., i_{n-1}]:
 *   byte_offset = storage_offset + sum(i_d * strides[d])
 * 
 * Examples:
 *   - Contiguous row-major (3,4): shape=[3,4], strides=[4*elem_size, elem_size]
 *   - Transposed (4,3): shape=[4,3], strides=[elem_size, 4*elem_size]
 *   - Sliced [1:3, 1:3]: offset adjusted, shape=[2,2], strides unchanged
 */
typedef struct {
    // === Raw tensor reference (shared storage) ===
    void*    raw_base;            // Pointer to raw tensor's base (for aliasing check)
    int64_t  raw_total_size;      // Total size of raw tensor in bytes
    
    // === Storage offset ===
    int64_t  storage_offset;      // Byte offset from raw_base to first element
    
    // === Shape and strides ===
    int64_t  shape[PTO2_MAX_TENSOR_DIM];    // Size in each dimension
    int64_t  strides[PTO2_MAX_TENSOR_DIM];  // Byte stride in each dimension
    int32_t  ndim;                          // Number of dimensions (0 = scalar)
    
    // === Precomputed bounding box (for fast overlap detection) ===
    int64_t  min_byte_offset;     // First byte accessed (relative to raw_base)
    int64_t  max_byte_offset;     // Last byte accessed (relative to raw_base)
    
    // === Element info ===
    int64_t  elem_size;           // Size of each element in bytes
    int64_t  numel;               // Total number of elements
    
    // === Extraction tracking ===
    PTO2TensorExtractionType extraction_type;  // How this tensor was created
    bool     is_contiguous;       // True if memory is contiguous (no gaps)
    
} PTO2LogicalTensor;

// =============================================================================
// Task Parameter
// =============================================================================

/**
 * Task parameter type enumeration
 */
typedef enum {
    PTO2_PARAM_INPUT = 0,     // Read-only input
    PTO2_PARAM_OUTPUT = 1,    // Write-only output
    PTO2_PARAM_INOUT = 2      // Read-write (accumulation)
} PTO2ParamType;

/**
 * Task parameter descriptor
 * Describes one input/output/inout buffer for a task
 */
typedef struct {
    PTO2ParamType type;       // Parameter type
    void*         buffer;     // Buffer base pointer
    int32_t       tile_index; // Tile index
    int32_t       size;       // Size in bytes
} PTO2TaskParam;

// =============================================================================
// Dependency List Entry
// =============================================================================

/**
 * Dependency list entry (singly-linked list node)
 * Stored in DepListPool ring buffer
 * 
 * Used for both fanin_list and fanout_list
 */
typedef struct {
    int32_t task_id;          // The dependent/dependency task ID
    int32_t next_offset;      // Offset to next entry (0 = end of list)
} PTO2DepListEntry;

// =============================================================================
// Task Descriptor
// =============================================================================

/**
 * Task descriptor structure
 * 
 * Stored in the TaskDescriptor ring buffer in shared memory.
 * Contains both static info (set at submission) and dynamic state.
 * 
 * Concurrency notes:
 * - fanout_head, fanout_count protected by fanout_lock (per-task spinlock)
 * - fanin_head, fanin_count set once at submission, read-only after
 * - Other fields set by Orchestrator, read by Scheduler
 */
typedef struct {
    // Task identification
    int32_t task_id;              // Unique task identifier (absolute, not wrapped)
    int32_t kernel_id;            // InCore function to execute
    int32_t worker_type;          // Target: CUBE, VECTOR, AI_CPU, ACCELERATOR
    int32_t scope_depth;          // Depth of scope when task was created
    
    // Dependency lists (linked list heads - offsets into DepListPool)
    // Fanin: producers this task depends on (set once at submission)
    int32_t fanin_head;           // Offset to first fanin entry (0 = empty)
    int32_t fanin_count;          // Number of producer dependencies
    
    // Fanout: consumers that depend on this task (grows as consumers submit)
    // PROTECTED BY fanout_lock
    volatile int32_t fanout_lock; // Per-task spinlock (0=unlocked, 1=locked)
    volatile int32_t fanout_head; // Offset to first fanout entry (0 = empty)
    volatile int32_t fanout_count;// Total consumers + scope_depth (for lifecycle)
    
    // Packed output buffer (all outputs packed into single contiguous buffer)
    void*    packed_buffer_base;  // Start of packed buffer in GM Heap
    void*    packed_buffer_end;   // End of packed buffer (for heap reclamation)
    int32_t  output_offsets[PTO2_MAX_OUTPUTS]; // Offset of each output within packed buffer
    int32_t  num_outputs;         // Number of output buffers
    
    // Input buffer pointers (for dependency resolution)
    int32_t  num_inputs;          // Number of input buffers
    
    // Function pointer (for execution)
    void*    func_ptr;            // InCore function pointer
    const char* func_name;        // Function name (for debugging/tracing)
    
    // Status flags
    bool     is_active;           // Task slot is in use
    
} PTO2TaskDescriptor;

// =============================================================================
// TensorMap Entry
// =============================================================================

/**
 * TensorMap entry structure
 * Maps tensor region -> producer task ID
 * 
 * Stored in ring buffer pool with lazy invalidation:
 * - Entry is valid only if producer_task_id >= last_task_alive
 * - Stale entries ignored during lookup
 * - Pool wraps around, overwriting stale entries
 * 
 * Chain truncation optimization:
 * - Entries in bucket chains sorted by task_id (newest first)
 * - When lookup hits stale entry, truncate rest of chain
 */
typedef struct {
    PTO2TensorRegion region;      // Tensor region key (legacy, for simple 1D)
    int32_t producer_task_id;     // Task that produces this region
    int32_t next_in_bucket;       // Offset to next entry in hash bucket (-1 = end)
    int32_t next_in_task;         // Offset to next entry for same task (-1 = end)
    bool    in_bucket;            // True if entry is linked in a bucket chain
                                  // CRITICAL: Must be set false before overwriting!
} PTO2TensorMapEntry;

/**
 * Extended TensorMap entry structure (for LogicalTensor support)
 * 
 * Supports multi-dimensional tensors with view/reshape/transpose operations.
 * Uses bounding box for fast overlap detection.
 * 
 * Hash strategy: 
 *   - Primary key: raw_base (groups all views of same storage)
 *   - Within bucket: check bounding box overlap
 */
typedef struct {
    // === Raw tensor identification (for grouping into same bucket) ===
    void*    raw_base;            // Base pointer of raw tensor storage
    int64_t  raw_total_size;      // Total size of raw tensor (for validation)
    
    // === Bounding box (precomputed for fast overlap check) ===
    // Overlap if: (A.min <= B.max) && (B.min <= A.max)
    int64_t  min_byte_offset;     // First byte accessed (relative to raw_base)
    int64_t  max_byte_offset;     // Last byte accessed (relative to raw_base)
    
    // === Full logical tensor info (for GCD-based exact check, optional) ===
    int64_t  storage_offset;      // Byte offset from raw_base
    int64_t  shape[PTO2_MAX_TENSOR_DIM];    // Shape in each dimension
    int64_t  strides[PTO2_MAX_TENSOR_DIM];  // Strides in each dimension
    int32_t  ndim;                // Number of dimensions
    
    // === Producer tracking ===
    int32_t  producer_task_id;    // Task that produces this tensor
    
    // === Chain links ===
    int32_t  next_in_bucket;      // Offset to next entry in hash bucket (-1 = end)
    int32_t  next_in_task;        // Offset to next entry for same task (-1 = end)
    bool     in_bucket;           // True if entry is linked in a bucket chain
    
    // === Flags ===
    bool     is_deep_copy;        // True if this is a deep copy (independent storage)
    
} PTO2TensorMapEntryEx;

// =============================================================================
// Cycle Cost Function Type
// =============================================================================

/**
 * Cycle cost function pointer type
 * Returns estimated cycle count for the InCore function
 */
typedef int64_t (*PTO2CycleCostFunc)(void** args, int32_t num_args);

// =============================================================================
// InCore Function Type
// =============================================================================

/**
 * InCore function signature
 * All InCore functions must match this signature
 */
typedef void (*PTO2InCoreFunc)(void** args, int32_t num_args);

// =============================================================================
// Utility Macros
// =============================================================================

/**
 * Memory barrier macros for different architectures
 */
#if defined(__aarch64__)
    #define PTO2_MEMORY_BARRIER()     __asm__ __volatile__("dmb sy" ::: "memory")
    #define PTO2_LOAD_ACQUIRE(ptr)    __atomic_load_n(ptr, __ATOMIC_ACQUIRE)
    #define PTO2_STORE_RELEASE(ptr, val) __atomic_store_n(ptr, val, __ATOMIC_RELEASE)
#elif defined(__x86_64__)
    #define PTO2_MEMORY_BARRIER()     __asm__ __volatile__("mfence" ::: "memory")
    #define PTO2_LOAD_ACQUIRE(ptr)    __atomic_load_n(ptr, __ATOMIC_ACQUIRE)
    #define PTO2_STORE_RELEASE(ptr, val) __atomic_store_n(ptr, val, __ATOMIC_RELEASE)
#else
    #define PTO2_MEMORY_BARRIER()     __sync_synchronize()
    #define PTO2_LOAD_ACQUIRE(ptr)    __atomic_load_n(ptr, __ATOMIC_ACQUIRE)
    #define PTO2_STORE_RELEASE(ptr, val) __atomic_store_n(ptr, val, __ATOMIC_RELEASE)
#endif

/**
 * Pause instruction for spin-wait loops
 * Include sched_yield() to prevent CPU starvation of other threads
 */
#include <sched.h>
#if defined(__aarch64__)
    #define PTO2_SPIN_PAUSE()         do { __asm__ __volatile__("yield" ::: "memory"); sched_yield(); } while(0)
#elif defined(__x86_64__)
    #define PTO2_SPIN_PAUSE()         do { __builtin_ia32_pause(); sched_yield(); } while(0)
#else
    #define PTO2_SPIN_PAUSE()         sched_yield()
#endif

/**
 * Atomic compare-and-swap
 */
#define PTO2_CAS(ptr, expected, desired) \
    __atomic_compare_exchange_n(ptr, expected, desired, false, \
                                __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)

/**
 * Atomic fetch-and-add
 */
#define PTO2_FETCH_ADD(ptr, val) \
    __atomic_fetch_add(ptr, val, __ATOMIC_ACQ_REL)

/**
 * Atomic exchange
 */
#define PTO2_EXCHANGE(ptr, val) \
    __atomic_exchange_n(ptr, val, __ATOMIC_ACQ_REL)

// =============================================================================
// Multi-Threading Support
// =============================================================================

#include <pthread.h>

// Maximum number of worker threads
#define PTO2_MAX_WORKERS          128
#define PTO2_MAX_CUBE_WORKERS     64
#define PTO2_MAX_VECTOR_WORKERS   64

// Forward declarations
struct PTO2Runtime;
struct PTO2SchedulerState;

/**
 * Worker context for each worker thread
 */
typedef struct {
    int32_t         worker_id;        // Unique worker ID
    int32_t         worker_type;      // PTO2WorkerType (CUBE, VECTOR, etc.)
    struct PTO2Runtime* runtime;      // Runtime reference
    
    // Thread control
    pthread_t       thread;           // Thread handle
    volatile bool   shutdown;         // Shutdown signal
    
    // Statistics
    int64_t         tasks_executed;   // Number of tasks executed
    int64_t         total_cycles;     // Total execution cycles
    int64_t         total_stall_cycles; // Cycles spent waiting
    
    // Current task (for tracing)
    int32_t         current_task_id;  // Currently executing task (-1 if idle)
    int64_t         task_start_cycle; // When current task started
    
} PTO2WorkerContext;

/**
 * Completion queue entry for worker->scheduler communication
 */
typedef struct {
    int32_t task_id;                  // Completed task ID
    int32_t worker_id;                // Worker that completed the task
    int64_t start_cycle;              // When task started
    int64_t end_cycle;                // When task completed
} PTO2CompletionEntry;

/**
 * Completion queue (lock-free SPSC or MPSC)
 */
typedef struct {
    PTO2CompletionEntry* entries;     // Circular buffer
    int32_t capacity;                 // Queue capacity
    volatile int32_t head;            // Consumer reads from here
    volatile int32_t tail;            // Producers write here
    pthread_mutex_t mutex;            // For MPSC synchronization
} PTO2CompletionQueue;

/**
 * Thread context for managing all runtime threads
 */
typedef struct {
    // Orchestrator thread
    pthread_t orchestrator_thread;
    void* (*orchestrator_func)(void*);
    void* orchestrator_arg;
    volatile bool orchestrator_done;
    
    // Scheduler thread
    pthread_t scheduler_thread;
    volatile bool scheduler_running;
    
    // Worker threads
    PTO2WorkerContext workers[PTO2_MAX_WORKERS];
    int32_t num_cube_workers;
    int32_t num_vector_workers;
    int32_t num_workers;              // Total workers
    
    // Ready queue synchronization (per worker type)
    pthread_mutex_t ready_mutex[PTO2_NUM_WORKER_TYPES];
    pthread_cond_t  ready_cond[PTO2_NUM_WORKER_TYPES];
    
    // Per-worker condition variables for selective wakeup
    // Only the worker with smallest clock gets signaled
    pthread_cond_t  worker_cond[PTO2_MAX_WORKERS];
    volatile bool   worker_waiting[PTO2_MAX_WORKERS];  // Track which workers are waiting
    
    // Completion queue (workers -> scheduler)
    PTO2CompletionQueue completion_queue;
    pthread_cond_t completion_cond;   // Signal scheduler when completions ready
    
    // Global shutdown signal
    volatile bool shutdown;
    
    // All-done signaling
    pthread_mutex_t done_mutex;
    pthread_cond_t  all_done_cond;
    volatile bool   all_done;
    
    // Cycle counter for simulation (atomic)
    volatile int64_t global_cycle;
    
    // Per-task end cycles for dependency tracking in simulation
    // task_end_cycles[task_id % TASK_WINDOW_SIZE] = end_cycle
    volatile int64_t* task_end_cycles;
    int32_t task_end_cycles_capacity;
    pthread_mutex_t task_end_mutex;   // Protect task_end_cycles updates
    
    // Per-worker current cycle (for proper scheduling)
    volatile int64_t* worker_current_cycle;
    
    // Thread startup synchronization
    // Ensures: workers start first, then scheduler, then orchestrator begins work
    pthread_mutex_t startup_mutex;
    pthread_cond_t  startup_cond;
    volatile int32_t workers_ready;        // Count of workers that have started
    volatile bool    scheduler_ready;      // Scheduler has started
    
} PTO2ThreadContext;

/**
 * Orchestrator context (passed to orchestrator thread)
 */
typedef struct {
    struct PTO2Runtime* runtime;
    void (*user_func)(struct PTO2Runtime*, void*);
    void* user_arg;
} PTO2OrchestratorContext;

/**
 * Scheduler context (passed to scheduler thread)
 */
typedef struct {
    struct PTO2Runtime* runtime;
    struct PTO2SchedulerState* scheduler;
    PTO2ThreadContext* thread_ctx;
} PTO2SchedulerContext;

// =============================================================================
// Task Parameter Convenience Macros
// =============================================================================

/**
 * Convenience macros for creating task parameters
 */
#define PTO2_INPUT(buf, tile_idx, sz) \
    (PTO2TaskParam){ .type = PTO2_PARAM_INPUT, .buffer = (buf), .tile_index = (tile_idx), .size = (sz) }

#define PTO2_OUTPUT(buf, tile_idx, sz) \
    (PTO2TaskParam){ .type = PTO2_PARAM_OUTPUT, .buffer = (buf), .tile_index = (tile_idx), .size = (sz) }

#define PTO2_INOUT(buf, tile_idx, sz) \
    (PTO2TaskParam){ .type = PTO2_PARAM_INOUT, .buffer = (buf), .tile_index = (tile_idx), .size = (sz) }

#endif // PTO_RUNTIME2_TYPES_H
