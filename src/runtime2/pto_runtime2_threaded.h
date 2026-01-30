/**
 * PTO Runtime2 - Multi-Threaded Interface
 * 
 * Extends the base runtime with multi-threaded execution support.
 * Orchestrator, Scheduler, and Workers run in separate threads.
 * 
 * Based on: docs/runtime_buffer_manager_methods.md
 */

#ifndef PTO_RUNTIME2_THREADED_H
#define PTO_RUNTIME2_THREADED_H

#include "pto_runtime2.h"
#include "pto_runtime2_types.h"
#include "pto_worker.h"

// =============================================================================
// Threaded Runtime Structure
// =============================================================================

/**
 * Trace event for recording task execution
 */
typedef struct {
    int32_t task_id;
    int32_t worker_id;
    int64_t start_cycle;
    int64_t end_cycle;
    const char* func_name;
} PTO2TraceEvent;

/**
 * Maximum number of trace events
 */
#define PTO2_MAX_TRACE_EVENTS 65536

/**
 * Extended runtime with thread context
 */
typedef struct PTO2RuntimeThreaded {
    PTO2Runtime base;                 // Base runtime (must be first)
    PTO2ThreadContext thread_ctx;     // Thread management
    
    // Contexts for threads
    PTO2OrchestratorContext orch_ctx;
    PTO2SchedulerContext sched_ctx;
    
    // Simulation mode flag
    bool simulation_mode;
    
    // Tracing
    bool trace_enabled;
    const char* trace_filename;
    PTO2TraceEvent* trace_events;     // Array of trace events
    volatile int32_t trace_count;     // Number of recorded events
    pthread_mutex_t trace_mutex;      // Mutex for thread-safe trace recording
    
} PTO2RuntimeThreaded;

// =============================================================================
// Threaded Runtime Creation
// =============================================================================

/**
 * Create a threaded runtime instance
 * 
 * @param num_cube_workers   Number of CUBE worker threads
 * @param num_vector_workers Number of VECTOR worker threads
 * @param simulation_mode    If true, use cycle simulation instead of execution
 * @return Runtime context, or NULL on failure
 */
PTO2RuntimeThreaded* pto2_runtime_create_threaded(int32_t num_cube_workers,
                                                   int32_t num_vector_workers,
                                                   bool simulation_mode);

/**
 * Create threaded runtime with custom sizes
 */
PTO2RuntimeThreaded* pto2_runtime_create_threaded_custom(int32_t num_cube_workers,
                                                          int32_t num_vector_workers,
                                                          bool simulation_mode,
                                                          int32_t task_window_size,
                                                          int32_t heap_size,
                                                          int32_t dep_list_size);

/**
 * Destroy threaded runtime
 */
void pto2_runtime_destroy_threaded(PTO2RuntimeThreaded* rt);

/**
 * Reset threaded runtime for reuse
 */
void pto2_runtime_reset_threaded(PTO2RuntimeThreaded* rt);

// =============================================================================
// Threaded Execution
// =============================================================================

/**
 * User orchestration function type
 */
typedef void (*PTO2OrchestrationFunc)(PTO2Runtime* rt, void* arg);

/**
 * Run runtime with multi-threading
 * 
 * Starts all threads, runs orchestration, waits for completion.
 * 
 * @param rt                 Threaded runtime
 * @param orchestration_func User's orchestration function
 * @param orchestration_arg  Argument to pass to orchestration function
 */
void pto2_runtime_run_threaded(PTO2RuntimeThreaded* rt,
                                PTO2OrchestrationFunc orchestration_func,
                                void* orchestration_arg);

/**
 * Run runtime with orchestration in current thread
 * 
 * Starts scheduler and worker threads, runs orchestration in current thread.
 * Useful for debugging or when orchestration needs main thread access.
 * 
 * @param rt                 Threaded runtime
 * @param orchestration_func User's orchestration function
 * @param orchestration_arg  Argument to pass to orchestration function
 */
void pto2_runtime_run_inline(PTO2RuntimeThreaded* rt,
                              PTO2OrchestrationFunc orchestration_func,
                              void* orchestration_arg);

// =============================================================================
// Thread Control
// =============================================================================

/**
 * Start all worker and scheduler threads
 * (Does not start orchestrator - call orchestration function directly)
 */
void pto2_runtime_start_threads(PTO2RuntimeThreaded* rt);

/**
 * Stop all threads and wait for completion
 */
void pto2_runtime_stop_threads(PTO2RuntimeThreaded* rt);

/**
 * Wait for all tasks to complete
 */
void pto2_runtime_wait_completion(PTO2RuntimeThreaded* rt);

/**
 * Check if all threads have finished
 */
bool pto2_runtime_threads_done(PTO2RuntimeThreaded* rt);

// =============================================================================
// Tracing
// =============================================================================

/**
 * Enable tracing to file
 */
void pto2_runtime_enable_trace(PTO2RuntimeThreaded* rt, const char* filename);

/**
 * Record a trace event (thread-safe)
 */
void pto2_runtime_record_trace(PTO2RuntimeThreaded* rt, int32_t task_id,
                                int32_t worker_id, int64_t start_cycle,
                                int64_t end_cycle, const char* func_name);

/**
 * Write trace to file
 */
void pto2_runtime_write_trace(PTO2RuntimeThreaded* rt, const char* filename);

// =============================================================================
// Statistics
// =============================================================================

/**
 * Print threaded runtime statistics
 */
void pto2_runtime_print_threaded_stats(PTO2RuntimeThreaded* rt);

/**
 * Get total simulated cycles
 */
int64_t pto2_runtime_get_total_cycles(PTO2RuntimeThreaded* rt);

#endif // PTO_RUNTIME2_THREADED_H
