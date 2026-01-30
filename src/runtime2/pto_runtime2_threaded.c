/**
 * PTO Runtime2 - Multi-Threaded Implementation
 * 
 * Implements threaded runtime with separate Orchestrator, Scheduler, and Worker threads.
 * 
 * Based on: docs/runtime_buffer_manager_methods.md
 */

#include "pto_runtime2_threaded.h"
#include "pto_runtime2.h"
#include "pto_worker.h"
#include "pto_scheduler.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

// =============================================================================
// Thread Context Initialization
// =============================================================================

static bool thread_ctx_init(PTO2ThreadContext* ctx, int32_t num_cube_workers,
                            int32_t num_vector_workers, int32_t task_window_size) {
    memset(ctx, 0, sizeof(PTO2ThreadContext));
    
    ctx->num_cube_workers = num_cube_workers;
    ctx->num_vector_workers = num_vector_workers;
    ctx->num_workers = num_cube_workers + num_vector_workers;
    
    if (ctx->num_workers > PTO2_MAX_WORKERS) {
        fprintf(stderr, "Too many workers requested: %d (max %d)\n",
                ctx->num_workers, PTO2_MAX_WORKERS);
        return false;
    }
    
    // Initialize ready queue mutexes and condition variables
    for (int i = 0; i < PTO2_NUM_WORKER_TYPES; i++) {
        if (pthread_mutex_init(&ctx->ready_mutex[i], NULL) != 0) {
            // Cleanup on failure
            for (int j = 0; j < i; j++) {
                pthread_mutex_destroy(&ctx->ready_mutex[j]);
                pthread_cond_destroy(&ctx->ready_cond[j]);
            }
            return false;
        }
        if (pthread_cond_init(&ctx->ready_cond[i], NULL) != 0) {
            pthread_mutex_destroy(&ctx->ready_mutex[i]);
            for (int j = 0; j < i; j++) {
                pthread_mutex_destroy(&ctx->ready_mutex[j]);
                pthread_cond_destroy(&ctx->ready_cond[j]);
            }
            return false;
        }
    }
    
    // Initialize completion queue (same size as ready queue to avoid bottleneck)
    int32_t completion_capacity = PTO2_READY_QUEUE_SIZE;
    if (!pto2_completion_queue_init(&ctx->completion_queue, completion_capacity)) {
        for (int i = 0; i < PTO2_NUM_WORKER_TYPES; i++) {
            pthread_mutex_destroy(&ctx->ready_mutex[i]);
            pthread_cond_destroy(&ctx->ready_cond[i]);
        }
        return false;
    }
    
    // Initialize done mutex and condition
    if (pthread_mutex_init(&ctx->done_mutex, NULL) != 0) {
        pto2_completion_queue_destroy(&ctx->completion_queue);
        for (int i = 0; i < PTO2_NUM_WORKER_TYPES; i++) {
            pthread_mutex_destroy(&ctx->ready_mutex[i]);
            pthread_cond_destroy(&ctx->ready_cond[i]);
        }
        return false;
    }
    
    if (pthread_cond_init(&ctx->all_done_cond, NULL) != 0) {
        pthread_mutex_destroy(&ctx->done_mutex);
        pto2_completion_queue_destroy(&ctx->completion_queue);
        for (int i = 0; i < PTO2_NUM_WORKER_TYPES; i++) {
            pthread_mutex_destroy(&ctx->ready_mutex[i]);
            pthread_cond_destroy(&ctx->ready_cond[i]);
        }
        return false;
    }
    
    if (pthread_cond_init(&ctx->completion_cond, NULL) != 0) {
        pthread_cond_destroy(&ctx->all_done_cond);
        pthread_mutex_destroy(&ctx->done_mutex);
        pto2_completion_queue_destroy(&ctx->completion_queue);
        for (int i = 0; i < PTO2_NUM_WORKER_TYPES; i++) {
            pthread_mutex_destroy(&ctx->ready_mutex[i]);
            pthread_cond_destroy(&ctx->ready_cond[i]);
        }
        return false;
    }
    
    ctx->shutdown = false;
    ctx->all_done = false;
    ctx->global_cycle = 0;
    
    // Initialize task end cycles tracking for simulation (dynamically sized)
    ctx->task_end_cycles_capacity = task_window_size;
    ctx->task_end_cycles = (volatile int64_t*)calloc(ctx->task_end_cycles_capacity, 
                                                      sizeof(int64_t));
    if (!ctx->task_end_cycles) {
        pthread_cond_destroy(&ctx->completion_cond);
        pthread_cond_destroy(&ctx->all_done_cond);
        pthread_mutex_destroy(&ctx->done_mutex);
        pto2_completion_queue_destroy(&ctx->completion_queue);
        for (int i = 0; i < PTO2_NUM_WORKER_TYPES; i++) {
            pthread_mutex_destroy(&ctx->ready_mutex[i]);
            pthread_cond_destroy(&ctx->ready_cond[i]);
        }
        return false;
    }
    
    if (pthread_mutex_init(&ctx->task_end_mutex, NULL) != 0) {
        free((void*)ctx->task_end_cycles);
        pthread_cond_destroy(&ctx->completion_cond);
        pthread_cond_destroy(&ctx->all_done_cond);
        pthread_mutex_destroy(&ctx->done_mutex);
        pto2_completion_queue_destroy(&ctx->completion_queue);
        for (int i = 0; i < PTO2_NUM_WORKER_TYPES; i++) {
            pthread_mutex_destroy(&ctx->ready_mutex[i]);
            pthread_cond_destroy(&ctx->ready_cond[i]);
        }
        return false;
    }
    
    // Initialize per-worker current cycle
    ctx->worker_current_cycle = (volatile int64_t*)calloc(ctx->num_workers, 
                                                           sizeof(int64_t));
    if (!ctx->worker_current_cycle) {
        pthread_mutex_destroy(&ctx->task_end_mutex);
        free((void*)ctx->task_end_cycles);
        pthread_cond_destroy(&ctx->completion_cond);
        pthread_cond_destroy(&ctx->all_done_cond);
        pthread_mutex_destroy(&ctx->done_mutex);
        pto2_completion_queue_destroy(&ctx->completion_queue);
        for (int i = 0; i < PTO2_NUM_WORKER_TYPES; i++) {
            pthread_mutex_destroy(&ctx->ready_mutex[i]);
            pthread_cond_destroy(&ctx->ready_cond[i]);
        }
        return false;
    }
    
    // Initialize startup synchronization
    if (pthread_mutex_init(&ctx->startup_mutex, NULL) != 0) {
        free((void*)ctx->worker_current_cycle);
        pthread_mutex_destroy(&ctx->task_end_mutex);
        free((void*)ctx->task_end_cycles);
        pthread_cond_destroy(&ctx->completion_cond);
        pthread_cond_destroy(&ctx->all_done_cond);
        pthread_mutex_destroy(&ctx->done_mutex);
        pto2_completion_queue_destroy(&ctx->completion_queue);
        for (int i = 0; i < PTO2_NUM_WORKER_TYPES; i++) {
            pthread_mutex_destroy(&ctx->ready_mutex[i]);
            pthread_cond_destroy(&ctx->ready_cond[i]);
        }
        return false;
    }
    
    if (pthread_cond_init(&ctx->startup_cond, NULL) != 0) {
        pthread_mutex_destroy(&ctx->startup_mutex);
        free((void*)ctx->worker_current_cycle);
        pthread_mutex_destroy(&ctx->task_end_mutex);
        free((void*)ctx->task_end_cycles);
        pthread_cond_destroy(&ctx->completion_cond);
        pthread_cond_destroy(&ctx->all_done_cond);
        pthread_mutex_destroy(&ctx->done_mutex);
        pto2_completion_queue_destroy(&ctx->completion_queue);
        for (int i = 0; i < PTO2_NUM_WORKER_TYPES; i++) {
            pthread_mutex_destroy(&ctx->ready_mutex[i]);
            pthread_cond_destroy(&ctx->ready_cond[i]);
        }
        return false;
    }
    
    ctx->workers_ready = 0;
    ctx->scheduler_ready = false;
    
    // Initialize per-worker condition variables for selective wakeup
    for (int i = 0; i < ctx->num_workers; i++) {
        if (pthread_cond_init(&ctx->worker_cond[i], NULL) != 0) {
            // Cleanup on failure
            for (int j = 0; j < i; j++) {
                pthread_cond_destroy(&ctx->worker_cond[j]);
            }
            pthread_cond_destroy(&ctx->startup_cond);
            pthread_mutex_destroy(&ctx->startup_mutex);
            free((void*)ctx->worker_current_cycle);
            pthread_mutex_destroy(&ctx->task_end_mutex);
            free((void*)ctx->task_end_cycles);
            pthread_cond_destroy(&ctx->completion_cond);
            pthread_cond_destroy(&ctx->all_done_cond);
            pthread_mutex_destroy(&ctx->done_mutex);
            pto2_completion_queue_destroy(&ctx->completion_queue);
            for (int j = 0; j < PTO2_NUM_WORKER_TYPES; j++) {
                pthread_mutex_destroy(&ctx->ready_mutex[j]);
                pthread_cond_destroy(&ctx->ready_cond[j]);
            }
            return false;
        }
        ctx->worker_waiting[i] = false;
    }
    
    return true;
}

static void thread_ctx_destroy(PTO2ThreadContext* ctx) {
    pthread_cond_destroy(&ctx->completion_cond);
    pthread_cond_destroy(&ctx->all_done_cond);
    pthread_mutex_destroy(&ctx->done_mutex);
    
    pto2_completion_queue_destroy(&ctx->completion_queue);
    
    for (int i = 0; i < PTO2_NUM_WORKER_TYPES; i++) {
        pthread_mutex_destroy(&ctx->ready_mutex[i]);
        pthread_cond_destroy(&ctx->ready_cond[i]);
    }
    
    // Destroy per-worker condition variables
    for (int i = 0; i < ctx->num_workers; i++) {
        pthread_cond_destroy(&ctx->worker_cond[i]);
    }
    
    // Free simulation tracking arrays
    if (ctx->task_end_cycles) {
        free((void*)ctx->task_end_cycles);
        ctx->task_end_cycles = NULL;
    }
    pthread_mutex_destroy(&ctx->task_end_mutex);
    
    if (ctx->worker_current_cycle) {
        free((void*)ctx->worker_current_cycle);
        ctx->worker_current_cycle = NULL;
    }
    
    // Destroy startup synchronization
    pthread_cond_destroy(&ctx->startup_cond);
    pthread_mutex_destroy(&ctx->startup_mutex);
}

static void thread_ctx_reset(PTO2ThreadContext* ctx) {
    ctx->shutdown = false;
    ctx->all_done = false;
    ctx->orchestrator_done = false;
    ctx->scheduler_running = false;
    ctx->global_cycle = 0;
    
    // Reset startup synchronization
    ctx->workers_ready = 0;
    ctx->scheduler_ready = false;
    
    // Reset completion queue (drain it)
    PTO2CompletionEntry entry;
    while (pto2_completion_queue_pop(&ctx->completion_queue, &entry)) {
        // Discard entries
    }
    
    // Reset simulation tracking arrays
    if (ctx->task_end_cycles) {
        memset((void*)ctx->task_end_cycles, 0, 
               ctx->task_end_cycles_capacity * sizeof(int64_t));
    }
    if (ctx->worker_current_cycle) {
        memset((void*)ctx->worker_current_cycle, 0, 
               ctx->num_workers * sizeof(int64_t));
    }
}

// =============================================================================
// Threaded Runtime Creation
// =============================================================================

PTO2RuntimeThreaded* pto2_runtime_create_threaded(int32_t num_cube_workers,
                                                   int32_t num_vector_workers,
                                                   bool simulation_mode) {
    return pto2_runtime_create_threaded_custom(num_cube_workers, num_vector_workers,
                                                simulation_mode,
                                                PTO2_TASK_WINDOW_SIZE,
                                                PTO2_HEAP_SIZE,
                                                PTO2_DEP_LIST_POOL_SIZE);
}

PTO2RuntimeThreaded* pto2_runtime_create_threaded_custom(int32_t num_cube_workers,
                                                          int32_t num_vector_workers,
                                                          bool simulation_mode,
                                                          int32_t task_window_size,
                                                          int32_t heap_size,
                                                          int32_t dep_list_size) {
    // Ensure task_window_size is a power of 2 (for fast modulo)
    if (task_window_size <= 0 || (task_window_size & (task_window_size - 1)) != 0) {
        fprintf(stderr, "ERROR: task_window_size (%d) must be a positive power of 2\n",
                task_window_size);
        return NULL;
    }
    
    // Allocate threaded runtime
    PTO2RuntimeThreaded* rt = (PTO2RuntimeThreaded*)calloc(1, sizeof(PTO2RuntimeThreaded));
    if (!rt) {
        return NULL;
    }
    
    // Initialize base runtime
    PTO2RuntimeMode mode = simulation_mode ? PTO2_MODE_SIMULATE : PTO2_MODE_EXECUTE;
    
    // Create shared memory
    rt->base.sm_handle = pto2_sm_create(task_window_size, heap_size, dep_list_size);
    if (!rt->base.sm_handle) {
        free(rt);
        return NULL;
    }
    
    // Allocate GM heap
    rt->base.gm_heap_size = heap_size;
    #if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
        if (posix_memalign(&rt->base.gm_heap, PTO2_ALIGN_SIZE, heap_size) != 0) {
            pto2_sm_destroy(rt->base.sm_handle);
            free(rt);
            return NULL;
        }
    #else
        rt->base.gm_heap = aligned_alloc(PTO2_ALIGN_SIZE, heap_size);
        if (!rt->base.gm_heap) {
            pto2_sm_destroy(rt->base.sm_handle);
            free(rt);
            return NULL;
        }
    #endif
    rt->base.gm_heap_owned = true;
    
    // Initialize orchestrator
    if (!pto2_orchestrator_init(&rt->base.orchestrator, rt->base.sm_handle,
                                 rt->base.gm_heap, heap_size)) {
        free(rt->base.gm_heap);
        pto2_sm_destroy(rt->base.sm_handle);
        free(rt);
        return NULL;
    }
    
    // Initialize scheduler
    if (!pto2_scheduler_init(&rt->base.scheduler, rt->base.sm_handle,
                              &rt->base.orchestrator.dep_pool)) {
        pto2_orchestrator_destroy(&rt->base.orchestrator);
        free(rt->base.gm_heap);
        pto2_sm_destroy(rt->base.sm_handle);
        free(rt);
        return NULL;
    }
    
    // Connect orchestrator to scheduler for scope_end notifications.
    // In multi-threaded mode:
    // - init_task_on_submit = false (scheduler thread polls for new tasks)
    // - scope_end still calls pto2_scheduler_on_scope_end (only modifies refcounts)
    pto2_orchestrator_set_scheduler_mode(&rt->base.orchestrator, &rt->base.scheduler, false);
    
    rt->base.mode = mode;
    rt->simulation_mode = simulation_mode;
    
    // Initialize thread context (pass task_window_size for dynamic allocation)
    if (!thread_ctx_init(&rt->thread_ctx, num_cube_workers, num_vector_workers, task_window_size)) {
        pto2_scheduler_destroy(&rt->base.scheduler);
        pto2_orchestrator_destroy(&rt->base.orchestrator);
        free(rt->base.gm_heap);
        pto2_sm_destroy(rt->base.sm_handle);
        free(rt);
        return NULL;
    }
    
    // Initialize worker contexts
    int worker_id = 0;
    
    // CUBE workers
    for (int i = 0; i < num_cube_workers; i++) {
        pto2_worker_init(&rt->thread_ctx.workers[worker_id], worker_id,
                          PTO2_WORKER_CUBE, (PTO2Runtime*)rt);
        worker_id++;
    }
    
    // VECTOR workers
    for (int i = 0; i < num_vector_workers; i++) {
        pto2_worker_init(&rt->thread_ctx.workers[worker_id], worker_id,
                          PTO2_WORKER_VECTOR, (PTO2Runtime*)rt);
        worker_id++;
    }
    
    // Setup scheduler context
    rt->sched_ctx.runtime = (PTO2Runtime*)rt;
    rt->sched_ctx.scheduler = &rt->base.scheduler;
    rt->sched_ctx.thread_ctx = &rt->thread_ctx;
    
    // Setup orchestrator context
    rt->orch_ctx.runtime = (PTO2Runtime*)rt;
    
    // Initialize trace recording
    rt->trace_enabled = true;  // Enable by default for simulation
    rt->trace_filename = NULL;
    rt->trace_events = (PTO2TraceEvent*)calloc(PTO2_MAX_TRACE_EVENTS, sizeof(PTO2TraceEvent));
    rt->trace_count = 0;
    pthread_mutex_init(&rt->trace_mutex, NULL);
    
    return rt;
}

void pto2_runtime_destroy_threaded(PTO2RuntimeThreaded* rt) {
    if (!rt) return;
    
    // Ensure threads are stopped
    pto2_runtime_stop_threads(rt);
    
    // Destroy worker contexts
    for (int i = 0; i < rt->thread_ctx.num_workers; i++) {
        pto2_worker_destroy(&rt->thread_ctx.workers[i]);
    }
    
    // Destroy trace resources
    if (rt->trace_events) {
        free(rt->trace_events);
        rt->trace_events = NULL;
    }
    pthread_mutex_destroy(&rt->trace_mutex);
    
    // Destroy thread context
    thread_ctx_destroy(&rt->thread_ctx);
    
    // Destroy base runtime components
    pto2_scheduler_destroy(&rt->base.scheduler);
    pto2_orchestrator_destroy(&rt->base.orchestrator);
    
    if (rt->base.gm_heap_owned && rt->base.gm_heap) {
        free(rt->base.gm_heap);
    }
    
    if (rt->base.sm_handle) {
        pto2_sm_destroy(rt->base.sm_handle);
    }
    
    free(rt);
}

void pto2_runtime_reset_threaded(PTO2RuntimeThreaded* rt) {
    if (!rt) return;
    
    // Reset base runtime
    pto2_orchestrator_reset(&rt->base.orchestrator);
    pto2_scheduler_reset(&rt->base.scheduler);
    pto2_sm_reset(rt->base.sm_handle);
    rt->base.total_cycles = 0;
    
    // Reset thread context
    thread_ctx_reset(&rt->thread_ctx);
    
    // Reset worker stats
    for (int i = 0; i < rt->thread_ctx.num_workers; i++) {
        pto2_worker_reset(&rt->thread_ctx.workers[i]);
    }
}

// =============================================================================
// Orchestrator Thread Entry Point
// =============================================================================

static void* orchestrator_thread_entry(void* arg) {
    PTO2OrchestratorContext* ctx = (PTO2OrchestratorContext*)arg;
    PTO2RuntimeThreaded* rt = (PTO2RuntimeThreaded*)ctx->runtime;
    
    // Call user's orchestration function
    ctx->user_func(ctx->runtime, ctx->user_arg);
    
    // Mark orchestration complete
    pto2_rt_orchestration_done(ctx->runtime);
    
    // Signal scheduler
    rt->thread_ctx.orchestrator_done = true;
    
    return NULL;
}

// =============================================================================
// Thread Control
// =============================================================================

void pto2_runtime_start_threads(PTO2RuntimeThreaded* rt) {
    PTO2ThreadContext* ctx = &rt->thread_ctx;
    
    // Reset startup synchronization state
    pthread_mutex_lock(&ctx->startup_mutex);
    ctx->workers_ready = 0;
    ctx->scheduler_ready = false;
    pthread_mutex_unlock(&ctx->startup_mutex);
    
    // === STEP 1: Start worker threads first ===
    for (int i = 0; i < ctx->num_workers; i++) {
        PTO2WorkerContext* worker = &ctx->workers[i];
        worker->shutdown = false;
        
        void* (*thread_func)(void*) = rt->simulation_mode ?
            pto2_worker_thread_func_sim : pto2_worker_thread_func;
        
        if (pthread_create(&worker->thread, NULL, thread_func, worker) != 0) {
            fprintf(stderr, "Failed to create worker thread %d\n", i);
        }
    }
    
    // === STEP 2: Start scheduler thread (it will wait for workers) ===
    ctx->scheduler_running = true;
    if (pthread_create(&ctx->scheduler_thread, NULL,
                       pto2_scheduler_thread_func, &rt->sched_ctx) != 0) {
        fprintf(stderr, "Failed to create scheduler thread\n");
        ctx->scheduler_running = false;
        return;
    }
    
    // === STEP 3: Wait for all workers and scheduler to be ready ===
    // This ensures orchestrator only starts after worker + scheduler are ready
    pthread_mutex_lock(&ctx->startup_mutex);
    while (!ctx->scheduler_ready) {
        pthread_cond_wait(&ctx->startup_cond, &ctx->startup_mutex);
    }
    pthread_mutex_unlock(&ctx->startup_mutex);
    
    // All threads are now ready - safe to start orchestration
}

void pto2_runtime_stop_threads(PTO2RuntimeThreaded* rt) {
    PTO2ThreadContext* ctx = &rt->thread_ctx;
    
    // Check if shutdown already signaled (threads already stopped or stopping)
    if (ctx->shutdown) {
        return;  // Already stopped
    }
    
    // Signal shutdown to all workers first (before waking them)
    for (int i = 0; i < ctx->num_workers; i++) {
        ctx->workers[i].shutdown = true;
    }
    ctx->shutdown = true;
    
    // Wake up all workers waiting on ready queues (both shared and per-worker conds)
    for (int i = 0; i < PTO2_NUM_WORKER_TYPES; i++) {
        pthread_mutex_lock(&ctx->ready_mutex[i]);
        pthread_cond_broadcast(&ctx->ready_cond[i]);
        pthread_mutex_unlock(&ctx->ready_mutex[i]);
    }
    
    // Wake up all per-worker condition variables
    for (int i = 0; i < ctx->num_workers; i++) {
        PTO2WorkerType type = ctx->workers[i].worker_type;
        pthread_mutex_lock(&ctx->ready_mutex[type]);
        pthread_cond_signal(&ctx->worker_cond[i]);
        pthread_mutex_unlock(&ctx->ready_mutex[type]);
    }
    
    // Wake up scheduler
    pthread_mutex_lock(&ctx->done_mutex);
    pthread_cond_broadcast(&ctx->completion_cond);
    pthread_cond_broadcast(&ctx->all_done_cond);
    pthread_mutex_unlock(&ctx->done_mutex);
    
    // Wait for all worker threads
    for (int i = 0; i < ctx->num_workers; i++) {
        PTO2WorkerContext* worker = &ctx->workers[i];
        if (worker->thread != 0) {
            pthread_join(worker->thread, NULL);
            worker->thread = 0;  // Mark as joined
        }
    }
    
    // Wait for scheduler thread
    if (ctx->scheduler_running) {
        pthread_join(ctx->scheduler_thread, NULL);
        ctx->scheduler_running = false;  // Mark as joined
    }
}

void pto2_runtime_wait_completion(PTO2RuntimeThreaded* rt) {
    PTO2ThreadContext* ctx = &rt->thread_ctx;
    
    pthread_mutex_lock(&ctx->done_mutex);
    while (!ctx->all_done && !ctx->shutdown) {
        pthread_cond_wait(&ctx->all_done_cond, &ctx->done_mutex);
    }
    pthread_mutex_unlock(&ctx->done_mutex);
}

bool pto2_runtime_threads_done(PTO2RuntimeThreaded* rt) {
    return rt->thread_ctx.all_done;
}

// =============================================================================
// Threaded Execution
// =============================================================================

void pto2_runtime_run_threaded(PTO2RuntimeThreaded* rt,
                                PTO2OrchestrationFunc orchestration_func,
                                void* orchestration_arg) {
    PTO2ThreadContext* ctx = &rt->thread_ctx;
    
    // Setup orchestrator context
    rt->orch_ctx.user_func = orchestration_func;
    rt->orch_ctx.user_arg = orchestration_arg;
    
    // Start worker and scheduler threads
    pto2_runtime_start_threads(rt);
    
    // Start orchestrator thread
    if (pthread_create(&ctx->orchestrator_thread, NULL,
                       orchestrator_thread_entry, &rt->orch_ctx) != 0) {
        fprintf(stderr, "Failed to create orchestrator thread\n");
        pto2_runtime_stop_threads(rt);
        return;
    }
    
    // Wait for orchestrator to complete
    pthread_join(ctx->orchestrator_thread, NULL);
    
    // Wait for all tasks to complete
    pto2_runtime_wait_completion(rt);
    
    // Stop all threads
    pto2_runtime_stop_threads(rt);
    
    // Collect total cycles from workers
    rt->base.total_cycles = ctx->global_cycle;
}

void pto2_runtime_run_inline(PTO2RuntimeThreaded* rt,
                              PTO2OrchestrationFunc orchestration_func,
                              void* orchestration_arg) {
    PTO2ThreadContext* ctx = &rt->thread_ctx;
    
    // Start worker and scheduler threads
    pto2_runtime_start_threads(rt);
    
    // Run orchestration in current thread
    orchestration_func((PTO2Runtime*)rt, orchestration_arg);
    
    // Mark orchestration complete
    pto2_rt_orchestration_done((PTO2Runtime*)rt);
    ctx->orchestrator_done = true;
    
    // Wait for all tasks to complete
    pto2_runtime_wait_completion(rt);
    
    // Stop all threads
    pto2_runtime_stop_threads(rt);
    
    // Collect total cycles
    rt->base.total_cycles = ctx->global_cycle;
}

// =============================================================================
// Tracing
// =============================================================================

void pto2_runtime_enable_trace(PTO2RuntimeThreaded* rt, const char* filename) {
    rt->trace_enabled = true;
    rt->trace_filename = filename;
}

void pto2_runtime_record_trace(PTO2RuntimeThreaded* rt, int32_t task_id,
                                int32_t worker_id, int64_t start_cycle,
                                int64_t end_cycle, const char* func_name) {
    if (!rt->trace_enabled || !rt->trace_events) return;
    
    pthread_mutex_lock(&rt->trace_mutex);
    
    if (rt->trace_count < PTO2_MAX_TRACE_EVENTS) {
        PTO2TraceEvent* e = &rt->trace_events[rt->trace_count];
        e->task_id = task_id;
        e->worker_id = worker_id;
        e->start_cycle = start_cycle;
        e->end_cycle = end_cycle;
        e->func_name = func_name;
        rt->trace_count++;
    }
    
    pthread_mutex_unlock(&rt->trace_mutex);
}

void pto2_runtime_write_trace(PTO2RuntimeThreaded* rt, const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Failed to open trace file: %s\n", filename);
        return;
    }
    
    fprintf(f, "[\n");
    
    // Write process and thread metadata
    fprintf(f, "  {\"name\": \"process_name\", \"ph\": \"M\", \"pid\": 0, "
            "\"args\": {\"name\": \"PTO Runtime2 Threaded\"}},\n");
    
    // Thread names for workers
    int tid = 0;
    for (int i = 0; i < rt->thread_ctx.num_cube_workers; i++) {
        fprintf(f, "  {\"name\": \"thread_name\", \"ph\": \"M\", \"pid\": 0, "
                "\"tid\": %d, \"args\": {\"name\": \"Cube%d\"}},\n", tid, i);
        tid++;
    }
    for (int i = 0; i < rt->thread_ctx.num_vector_workers; i++) {
        fprintf(f, "  {\"name\": \"thread_name\", \"ph\": \"M\", \"pid\": 0, "
                "\"tid\": %d, \"args\": {\"name\": \"Vector%d\"}},\n", tid, i);
        tid++;
    }
    
    // Write task trace events
    for (int32_t i = 0; i < rt->trace_count; i++) {
        PTO2TraceEvent* e = &rt->trace_events[i];
        const char* name = e->func_name ? e->func_name : "task";
        
        // Convert cycles to microseconds for Perfetto display
        // Scale: 1 cycle = 1000 microseconds for better visibility
        int64_t ts_us = e->start_cycle * 1000;
        int64_t dur_us = (e->end_cycle - e->start_cycle) * 1000;
        
        fprintf(f, "  {\"name\": \"%s\", \"cat\": \"task\", \"ph\": \"X\", "
                "\"pid\": 0, \"tid\": %d, \"ts\": %lld, \"dur\": %lld, "
                "\"args\": {\"task_id\": %d}}",
                name, e->worker_id, (long long)ts_us, (long long)dur_us, e->task_id);
        
        if (i < rt->trace_count - 1) {
            fprintf(f, ",\n");
        } else {
            fprintf(f, "\n");
        }
    }
    
    fprintf(f, "]\n");
    fclose(f);
    
    printf("Trace written to: %s (%d events)\n", filename, rt->trace_count);
}

// =============================================================================
// Statistics
// =============================================================================

void pto2_runtime_print_threaded_stats(PTO2RuntimeThreaded* rt) {
    printf("\n========== PTO Runtime2 Threaded Statistics ==========\n\n");
    
    // Thread configuration
    printf("=== Thread Configuration ===\n");
    printf("CUBE workers:   %d\n", rt->thread_ctx.num_cube_workers);
    printf("VECTOR workers: %d\n", rt->thread_ctx.num_vector_workers);
    printf("Total workers:  %d\n", rt->thread_ctx.num_workers);
    printf("Simulation:     %s\n", rt->simulation_mode ? "yes" : "no");
    printf("===========================\n\n");
    
    // Worker stats
    printf("=== Worker Statistics ===\n");
    for (int i = 0; i < rt->thread_ctx.num_workers; i++) {
        pto2_worker_print_stats(&rt->thread_ctx.workers[i]);
    }
    printf("=========================\n\n");
    
    // Base runtime stats
    pto2_runtime_print_stats((PTO2Runtime*)rt);
    
    // Overall
    printf("=== Overall ===\n");
    printf("Global cycles: %lld\n", (long long)rt->thread_ctx.global_cycle);
    printf("================\n");
    
    printf("\n====================================================\n");
}

int64_t pto2_runtime_get_total_cycles(PTO2RuntimeThreaded* rt) {
    return rt->thread_ctx.global_cycle;
}
