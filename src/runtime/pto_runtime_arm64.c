/**
 * PTO Runtime System - ARM64 Platform Implementation
 * 
 * ARM64-specific implementations:
 * - Single ready queue management (no cube/vector separation)
 * - Distributed dependency tracking via fanout lists
 * - Multi-threaded worker execution with pthreads
 * - Task completion with dependency propagation
 */

#include "pto_runtime_arm64.h"
#include <time.h>

// =============================================================================
// ARM64 Ready Queue Implementation (Single Queue)
// =============================================================================

static void ready_queue_push(PTORuntime* rt, int32_t task_id) {
    if (rt->ready_count >= PTO_MAX_READY_QUEUE) {
        fprintf(stderr, "[PTO Runtime ARM64] ERROR: Ready queue overflow\n");
        return;
    }
    
    rt->ready_queue[rt->ready_tail] = task_id;
    rt->ready_tail = (rt->ready_tail + 1) % PTO_MAX_READY_QUEUE;
    rt->ready_count++;
}

static int32_t ready_queue_pop(PTORuntime* rt) {
    if (rt->ready_count == 0) {
        return -1;
    }
    
    int32_t task_id = rt->ready_queue[rt->ready_head];
    rt->ready_head = (rt->ready_head + 1) % PTO_MAX_READY_QUEUE;
    rt->ready_count--;
    return task_id;
}

// Thread-safe ready queue push
static void ready_queue_push_threadsafe(PTORuntime* rt, int32_t task_id) {
    DEBUG_PRINT("[Queue] push_threadsafe: trying to lock for task %d\n", task_id);
    fflush(stdout);
    
    pthread_mutex_lock(&rt->queue_mutex);
    
    DEBUG_PRINT("[Queue] push_threadsafe: got lock for task %d, ready_count=%d\n", task_id, rt->ready_count);
    fflush(stdout);
    
    if (rt->ready_count >= PTO_MAX_READY_QUEUE) {
        fprintf(stderr, "[PTO Runtime ARM64] ERROR: Ready queue overflow\n");
        pthread_mutex_unlock(&rt->queue_mutex);
        return;
    }
    
    rt->ready_queue[rt->ready_tail] = task_id;
    rt->ready_tail = (rt->ready_tail + 1) % PTO_MAX_READY_QUEUE;
    rt->ready_count++;
    
    // Broadcast to wake up all waiting workers
    pthread_cond_broadcast(&rt->queue_not_empty);
    
    pthread_mutex_unlock(&rt->queue_mutex);
    
    DEBUG_PRINT("[Queue] push_threadsafe: released lock, task %d queued\n", task_id);
    fflush(stdout);
}

// =============================================================================
// ARM64 Platform API Implementation
// =============================================================================

void pto_runtime_enable_simulation(PTORuntime* rt, int32_t num_workers) {
    if (!rt) return;
    rt->simulation_mode = true;
    rt->dual_queue_mode = false;
    pto_trace_init(num_workers > 0 ? num_workers : 1);
    DEBUG_PRINT("[PTO Runtime ARM64] Simulation mode enabled with %d workers\n", num_workers);
}

int32_t pto_get_ready_task(PTORuntime* rt) {
    return ready_queue_pop(rt);
}

int32_t pto_get_ready_task_blocking(PTORuntime* rt) {
    DEBUG_PRINT("[Queue] get_blocking: trying to lock\n");
    fflush(stdout);
    
    pthread_mutex_lock(&rt->queue_mutex);
    
    DEBUG_PRINT("[Queue] get_blocking: got lock, ready_count=%d, started=%d, threshold=%d\n", 
           rt->ready_count, rt->execution_started, rt->execution_task_threshold);
    fflush(stdout);
    
    // Determine if we can start execution:
    // - threshold == 0: must wait for execution_started (orchestration complete)
    // - threshold > 0: can start when active_task_count > threshold OR execution_started
    bool can_execute = rt->execution_started || 
                       (rt->execution_task_threshold > 0 && 
                        rt->total_tasks_scheduled > rt->execution_task_threshold);
    
    // Wait until: (can_execute AND task available) OR shutdown OR all done
    while ((rt->ready_count == 0 || !can_execute) && !rt->shutdown_requested) {
        // Check if all tasks are completed
        if (rt->execution_started && rt->total_tasks_completed >= rt->total_tasks_scheduled) {
            pthread_mutex_unlock(&rt->queue_mutex);
            return -1;  // All done
        }
        
        // Reduce debug spam - only print occasionally
        static __thread int wait_count = 0;
        if (++wait_count % 1000 == 1) {
            DEBUG_PRINT("[Queue] get_blocking: waiting (ready=%d, can_exec=%d, shutdown=%d, count=%d)\n", 
                   rt->ready_count, can_execute, rt->shutdown_requested, wait_count);
            fflush(stdout);
        }
        
        // Wait for signal (with short timeout for responsiveness)
        struct timespec timeout;
        clock_gettime(CLOCK_REALTIME, &timeout);
        timeout.tv_nsec += 100000;  // 100µs timeout
        if (timeout.tv_nsec >= 1000000000) {
            timeout.tv_sec++;
            timeout.tv_nsec -= 1000000000;
        }
        pthread_cond_timedwait(&rt->queue_not_empty, &rt->queue_mutex, &timeout);
        
        // Re-check execution condition after wakeup
        can_execute = rt->execution_started || 
                      (rt->execution_task_threshold > 0 && 
                       rt->total_tasks_scheduled > rt->execution_task_threshold);
    }
    
    if (rt->shutdown_requested || rt->ready_count == 0) {
        pthread_mutex_unlock(&rt->queue_mutex);
        return -1;
    }
    
    int32_t task_id = rt->ready_queue[rt->ready_head];
    rt->ready_head = (rt->ready_head + 1) % PTO_MAX_READY_QUEUE;
    rt->ready_count--;
    
    pthread_mutex_unlock(&rt->queue_mutex);
    return task_id;
}

// =============================================================================
// ARM64 Task Submit Implementation
// =============================================================================

void pto_task_submit(PTORuntime* rt, int32_t task_id) {
    if (task_id < 0 || task_id >= rt->next_task_id) {
        fprintf(stderr, "[PTO Runtime ARM64] ERROR: Invalid task_id %d\n", task_id);
        return;
    }
    
    PendingTask* task = &rt->pend_task[PTO_TASK_SLOT(task_id)];
    
    bool ready = pto_task_prepare_submit(rt, task_id);
    int32_t slot = PTO_TASK_SLOT(task_id);
    int32_t remaining = task->fanin_count - rt->fanin_refcount[slot];
    DEBUG_PRINT("[PTO Runtime ARM64] Submitted task %d: %s (fanin_rem=%d, fanout_cons=%d)\n",
           task_id, task->func_name, remaining, task->fanout_consumer_count);
    
    if (ready) {
        ready_queue_push_threadsafe(rt, task_id);
        DEBUG_PRINT("[PTO Runtime ARM64] Task %d is ready\n", task_id);
    }
}

// =============================================================================
// ARM64 Task Complete Implementation - Distributed Dependency Management
// =============================================================================

void pto_task_complete(PTORuntime* rt, int32_t task_id) {
    // For correctness, reuse the threadsafe completion logic.
    pto_task_complete_threadsafe(rt, task_id);
}

void pto_task_complete_threadsafe(PTORuntime* rt, int32_t task_id) {
    if (task_id < 0 || task_id >= rt->next_task_id) {
        fprintf(stderr, "[PTO Runtime ARM64] ERROR: Invalid task_id %d\n", task_id);
        return;
    }
    
    pthread_mutex_lock(&rt->task_mutex);
    
    int32_t slot = PTO_TASK_SLOT(task_id);
    PendingTask* task = &rt->pend_task[slot];

    // Skip stale or already-complete slots (can happen in pipelined mode)
    if (!task->is_active || task->task_id != task_id || task->is_complete) {
        pthread_mutex_unlock(&rt->task_mutex);
        return;
    }

    task->is_complete = true;
    rt->task_state[slot] = PTO_TASK_COMPLETED;
    rt->active_task_count--;
    rt->total_tasks_completed++;

    // Notify dependents: increment fanin_refcount, enqueue when ready.
    int32_t off = task->fanout_head;
    int32_t seen = 0;
    while (off != 0 && seen < task->fanout_consumer_count) {
        int32_t consumer_id = rt->dep_list_pool[off].task_id;
        int32_t cslot = PTO_TASK_SLOT(consumer_id);
        PendingTask* consumer = &rt->pend_task[cslot];

        if (consumer->is_active && consumer->task_id == consumer_id) {
            rt->fanin_refcount[cslot]++;
            if (task->end_cycle > consumer->earliest_start_cycle) {
                consumer->earliest_start_cycle = task->end_cycle;
            }

            if (consumer->is_submitted &&
                rt->task_state[cslot] == PTO_TASK_PENDING &&
                rt->fanin_refcount[cslot] == consumer->fanin_count) {
                rt->task_state[cslot] = PTO_TASK_READY;
                ready_queue_push_threadsafe(rt, consumer_id);
            }
        }

        off = rt->dep_list_pool[off].next_offset;
        seen++;
    }

    // Release references to producers (fanout_refcount increments).
    off = task->fanin_head;
    while (off != 0) {
        int32_t producer_id = rt->dep_list_pool[off].task_id;
        int32_t pslot = PTO_TASK_SLOT(producer_id);
        PendingTask* producer = &rt->pend_task[pslot];

        if (producer->is_active && producer->task_id == producer_id) {
            rt->fanout_refcount[pslot]++;
            pto_try_mark_consumed_locked(rt, producer_id);
        }

        off = rt->dep_list_pool[off].next_offset;
    }

    // Check if this task can be consumed.
    pto_try_mark_consumed_locked(rt, task_id);

    bool window_advanced = pto_advance_last_task_alive_locked(rt);
    bool all_done = (rt->total_tasks_completed >= rt->total_tasks_scheduled);

    if (window_advanced) {
        pthread_cond_broadcast(&rt->window_not_full);
    }

    pthread_mutex_unlock(&rt->task_mutex);

    if (all_done) {
        pthread_mutex_lock(&rt->queue_mutex);
        pthread_cond_broadcast(&rt->all_done);
        pthread_cond_broadcast(&rt->queue_not_empty);
        pthread_mutex_unlock(&rt->queue_mutex);
    }
}

// =============================================================================
// ARM64 Task Execution
// =============================================================================

// Generic function pointer type for InCore functions
typedef void (*InCoreFuncPtr)(void);

static void execute_task_internal(PTORuntime* rt, int32_t task_id, int32_t worker_id) {
    int32_t slot = PTO_TASK_SLOT(task_id);
    PendingTask* task = &rt->pend_task[slot];
    
    DEBUG_PRINT("[Worker ARM64] Executing task %d: %s\n", task_id, task->func_name);
    
    // Build argument array from task arguments
    void* args[PTO_MAX_ARGS * 2];
    int arg_idx = 0;
    
    for (int i = 0; i < task->num_args; i++) {
        TaskArg* arg = &task->args[i];
        float* base_ptr = (float*)arg->region.raw_tensor;
        int64_t offset = arg->region.row_offset * arg->region.cols + arg->region.col_offset;
        args[arg_idx++] = (void*)(base_ptr + offset);
    }
    
    // Simulation mode: call cycle function and record trace
    if (rt->simulation_mode && task->cycle_func) {
        int64_t cycle_cost = task->cycle_func(args, task->num_args);
        
        // Get worker's current cycle
        int64_t worker_current = pto_trace_get_cycle(worker_id);
        
        // Actual start time = max(worker ready, dependencies satisfied)
        int64_t actual_start = (worker_current > task->earliest_start_cycle) ? 
            worker_current : task->earliest_start_cycle;
        int64_t actual_end = actual_start + cycle_cost;
        
        // Update task's end_cycle
        task->end_cycle = actual_end;
        
        // Record with dependency-aware timing
        pto_trace_record_with_time(worker_id, task->func_name, actual_start, actual_end);
        DEBUG_PRINT("[Worker ARM64] Task %d: %s (simulated, %lld cycles, start=%lld)\n", 
               task_id, task->func_name, (long long)cycle_cost, (long long)actual_start);
    }
    // Normal mode: call the InCore function
    else if (task->func_ptr) {
        PTOInCoreFunc func = (PTOInCoreFunc)task->func_ptr;
        func(args, task->num_args);
    } else {
        DEBUG_PRINT("[Worker ARM64] Task %d: %s (no execution - no func_ptr)\n", 
               task_id, task->func_name);
    }
}

void pto_execute_task_with_worker(PTORuntime* rt, int32_t task_id, int32_t worker_id) {
    execute_task_internal(rt, task_id, worker_id);
}

void pto_execute_all(PTORuntime* rt) {
    printf("\n[PTO Runtime ARM64] ======== Executing all tasks ========\n");
    
    while (rt->total_tasks_completed < rt->total_tasks_scheduled) {
        int32_t task_id = pto_get_ready_task(rt);
        
        if (task_id < 0) {
            // No ready tasks - check for deadlock
            fprintf(stderr, "[PTO Runtime ARM64] WARNING: No ready tasks but %lld/%lld tasks completed - possible deadlock\n",
                    (long long)rt->total_tasks_completed, (long long)rt->total_tasks_scheduled);
            break;
            continue;
        }
        
        PendingTask* task = &rt->pend_task[PTO_TASK_SLOT(task_id)];
        
        DEBUG_PRINT("[PTO Runtime ARM64] Executing task %d: %s\n", task_id, task->func_name);
        
        // Execute the task
        if (task->func_ptr) {
            InCoreFuncPtr func __attribute__((unused)) = (InCoreFuncPtr)task->func_ptr;
            DEBUG_PRINT("[PTO Runtime ARM64] (Simulated execution of %s)\n", task->func_name);
        }
        
        // Mark complete and update dependencies
        pto_task_complete_threadsafe(rt, task_id);
    }
    
    printf("Execution complete!\n");
}

void pto_register_incore_func(PTORuntime* rt, const char* func_name, PTOInCoreFunc func_ptr) {
    DEBUG_PRINT("[PTO Runtime ARM64] Registered InCore function: %s\n", func_name);
    (void)rt;
    (void)func_name;
    (void)func_ptr;
}

// =============================================================================
// ARM64 Worker Thread Implementation
// =============================================================================

typedef struct {
    PTORuntime* rt;
    int worker_id;
} WorkerContext;

static void* worker_thread_func(void* arg) {
    WorkerContext* ctx = (WorkerContext*)arg;
    PTORuntime* rt = ctx->rt;
    int worker_id = ctx->worker_id;
    
    DEBUG_PRINT("[Worker ARM64 %d] Started\n", worker_id);
    fflush(stdout);
    
    while (!rt->shutdown_requested) {
        // Get next ready task (blocking)
        int32_t task_id = pto_get_ready_task_blocking(rt);
        
        if (task_id < 0) {
            // No task available - check if we should exit
            if (rt->shutdown_requested) {
                break;
            }
            // Check if all tasks are done
            if (rt->execution_started && 
                rt->total_tasks_completed >= rt->total_tasks_scheduled) {
                break;
            }
            continue;
        }
        
        // Execute the task
        execute_task_internal(rt, task_id, worker_id);
        
        // Mark task as complete
        pto_task_complete_threadsafe(rt, task_id);
    }
    
    DEBUG_PRINT("[Worker ARM64 %d] Exiting\n", worker_id);
    free(ctx);
    return NULL;
}

// =============================================================================
// ARM64 Runtime Entry Point
// =============================================================================

int runtime_entry_arm64(PTOOrchFunc orch_func, void* user_data, int num_workers,
                        int execution_task_threshold) {
    if (!orch_func) {
        fprintf(stderr, "[PTO Runtime ARM64] ERROR: No orchestration function provided\n");
        return -1;
    }
    
    if (num_workers < 1) num_workers = 1;
    if (num_workers > PTO_MAX_WORKERS) num_workers = PTO_MAX_WORKERS;
    if (execution_task_threshold < 0) execution_task_threshold = 0;
    
    printf("[PTO Runtime ARM64] ========================================\n");
    printf("[PTO Runtime ARM64] Multi-threaded Execution\n");
    printf("[PTO Runtime ARM64] Workers: %d\n", num_workers);
    if (execution_task_threshold > 0) {
        printf("[PTO Runtime ARM64] Execution threshold: %d tasks (pipelined)\n", execution_task_threshold);
    } else {
        printf("[PTO Runtime ARM64] Execution mode: wait for orchestration\n");
    }
    printf("[PTO Runtime ARM64] ========================================\n");
    
    // Allocate runtime on heap
    PTORuntime* rt = (PTORuntime*)malloc(sizeof(PTORuntime));
    if (!rt) {
        fprintf(stderr, "[PTO Runtime ARM64] ERROR: Failed to allocate runtime\n");
        return -1;
    }
    
    // Initialize runtime
    pto_runtime_init(rt);
    rt->num_workers = num_workers;
    rt->shutdown_requested = false;
    rt->execution_started = false;
    rt->execution_task_threshold = execution_task_threshold;
    
    // Spawn worker threads
    printf("[PTO Runtime ARM64] Spawning %d worker threads...\n", num_workers);
    for (int i = 0; i < num_workers; i++) {
        WorkerContext* ctx = (WorkerContext*)malloc(sizeof(WorkerContext));
        if (!ctx) {
            fprintf(stderr, "[PTO Runtime ARM64] ERROR: Failed to allocate worker context\n");
            rt->shutdown_requested = true;
            pthread_cond_broadcast(&rt->queue_not_empty);
            for (int j = 0; j < i; j++) {
                pthread_join(rt->workers[j], NULL);
            }
            pto_runtime_shutdown(rt);
            free(rt);
            return -1;
        }
        ctx->rt = rt;
        ctx->worker_id = i;
        
        if (pthread_create(&rt->workers[i], NULL, worker_thread_func, ctx) != 0) {
            fprintf(stderr, "[PTO Runtime ARM64] ERROR: Failed to create worker thread %d\n", i);
            free(ctx);
            rt->shutdown_requested = true;
            pthread_cond_broadcast(&rt->queue_not_empty);
            for (int j = 0; j < i; j++) {
                pthread_join(rt->workers[j], NULL);
            }
            pto_runtime_shutdown(rt);
            free(rt);
            return -1;
        }
        printf("[PTO Runtime ARM64] Created worker thread %d\n", i);
        fflush(stdout);
    }
    
    // Give workers a moment to start
    struct timespec start_delay = {0, 10000000};  // 10ms
    nanosleep(&start_delay, NULL);
    printf("[PTO Runtime ARM64] Workers started, now building task graph...\n");
    fflush(stdout);
    
    // Build task graph by calling orchestration function
    printf("[PTO Runtime ARM64] Building task graph...\n");
    fflush(stdout);
    orch_func(rt, user_data);
    
    // Mark that orchestration is complete
    pthread_mutex_lock(&rt->task_mutex);
    rt->execution_started = true;
    int64_t total_tasks = rt->total_tasks_scheduled;
    pthread_mutex_unlock(&rt->task_mutex);
    
    printf("[PTO Runtime ARM64] Task graph built: %lld tasks\n", (long long)total_tasks);
    printf("[PTO Runtime ARM64] Executing tasks...\n");
    
    // Wake up any workers that might be waiting
    pthread_mutex_lock(&rt->queue_mutex);
    pthread_cond_broadcast(&rt->queue_not_empty);
    pthread_mutex_unlock(&rt->queue_mutex);
    
    // Wait for all tasks to complete
    struct timespec poll_interval = {0, 1000000};  // 1ms
    while (1) {
        pthread_mutex_lock(&rt->task_mutex);
        bool all_done = (rt->total_tasks_completed >= rt->total_tasks_scheduled);
        int64_t completed = rt->total_tasks_completed;
        pthread_mutex_unlock(&rt->task_mutex);
        
        if (all_done) {
            printf("[PTO Runtime ARM64] All %lld tasks completed!\n", (long long)completed);
            break;
        }
        
        // Progress report
        static int64_t last_reported = 0;
        if (completed > last_reported + 1000 || completed == total_tasks) {
            printf("[PTO Runtime ARM64] Progress: %lld / %lld tasks (%.1f%%)\n",
                   (long long)completed, (long long)total_tasks,
                   100.0 * completed / total_tasks);
            last_reported = completed;
        }
        
        nanosleep(&poll_interval, NULL);
    }
    
    // Signal workers to shutdown
    printf("[PTO Runtime ARM64] Shutting down workers...\n");
    rt->shutdown_requested = true;
    
    // Wake up all workers
    pthread_mutex_lock(&rt->queue_mutex);
    pthread_cond_broadcast(&rt->queue_not_empty);
    pthread_mutex_unlock(&rt->queue_mutex);
    
    // Wait for all workers to exit
    for (int i = 0; i < num_workers; i++) {
        pthread_join(rt->workers[i], NULL);
    }
    
    // Print statistics
    printf("[PTO Runtime ARM64] ========================================\n");
    printf("[PTO Runtime ARM64] Execution Statistics\n");
    printf("[PTO Runtime ARM64]   Total tasks: %lld\n", (long long)rt->total_tasks_scheduled);
    printf("[PTO Runtime ARM64]   Completed:   %lld\n", (long long)rt->total_tasks_completed);
    printf("[PTO Runtime ARM64]   Workers:     %d\n", num_workers);
    printf("[PTO Runtime ARM64] ========================================\n");
    
    // Cleanup
    pto_runtime_shutdown(rt);
    free(rt);
    
    return 0;
}
