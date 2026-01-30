/**
 * PTO Runtime2 - Worker Thread Implementation
 * 
 * Implements worker threads that execute tasks dispatched by the scheduler.
 * 
 * Based on: docs/runtime_buffer_manager_methods.md
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif

#include "pto_worker.h"
#include "pto_runtime2.h"
#include "pto_runtime2_threaded.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <sched.h>


// =============================================================================
// Worker Type Names
// =============================================================================

const char* pto2_worker_type_name(PTO2WorkerType type) {
    switch (type) {
        case PTO2_WORKER_CUBE:        return "CUBE";
        case PTO2_WORKER_VECTOR:      return "VECTOR";
        case PTO2_WORKER_AI_CPU:      return "AI_CPU";
        case PTO2_WORKER_ACCELERATOR: return "ACCELERATOR";
        default:                      return "UNKNOWN";
    }
}

// =============================================================================
// Cycle Estimation (for simulation mode)
// =============================================================================

// Default cycle costs
#define DEFAULT_CUBE_CYCLES      100   // Matrix multiply
#define DEFAULT_VECTOR_CYCLES    50    // Element-wise ops
#define DEFAULT_DMA_CYCLES       80    // DMA transfer
#define DEFAULT_AICPU_CYCLES     200   // AI_CPU operations

static int64_t estimate_cycles_by_name(const char* func_name, int64_t data_size) {
    if (!func_name) {
        return DEFAULT_VECTOR_CYCLES;
    }
    
    // Matrix operations (Cube)
    if (strstr(func_name, "gemm") || strstr(func_name, "matmul") ||
        strstr(func_name, "conv")) {
        return DEFAULT_CUBE_CYCLES + data_size / 1024;
    }
    
    // DMA / Copy operations
    if (strstr(func_name, "dma") || strstr(func_name, "copy") ||
        strstr(func_name, "transfer")) {
        return DEFAULT_DMA_CYCLES + data_size / 512;
    }
    
    // Vector operations
    if (strstr(func_name, "add") || strstr(func_name, "mul") ||
        strstr(func_name, "relu") || strstr(func_name, "sigmoid") ||
        strstr(func_name, "vector")) {
        return DEFAULT_VECTOR_CYCLES + data_size / 2048;
    }
    
    // Default: vector operation
    return DEFAULT_VECTOR_CYCLES + data_size / 2048;
}

// =============================================================================
// Worker Initialization
// =============================================================================

bool pto2_worker_init(PTO2WorkerContext* worker, int32_t worker_id,
                       PTO2WorkerType worker_type, struct PTO2Runtime* runtime) {
    memset(worker, 0, sizeof(PTO2WorkerContext));
    
    worker->worker_id = worker_id;
    worker->worker_type = worker_type;
    worker->runtime = runtime;
    worker->shutdown = false;
    worker->current_task_id = -1;
    
    return true;
}

void pto2_worker_destroy(PTO2WorkerContext* worker) {
    // Nothing to free for now
    (void)worker;
}

void pto2_worker_reset(PTO2WorkerContext* worker) {
    worker->tasks_executed = 0;
    worker->total_cycles = 0;
    worker->total_stall_cycles = 0;
    worker->current_task_id = -1;
}

// =============================================================================
// Completion Queue Implementation
// =============================================================================

bool pto2_completion_queue_init(PTO2CompletionQueue* queue, int32_t capacity) {
    queue->entries = (PTO2CompletionEntry*)calloc(capacity, sizeof(PTO2CompletionEntry));
    if (!queue->entries) {
        return false;
    }
    
    queue->capacity = capacity;
    queue->head = 0;
    queue->tail = 0;
    
    if (pthread_mutex_init(&queue->mutex, NULL) != 0) {
        free(queue->entries);
        return false;
    }
    
    return true;
}

void pto2_completion_queue_destroy(PTO2CompletionQueue* queue) {
    if (queue->entries) {
        free(queue->entries);
        queue->entries = NULL;
    }
    pthread_mutex_destroy(&queue->mutex);
}

bool pto2_completion_queue_push(PTO2CompletionQueue* queue,
                                 int32_t task_id, int32_t worker_id,
                                 int64_t start_cycle, int64_t end_cycle) {
    pthread_mutex_lock(&queue->mutex);
    
    int32_t next_tail = (queue->tail + 1) % queue->capacity;
    if (next_tail == queue->head) {
        // Queue full
        pthread_mutex_unlock(&queue->mutex);
        return false;
    }
    
    PTO2CompletionEntry* entry = &queue->entries[queue->tail];
    entry->task_id = task_id;
    entry->worker_id = worker_id;
    entry->start_cycle = start_cycle;
    entry->end_cycle = end_cycle;
    
    queue->tail = next_tail;
    
    pthread_mutex_unlock(&queue->mutex);
    return true;
}

bool pto2_completion_queue_pop(PTO2CompletionQueue* queue, PTO2CompletionEntry* entry) {
    pthread_mutex_lock(&queue->mutex);
    
    if (queue->head == queue->tail) {
        // Queue empty
        pthread_mutex_unlock(&queue->mutex);
        return false;
    }
    
    *entry = queue->entries[queue->head];
    queue->head = (queue->head + 1) % queue->capacity;
    
    pthread_mutex_unlock(&queue->mutex);
    return true;
}

bool pto2_completion_queue_empty(PTO2CompletionQueue* queue) {
    pthread_mutex_lock(&queue->mutex);
    bool empty = (queue->head == queue->tail);
    pthread_mutex_unlock(&queue->mutex);
    return empty;
}

// =============================================================================
// Task Acquisition and Execution
// =============================================================================

// Check if this worker has the smallest clock among ALL workers of same type
// Compare with ALL workers - if any worker (waiting or not) has smaller clock,
// it should get the task first (it will finish and enter waiting state soon)
// When clocks are equal, allow competition (multi-threading fairness)
static bool worker_has_min_clock(PTO2WorkerContext* worker, PTO2ThreadContext* ctx) {
    int64_t my_clock = PTO2_LOAD_ACQUIRE(&ctx->worker_current_cycle[worker->worker_id]);
    
    int32_t start_id, end_id;
    if (worker->worker_type == PTO2_WORKER_CUBE) {
        start_id = 0;
        end_id = ctx->num_cube_workers;
    } else {
        start_id = ctx->num_cube_workers;
        end_id = ctx->num_cube_workers + ctx->num_vector_workers;
    }
    
    for (int32_t i = start_id; i < end_id; i++) {
        if (i == worker->worker_id) continue;
        int64_t other_clock = PTO2_LOAD_ACQUIRE(&ctx->worker_current_cycle[i]);
        if (other_clock < my_clock) {
            return false;  // Another worker has smaller clock - must wait
        }
        // When clocks are equal, allow competition
    }
    return true;  // I have the smallest clock (or equal)
}

int32_t pto2_worker_get_task(PTO2WorkerContext* worker) {
    PTO2RuntimeThreaded* rt = (PTO2RuntimeThreaded*)worker->runtime;
    PTO2ThreadContext* ctx = &rt->thread_ctx;
    PTO2SchedulerState* sched = &rt->base.scheduler;
    
    // Get the ready queue for this worker type
    PTO2ReadyQueue* queue = &sched->ready_queues[worker->worker_type];
    pthread_mutex_t* mutex = &ctx->ready_mutex[worker->worker_type];
    
    // Use per-worker condition variable
    pthread_cond_t* my_cond = &ctx->worker_cond[worker->worker_id];
    
    while (!worker->shutdown) {
        // Mark this worker as waiting BEFORE acquiring mutex
        // This ensures we're considered when tasks are enqueued
        __atomic_store_n(&ctx->worker_waiting[worker->worker_id], true, __ATOMIC_RELEASE);
        
        pthread_mutex_lock(mutex);
        
        // Wait while:
        // 1. Queue is empty, OR
        // 2. Queue has tasks but another waiting worker has smaller clock
        // Use timed wait to avoid lost signals causing permanent waits
        struct timespec timeout;
        while (!worker->shutdown) {
            if (pto2_ready_queue_empty(queue)) {
                // Queue empty - wait for task
                pthread_cond_wait(my_cond, mutex);
            } else if (!worker_has_min_clock(worker, ctx)) {
                // Queue has tasks but I don't have the smallest clock
                // Signal ALL workers with min clock and wait
                int32_t start_id, end_id;
                if (worker->worker_type == PTO2_WORKER_CUBE) {
                    start_id = 0;
                    end_id = ctx->num_cube_workers;
                } else {
                    start_id = ctx->num_cube_workers;
                    end_id = ctx->num_cube_workers + ctx->num_vector_workers;
                }
                int64_t my_clock = PTO2_LOAD_ACQUIRE(&ctx->worker_current_cycle[worker->worker_id]);
                // Find global min clock
                int64_t global_min = INT64_MAX;
                for (int32_t i = start_id; i < end_id; i++) {
                    int64_t clock = PTO2_LOAD_ACQUIRE(&ctx->worker_current_cycle[i]);
                    if (clock < global_min) {
                        global_min = clock;
                    }
                }
                // Broadcast to wake all workers (they will re-check)
                pthread_cond_broadcast(&ctx->ready_cond[worker->worker_type]);
                // Wait until my clock becomes min
                struct timespec timeout;
                clock_gettime(CLOCK_REALTIME, &timeout);
                timeout.tv_nsec += 100000;  // 100us timeout
                if (timeout.tv_nsec >= 1000000000) {
                    timeout.tv_sec += 1;
                    timeout.tv_nsec -= 1000000000;
                }
                pthread_cond_timedwait(my_cond, mutex, &timeout);
            } else {
                // Queue has tasks AND I have smallest clock - take task
                break;
            }
        }
        
        // No longer waiting
        __atomic_store_n(&ctx->worker_waiting[worker->worker_id], false, __ATOMIC_RELEASE);
        
        if (worker->shutdown && pto2_ready_queue_empty(queue)) {
            pthread_mutex_unlock(mutex);
            return -1;
        }
        
        // Take the task (we have the smallest clock among waiting workers)
        int32_t task_id = pto2_ready_queue_pop(queue);
        
        // Debug: Print all worker states when successfully getting a task
        #ifdef PTO2_DEBUG_LOAD_BALANCE
        {
            int64_t my_clock = PTO2_LOAD_ACQUIRE(&ctx->worker_current_cycle[worker->worker_id]);
            int32_t start_id, end_id;
            if (worker->worker_type == PTO2_WORKER_CUBE) {
                start_id = 0;
                end_id = ctx->num_cube_workers;
            } else {
                start_id = ctx->num_cube_workers;
                end_id = ctx->num_cube_workers + ctx->num_vector_workers;
            }
            
            fprintf(stderr, "\n[TASK_GET] Worker %d (clock=%ld) got task %d\n",
                    worker->worker_id, (long)my_clock, task_id);
            fprintf(stderr, "  All %s workers:\n", 
                    worker->worker_type == PTO2_WORKER_CUBE ? "CUBE" : "VECTOR");
            for (int32_t i = start_id; i < end_id; i++) {
                int64_t clock = PTO2_LOAD_ACQUIRE(&ctx->worker_current_cycle[i]);
                bool waiting = __atomic_load_n(&ctx->worker_waiting[i], __ATOMIC_ACQUIRE);
                const char* status = (i == worker->worker_id) ? "GOT_TASK" :
                                     waiting ? "WAITING" : "BUSY";
                fprintf(stderr, "    Worker %2d: clock=%7ld  %s%s\n", 
                        i, (long)clock, status,
                        (clock < my_clock) ? " <-- SMALLER CLOCK!" : "");
            }
        }
        #endif
        
        // If queue still has tasks, wake up ALL waiting workers with smallest clock
        if (!pto2_ready_queue_empty(queue)) {
            int32_t start_id, end_id;
            if (worker->worker_type == PTO2_WORKER_CUBE) {
                start_id = 0;
                end_id = ctx->num_cube_workers;
            } else {
                start_id = ctx->num_cube_workers;
                end_id = ctx->num_cube_workers + ctx->num_vector_workers;
            }
            
            // Find minimum clock among waiting workers
            int64_t min_clock = INT64_MAX;
            for (int32_t i = start_id; i < end_id; i++) {
                if (__atomic_load_n(&ctx->worker_waiting[i], __ATOMIC_ACQUIRE)) {
                    int64_t clock = PTO2_LOAD_ACQUIRE(&ctx->worker_current_cycle[i]);
                    if (clock < min_clock) {
                        min_clock = clock;
                    }
                }
            }
            // Signal ALL waiting workers with min clock
            if (min_clock < INT64_MAX) {
                for (int32_t i = start_id; i < end_id; i++) {
                    if (__atomic_load_n(&ctx->worker_waiting[i], __ATOMIC_ACQUIRE)) {
                        int64_t clock = PTO2_LOAD_ACQUIRE(&ctx->worker_current_cycle[i]);
                        if (clock == min_clock) {
                            pthread_cond_signal(&ctx->worker_cond[i]);
                        }
                    }
                }
            }
        }
        
        pthread_mutex_unlock(mutex);
        return task_id;
    }
    
    return -1;
}

int32_t pto2_worker_try_get_task(PTO2WorkerContext* worker) {
    PTO2RuntimeThreaded* rt = (PTO2RuntimeThreaded*)worker->runtime;
    PTO2ThreadContext* ctx = &rt->thread_ctx;
    PTO2SchedulerState* sched = &rt->base.scheduler;
    
    PTO2ReadyQueue* queue = &sched->ready_queues[worker->worker_type];
    pthread_mutex_t* mutex = &ctx->ready_mutex[worker->worker_type];
    
    return pto2_ready_queue_try_pop_threadsafe(queue, mutex);
}

void pto2_worker_execute_task(PTO2WorkerContext* worker, int32_t task_id) {
    PTO2RuntimeThreaded* rt = (PTO2RuntimeThreaded*)worker->runtime;
    PTO2TaskDescriptor* task = pto2_sm_get_task(rt->base.sm_handle, task_id);
    
    worker->current_task_id = task_id;
    
    // Call the InCore function if provided
    if (task->func_ptr) {
        // Build args array from task outputs
        void* args[PTO2_MAX_OUTPUTS + PTO2_MAX_INPUTS];
        int num_args = 0;
        
        // Add output pointers first
        for (int i = 0; i < task->num_outputs; i++) {
            args[num_args++] = (char*)task->packed_buffer_base + task->output_offsets[i];
        }
        
        // Call the function
        PTO2InCoreFunc func = (PTO2InCoreFunc)task->func_ptr;
        func(args, num_args);
    }
    
    worker->current_task_id = -1;
    worker->tasks_executed++;
}

int64_t pto2_worker_simulate_task(PTO2WorkerContext* worker, int32_t task_id) {
    PTO2RuntimeThreaded* rt = (PTO2RuntimeThreaded*)worker->runtime;
    PTO2TaskDescriptor* task = pto2_sm_get_task(rt->base.sm_handle, task_id);
    
    worker->current_task_id = task_id;
    
    // Calculate data size from packed buffer
    int64_t data_size = 0;
    if (task->packed_buffer_end && task->packed_buffer_base) {
        data_size = (int64_t)((char*)task->packed_buffer_end - 
                              (char*)task->packed_buffer_base);
    }
    
    // Estimate cycles
    int64_t cycles = estimate_cycles_by_name(task->func_name, data_size);
    
    worker->current_task_id = -1;
    worker->tasks_executed++;
    worker->total_cycles += cycles;
    
    return cycles;
}

void pto2_worker_task_complete(PTO2WorkerContext* worker, int32_t task_id, 
                                int64_t start_cycle, int64_t end_cycle) {
    PTO2RuntimeThreaded* rt = (PTO2RuntimeThreaded*)worker->runtime;
    PTO2ThreadContext* ctx = &rt->thread_ctx;
    
    // Push completion to queue (retry if full)
    int retry_count = 0;
    while (!pto2_completion_queue_push(&ctx->completion_queue,
                                        task_id, worker->worker_id,
                                        start_cycle, end_cycle)) {
        // Queue is full - signal scheduler and wait briefly
        pthread_mutex_lock(&ctx->done_mutex);
        pthread_cond_signal(&ctx->completion_cond);
        pthread_mutex_unlock(&ctx->done_mutex);
        
        retry_count++;
        if (retry_count % 1000 == 0) {
            fprintf(stderr, "[Worker %d] Completion queue full, retrying (%d)...\n",
                    worker->worker_id, retry_count);
        }
        PTO2_SPIN_PAUSE();
    }
    
    // Signal completion condition
    pthread_mutex_lock(&ctx->done_mutex);
    pthread_cond_signal(&ctx->completion_cond);
    pthread_mutex_unlock(&ctx->done_mutex);
}

// =============================================================================
// Worker Thread Functions
// =============================================================================

void* pto2_worker_thread_func(void* arg) {
    PTO2WorkerContext* worker = (PTO2WorkerContext*)arg;
    PTO2RuntimeThreaded* rt = (PTO2RuntimeThreaded*)worker->runtime;
    PTO2ThreadContext* ctx = &rt->thread_ctx;
    
    // Signal that this worker is ready
    pthread_mutex_lock(&ctx->startup_mutex);
    ctx->workers_ready++;
    pthread_cond_broadcast(&ctx->startup_cond);
    pthread_mutex_unlock(&ctx->startup_mutex);
    
    while (!worker->shutdown) {
        // Get next task (blocks if queue empty)
        int32_t task_id = pto2_worker_get_task(worker);
        if (task_id < 0) {
            // Shutdown or error
            break;
        }
        
        // Execute the task
        pto2_worker_execute_task(worker, task_id);
        
        // Signal completion (with 0 cycles since not simulating)
        pto2_worker_task_complete(worker, task_id, 0, 0);
    }
    
    return NULL;
}

void* pto2_worker_thread_func_sim(void* arg) {
    PTO2WorkerContext* worker = (PTO2WorkerContext*)arg;
    PTO2RuntimeThreaded* rt = (PTO2RuntimeThreaded*)worker->runtime;
    PTO2ThreadContext* ctx = &rt->thread_ctx;
    
    // Signal that this worker is ready
    pthread_mutex_lock(&ctx->startup_mutex);
    ctx->workers_ready++;
    pthread_cond_broadcast(&ctx->startup_cond);
    pthread_mutex_unlock(&ctx->startup_mutex);
    
    while (!worker->shutdown) {
        // Get next task (blocks if queue empty)
        // Clock-based load balancing is now inside pto2_worker_get_task
        int32_t task_id = pto2_worker_get_task(worker);
        if (task_id < 0) {
            // Shutdown or error
            break;
        }
        
        // Get task descriptor to check dependencies
        PTO2TaskDescriptor* task = pto2_sm_get_task(rt->base.sm_handle, task_id);
        
        // === Calculate earliest start based on dependencies ===
        int64_t earliest_start = 0;
        
        // Check all fanin tasks for their completion times
        pthread_mutex_lock(&ctx->task_end_mutex);
        int32_t fanin_current = task->fanin_head;
        while (fanin_current > 0) {
            PTO2DepListEntry* entry = pto2_dep_pool_get(&rt->base.orchestrator.dep_pool, 
                                                         fanin_current);
            if (!entry) break;
            
            int32_t dep_task_id = entry->task_id;
            int32_t window_mask = rt->base.scheduler.task_window_mask;
            int32_t dep_slot = dep_task_id & window_mask;
            if (dep_slot >= 0 && dep_slot < ctx->task_end_cycles_capacity) {
                int64_t dep_end = ctx->task_end_cycles[dep_slot];
                if (dep_end > earliest_start) {
                    earliest_start = dep_end;
                }
            }
            
            fanin_current = entry->next_offset;
        }
        pthread_mutex_unlock(&ctx->task_end_mutex);
        
        // Get worker's current cycle (when this worker will be free)
        int64_t worker_free_cycle = PTO2_LOAD_ACQUIRE(&ctx->worker_current_cycle[worker->worker_id]);
        
        // Start time is max of dependency completion and worker availability
        int64_t start_cycle = (earliest_start > worker_free_cycle) ? 
                               earliest_start : worker_free_cycle;
        
        worker->task_start_cycle = start_cycle;
        
        // Simulate the task (estimate cycles)
        int64_t cycles = pto2_worker_simulate_task(worker, task_id);
        
        int64_t end_cycle = start_cycle + cycles;
        
        // IMPORTANT: Update worker's clock IMMEDIATELY after calculating end_cycle
        // This allows other workers to see the correct expected end time
        // and avoid taking tasks that should go to workers with smaller clocks
        PTO2_STORE_RELEASE(&ctx->worker_current_cycle[worker->worker_id], end_cycle);
        
        // Update task end cycle for dependency tracking
        int32_t window_mask = rt->base.scheduler.task_window_mask;
        int32_t slot = task_id & window_mask;
        pthread_mutex_lock(&ctx->task_end_mutex);
        if (slot >= 0 && slot < ctx->task_end_cycles_capacity) {
            ctx->task_end_cycles[slot] = end_cycle;
        }
        pthread_mutex_unlock(&ctx->task_end_mutex);
        
        // Track stall cycles
        int64_t stall_cycles = start_cycle - worker_free_cycle;
        if (stall_cycles > 0) {
            worker->total_stall_cycles += stall_cycles;
        }
        
        // Record trace event
        pto2_runtime_record_trace(rt, task_id, worker->worker_id, 
                                   start_cycle, end_cycle, task->func_name);
        
        // Signal completion with actual timing
        pto2_worker_task_complete(worker, task_id, start_cycle, end_cycle);
    }
    
    return NULL;
}

// =============================================================================
// Statistics
// =============================================================================

void pto2_worker_print_stats(PTO2WorkerContext* worker) {
    printf("Worker %d (%s):\n", worker->worker_id, 
           pto2_worker_type_name(worker->worker_type));
    printf("  Tasks executed:     %lld\n", (long long)worker->tasks_executed);
    printf("  Total cycles:       %lld\n", (long long)worker->total_cycles);
    printf("  Total stall cycles: %lld\n", (long long)worker->total_stall_cycles);
    
    if (worker->tasks_executed > 0) {
        printf("  Avg cycles/task:    %lld\n", 
               (long long)(worker->total_cycles / worker->tasks_executed));
    }
}
