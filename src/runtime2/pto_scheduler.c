/**
 * PTO Runtime2 - Scheduler Implementation
 * 
 * Implements scheduler state management, ready queues, and task lifecycle.
 * 
 * Based on: docs/runtime_buffer_manager_methods.md
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif

#include "pto_scheduler.h"
#include "pto_worker.h"
#include "pto_runtime2.h"
#include "pto_runtime2_threaded.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>
#include <time.h>

// =============================================================================
// Task State Names
// =============================================================================

const char* pto2_task_state_name(PTO2TaskState state) {
    switch (state) {
        case PTO2_TASK_PENDING:   return "PENDING";
        case PTO2_TASK_READY:     return "READY";
        case PTO2_TASK_RUNNING:   return "RUNNING";
        case PTO2_TASK_COMPLETED: return "COMPLETED";
        case PTO2_TASK_CONSUMED:  return "CONSUMED";
        default:                  return "UNKNOWN";
    }
}

// =============================================================================
// Ready Queue Implementation
// =============================================================================

bool pto2_ready_queue_init(PTO2ReadyQueue* queue, int32_t capacity) {
    queue->task_ids = (int32_t*)malloc(capacity * sizeof(int32_t));
    if (!queue->task_ids) {
        return false;
    }
    
    queue->head = 0;
    queue->tail = 0;
    queue->capacity = capacity;
    queue->count = 0;
    
    return true;
}

void pto2_ready_queue_destroy(PTO2ReadyQueue* queue) {
    if (queue->task_ids) {
        free(queue->task_ids);
        queue->task_ids = NULL;
    }
}

void pto2_ready_queue_reset(PTO2ReadyQueue* queue) {
    queue->head = 0;
    queue->tail = 0;
    queue->count = 0;
}

bool pto2_ready_queue_push(PTO2ReadyQueue* queue, int32_t task_id) {
    if (pto2_ready_queue_full(queue)) {
        return false;
    }
    
    queue->task_ids[queue->tail] = task_id;
    queue->tail = (queue->tail + 1) % queue->capacity;
    queue->count++;
    
    return true;
}

int32_t pto2_ready_queue_pop(PTO2ReadyQueue* queue) {
    if (pto2_ready_queue_empty(queue)) {
        return -1;
    }
    
    int32_t task_id = queue->task_ids[queue->head];
    queue->head = (queue->head + 1) % queue->capacity;
    queue->count--;
    
    return task_id;
}

// =============================================================================
// Thread-Safe Ready Queue Operations
// =============================================================================

bool pto2_ready_queue_push_threadsafe(PTO2ReadyQueue* queue, int32_t task_id,
                                       pthread_mutex_t* mutex, pthread_cond_t* cond) {
    pthread_mutex_lock(mutex);
    
    bool success = pto2_ready_queue_push(queue, task_id);
    
    if (success) {
        // Wake ALL waiting workers to compete fairly for tasks
        // Using signal() tends to wake the same worker repeatedly (OS scheduling bias)
        pthread_cond_broadcast(cond);
    } else {
        fprintf(stderr, "[ERROR] Ready queue full! Task %d dropped!\n", task_id);
    }
    
    pthread_mutex_unlock(mutex);
    return success;
}

// Push task and only wake the worker with smallest simulated clock
// This ensures fair task distribution based on simulation time
// Static counter for round-robin among workers with same min clock
static volatile int32_t s_round_robin_counter = 0;

bool pto2_ready_queue_push_wake_min_clock(PTO2ReadyQueue* queue, int32_t task_id,
                                           pthread_mutex_t* mutex,
                                           volatile int64_t* worker_clocks,
                                           volatile bool* worker_waiting,
                                           pthread_cond_t* worker_conds,
                                           int32_t worker_start, int32_t worker_end) {
    pthread_mutex_lock(mutex);
    
    bool success = pto2_ready_queue_push(queue, task_id);
    
    if (success) {
        // Broadcast to ALL waiting workers
        // Each worker will check if it has the smallest clock before taking task
        // This ensures the worker with smallest clock gets the task
        for (int32_t i = worker_start; i < worker_end; i++) {
            if (__atomic_load_n(&worker_waiting[i], __ATOMIC_ACQUIRE)) {
                pthread_cond_signal(&worker_conds[i]);
            }
        }
    } else {
        fprintf(stderr, "[ERROR] Ready queue full! Task %d dropped!\n", task_id);
    }
    
    pthread_mutex_unlock(mutex);
    return success;
}

int32_t pto2_ready_queue_pop_threadsafe(PTO2ReadyQueue* queue,
                                         pthread_mutex_t* mutex, pthread_cond_t* cond,
                                         volatile bool* shutdown) {
    pthread_mutex_lock(mutex);
    
    // Wait while queue is empty and not shutting down
    while (pto2_ready_queue_empty(queue) && !(*shutdown)) {
        pthread_cond_wait(cond, mutex);
    }
    
    // Check if we woke up due to shutdown
    if (*shutdown && pto2_ready_queue_empty(queue)) {
        pthread_mutex_unlock(mutex);
        return -1;
    }
    
    int32_t task_id = pto2_ready_queue_pop(queue);
    
    pthread_mutex_unlock(mutex);
    return task_id;
}

// Extended version that allows caller to check additional conditions after wakeup
// Returns -2 if should_yield callback returns true (caller should retry)
int32_t pto2_ready_queue_pop_with_yield_check(PTO2ReadyQueue* queue,
                                               pthread_mutex_t* mutex, pthread_cond_t* cond,
                                               volatile bool* shutdown,
                                               bool (*should_yield)(void* ctx), void* yield_ctx) {
    pthread_mutex_lock(mutex);
    
    while (true) {
        // Wait while queue is empty and not shutting down
        while (pto2_ready_queue_empty(queue) && !(*shutdown)) {
            pthread_cond_wait(cond, mutex);
        }
        
        // Check if we woke up due to shutdown
        if (*shutdown && pto2_ready_queue_empty(queue)) {
            pthread_mutex_unlock(mutex);
            return -1;
        }
        
        // After wakeup, check if we should yield to other workers
        // (e.g., if our simulated clock is not the smallest)
        if (should_yield && should_yield(yield_ctx)) {
            // Signal other workers and go back to waiting
            pthread_cond_broadcast(cond);
            pthread_cond_wait(cond, mutex);
            continue;  // Re-check conditions
        }
        
        break;  // We can take the task
    }
    
    int32_t task_id = pto2_ready_queue_pop(queue);
    
    pthread_mutex_unlock(mutex);
    return task_id;
}

int32_t pto2_ready_queue_try_pop_threadsafe(PTO2ReadyQueue* queue,
                                             pthread_mutex_t* mutex) {
    pthread_mutex_lock(mutex);
    
    int32_t task_id = pto2_ready_queue_pop(queue);
    
    pthread_mutex_unlock(mutex);
    return task_id;
}

int32_t pto2_ready_queue_count_threadsafe(PTO2ReadyQueue* queue, pthread_mutex_t* mutex) {
    pthread_mutex_lock(mutex);
    int32_t count = queue->count;
    pthread_mutex_unlock(mutex);
    return count;
}

bool pto2_ready_queue_empty_threadsafe(PTO2ReadyQueue* queue, pthread_mutex_t* mutex) {
    pthread_mutex_lock(mutex);
    bool empty = pto2_ready_queue_empty(queue);
    pthread_mutex_unlock(mutex);
    return empty;
}

// =============================================================================
// Scheduler Initialization
// =============================================================================

bool pto2_scheduler_init(PTO2SchedulerState* sched, 
                          PTO2SharedMemoryHandle* sm_handle,
                          PTO2DepListPool* dep_pool) {
    memset(sched, 0, sizeof(PTO2SchedulerState));
    
    sched->sm_handle = sm_handle;
    sched->dep_pool = dep_pool;
    
    // Get runtime task_window_size from shared memory header
    int32_t window_size = sm_handle->header->task_window_size;
    sched->task_window_size = window_size;
    sched->task_window_mask = window_size - 1;  // For fast modulo (window_size must be power of 2)
    
    // Initialize local copies of ring pointers
    sched->last_task_alive = 0;
    sched->heap_tail = 0;
    
    // Allocate per-task state arrays (dynamically sized based on runtime window_size)
    sched->task_state = (PTO2TaskState*)calloc(window_size, sizeof(PTO2TaskState));
    if (!sched->task_state) {
        return false;
    }
    
    sched->fanin_refcount = (int32_t*)calloc(window_size, sizeof(int32_t));
    if (!sched->fanin_refcount) {
        free(sched->task_state);
        return false;
    }
    
    sched->fanout_refcount = (int32_t*)calloc(window_size, sizeof(int32_t));
    if (!sched->fanout_refcount) {
        free(sched->fanin_refcount);
        free(sched->task_state);
        return false;
    }
    
    // Allocate per-slot locks (all initialized to 0 = unlocked)
    sched->slot_locks = (volatile int32_t*)calloc(window_size, sizeof(int32_t));
    if (!sched->slot_locks) {
        free(sched->fanout_refcount);
        free(sched->fanin_refcount);
        free(sched->task_state);
        return false;
    }
    
    // Initialize ready queues
    for (int i = 0; i < PTO2_NUM_WORKER_TYPES; i++) {
        if (!pto2_ready_queue_init(&sched->ready_queues[i], PTO2_READY_QUEUE_SIZE)) {
            // Cleanup on failure
            for (int j = 0; j < i; j++) {
                pto2_ready_queue_destroy(&sched->ready_queues[j]);
            }
            free((void*)sched->slot_locks);
            free(sched->fanout_refcount);
            free(sched->fanin_refcount);
            free(sched->task_state);
            return false;
        }
    }
    
    return true;
}

void pto2_scheduler_destroy(PTO2SchedulerState* sched) {
    if (sched->task_state) {
        free(sched->task_state);
        sched->task_state = NULL;
    }
    
    if (sched->fanin_refcount) {
        free(sched->fanin_refcount);
        sched->fanin_refcount = NULL;
    }
    
    if (sched->fanout_refcount) {
        free(sched->fanout_refcount);
        sched->fanout_refcount = NULL;
    }
    
    if (sched->slot_locks) {
        free((void*)sched->slot_locks);
        sched->slot_locks = NULL;
    }
    
    for (int i = 0; i < PTO2_NUM_WORKER_TYPES; i++) {
        pto2_ready_queue_destroy(&sched->ready_queues[i]);
    }
}

void pto2_scheduler_reset(PTO2SchedulerState* sched) {
    sched->last_task_alive = 0;
    sched->heap_tail = 0;
    
    memset(sched->task_state, 0, PTO2_TASK_WINDOW_SIZE * sizeof(PTO2TaskState));
    memset(sched->fanin_refcount, 0, PTO2_TASK_WINDOW_SIZE * sizeof(int32_t));
    memset(sched->fanout_refcount, 0, PTO2_TASK_WINDOW_SIZE * sizeof(int32_t));
    
    for (int i = 0; i < PTO2_NUM_WORKER_TYPES; i++) {
        pto2_ready_queue_reset(&sched->ready_queues[i]);
    }
    
    sched->tasks_completed = 0;
    sched->tasks_consumed = 0;
}

// =============================================================================
// Task State Management
// =============================================================================

void pto2_scheduler_init_task(PTO2SchedulerState* sched, int32_t task_id,
                               PTO2TaskDescriptor* task) {
    int32_t slot = pto2_task_slot(sched, task_id);
    
    // Initialize scheduler state for this task
    sched->task_state[slot] = PTO2_TASK_PENDING;
    sched->fanin_refcount[slot] = 0;
    sched->fanout_refcount[slot] = 0;
    
    // Check if task is immediately ready (no dependencies)
    if (task->fanin_count == 0) {
        sched->task_state[slot] = PTO2_TASK_READY;
        pto2_ready_queue_push(&sched->ready_queues[task->worker_type], task_id);
    }
}

void pto2_scheduler_check_ready(PTO2SchedulerState* sched, int32_t task_id,
                                 PTO2TaskDescriptor* task) {
    int32_t slot = pto2_task_slot(sched, task_id);
    
    // Only transition PENDING -> READY
    if (sched->task_state[slot] != PTO2_TASK_PENDING) {
        return;
    }
    
    // Check if all producers have completed
    if (sched->fanin_refcount[slot] == task->fanin_count) {
        sched->task_state[slot] = PTO2_TASK_READY;
        pto2_ready_queue_push(&sched->ready_queues[task->worker_type], task_id);
    }
}

void pto2_scheduler_mark_running(PTO2SchedulerState* sched, int32_t task_id) {
    int32_t slot = pto2_task_slot(sched, task_id);
    sched->task_state[slot] = PTO2_TASK_RUNNING;
}

int32_t pto2_scheduler_get_ready_task(PTO2SchedulerState* sched, 
                                       PTO2WorkerType worker_type) {
    return pto2_ready_queue_pop(&sched->ready_queues[worker_type]);
}

// =============================================================================
// Task Completion Handling
// =============================================================================

/**
 * Check if task can transition to CONSUMED and handle if so
 * 
 * NOTE: fanout_refcount is accessed atomically because it can be modified
 * by both orchestrator thread (via scope_end) and scheduler thread (via task_complete).
 */
static void check_and_handle_consumed(PTO2SchedulerState* sched, 
                                       int32_t task_id,
                                       PTO2TaskDescriptor* task) {
    int32_t slot = pto2_task_slot(sched, task_id);
    
    // Read fanout_count (set by orchestrator, only grows)
    int32_t fanout_count = __atomic_load_n(&task->fanout_count, __ATOMIC_ACQUIRE);
    
    // Read fanout_refcount atomically (modified by both orchestrator and scheduler threads)
    int32_t refcount = __atomic_load_n(&sched->fanout_refcount[slot], __ATOMIC_SEQ_CST);
    
    if (refcount != fanout_count) {
        return;  // Not all references released yet
    }
    
    // Use CAS to atomically transition COMPLETED -> CONSUMED
    // This prevents multiple threads from transitioning the same task
    int32_t expected = PTO2_TASK_COMPLETED;
    if (!__atomic_compare_exchange_n(&sched->task_state[slot], &expected, PTO2_TASK_CONSUMED,
                                      false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
        // CAS failed - either not COMPLETED or another thread already transitioned
        return;
    }
    
    // Successfully transitioned to CONSUMED
    __atomic_fetch_add(&sched->tasks_consumed, 1, __ATOMIC_SEQ_CST);
    
    // Reset refcounts for slot reuse (ring buffer will reuse this slot)
    // Use atomic store for fanout_refcount
    __atomic_store_n(&sched->fanout_refcount[slot], 0, __ATOMIC_SEQ_CST);
    __atomic_store_n(&sched->fanin_refcount[slot], 0, __ATOMIC_SEQ_CST);
    
    // Try to advance ring pointers
    if (task_id == sched->last_task_alive) {
        pto2_scheduler_advance_ring_pointers(sched);
    }
}

void pto2_scheduler_on_task_complete(PTO2SchedulerState* sched, int32_t task_id) {
    int32_t slot = pto2_task_slot(sched, task_id);
    PTO2TaskDescriptor* task = pto2_sm_get_task(sched->sm_handle, task_id);
    
    // Mark task as completed
    sched->task_state[slot] = PTO2_TASK_COMPLETED;
    sched->tasks_completed++;
    
    // === STEP 1: Update fanin_refcount of all consumers ===
    // Read fanout_list and increment each consumer's fanin_refcount
    int32_t fanout_head = PTO2_LOAD_ACQUIRE(&task->fanout_head);
    int32_t current = fanout_head;
    
    while (current > 0) {
        PTO2DepListEntry* entry = pto2_dep_pool_get(sched->dep_pool, current);
        if (!entry) break;
        
        int32_t consumer_id = entry->task_id;
        int32_t consumer_slot = pto2_task_slot(sched, consumer_id);
        PTO2TaskDescriptor* consumer = pto2_sm_get_task(sched->sm_handle, consumer_id);
        
        // Increment consumer's fanin_refcount
        sched->fanin_refcount[consumer_slot]++;
        
        // Check if consumer is now ready
        pto2_scheduler_check_ready(sched, consumer_id, consumer);
        
        current = entry->next_offset;
    }
    
    // === STEP 2: Update fanout_refcount of all producers ===
    // This task is a consumer of its fanin producers - release references
    current = task->fanin_head;
    
    while (current > 0) {
        PTO2DepListEntry* entry = pto2_dep_pool_get(sched->dep_pool, current);
        if (!entry) break;
        
        int32_t producer_id = entry->task_id;
        pto2_scheduler_release_producer(sched, producer_id);
        
        current = entry->next_offset;
    }
    
    // === STEP 3: Check if this task can transition to CONSUMED ===
    check_and_handle_consumed(sched, task_id, task);
}

void pto2_scheduler_on_scope_end(PTO2SchedulerState* sched, 
                                  int32_t begin_pos, int32_t end_pos) {
    // Note: In multi-threaded mode, scope_end may be called from orchestrator thread
    // before scheduler has initialized the tasks. The refcount update should be
    // deferred until after task initialization.
    // 
    // For now, we increment refcount directly. This works if scope_end is called
    // after scheduler processes new tasks, or if we don't reinitialize refcount.
    for (int32_t task_id = begin_pos; task_id < end_pos; task_id++) {
        pto2_scheduler_release_producer(sched, task_id);
    }
}

void pto2_scheduler_release_producer(PTO2SchedulerState* sched, int32_t producer_id) {
    int32_t slot = pto2_task_slot(sched, producer_id);
    PTO2TaskDescriptor* producer = pto2_sm_get_task(sched->sm_handle, producer_id);
    
    // Increment fanout_refcount atomically (called from both orchestrator and scheduler threads)
    __atomic_fetch_add(&sched->fanout_refcount[slot], 1, __ATOMIC_SEQ_CST);
    
    // Check if producer can transition to CONSUMED
    check_and_handle_consumed(sched, producer_id, producer);
}

// =============================================================================
// Ring Pointer Management
// =============================================================================

void pto2_scheduler_advance_ring_pointers(PTO2SchedulerState* sched) {
    PTO2SharedMemoryHeader* header = sched->sm_handle->header;
    int32_t current_task_index = PTO2_LOAD_ACQUIRE(&header->current_task_index);
    
    // Advance last_task_alive while tasks at that position are CONSUMED
    while (sched->last_task_alive < current_task_index) {
        int32_t slot = pto2_task_slot(sched, sched->last_task_alive);
        
        if (sched->task_state[slot] != PTO2_TASK_CONSUMED) {
            break;  // Found non-consumed task, stop advancing
        }
        
        sched->last_task_alive++;
    }
    
    // Update heap_tail based on last consumed task's buffer
    if (sched->last_task_alive > 0) {
        int32_t last_consumed_id = sched->last_task_alive - 1;
        PTO2TaskDescriptor* last_consumed = pto2_sm_get_task(sched->sm_handle, last_consumed_id);
        
        if (last_consumed->packed_buffer_end != NULL) {
            // heap_tail = offset of end of last consumed task's buffer
            // Note: This requires knowing the heap base, which should be passed in
            // For now, we just track the relative position
            sched->heap_tail = (int32_t)(intptr_t)last_consumed->packed_buffer_end;
        }
    }
    
    // Write to shared memory for orchestrator flow control
    pto2_scheduler_sync_to_sm(sched);
}

void pto2_scheduler_sync_to_sm(PTO2SchedulerState* sched) {
    PTO2SharedMemoryHeader* header = sched->sm_handle->header;
    
    PTO2_STORE_RELEASE(&header->last_task_alive, sched->last_task_alive);
    PTO2_STORE_RELEASE(&header->heap_tail, sched->heap_tail);
}

// =============================================================================
// Scheduler Main Loop Helpers
// =============================================================================

bool pto2_scheduler_is_done(PTO2SchedulerState* sched) {
    PTO2SharedMemoryHeader* header = sched->sm_handle->header;
    
    // Check if orchestrator has finished
    int32_t orch_done = PTO2_LOAD_ACQUIRE(&header->orchestrator_done);
    if (!orch_done) {
        return false;
    }
    
    // Check if all tasks have been consumed
    int32_t current_task_index = PTO2_LOAD_ACQUIRE(&header->current_task_index);
    return sched->last_task_alive >= current_task_index;
}

int32_t pto2_scheduler_process_new_tasks(PTO2SchedulerState* sched) {
    // In simulated mode with shared address space, tasks are already
    // initialized by the orchestrator during pto2_submit_task().
    // This function is kept for compatibility with decoupled mode
    // where orchestrator and scheduler run on different processors.
    (void)sched;
    return 0;
}

// =============================================================================
// Debug Utilities
// =============================================================================

void pto2_scheduler_print_stats(PTO2SchedulerState* sched) {
    printf("=== Scheduler Statistics ===\n");
    printf("last_task_alive:   %d\n", sched->last_task_alive);
    printf("heap_tail:         %d\n", sched->heap_tail);
    printf("tasks_completed:   %lld\n", (long long)sched->tasks_completed);
    printf("tasks_consumed:    %lld\n", (long long)sched->tasks_consumed);
    printf("============================\n");
}

void pto2_scheduler_print_queues(PTO2SchedulerState* sched) {
    printf("=== Ready Queues ===\n");
    
    const char* worker_names[] = {"CUBE", "VECTOR", "AI_CPU", "ACCELERATOR"};
    
    for (int i = 0; i < PTO2_NUM_WORKER_TYPES; i++) {
        printf("  %s: count=%d\n", worker_names[i], 
               pto2_ready_queue_count(&sched->ready_queues[i]));
    }
    
    printf("====================\n");
}

// =============================================================================
// Scheduler Thread Implementation
// =============================================================================

/**
 * Thread-safe version of check_ready that signals workers
 */
static void check_ready_threadsafe(PTO2SchedulerState* sched, int32_t task_id,
                                    PTO2TaskDescriptor* task,
                                    PTO2ThreadContext* thread_ctx) {
    int32_t slot = pto2_task_slot(sched, task_id);
    
    // Read state with acquire semantics
    int32_t state = __atomic_load_n(&sched->task_state[slot], __ATOMIC_ACQUIRE);
    
    // Only transition PENDING -> READY
    if (state != PTO2_TASK_PENDING) {
        return;
    }
    
    // Read fanin_count with acquire semantics to ensure visibility
    int32_t fanin_count = __atomic_load_n(&task->fanin_count, __ATOMIC_ACQUIRE);
    
    // Check if all producers have completed
    int32_t refcount = __atomic_load_n(&sched->fanin_refcount[slot], __ATOMIC_ACQUIRE);
    if (refcount >= fanin_count) {
        // Use CAS to atomically transition PENDING -> READY
        int32_t expected = PTO2_TASK_PENDING;
        if (__atomic_compare_exchange_n(&sched->task_state[slot], &expected, PTO2_TASK_READY,
                                         false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
            // Successfully transitioned to READY
            pto2_scheduler_enqueue_ready_threadsafe(sched, task_id, 
                                                     task->worker_type, thread_ctx);
        } else {
            // CAS failed - log if the state was unexpected
            int32_t actual = __atomic_load_n(&sched->task_state[slot], __ATOMIC_ACQUIRE);
            if (actual != PTO2_TASK_READY && actual != PTO2_TASK_RUNNING && 
                actual != PTO2_TASK_COMPLETED && actual != PTO2_TASK_CONSUMED) {
                fprintf(stderr, "[WARN] Task %d CAS failed: expected PENDING, got %d\n",
                        task_id, expected);
            }
        }
    }
}

/**
 * Thread-safe task completion handling
 */
static void on_task_complete_threadsafe(PTO2SchedulerState* sched, int32_t task_id,
                                         PTO2ThreadContext* thread_ctx) {
    int32_t slot = pto2_task_slot(sched, task_id);
    PTO2TaskDescriptor* task = pto2_sm_get_task(sched->sm_handle, task_id);
    
    // Check if already completed (prevent duplicate processing)
    int32_t old_state = sched->task_state[slot];
    if (old_state >= PTO2_TASK_COMPLETED) {
        return;  // Skip duplicate processing
    }
    
    // Mark task as completed
    sched->task_state[slot] = PTO2_TASK_COMPLETED;
    sched->tasks_completed++;
    
    // === STEP 1: Update fanin_refcount of all consumers ===
    // CRITICAL: Lock the task's fanout to synchronize with orchestrator adding consumers
    // This prevents race where:
    //   1. Orchestrator is adding consumer to this task's fanout list
    //   2. We're iterating the fanout list and miss the new consumer
    while (PTO2_EXCHANGE(&task->fanout_lock, 1) != 0) {
        PTO2_SPIN_PAUSE();
    }
    
    int32_t fanout_head = task->fanout_head;  // Safe to read without atomic after acquiring lock
    int32_t current = fanout_head;
    int consumers_updated = 0;
    
    while (current > 0) {
        PTO2DepListEntry* entry = pto2_dep_pool_get(sched->dep_pool, current);
        if (!entry) {
            fprintf(stderr, "[ERROR] Task %d: NULL entry at offset %d!\n", task_id, current);
            break;
        }
        
        int32_t consumer_id = entry->task_id;
        int32_t consumer_slot = pto2_task_slot(sched, consumer_id);
        PTO2TaskDescriptor* consumer = pto2_sm_get_task(sched->sm_handle, consumer_id);
        
        // Increment consumer's fanin_refcount atomically
        int32_t new_refcount = __atomic_add_fetch(&sched->fanin_refcount[consumer_slot], 1, __ATOMIC_SEQ_CST);
        consumers_updated++;
        
        // Read fanin_count with proper synchronization
        int32_t fanin_count = __atomic_load_n(&consumer->fanin_count, __ATOMIC_ACQUIRE);
        
        // Check if consumer is now ready - inline the check here to avoid race
        if (new_refcount >= fanin_count) {
            // Try to transition to READY
            int32_t expected = PTO2_TASK_PENDING;
            if (__atomic_compare_exchange_n(&sched->task_state[consumer_slot], &expected, PTO2_TASK_READY,
                                             false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
                pto2_scheduler_enqueue_ready_threadsafe(sched, consumer_id, 
                                                         consumer->worker_type, thread_ctx);
            }
            // CAS failure is OK - means someone else already transitioned to READY
        }
        
        current = entry->next_offset;
    }
    
    // Release the fanout lock
    PTO2_STORE_RELEASE(&task->fanout_lock, 0);
    
    // Warn if fanout_head > 0 but no consumers updated (indicates corrupt list)
    int32_t expected_consumers = task->fanout_count - task->scope_depth;
    if (fanout_head > 0 && consumers_updated == 0) {
        fprintf(stderr, "[ERROR] Task %d: fanout_head=%d but no consumers updated!\n",
                task_id, fanout_head);
    }
    
    // Debug: warn if no consumers were updated but fanout_count > scope_depth
    if (consumers_updated == 0 && task->fanout_count > task->scope_depth) {
        fprintf(stderr, "[WARNING] Task %d (%s) completed with fanout_head=%d but no consumers updated! "
                "fanout_count=%d, scope_depth=%d\n",
                task_id, task->func_name, fanout_head, task->fanout_count, task->scope_depth);
    }
    
    // === STEP 2: Update fanout_refcount of all producers ===
    current = task->fanin_head;
    
    while (current > 0) {
        PTO2DepListEntry* entry = pto2_dep_pool_get(sched->dep_pool, current);
        if (!entry) break;
        
        int32_t producer_id = entry->task_id;
        pto2_scheduler_release_producer(sched, producer_id);
        
        current = entry->next_offset;
    }
    
    // === STEP 3: Check if this task can transition to CONSUMED ===
    // Delegate to shared function that uses proper synchronization
    check_and_handle_consumed(sched, task_id, task);
}

void pto2_scheduler_enqueue_ready_threadsafe(PTO2SchedulerState* sched,
                                              int32_t task_id,
                                              PTO2WorkerType worker_type,
                                              PTO2ThreadContext* thread_ctx) {
    PTO2ReadyQueue* queue = &sched->ready_queues[worker_type];
    pthread_mutex_t* mutex = &thread_ctx->ready_mutex[worker_type];
    
    // Determine worker range for this type
    int32_t worker_start, worker_end;
    if (worker_type == PTO2_WORKER_CUBE) {
        worker_start = 0;
        worker_end = thread_ctx->num_cube_workers;
    } else {
        worker_start = thread_ctx->num_cube_workers;
        worker_end = thread_ctx->num_cube_workers + thread_ctx->num_vector_workers;
    }
    
    // Push task and only wake the worker with smallest clock
    pto2_ready_queue_push_wake_min_clock(queue, task_id, mutex,
                                          thread_ctx->worker_current_cycle,
                                          thread_ctx->worker_waiting,
                                          thread_ctx->worker_cond,
                                          worker_start, worker_end);
}

int32_t pto2_scheduler_process_completions(PTO2SchedulerContext* ctx) {
    PTO2SchedulerState* sched = ctx->scheduler;
    PTO2ThreadContext* thread_ctx = ctx->thread_ctx;
    
    int32_t count = 0;
    PTO2CompletionEntry entry;
    
    // Process all available completions
    while (pto2_completion_queue_pop(&thread_ctx->completion_queue, &entry)) {
        on_task_complete_threadsafe(sched, entry.task_id, thread_ctx);
        count++;
    }
    
    return count;
}

/**
 * Process new tasks from orchestrator (thread-safe version)
 * 
 * Called by scheduler thread to initialize newly submitted tasks.
 */
static int32_t process_new_tasks_threadsafe(PTO2SchedulerState* sched,
                                             PTO2ThreadContext* thread_ctx,
                                             int32_t* last_processed) {
    PTO2SharedMemoryHeader* header = sched->sm_handle->header;
    int32_t current_task_index = PTO2_LOAD_ACQUIRE(&header->current_task_index);
    
    int32_t count = 0;
    
    while (*last_processed < current_task_index) {
        int32_t task_id = *last_processed;
        PTO2TaskDescriptor* task = pto2_sm_get_task(sched->sm_handle, task_id);
        
        int32_t slot = pto2_task_slot(sched, task_id);
        
        // Check current state - skip if already processed
        // This can happen if on_task_complete_threadsafe processed this task before us
        int32_t current_state = __atomic_load_n(&sched->task_state[slot], __ATOMIC_ACQUIRE);
        if (current_state != PTO2_TASK_PENDING) {
            // Task already processed (READY, RUNNING, COMPLETED, or CONSUMED)
            // Skip initialization - just move to next task
            (*last_processed)++;
            count++;
            continue;
        }
        
        // Task is in PENDING state - check if it's ready
        // NOTE: DO NOT reset fanin_refcount or fanout_refcount here!
        // In multi-threaded mode:
        //   - scope_end may have already updated fanout_refcount
        //   - producer completion may have already updated fanin_refcount
        // Both refcounts are reset when task transitions to CONSUMED state.
        
        // Check if task is immediately ready (no dependencies OR all producers already completed)
        int32_t fanin_count = __atomic_load_n(&task->fanin_count, __ATOMIC_ACQUIRE);
        int32_t fanin_refcount = __atomic_load_n(&sched->fanin_refcount[slot], __ATOMIC_ACQUIRE);
        if (fanin_count == 0 || fanin_refcount >= fanin_count) {
            // Use CAS to transition to READY (another thread might have already done this)
            int32_t expected = PTO2_TASK_PENDING;
            if (__atomic_compare_exchange_n(&sched->task_state[slot], &expected, PTO2_TASK_READY,
                                             false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
                pto2_scheduler_enqueue_ready_threadsafe(sched, task_id,
                                                         task->worker_type, thread_ctx);
            }
        }
        
        (*last_processed)++;
        count++;
    }
    
    return count;
}

// Progress report interval in seconds
#define PTO2_PROGRESS_REPORT_INTERVAL_SEC  2

void* pto2_scheduler_thread_func(void* arg) {
    PTO2SchedulerContext* ctx = (PTO2SchedulerContext*)arg;
    PTO2SchedulerState* sched = ctx->scheduler;
    PTO2ThreadContext* thread_ctx = ctx->thread_ctx;
    
    // Wait for all workers to be ready first
    pthread_mutex_lock(&thread_ctx->startup_mutex);
    while (thread_ctx->workers_ready < thread_ctx->num_workers) {
        pthread_cond_wait(&thread_ctx->startup_cond, &thread_ctx->startup_mutex);
    }
    // Signal that scheduler is ready
    thread_ctx->scheduler_ready = true;
    pthread_cond_broadcast(&thread_ctx->startup_cond);
    pthread_mutex_unlock(&thread_ctx->startup_mutex);
    
    int32_t last_processed_task = 0;
    
    // Progress tracking
    struct timespec last_progress_time;
    clock_gettime(CLOCK_REALTIME, &last_progress_time);
    int64_t last_reported_completed = 0;
    
    while (!thread_ctx->shutdown) {
        bool did_work = false;
        
        // === STEP 1: Process new tasks from orchestrator ===
        int32_t new_tasks = process_new_tasks_threadsafe(sched, thread_ctx, 
                                                          &last_processed_task);
        if (new_tasks > 0) {
            did_work = true;
        }
        
        // === STEP 2: Process completions from workers ===
        int32_t completions = pto2_scheduler_process_completions(ctx);
        if (completions > 0) {
            did_work = true;
        }
        
        // === STEP 3: Advance ring pointers and sync to shared memory ===
        pto2_scheduler_advance_ring_pointers(sched);
        pto2_scheduler_sync_to_sm(sched);  // Critical: sync to shared memory for orchestrator flow control
        
        // === STEP 4: Periodic progress report ===
        struct timespec now;
        clock_gettime(CLOCK_REALTIME, &now);
        double elapsed = (now.tv_sec - last_progress_time.tv_sec) + 
                         (now.tv_nsec - last_progress_time.tv_nsec) / 1e9;
        if (elapsed >= PTO2_PROGRESS_REPORT_INTERVAL_SEC) {
            int64_t completed = sched->tasks_completed;
            int64_t consumed = sched->tasks_consumed;
            int32_t current_task_index = PTO2_LOAD_ACQUIRE(&sched->sm_handle->header->current_task_index);
            
            // Always print if there's progress, or if stuck (completed != consumed)
            if (completed > last_reported_completed || completed != consumed) {
                fprintf(stderr, "[Progress] completed=%lld, consumed=%lld, submitted=%d, last_alive=%d",
                        (long long)completed, (long long)consumed, current_task_index, sched->last_task_alive);
                
                // If stuck, show more details
                if (completed == last_reported_completed && completed != consumed) {
                    fprintf(stderr, " (STUCK: %lld tasks not consumed)", (long long)(completed - consumed));
                }
                fprintf(stderr, "\n");
                
                last_reported_completed = completed;
            }
            last_progress_time = now;
        }
        
        // === STEP 5: Check if all done ===
        if (pto2_scheduler_is_done(sched)) {
            pthread_mutex_lock(&thread_ctx->done_mutex);
            thread_ctx->all_done = true;
            pthread_cond_broadcast(&thread_ctx->all_done_cond);
            pthread_mutex_unlock(&thread_ctx->done_mutex);
            break;
        }
        
        // If no work done, brief pause to avoid busy-waiting
        if (!did_work) {
            // Wait for completion signals with timeout
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_nsec += 1000000;  // 1ms timeout
            if (ts.tv_nsec >= 1000000000) {
                ts.tv_sec++;
                ts.tv_nsec -= 1000000000;
            }
            
            pthread_mutex_lock(&thread_ctx->done_mutex);
            pthread_cond_timedwait(&thread_ctx->completion_cond, 
                                   &thread_ctx->done_mutex, &ts);
            pthread_mutex_unlock(&thread_ctx->done_mutex);
        }
    }
    
    return NULL;
}
