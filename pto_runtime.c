/**
 * PTO Runtime System - Implementation
 * 
 * Task scheduling runtime for PTO programs.
 * Manages dependencies between InCore function calls in Orchestration functions.
 */

#include "pto_runtime.h"
#include <time.h>  // For nanosleep, clock_gettime

// Debug output control - set to 0 to disable debug prints
#ifndef PTO_DEBUG
#define PTO_DEBUG 0
#endif

#if PTO_DEBUG
#define DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
#define DEBUG_PRINT(...) ((void)0)
#endif

// =============================================================================
// Runtime Initialization
// =============================================================================

void pto_runtime_init(PTORuntime* rt) {
    if (!rt) return;
    
    // Initialize task table
    memset(rt->pend_task, 0, sizeof(rt->pend_task));
    rt->next_task_id = 0;
    rt->active_task_count = 0;
    
    // Initialize tensor map
    memset(rt->tensor_map, 0, sizeof(rt->tensor_map));
    
    // Initialize ready queue
    memset(rt->ready_queue, 0, sizeof(rt->ready_queue));
    rt->ready_head = 0;
    rt->ready_tail = 0;
    rt->ready_count = 0;
    
    // Initialize statistics
    rt->total_tasks_scheduled = 0;
    rt->total_tasks_completed = 0;
    
    // Initialize thread synchronization primitives
    pthread_mutex_init(&rt->queue_mutex, NULL);
    pthread_mutex_init(&rt->task_mutex, NULL);
    pthread_cond_init(&rt->queue_not_empty, NULL);
    pthread_cond_init(&rt->all_done, NULL);
    
    // Initialize worker state
    rt->num_workers = 0;
    rt->shutdown_requested = false;
    rt->execution_started = false;
    memset(rt->workers, 0, sizeof(rt->workers));
    memset(rt->func_registry, 0, sizeof(rt->func_registry));
    
    DEBUG_PRINT("[PTO Runtime] Initialized (max_tasks=%d, tensormap_size=%d)\n",
           PTO_MAX_TASKS, PTO_TENSORMAP_SIZE);
}

void pto_runtime_shutdown(PTORuntime* rt) {
    if (!rt) return;
    
    // Free tensor map entries
    pto_tensormap_clear(rt);
    
    // Destroy thread synchronization primitives
    pthread_mutex_destroy(&rt->queue_mutex);
    pthread_mutex_destroy(&rt->task_mutex);
    pthread_cond_destroy(&rt->queue_not_empty);
    pthread_cond_destroy(&rt->all_done);
    
    DEBUG_PRINT("[PTO Runtime] Shutdown (scheduled=%lld, completed=%lld)\n",
           (long long)rt->total_tasks_scheduled,
           (long long)rt->total_tasks_completed);
}

// =============================================================================
// TensorMap Implementation
// =============================================================================

uint32_t pto_tensormap_hash(TensorRegion* region) {
    // Simple hash combining pointer and offsets
    uint64_t ptr_val = (uint64_t)region->raw_tensor;
    uint32_t hash = (uint32_t)(ptr_val ^ (ptr_val >> 32));
    hash ^= (uint32_t)region->row_offset * 31;
    hash ^= (uint32_t)region->col_offset * 17;
    hash ^= (uint32_t)region->rows * 13;
    hash ^= (uint32_t)region->cols * 7;
    return hash % PTO_TENSORMAP_SIZE;
}

bool pto_region_match(TensorRegion* a, TensorRegion* b) {
    return a->raw_tensor == b->raw_tensor &&
           a->row_offset == b->row_offset &&
           a->col_offset == b->col_offset &&
           a->rows == b->rows &&
           a->cols == b->cols;
}

int32_t pto_tensormap_lookup(PTORuntime* rt, TensorRegion* region) {
    uint32_t hash = pto_tensormap_hash(region);
    TensorMapEntry* entry = rt->tensor_map[hash];
    
    while (entry) {
        if (pto_region_match(&entry->region, region)) {
            return entry->producer_id;
        }
        entry = entry->next;
    }
    
    return -1; // Not found
}

void pto_tensormap_insert(PTORuntime* rt, TensorRegion* region, int32_t task_id) {
    uint32_t hash = pto_tensormap_hash(region);
    
    // Check if entry already exists
    TensorMapEntry* entry = rt->tensor_map[hash];
    while (entry) {
        if (pto_region_match(&entry->region, region)) {
            // Update existing entry
            entry->producer_id = task_id;
            return;
        }
        entry = entry->next;
    }
    
    // Create new entry
    TensorMapEntry* new_entry = (TensorMapEntry*)malloc(sizeof(TensorMapEntry));
    if (!new_entry) {
        fprintf(stderr, "[PTO Runtime] ERROR: Failed to allocate TensorMapEntry\n");
        return;
    }
    
    new_entry->region = *region;
    new_entry->producer_id = task_id;
    new_entry->next = rt->tensor_map[hash];
    rt->tensor_map[hash] = new_entry;
}

void pto_tensormap_clear(PTORuntime* rt) {
    for (int i = 0; i < PTO_TENSORMAP_SIZE; i++) {
        TensorMapEntry* entry = rt->tensor_map[i];
        while (entry) {
            TensorMapEntry* next = entry->next;
            free(entry);
            entry = next;
        }
        rt->tensor_map[i] = NULL;
    }
}

// =============================================================================
// Ready Queue Implementation
// =============================================================================

static void ready_queue_push(PTORuntime* rt, int32_t task_id) {
    if (rt->ready_count >= PTO_MAX_READY_QUEUE) {
        fprintf(stderr, "[PTO Runtime] ERROR: Ready queue overflow\n");
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

// =============================================================================
// Thread-safe Ready Queue Operations
// =============================================================================

static void ready_queue_push_threadsafe(PTORuntime* rt, int32_t task_id) {
    DEBUG_PRINT("[Queue] push_threadsafe: trying to lock for task %d\n", task_id);
    fflush(stdout);
    
    pthread_mutex_lock(&rt->queue_mutex);
    
    DEBUG_PRINT("[Queue] push_threadsafe: got lock for task %d, ready_count=%d\n", task_id, rt->ready_count);
    fflush(stdout);
    
    if (rt->ready_count >= PTO_MAX_READY_QUEUE) {
        fprintf(stderr, "[PTO Runtime] ERROR: Ready queue overflow\n");
        pthread_mutex_unlock(&rt->queue_mutex);
        return;
    }
    
    rt->ready_queue[rt->ready_tail] = task_id;
    rt->ready_tail = (rt->ready_tail + 1) % PTO_MAX_READY_QUEUE;
    rt->ready_count++;
    
    // Broadcast to wake up all waiting workers (more responsive than signal)
    pthread_cond_broadcast(&rt->queue_not_empty);
    
    pthread_mutex_unlock(&rt->queue_mutex);
    
    DEBUG_PRINT("[Queue] push_threadsafe: released lock, task %d queued\n", task_id);
    fflush(stdout);
}

int32_t pto_get_ready_task_blocking(PTORuntime* rt) {
    DEBUG_PRINT("[Queue] get_blocking: trying to lock\n");
    fflush(stdout);
    
    pthread_mutex_lock(&rt->queue_mutex);
    
    DEBUG_PRINT("[Queue] get_blocking: got lock, ready_count=%d, started=%d\n", 
           rt->ready_count, rt->execution_started);
    fflush(stdout);
    
    // Wait until: (execution started AND task available) OR shutdown OR all done
    while ((rt->ready_count == 0 || !rt->execution_started) && !rt->shutdown_requested) {
        // Check if all tasks are completed
        if (rt->execution_started && rt->total_tasks_completed >= rt->total_tasks_scheduled) {
            pthread_mutex_unlock(&rt->queue_mutex);
            return -1;  // All done
        }
        
        // Reduce debug spam - only print occasionally
        static __thread int wait_count = 0;
        if (++wait_count % 1000 == 1) {
            DEBUG_PRINT("[Queue] get_blocking: waiting (ready=%d, started=%d, shutdown=%d, count=%d)\n", 
                   rt->ready_count, rt->execution_started, rt->shutdown_requested, wait_count);
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
// Task Management
// =============================================================================

int32_t pto_task_alloc(PTORuntime* rt, const char* func_name, void* func_ptr,
                       int32_t buffer_bytes, int32_t reuse_bytes) {
    if (rt->next_task_id >= PTO_MAX_TASKS) {
        fprintf(stderr, "[PTO Runtime] ERROR: Task table full\n");
        return -1;
    }
    
    int32_t task_id = rt->next_task_id++;
    PendingTask* task = &rt->pend_task[task_id];
    
    // Initialize task
    task->task_id = task_id;
    task->func_name = func_name;
    task->func_ptr = func_ptr;
    task->num_args = 0;
    task->buffer_size_bytes = buffer_bytes;
    task->buffer_size_with_reuse = reuse_bytes;
    task->fanin = 0;
    task->fanout_count = 0;
    task->is_active = true;
    task->is_complete = false;
    
    // Clear fanout list
    memset(task->fanout, 0, sizeof(task->fanout));
    
    rt->active_task_count++;
    rt->total_tasks_scheduled++;
    
    DEBUG_PRINT("[PTO Runtime] Allocated task %d: %s (buf=%d B, reuse=%d B)\n", 
           task_id, func_name, buffer_bytes, reuse_bytes);
    
    return task_id;
}

void pto_task_add_input(PTORuntime* rt, int32_t task_id,
                        void* tensor, int64_t row_off, int64_t col_off,
                        int64_t rows, int64_t cols) {
    if (task_id < 0 || task_id >= rt->next_task_id) {
        fprintf(stderr, "[PTO Runtime] ERROR: Invalid task_id %d\n", task_id);
        return;
    }
    
    PendingTask* task = &rt->pend_task[task_id];
    
    if (task->num_args >= PTO_MAX_ARGS) {
        fprintf(stderr, "[PTO Runtime] ERROR: Too many arguments for task %d\n", task_id);
        return;
    }
    
    // Create tensor region
    TensorRegion region = {
        .raw_tensor = tensor,
        .row_offset = row_off,
        .col_offset = col_off,
        .rows = rows,
        .cols = cols
    };
    
    // Add argument
    TaskArg* arg = &task->args[task->num_args++];
    arg->region = region;
    arg->is_output = false;
    
    // Look up producer in TensorMap
    int32_t producer_id = pto_tensormap_lookup(rt, &region);
    
    if (producer_id >= 0 && producer_id != task_id) {
        // Found producer - add dependency
        PendingTask* producer = &rt->pend_task[producer_id];
        
        // Add current task to producer's fanout
        if (producer->fanout_count < PTO_MAX_FANOUT) {
            producer->fanout[producer->fanout_count++] = task_id;
        } else {
            fprintf(stderr, "[PTO Runtime] WARNING: Fanout overflow for task %d\n", producer_id);
        }
        
        // Increment fanin (dependency count)
        task->fanin++;
        
        DEBUG_PRINT("[PTO Runtime] Task %d depends on task %d (tensor=%p, offset=[%lld,%lld])\n",
               task_id, producer_id, tensor, (long long)row_off, (long long)col_off);
    } else {
        DEBUG_PRINT("[PTO Runtime] Task %d input (tensor=%p, offset=[%lld,%lld]) - no producer\n",
               task_id, tensor, (long long)row_off, (long long)col_off);
    }
}

void pto_task_add_output(PTORuntime* rt, int32_t task_id,
                         void* tensor, int64_t row_off, int64_t col_off,
                         int64_t rows, int64_t cols) {
    if (task_id < 0 || task_id >= rt->next_task_id) {
        fprintf(stderr, "[PTO Runtime] ERROR: Invalid task_id %d\n", task_id);
        return;
    }
    
    PendingTask* task = &rt->pend_task[task_id];
    
    if (task->num_args >= PTO_MAX_ARGS) {
        fprintf(stderr, "[PTO Runtime] ERROR: Too many arguments for task %d\n", task_id);
        return;
    }
    
    // Create tensor region
    TensorRegion region = {
        .raw_tensor = tensor,
        .row_offset = row_off,
        .col_offset = col_off,
        .rows = rows,
        .cols = cols
    };
    
    // Add argument
    TaskArg* arg = &task->args[task->num_args++];
    arg->region = region;
    arg->is_output = true;
    
    // Register in TensorMap (this task produces this region)
    pto_tensormap_insert(rt, &region, task_id);
    
    DEBUG_PRINT("[PTO Runtime] Task %d output (tensor=%p, offset=[%lld,%lld], shape=[%lld,%lld])\n",
           task_id, tensor, (long long)row_off, (long long)col_off,
           (long long)rows, (long long)cols);
}

void pto_task_submit(PTORuntime* rt, int32_t task_id) {
    if (task_id < 0 || task_id >= rt->next_task_id) {
        fprintf(stderr, "[PTO Runtime] ERROR: Invalid task_id %d\n", task_id);
        return;
    }
    
    PendingTask* task = &rt->pend_task[task_id];
    
    DEBUG_PRINT("[PTO Runtime] Submitted task %d: %s (fanin=%d, fanout=%d)\n",
           task_id, task->func_name, task->fanin, task->fanout_count);
    
    // If no dependencies, add directly to ready queue (thread-safe)
    // This allows workers to start executing immediately
    if (task->fanin == 0) {
        ready_queue_push_threadsafe(rt, task_id);
        DEBUG_PRINT("[PTO Runtime] Task %d is ready (no dependencies)\n", task_id);
    }
    // Tasks with fanin > 0 stay in pend_task until dependencies complete
}

void pto_task_complete(PTORuntime* rt, int32_t task_id) {
    if (task_id < 0 || task_id >= rt->next_task_id) {
        fprintf(stderr, "[PTO Runtime] ERROR: Invalid task_id %d\n", task_id);
        return;
    }
    
    PendingTask* task = &rt->pend_task[task_id];
    task->is_complete = true;
    rt->active_task_count--;
    rt->total_tasks_completed++;
    
    DEBUG_PRINT("[PTO Runtime] Completed task %d: %s\n", task_id, task->func_name);
    
    // Update dependent tasks
    for (int i = 0; i < task->fanout_count; i++) {
        int32_t dep_id = task->fanout[i];
        PendingTask* dep = &rt->pend_task[dep_id];
        
        dep->fanin--;
        DEBUG_PRINT("[PTO Runtime] Task %d fanin decremented to %d\n", dep_id, dep->fanin);
        
        // If all dependencies satisfied, add to ready queue
        if (dep->fanin == 0 && !dep->is_complete) {
            ready_queue_push(rt, dep_id);
            DEBUG_PRINT("[PTO Runtime] Task %d is now ready\n", dep_id);
        }
    }
}

void pto_task_complete_threadsafe(PTORuntime* rt, int32_t task_id) {
    if (task_id < 0 || task_id >= rt->next_task_id) {
        fprintf(stderr, "[PTO Runtime] ERROR: Invalid task_id %d\n", task_id);
        return;
    }
    
    pthread_mutex_lock(&rt->task_mutex);
    
    PendingTask* task = &rt->pend_task[task_id];
    task->is_complete = true;
    rt->active_task_count--;
    rt->total_tasks_completed++;
    
    DEBUG_PRINT("[PTO Runtime] Completed task %d: %s (completed=%lld/%lld)\n", 
           task_id, task->func_name, 
           (long long)rt->total_tasks_completed, 
           (long long)rt->total_tasks_scheduled);
    
    // Collect tasks that become ready
    int32_t newly_ready[PTO_MAX_FANOUT];
    int32_t newly_ready_count = 0;
    
    for (int i = 0; i < task->fanout_count; i++) {
        int32_t dep_id = task->fanout[i];
        PendingTask* dep = &rt->pend_task[dep_id];
        
        dep->fanin--;
        DEBUG_PRINT("[PTO Runtime] Task %d fanin decremented to %d\n", dep_id, dep->fanin);
        
        if (dep->fanin == 0 && !dep->is_complete) {
            newly_ready[newly_ready_count++] = dep_id;
        }
    }
    
    // Check if all tasks completed
    bool all_done = (rt->total_tasks_completed >= rt->total_tasks_scheduled);
    
    pthread_mutex_unlock(&rt->task_mutex);
    
    // Add newly ready tasks to queue (outside task_mutex to avoid deadlock)
    for (int i = 0; i < newly_ready_count; i++) {
        ready_queue_push_threadsafe(rt, newly_ready[i]);
        DEBUG_PRINT("[PTO Runtime] Task %d is now ready\n", newly_ready[i]);
    }
    
    // Signal if all tasks are done
    if (all_done) {
        pthread_mutex_lock(&rt->queue_mutex);
        pthread_cond_broadcast(&rt->all_done);
        pthread_cond_broadcast(&rt->queue_not_empty);  // Wake up waiting workers
        pthread_mutex_unlock(&rt->queue_mutex);
    }
}

int32_t pto_get_ready_task(PTORuntime* rt) {
    return ready_queue_pop(rt);
}

// =============================================================================
// Execution
// =============================================================================

// Generic function pointer type for InCore functions
typedef void (*InCoreFuncPtr)(void);

void pto_execute_all(PTORuntime* rt) {
    printf("\n[PTO Runtime] ======== Executing all tasks ========\n");
    
    while (rt->ready_count > 0 || rt->active_task_count > (int32_t)rt->total_tasks_completed) {
        int32_t task_id = pto_get_ready_task(rt);
        
        if (task_id < 0) {
            // No ready tasks - check for deadlock
            if (rt->active_task_count > (int32_t)rt->total_tasks_completed) {
                fprintf(stderr, "[PTO Runtime] WARNING: No ready tasks but %d tasks pending - possible deadlock\n",
                        rt->active_task_count - (int32_t)rt->total_tasks_completed);
                break;
            }
            continue;
        }
        
        PendingTask* task = &rt->pend_task[task_id];
        
        DEBUG_PRINT("[PTO Runtime] Executing task %d: %s\n", task_id, task->func_name);
        
        // Execute the task (in a real implementation, this would call the actual function)
        if (task->func_ptr) {
            // For now, just call the function pointer
            // In a real implementation, we would pass the arguments
            InCoreFuncPtr func __attribute__((unused)) = (InCoreFuncPtr)task->func_ptr;
            // Note: actual argument passing would require more sophisticated handling
            // func();  // Uncomment when function signatures are properly handled
            DEBUG_PRINT("[PTO Runtime] (Simulated execution of %s)\n", task->func_name);
        }
        
        // Mark complete and update dependencies
        pto_task_complete(rt, task_id);
    }
    
    DEBUG_PRINT("[PTO Runtime] ======== Execution complete ========\n\n");
}

void pto_runtime_stats(PTORuntime* rt) {
    printf("\n[PTO Runtime Statistics]\n");
    printf("  Total tasks scheduled: %lld\n", (long long)rt->total_tasks_scheduled);
    printf("  Total tasks completed: %lld\n", (long long)rt->total_tasks_completed);
    printf("  Active tasks:          %d\n", rt->active_task_count);
    printf("  Ready queue size:      %d\n", rt->ready_count);
    printf("\n");
}

// =============================================================================
// Dump Function - Export Runtime State to Text File
// =============================================================================

int pto_runtime_dump(PTORuntime* rt, const char* filename) {
    if (!rt || !filename) return -1;
    
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "[PTO Runtime] ERROR: Cannot open file %s for writing\n", filename);
        return -1;
    }
    
    // Header
    fprintf(fp, "================================================================================\n");
    fprintf(fp, "PTO RUNTIME DUMP\n");
    fprintf(fp, "================================================================================\n\n");
    
    // Summary statistics
    fprintf(fp, "SUMMARY\n");
    fprintf(fp, "--------------------------------------------------------------------------------\n");
    fprintf(fp, "  Total tasks scheduled:  %lld\n", (long long)rt->total_tasks_scheduled);
    fprintf(fp, "  Total tasks completed:  %lld\n", (long long)rt->total_tasks_completed);
    fprintf(fp, "  Active task count:      %d\n", rt->active_task_count);
    fprintf(fp, "  Next task ID:           %d\n", rt->next_task_id);
    fprintf(fp, "  Ready queue size:       %d\n", rt->ready_count);
    fprintf(fp, "\n");
    
    // Task Table
    fprintf(fp, "================================================================================\n");
    fprintf(fp, "TASK TABLE (pend_task)\n");
    fprintf(fp, "================================================================================\n\n");
    
    for (int32_t i = 0; i < rt->next_task_id; i++) {
        PendingTask* task = &rt->pend_task[i];
        
        fprintf(fp, "--------------------------------------------------------------------------------\n");
        fprintf(fp, "TASK %d\n", task->task_id);
        fprintf(fp, "--------------------------------------------------------------------------------\n");
        fprintf(fp, "  Function:     %s\n", task->func_name ? task->func_name : "(null)");
        fprintf(fp, "  Func Ptr:     %p\n", task->func_ptr);
        fprintf(fp, "  Is Active:    %s\n", task->is_active ? "true" : "false");
        fprintf(fp, "  Is Complete:  %s\n", task->is_complete ? "true" : "false");
        fprintf(fp, "\n");
        
        // Buffer size estimation
        fprintf(fp, "  BUFFER SIZE (InCore Tile Buffers)\n");
        fprintf(fp, "  ----------------------------------\n");
        fprintf(fp, "    without_reuse = %d bytes (%.2f KB)\n", 
                task->buffer_size_bytes, task->buffer_size_bytes / 1024.0);
        fprintf(fp, "    with_reuse    = %d bytes (%.2f KB)\n", 
                task->buffer_size_with_reuse, task->buffer_size_with_reuse / 1024.0);
        if (task->buffer_size_bytes > 0) {
            int savings = task->buffer_size_bytes - task->buffer_size_with_reuse;
            float pct = 100.0 * savings / task->buffer_size_bytes;
            fprintf(fp, "    savings       = %d bytes (%.1f%%)\n", savings, pct);
        }
        fprintf(fp, "\n");
        
        // Fanin counter
        fprintf(fp, "  FANIN COUNTER\n");
        fprintf(fp, "  -------------\n");
        fprintf(fp, "    fanin = %d\n", task->fanin);
        fprintf(fp, "\n");
        
        // Fanout list
        fprintf(fp, "  FANOUT LIST (consumers that depend on this task)\n");
        fprintf(fp, "  ------------------------------------------------\n");
        fprintf(fp, "    fanout_count = %d\n", task->fanout_count);
        if (task->fanout_count > 0) {
            fprintf(fp, "    fanout[] = [");
            for (int j = 0; j < task->fanout_count; j++) {
                fprintf(fp, "%d", task->fanout[j]);
                if (j < task->fanout_count - 1) fprintf(fp, ", ");
            }
            fprintf(fp, "]\n");
            
            // Detailed fanout info
            fprintf(fp, "    Consumers:\n");
            for (int j = 0; j < task->fanout_count; j++) {
                int32_t consumer_id = task->fanout[j];
                PendingTask* consumer = &rt->pend_task[consumer_id];
                fprintf(fp, "      -> Task %d (%s)\n", consumer_id, 
                        consumer->func_name ? consumer->func_name : "(null)");
            }
        } else {
            fprintf(fp, "    fanout[] = [] (no consumers)\n");
        }
        fprintf(fp, "\n");
        
        // Arguments
        fprintf(fp, "  ARGUMENTS (num_args = %d)\n", task->num_args);
        fprintf(fp, "  -------------------------\n");
        for (int j = 0; j < task->num_args; j++) {
            TaskArg* arg = &task->args[j];
            fprintf(fp, "    [%d] %s:\n", j, arg->is_output ? "OUTPUT" : "INPUT");
            fprintf(fp, "        tensor:     %p\n", arg->region.raw_tensor);
            fprintf(fp, "        row_offset: %lld\n", (long long)arg->region.row_offset);
            fprintf(fp, "        col_offset: %lld\n", (long long)arg->region.col_offset);
            fprintf(fp, "        rows:       %lld\n", (long long)arg->region.rows);
            fprintf(fp, "        cols:       %lld\n", (long long)arg->region.cols);
        }
        fprintf(fp, "\n");
    }
    
    // Ready Queue
    fprintf(fp, "================================================================================\n");
    fprintf(fp, "READY QUEUE\n");
    fprintf(fp, "================================================================================\n\n");
    fprintf(fp, "  Head:  %d\n", rt->ready_head);
    fprintf(fp, "  Tail:  %d\n", rt->ready_tail);
    fprintf(fp, "  Count: %d\n", rt->ready_count);
    if (rt->ready_count > 0) {
        fprintf(fp, "  Queue: [");
        int idx = rt->ready_head;
        for (int i = 0; i < rt->ready_count; i++) {
            fprintf(fp, "%d", rt->ready_queue[idx]);
            if (i < rt->ready_count - 1) fprintf(fp, ", ");
            idx = (idx + 1) % PTO_MAX_READY_QUEUE;
        }
        fprintf(fp, "]\n");
    } else {
        fprintf(fp, "  Queue: [] (empty)\n");
    }
    fprintf(fp, "\n");
    
    // TensorMap (active entries)
    fprintf(fp, "================================================================================\n");
    fprintf(fp, "TENSOR MAP (non-empty buckets)\n");
    fprintf(fp, "================================================================================\n\n");
    int tensor_count = 0;
    for (int i = 0; i < PTO_TENSORMAP_SIZE; i++) {
        TensorMapEntry* entry = rt->tensor_map[i];
        while (entry) {
            fprintf(fp, "  [bucket %d] tensor=%p, offset=[%lld,%lld], shape=[%lld,%lld] -> producer: Task %d\n",
                    i,
                    entry->region.raw_tensor,
                    (long long)entry->region.row_offset,
                    (long long)entry->region.col_offset,
                    (long long)entry->region.rows,
                    (long long)entry->region.cols,
                    entry->producer_id);
            tensor_count++;
            entry = entry->next;
        }
    }
    if (tensor_count == 0) {
        fprintf(fp, "  (empty)\n");
    }
    fprintf(fp, "\n  Total tensor entries: %d\n", tensor_count);
    fprintf(fp, "\n");
    
    // Dependency Graph (ASCII representation)
    fprintf(fp, "================================================================================\n");
    fprintf(fp, "DEPENDENCY GRAPH (Producer -> Consumer)\n");
    fprintf(fp, "================================================================================\n\n");
    for (int32_t i = 0; i < rt->next_task_id; i++) {
        PendingTask* task = &rt->pend_task[i];
        if (!task->is_active) continue;
        
        // Status indicator
        const char* status = task->is_complete ? "[DONE]" : 
                            (task->fanin == 0 ? "[READY]" : "[WAIT]");
        
        fprintf(fp, "  Task %d (%s) %s\n", i, 
                task->func_name ? task->func_name : "?", status);
        
        if (task->fanout_count > 0) {
            for (int j = 0; j < task->fanout_count; j++) {
                int32_t consumer_id = task->fanout[j];
                PendingTask* consumer = &rt->pend_task[consumer_id];
                fprintf(fp, "    └──> Task %d (%s)\n", consumer_id,
                        consumer->func_name ? consumer->func_name : "?");
            }
        }
    }
    fprintf(fp, "\n");
    
    fprintf(fp, "================================================================================\n");
    fprintf(fp, "END OF DUMP\n");
    fprintf(fp, "================================================================================\n");
    
    fclose(fp);
    DEBUG_PRINT("[PTO Runtime] Dumped runtime state to %s\n", filename);
    return 0;
}

int pto_runtime_dump_stdout(PTORuntime* rt) {
    if (!rt) return -1;
    
    // Reuse dump logic but write to stdout
    printf("================================================================================\n");
    printf("PTO RUNTIME DUMP\n");
    printf("================================================================================\n\n");
    
    printf("SUMMARY\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("  Total tasks scheduled:  %lld\n", (long long)rt->total_tasks_scheduled);
    printf("  Total tasks completed:  %lld\n", (long long)rt->total_tasks_completed);
    printf("  Active task count:      %d\n", rt->active_task_count);
    printf("  Next task ID:           %d\n", rt->next_task_id);
    printf("  Ready queue size:       %d\n", rt->ready_count);
    printf("\n");
    
    printf("TASK TABLE\n");
    printf("--------------------------------------------------------------------------------\n");
    for (int32_t i = 0; i < rt->next_task_id; i++) {
        PendingTask* task = &rt->pend_task[i];
        const char* status = task->is_complete ? "DONE" : 
                            (task->fanin == 0 ? "READY" : "WAIT");
        
        printf("  Task %d: %-20s [%s] fanin=%d buf=%.1fKB fanout=[",
               i, task->func_name ? task->func_name : "?", status, task->fanin,
               task->buffer_size_with_reuse / 1024.0);
        for (int j = 0; j < task->fanout_count; j++) {
            printf("%d", task->fanout[j]);
            if (j < task->fanout_count - 1) printf(",");
        }
        printf("]\n");
    }
    printf("\n");
    
    return 0;
}

// =============================================================================
// Helper Macros for Code Generation
// =============================================================================

/**
 * Convenience macro to schedule an InCore function call
 * 
 * Usage:
 *   PTO_SCHEDULE_INCORE(rt, rowmax, input, 0, 0, 8, 8, output, 0, 0, 8, 1);
 */
#define PTO_SCHEDULE_INCORE_1IN_1OUT(rt, func, in_ptr, in_row, in_col, in_rows, in_cols, \
                                      out_ptr, out_row, out_col, out_rows, out_cols) \
    do { \
        int32_t _tid = pto_task_alloc(rt, #func, (void*)func); \
        if (_tid >= 0) { \
            pto_task_add_input(rt, _tid, in_ptr, in_row, in_col, in_rows, in_cols); \
            pto_task_add_output(rt, _tid, out_ptr, out_row, out_col, out_rows, out_cols); \
            pto_task_submit(rt, _tid); \
        } \
    } while(0)

#define PTO_SCHEDULE_INCORE_2IN_1OUT(rt, func, in1_ptr, in1_row, in1_col, in1_rows, in1_cols, \
                                      in2_ptr, in2_row, in2_col, in2_rows, in2_cols, \
                                      out_ptr, out_row, out_col, out_rows, out_cols) \
    do { \
        int32_t _tid = pto_task_alloc(rt, #func, (void*)func); \
        if (_tid >= 0) { \
            pto_task_add_input(rt, _tid, in1_ptr, in1_row, in1_col, in1_rows, in1_cols); \
            pto_task_add_input(rt, _tid, in2_ptr, in2_row, in2_col, in2_rows, in2_cols); \
            pto_task_add_output(rt, _tid, out_ptr, out_row, out_col, out_rows, out_cols); \
            pto_task_submit(rt, _tid); \
        } \
    } while(0)

// =============================================================================
// Multi-threaded Execution - Worker Thread and Runtime Entry
// =============================================================================

/**
 * Worker thread context
 */
typedef struct {
    PTORuntime* rt;
    int worker_id;
} WorkerContext;

/**
 * Execute a single task by calling its InCore function
 */
static void execute_task(PTORuntime* rt, int32_t task_id) {
    PendingTask* task = &rt->pend_task[task_id];
    
    DEBUG_PRINT("[Worker] Executing task %d: %s\n", task_id, task->func_name);
    
    if (task->func_ptr) {
        // Build argument array from task arguments
        // Each TaskArg contains a TensorRegion with raw_tensor pointer and offsets
        void* args[PTO_MAX_ARGS * 2];  // ptr, offset pairs
        int arg_idx = 0;
        
        for (int i = 0; i < task->num_args; i++) {
            TaskArg* arg = &task->args[i];
            // Calculate actual pointer with offset
            // Assuming float* tensors with row-major layout
            float* base_ptr = (float*)arg->region.raw_tensor;
            int64_t offset = arg->region.row_offset * arg->region.cols + arg->region.col_offset;
            args[arg_idx++] = (void*)(base_ptr + offset);
        }
        
        // Call the InCore function
        // The function signature depends on the specific InCore function
        // For simplicity, we use a generic approach here
        PTOInCoreFunc func = (PTOInCoreFunc)task->func_ptr;
        func(args, task->num_args);
    } else {
        // No function pointer - simulate execution
        DEBUG_PRINT("[Worker] Task %d: %s (simulated - no func_ptr)\n", 
               task_id, task->func_name);
    }
}

/**
 * Worker thread function
 * Continuously fetches and executes tasks until shutdown
 */
static void* worker_thread_func(void* arg) {
    WorkerContext* ctx = (WorkerContext*)arg;
    PTORuntime* rt = ctx->rt;
    int worker_id __attribute__((unused)) = ctx->worker_id;
    
    DEBUG_PRINT("[Worker %d] Started\n", worker_id);
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
        execute_task(rt, task_id);
        
        // Mark task as complete (updates dependencies, may wake other workers)
        pto_task_complete_threadsafe(rt, task_id);
    }
    
    DEBUG_PRINT("[Worker %d] Exiting\n", worker_id);
    free(ctx);
    return NULL;
}

/**
 * ARM64 Runtime Entry Point
 */
int runtime_entry_arm64(PTOOrchFunc orch_func, void* user_data, int num_workers) {
    if (!orch_func) {
        fprintf(stderr, "[PTO Runtime] ERROR: No orchestration function provided\n");
        return -1;
    }
    
    if (num_workers < 1) num_workers = 1;
    if (num_workers > PTO_MAX_WORKERS) num_workers = PTO_MAX_WORKERS;
    
    printf("[PTO Runtime] ========================================\n");
    printf("[PTO Runtime] ARM64 Multi-threaded Execution\n");
    printf("[PTO Runtime] Workers: %d\n", num_workers);
    printf("[PTO Runtime] ========================================\n");
    
    // Allocate runtime on heap (PTORuntime can be large)
    PTORuntime* rt = (PTORuntime*)malloc(sizeof(PTORuntime));
    if (!rt) {
        fprintf(stderr, "[PTO Runtime] ERROR: Failed to allocate runtime\n");
        return -1;
    }
    
    // Initialize runtime
    pto_runtime_init(rt);
    rt->num_workers = num_workers;
    rt->shutdown_requested = false;
    rt->execution_started = false;
    
    // Spawn worker threads
    printf("[PTO Runtime] Spawning %d worker threads...\n", num_workers);
    for (int i = 0; i < num_workers; i++) {
        WorkerContext* ctx = (WorkerContext*)malloc(sizeof(WorkerContext));
        if (!ctx) {
            fprintf(stderr, "[PTO Runtime] ERROR: Failed to allocate worker context\n");
            // Cleanup already spawned threads
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
            fprintf(stderr, "[PTO Runtime] ERROR: Failed to create worker thread %d\n", i);
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
        printf("[PTO Runtime] Created worker thread %d\n", i);
        fflush(stdout);
    }
    
    // Give workers a moment to start
    struct timespec start_delay = {0, 10000000};  // 10ms
    nanosleep(&start_delay, NULL);
    printf("[PTO Runtime] Workers started, now building task graph...\n");
    fflush(stdout);
    
    // Build task graph by calling orchestration function
    printf("[PTO Runtime] Building task graph...\n");
    fflush(stdout);
    orch_func(rt, user_data);
    
    // Mark that orchestration is complete - all tasks are now submitted
    pthread_mutex_lock(&rt->task_mutex);
    rt->execution_started = true;
    int64_t total_tasks = rt->total_tasks_scheduled;
    pthread_mutex_unlock(&rt->task_mutex);
    
    printf("[PTO Runtime] Task graph built: %lld tasks\n", (long long)total_tasks);
    printf("[PTO Runtime] Executing tasks...\n");
    
    // Wake up any workers that might be waiting
    pthread_mutex_lock(&rt->queue_mutex);
    pthread_cond_broadcast(&rt->queue_not_empty);
    pthread_mutex_unlock(&rt->queue_mutex);
    
    // Wait for all tasks to complete
    // We poll active_task_count periodically
    struct timespec poll_interval = {0, 1000000};  // 1ms (was 50ms - too slow!)
    while (1) {
        pthread_mutex_lock(&rt->task_mutex);
        bool all_done = (rt->total_tasks_completed >= rt->total_tasks_scheduled);
        int64_t completed = rt->total_tasks_completed;
        pthread_mutex_unlock(&rt->task_mutex);
        
        if (all_done) {
            printf("[PTO Runtime] All %lld tasks completed!\n", (long long)completed);
            break;
        }
        
        // Progress report every 100ms or so
        static int64_t last_reported = 0;
        if (completed > last_reported + 1000 || completed == total_tasks) {
            printf("[PTO Runtime] Progress: %lld / %lld tasks (%.1f%%)\n",
                   (long long)completed, (long long)total_tasks,
                   100.0 * completed / total_tasks);
            last_reported = completed;
        }
        
        nanosleep(&poll_interval, NULL);
    }
    
    // Signal workers to shutdown
    printf("[PTO Runtime] Shutting down workers...\n");
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
    printf("[PTO Runtime] ========================================\n");
    printf("[PTO Runtime] Execution Statistics\n");
    printf("[PTO Runtime]   Total tasks: %lld\n", (long long)rt->total_tasks_scheduled);
    printf("[PTO Runtime]   Completed:   %lld\n", (long long)rt->total_tasks_completed);
    printf("[PTO Runtime]   Workers:     %d\n", num_workers);
    printf("[PTO Runtime] ========================================\n");
    
    // Cleanup
    pto_runtime_shutdown(rt);
    free(rt);
    
    return 0;
}

/**
 * Register an InCore function (for lookup by name)
 */
void pto_register_incore_func(PTORuntime* rt, const char* func_name, PTOInCoreFunc func_ptr) {
    // For now, we store the function pointer directly in task->func_ptr when allocating
    // This function is a placeholder for a more sophisticated registry
    DEBUG_PRINT("[PTO Runtime] Registered InCore function: %s\n", func_name);
    (void)rt;
    (void)func_name;
    (void)func_ptr;
}

// =============================================================================
// Example: Generated Orchestration Function
// =============================================================================

#ifdef PTO_RUNTIME_EXAMPLE

// Forward declarations for InCore functions
void rowmax(float* input, float* output);
void rowexpandsub(float* input_x, float* input_row, float* output);
void elem_exp(float* input, float* output);
void rowsum(float* input, float* output);
void rowexpanddiv(float* input_x, float* input_row, float* output);

/**
 * Example: Generated Orchestration function for fused_softmax
 * 
 * This shows how an Orchestration function would be generated
 * to use the PTO runtime for scheduling InCore calls.
 */
void fused_softmax_orchestration(PTORuntime* rt,
                                  float* input, float* output,
                                  float* temp_rowmax, float* temp_shifted,
                                  float* temp_exp, float* temp_rowsum) {
    // Task 0: rowmax(input) -> temp_rowmax
    int32_t t0 = pto_task_alloc(rt, "rowmax", (void*)rowmax);
    pto_task_add_input(rt, t0, input, 0, 0, 8, 8);
    pto_task_add_output(rt, t0, temp_rowmax, 0, 0, 8, 1);
    pto_task_submit(rt, t0);
    
    // Task 1: rowexpandsub(input, temp_rowmax) -> temp_shifted
    int32_t t1 = pto_task_alloc(rt, "rowexpandsub", (void*)rowexpandsub);
    pto_task_add_input(rt, t1, input, 0, 0, 8, 8);
    pto_task_add_input(rt, t1, temp_rowmax, 0, 0, 8, 1);  // Creates dependency on t0
    pto_task_add_output(rt, t1, temp_shifted, 0, 0, 8, 8);
    pto_task_submit(rt, t1);
    
    // Task 2: elem_exp(temp_shifted) -> temp_exp
    int32_t t2 = pto_task_alloc(rt, "elem_exp", (void*)elem_exp);
    pto_task_add_input(rt, t2, temp_shifted, 0, 0, 8, 8);  // Creates dependency on t1
    pto_task_add_output(rt, t2, temp_exp, 0, 0, 8, 8);
    pto_task_submit(rt, t2);
    
    // Task 3: rowsum(temp_exp) -> temp_rowsum
    int32_t t3 = pto_task_alloc(rt, "rowsum", (void*)rowsum);
    pto_task_add_input(rt, t3, temp_exp, 0, 0, 8, 8);  // Creates dependency on t2
    pto_task_add_output(rt, t3, temp_rowsum, 0, 0, 8, 1);
    pto_task_submit(rt, t3);
    
    // Task 4: rowexpanddiv(temp_exp, temp_rowsum) -> output
    int32_t t4 = pto_task_alloc(rt, "rowexpanddiv", (void*)rowexpanddiv);
    pto_task_add_input(rt, t4, temp_exp, 0, 0, 8, 8);     // Creates dependency on t2
    pto_task_add_input(rt, t4, temp_rowsum, 0, 0, 8, 1);  // Creates dependency on t3
    pto_task_add_output(rt, t4, output, 0, 0, 8, 8);
    pto_task_submit(rt, t4);
    
    // Execute all scheduled tasks
    pto_execute_all(rt);
}

// Example main function
int main() {
    // Initialize runtime
    PTORuntime rt;
    pto_runtime_init(&rt);
    
    // Allocate buffers
    float input[64];
    float output[64];
    float temp_rowmax[8];
    float temp_shifted[64];
    float temp_exp[64];
    float temp_rowsum[8];
    
    // Initialize input (example)
    for (int i = 0; i < 64; i++) {
        input[i] = (float)i / 64.0f;
    }
    
    // Execute orchestration function
    fused_softmax_orchestration(&rt, input, output,
                                 temp_rowmax, temp_shifted,
                                 temp_exp, temp_rowsum);
    
    // Print statistics
    pto_runtime_stats(&rt);
    
    // Shutdown runtime
    pto_runtime_shutdown(&rt);
    
    return 0;
}

#endif // PTO_RUNTIME_EXAMPLE
