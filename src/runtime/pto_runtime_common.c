/**
 * PTO Runtime System - Common Implementation (Platform Independent)
 * 
 * This file contains platform-independent implementations:
 * - Runtime initialization (basic structure)
 * - Task allocation and argument management
 * - TensorMap operations (dependency discovery)
 * - Record & Replay core logic
 * - Cycle trace recording
 * - Debug dump functions
 */

#include "pto_runtime_common.h"
#include <time.h>

// =============================================================================
// Global Variables
// =============================================================================

CycleTrace* pto_global_trace = NULL;

// =============================================================================
// Runtime Initialization (Platform Independent Parts)
// =============================================================================

void pto_runtime_init(PTORuntime* rt) {
    if (!rt) return;
    
    // Initialize task table (sliding window)
    memset(rt->pend_task, 0, sizeof(rt->pend_task));
    
    rt->next_task_id = 0;
    rt->active_task_count = 0;
    
    // Initialize sliding window tracking
    rt->last_task_alive = 0;
    rt->window_aborted = false;
    rt->runtime_mode = PTO_MODE_BENCHMARK_ONLY;  // Default: no window check
    
    // Initialize dependency list pool
    memset(rt->dep_list_pool, 0, sizeof(rt->dep_list_pool));
    rt->dep_list_top = 1;   // 0 reserved as NULL
    rt->dep_list_tail = 1;

    // Initialize TensorMap
    for (int i = 0; i < PTO_TENSORMAP_SIZE; i++) {
        rt->tensor_map.buckets[i] = -1;
    }
    for (int i = 0; i < PTO_TASK_WINDOW_SIZE; i++) {
        rt->tensor_map.task_entry_head[i] = -1;
        rt->fanin_refcount[i] = 0;
        rt->fanout_refcount[i] = 0;
        rt->task_state[i] = PTO_TASK_PENDING;
    }
    for (int i = 0; i < PTO_TENSORMAP_POOL_SIZE; i++) {
        rt->tensor_map.entry_pool[i].in_bucket = false;
        rt->tensor_map.entry_pool[i].next_in_bucket = -1;
        rt->tensor_map.entry_pool[i].next_in_task = -1;
        rt->tensor_map.entry_pool[i].producer_task_id = -1;
        memset(&rt->tensor_map.entry_pool[i].region, 0, sizeof(TensorRegion));
    }
    rt->tensor_map.pool_head = 0;
    rt->tensor_map.last_task_alive = 0;

    // Initialize packed output heap ring (host/sim)
    rt->heap_size = PTO_HEAP_SIZE_BYTES;
    rt->heap_top = 0;
    rt->heap_tail = 0;
    rt->heap_base = (uint8_t*)aligned_alloc(64, (size_t)rt->heap_size);
    if (rt->heap_base) {
        memset(rt->heap_base, 0, (size_t)rt->heap_size);
    } else {
        rt->heap_base = (uint8_t*)calloc(1, (size_t)rt->heap_size);
    }
    
    // Initialize legacy ready queue
    memset(rt->ready_queue, 0, sizeof(rt->ready_queue));
    rt->ready_head = 0;
    rt->ready_tail = 0;
    rt->ready_count = 0;
    
    // Initialize dual ready queues (for a2a3_sim mode)
    memset(rt->vector_ready_queue, 0, sizeof(rt->vector_ready_queue));
    rt->vector_ready_head = 0;
    rt->vector_ready_tail = 0;
    rt->vector_ready_count = 0;
    
    memset(rt->cube_ready_queue, 0, sizeof(rt->cube_ready_queue));
    rt->cube_ready_head = 0;
    rt->cube_ready_tail = 0;
    rt->cube_ready_count = 0;
    
    // Initialize statistics
    rt->total_tasks_scheduled = 0;
    rt->total_tasks_completed = 0;
    
    // Initialize thread synchronization primitives
    pthread_mutex_init(&rt->queue_mutex, NULL);
    pthread_mutex_init(&rt->task_mutex, NULL);
    pthread_cond_init(&rt->queue_not_empty, NULL);
    pthread_cond_init(&rt->all_done, NULL);
    pthread_cond_init(&rt->vector_queue_not_empty, NULL);
    pthread_cond_init(&rt->cube_queue_not_empty, NULL);
    pthread_cond_init(&rt->window_not_full, NULL);
    
    // Initialize worker state
    rt->num_workers = 0;
    rt->num_vector_workers = 0;
    rt->num_cube_workers = 0;
    rt->shutdown_requested = false;
    rt->execution_started = false;
    rt->execution_task_threshold = 0;
    rt->simulation_mode = false;
    rt->dual_queue_mode = false;
    memset(rt->workers, 0, sizeof(rt->workers));
    memset(rt->func_registry, 0, sizeof(rt->func_registry));

    // Initialize scope stack
    rt->scope_stack_top = -1;
    
    DEBUG_PRINT("[PTO Runtime] Initialized (window_size=%d, tensormap_size=%d)\n",
           PTO_TASK_WINDOW_SIZE, PTO_TENSORMAP_SIZE);
}

void pto_runtime_shutdown(PTORuntime* rt) {
    if (!rt) return;
    
    // Cleanup core simulator (if used)
    pto_cleanup_core_sim();
    
    // Free packed output heap
    if (rt->heap_base) {
        free(rt->heap_base);
        rt->heap_base = NULL;
    }
    
    // Destroy thread synchronization primitives
    pthread_mutex_destroy(&rt->queue_mutex);
    pthread_mutex_destroy(&rt->task_mutex);
    pthread_cond_destroy(&rt->queue_not_empty);
    pthread_cond_destroy(&rt->all_done);
    pthread_cond_destroy(&rt->vector_queue_not_empty);
    pthread_cond_destroy(&rt->cube_queue_not_empty);
    pthread_cond_destroy(&rt->window_not_full);
    
    DEBUG_PRINT("[PTO Runtime] Shutdown (scheduled=%lld, completed=%lld)\n",
           (long long)rt->total_tasks_scheduled,
           (long long)rt->total_tasks_completed);
}

void pto_runtime_set_mode(PTORuntime* rt, PTORuntimeMode mode) {
    if (!rt) return;
    rt->runtime_mode = mode;
    DEBUG_PRINT("[PTO Runtime] Mode set to %d\n", mode);
}

bool pto_runtime_was_aborted(PTORuntime* rt) {
    if (!rt) return false;
    return rt->window_aborted;
}

void pto_runtime_reset(PTORuntime* rt) {
    if (!rt) return;
    
    // Clear task table
    memset(rt->pend_task, 0, sizeof(rt->pend_task));
    
    rt->next_task_id = 0;
    rt->active_task_count = 0;
    rt->last_task_alive = 0;
    rt->window_aborted = false;
    
    // Clear dependency list pool
    memset(rt->dep_list_pool, 0, sizeof(rt->dep_list_pool));
    rt->dep_list_top = 1;
    rt->dep_list_tail = 1;

    // Clear tensor map
    for (int i = 0; i < PTO_TENSORMAP_SIZE; i++) {
        rt->tensor_map.buckets[i] = -1;
    }
    for (int i = 0; i < PTO_TENSORMAP_POOL_SIZE; i++) {
        rt->tensor_map.entry_pool[i].in_bucket = false;
        rt->tensor_map.entry_pool[i].next_in_bucket = -1;
        rt->tensor_map.entry_pool[i].next_in_task = -1;
        rt->tensor_map.entry_pool[i].producer_task_id = -1;
        memset(&rt->tensor_map.entry_pool[i].region, 0, sizeof(TensorRegion));
    }
    rt->tensor_map.pool_head = 0;
    rt->tensor_map.last_task_alive = 0;
    for (int i = 0; i < PTO_TASK_WINDOW_SIZE; i++) {
        rt->tensor_map.task_entry_head[i] = -1;
        rt->fanin_refcount[i] = 0;
        rt->fanout_refcount[i] = 0;
        rt->task_state[i] = PTO_TASK_PENDING;
    }

    // Reset heap ring pointers
    rt->heap_top = 0;
    rt->heap_tail = 0;
    
    // Reset ready queues
    rt->ready_head = 0;
    rt->ready_tail = 0;
    rt->ready_count = 0;
    rt->vector_ready_head = 0;
    rt->vector_ready_tail = 0;
    rt->vector_ready_count = 0;
    rt->cube_ready_head = 0;
    rt->cube_ready_tail = 0;
    rt->cube_ready_count = 0;
    
    // Reset statistics
    rt->total_tasks_scheduled = 0;
    rt->total_tasks_completed = 0;

    // Reset scope stack
    rt->scope_stack_top = -1;
    
    DEBUG_PRINT("[PTO Runtime] Reset complete\n");
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
// TensorMap Implementation (Platform Independent)
//
// Design:
// - Hash buckets are singly-linked lists of entry offsets into a ring-buffer pool.
// - Entries become stale when producer_task_id < last_task_alive (lazy invalidation).
// - On lookup, encountering the first stale entry allows chain truncation since
//   inserts always go to the head (newest-to-oldest ordering by insertion time).
// - On pool reuse, we must unlink the old entry from its bucket chain before
//   overwriting to avoid corrupting chains.
// =============================================================================

uint32_t pto_tensormap_hash(TensorRegion* region) {
    uint64_t ptr_val = (uint64_t)region->raw_tensor;
    uint64_t h = ptr_val;
    h ^= (uint64_t)region->row_offset * 0x9E3779B97F4A7C15ULL;
    h ^= (uint64_t)region->col_offset * 0xC6A4A7935BD1E995ULL;
    h ^= ((uint64_t)region->rows << 32) | (uint64_t)region->cols;
    h ^= h >> 33;
    h *= 0xFF51AFD7ED558CCDULL;
    h ^= h >> 33;
    return (uint32_t)h & (PTO_TENSORMAP_SIZE - 1);
}

bool pto_region_match(TensorRegion* a, TensorRegion* b) {
    return a->raw_tensor == b->raw_tensor &&
           a->row_offset == b->row_offset &&
           a->col_offset == b->col_offset &&
           a->rows == b->rows &&
           a->cols == b->cols;
}

static inline bool pto_tensormap_entry_valid(PTORuntime* rt, TensorMapEntry* entry) {
    return entry->producer_task_id >= rt->last_task_alive;
}

static void pto_tensormap_remove_from_bucket(PTORuntime* rt, int32_t entry_offset) {
    TensorMap* tm = &rt->tensor_map;
    TensorMapEntry* entry = &tm->entry_pool[entry_offset];
    uint32_t bucket = pto_tensormap_hash(&entry->region);

    int32_t* prev_ptr = &tm->buckets[bucket];
    int32_t cur = *prev_ptr;
    while (cur >= 0) {
        if (cur == entry_offset) {
            *prev_ptr = tm->entry_pool[cur].next_in_bucket;
            entry->in_bucket = false;
            entry->next_in_bucket = -1;
            return;
        }
        prev_ptr = &tm->entry_pool[cur].next_in_bucket;
        cur = *prev_ptr;
    }
    // Not found (bucket chain may have been cleared); still mark as unlinked.
    entry->in_bucket = false;
    entry->next_in_bucket = -1;
}

int32_t pto_tensormap_lookup(PTORuntime* rt, TensorRegion* region) {
    TensorMap* tm = &rt->tensor_map;
    tm->last_task_alive = rt->last_task_alive;
    uint32_t bucket = pto_tensormap_hash(region);

    int32_t* prev_ptr = &tm->buckets[bucket];
    int32_t offset = *prev_ptr;

    while (offset >= 0) {
        TensorMapEntry* entry = &tm->entry_pool[offset];

        if (!pto_tensormap_entry_valid(rt, entry)) {
            // Truncate stale tail: all following entries are older.
            *prev_ptr = -1;
            while (offset >= 0) {
                TensorMapEntry* stale = &tm->entry_pool[offset];
                int32_t next = stale->next_in_bucket;
                stale->in_bucket = false;
                stale->next_in_bucket = -1;
                offset = next;
            }
            return -1;
        }

        if (pto_region_match(&entry->region, region)) {
            return entry->producer_task_id;
        }

        prev_ptr = &entry->next_in_bucket;
        offset = *prev_ptr;
    }

    return -1;
}

void pto_tensormap_insert(PTORuntime* rt, TensorRegion* region, int32_t task_id) {
    TensorMap* tm = &rt->tensor_map;
    tm->last_task_alive = rt->last_task_alive;

    int32_t entry_offset = tm->pool_head;
    TensorMapEntry* entry = &tm->entry_pool[entry_offset];
    tm->pool_head = (tm->pool_head + 1) % PTO_TENSORMAP_POOL_SIZE;

    if (entry->in_bucket) {
        pto_tensormap_remove_from_bucket(rt, entry_offset);
    }

    entry->region = *region;
    entry->producer_task_id = task_id;

    uint32_t bucket = pto_tensormap_hash(region);
    entry->next_in_bucket = tm->buckets[bucket];
    tm->buckets[bucket] = entry_offset;
    entry->in_bucket = true;

    int32_t task_slot = PTO_TASK_SLOT(task_id);
    entry->next_in_task = tm->task_entry_head[task_slot];
    tm->task_entry_head[task_slot] = entry_offset;
}

void pto_tensormap_clear(PTORuntime* rt) {
    TensorMap* tm = &rt->tensor_map;
    for (int i = 0; i < PTO_TENSORMAP_SIZE; i++) {
        tm->buckets[i] = -1;
    }
    for (int i = 0; i < PTO_TASK_WINDOW_SIZE; i++) {
        tm->task_entry_head[i] = -1;
    }
    for (int i = 0; i < PTO_TENSORMAP_POOL_SIZE; i++) {
        tm->entry_pool[i].in_bucket = false;
        tm->entry_pool[i].next_in_bucket = -1;
        tm->entry_pool[i].next_in_task = -1;
        tm->entry_pool[i].producer_task_id = -1;
        memset(&tm->entry_pool[i].region, 0, sizeof(TensorRegion));
    }
    tm->pool_head = 0;
    tm->last_task_alive = rt->last_task_alive;
}

void pto_tensormap_gc(PTORuntime* rt) {
    // No-op: stale tails are truncated during lookup; entries are unlinked on reuse.
    (void)rt;
}

// =============================================================================
// Task Allocation and Arguments (Platform Independent)
// =============================================================================

static inline int32_t pto_current_scope_depth(PTORuntime* rt) {
    return rt->scope_stack_top + 1;
}

static inline void pto_task_fanout_lock(PendingTask* task) {
    while (__atomic_exchange_n(&task->fanout_lock, 1, __ATOMIC_ACQUIRE) != 0) {
        // spin
    }
}

static inline void pto_task_fanout_unlock(PendingTask* task) {
    __atomic_store_n(&task->fanout_lock, 0, __ATOMIC_RELEASE);
}

static inline int32_t pto_dep_list_next(int32_t off) {
    off++;
    if (off >= PTO_DEP_LIST_POOL_SIZE) return 1;
    return off;
}

static int32_t pto_dep_list_alloc_one_locked(PTORuntime* rt) {
    int32_t next = pto_dep_list_next(rt->dep_list_top);
    if (next == rt->dep_list_tail) {
        // Pool full: stall in execute/simulate; abort in dump/benchmark.
        while (next == rt->dep_list_tail &&
               (rt->runtime_mode == PTO_MODE_EXECUTE || rt->runtime_mode == PTO_MODE_SIMULATE)) {
            pthread_cond_wait(&rt->window_not_full, &rt->task_mutex);
            next = pto_dep_list_next(rt->dep_list_top);
        }
        if (next == rt->dep_list_tail) {
            fprintf(stderr, "[PTO Runtime] ERROR: DepList pool full\n");
            return 0;
        }
    }
    int32_t off = rt->dep_list_top;
    rt->dep_list_top = next;
    return off;
}

static inline int32_t pto_align_up_i32(int32_t v, int32_t a) {
    return (v + (a - 1)) & ~(a - 1);
}

static void* pto_heap_alloc_locked(PTORuntime* rt, int32_t size_bytes) {
    if (!rt->heap_base || rt->heap_size <= 0) return NULL;
    size_bytes = pto_align_up_i32(size_bytes, 64);
    if (size_bytes <= 0 || size_bytes >= rt->heap_size) return NULL;

    while (1) {
        int32_t top = rt->heap_top;
        int32_t tail = rt->heap_tail;

        if (top >= tail) {
            int32_t space_end = rt->heap_size - top;
            if (space_end > size_bytes) {
                void* ptr = rt->heap_base + top;
                top += size_bytes;
                if (top == rt->heap_size) top = 0;
                rt->heap_top = top;
                return ptr;
            }
            // wrap to 0
            if (tail > size_bytes) {
                rt->heap_top = size_bytes;
                return rt->heap_base;
            }
        } else {
            int32_t gap = tail - top;
            if (gap > size_bytes) {
                void* ptr = rt->heap_base + top;
                rt->heap_top = top + size_bytes;
                return ptr;
            }
        }

        if (rt->runtime_mode == PTO_MODE_EXECUTE || rt->runtime_mode == PTO_MODE_SIMULATE) {
            pthread_cond_wait(&rt->window_not_full, &rt->task_mutex);
            continue;
        }
        return NULL;
    }
}

bool pto_try_mark_consumed_locked(PTORuntime* rt, int32_t task_id) {
    int32_t slot = PTO_TASK_SLOT(task_id);
    PendingTask* task = &rt->pend_task[slot];
    if (!task->is_active || task->task_id != task_id) return false;
    if (task->is_consumed) return false;
    if (!task->is_complete) return false;

    pto_task_fanout_lock(task);
    int32_t fanout_count = task->fanout_count;
    pto_task_fanout_unlock(task);

    if (rt->fanout_refcount[slot] != fanout_count) return false;

    task->is_consumed = true;
    rt->task_state[slot] = PTO_TASK_CONSUMED;
    return true;
}

bool pto_advance_last_task_alive_locked(PTORuntime* rt) {
    bool advanced = false;
    while (rt->last_task_alive < rt->next_task_id) {
        int32_t tid = rt->last_task_alive;
        int32_t slot = PTO_TASK_SLOT(tid);
        PendingTask* task = &rt->pend_task[slot];
        if (!task->is_active || task->task_id != tid || !task->is_consumed) {
            break;
        }

        // Advance heap_tail to end of last consumed packed buffer (if any)
        if (task->packed_buffer_end && rt->heap_base) {
            intptr_t off = (uint8_t*)task->packed_buffer_end - rt->heap_base;
            if (off >= 0 && off <= rt->heap_size) {
                rt->heap_tail = (int32_t)off;
                if (rt->heap_tail == rt->heap_size) rt->heap_tail = 0;
            }
        }

        // Advance dep list tail
        if (task->dep_pool_end > 0) {
            rt->dep_list_tail = task->dep_pool_end;
        }

        rt->last_task_alive++;
        advanced = true;
    }

    // Sync TensorMap validity threshold
    rt->tensor_map.last_task_alive = rt->last_task_alive;
    return advanced;
}

void pto_scope_begin(PTORuntime* rt) {
    if (!rt) return;
    if (rt->scope_stack_top >= (int32_t)(sizeof(rt->scope_stack) / sizeof(rt->scope_stack[0])) - 1) {
        fprintf(stderr, "[PTO Runtime] ERROR: scope stack overflow\n");
        return;
    }
    // Scope begins at current task id (absolute).
    rt->scope_stack[++rt->scope_stack_top] = rt->next_task_id;
}

int32_t pto_get_scope_depth(PTORuntime* rt) {
    if (!rt) return 0;
    return rt->scope_stack_top + 1;
}

void pto_scope_end(PTORuntime* rt) {
    if (!rt) return;
    if (rt->scope_stack_top < 0) {
        fprintf(stderr, "[PTO Runtime] ERROR: scope_end with empty stack\n");
        return;
    }

    pthread_mutex_lock(&rt->task_mutex);
    int32_t begin = rt->scope_stack[rt->scope_stack_top--];
    int32_t end = rt->next_task_id;

    for (int32_t task_id = begin; task_id < end; task_id++) {
        int32_t slot = PTO_TASK_SLOT(task_id);
        PendingTask* task = &rt->pend_task[slot];
        if (!task->is_active || task->task_id != task_id) continue;

        // This scope releases one reference to the task's output buffer.
        rt->fanout_refcount[slot]++;
        pto_try_mark_consumed_locked(rt, task_id);
    }

    bool window_advanced = pto_advance_last_task_alive_locked(rt);
    if (window_advanced) {
        pthread_cond_broadcast(&rt->window_not_full);
    }
    pthread_mutex_unlock(&rt->task_mutex);
}

int32_t pto_task_alloc_impl(PTORuntime* rt, const char* func_name, void* func_ptr,
                            int32_t buffer_bytes, int32_t reuse_bytes, bool is_cube) {
    // Check if window is full
    int32_t tasks_in_flight = rt->next_task_id - rt->last_task_alive;
    
    if (tasks_in_flight >= PTO_TASK_WINDOW_SIZE) {
        // Window is full - behavior depends on runtime mode
        switch (rt->runtime_mode) {
            case PTO_MODE_BENCHMARK_ONLY:
                // Benchmark mode: simulate window advancement to enable TensorMap cleanup
                // Since no tasks actually complete, last_task_alive stays at 0,
                // causing TensorMap to grow unboundedly. By advancing it here, we allow
                // stale entries to be reclaimed, keeping TensorMap size bounded.
                rt->last_task_alive = rt->next_task_id - (PTO_TASK_WINDOW_SIZE / 2);
                rt->tensor_map.last_task_alive = rt->last_task_alive;
                break;
                
            case PTO_MODE_DUMP_GRAPH:
                // Dump/graph mode: abort orchestration
                if (!rt->window_aborted) {
                    rt->window_aborted = true;
                    fprintf(stderr, "[PTO Runtime] Window full (size=%d), aborting orchestration for dump/graph\n",
                            PTO_TASK_WINDOW_SIZE);
                }
                return -1;
                
            case PTO_MODE_EXECUTE:
            case PTO_MODE_SIMULATE:
                // Execute/simulate mode: wait for workers to complete tasks
                pthread_mutex_lock(&rt->task_mutex);
                while ((rt->next_task_id - rt->last_task_alive) >= PTO_TASK_WINDOW_SIZE) {
                    DEBUG_PRINT("[PTO Runtime] Window full, waiting... (oldest=%d, next=%d)\n",
                           rt->last_task_alive, rt->next_task_id);
                    pthread_cond_wait(&rt->window_not_full, &rt->task_mutex);
                }
                pthread_mutex_unlock(&rt->task_mutex);
                break;
        }
    }
    
    // Allocate task ID and get slot in window
    int32_t task_id = rt->next_task_id++;
    int32_t slot = PTO_TASK_SLOT(task_id);
    PendingTask* task = &rt->pend_task[slot];
    
    // Initialize task
    task->task_id = task_id;
    task->func_name = func_name;
    task->func_ptr = func_ptr;
    task->cycle_func = NULL;  // Set via pto_task_set_cycle_func if needed
    task->num_args = 0;
    task->num_outputs = 0;
    task->packed_buffer_base = NULL;
    task->packed_buffer_end = NULL;
    task->dep_pool_end = rt->dep_list_top;
    task->buffer_size_bytes = buffer_bytes;
    task->buffer_size_with_reuse = reuse_bytes;
    task->fanin_head = 0;
    task->fanin_count = 0;
    task->fanout_head = 0;
    task->fanout_consumer_count = 0;
    task->scope_depth = pto_current_scope_depth(rt);
    task->fanout_count = task->scope_depth;
    task->fanout_lock = 0;
    task->is_active = true;
    task->is_submitted = false;
    task->is_complete = false;
    task->is_consumed = false;
    task->is_cube = is_cube;
    task->earliest_start_cycle = 0;
    task->end_cycle = 0;
    
    for (int i = 0; i < PTO_MAX_ARGS; i++) {
        task->output_arg_index[i] = -1;
        task->output_offsets[i] = -1;
    }

    // Initialize scheduler state for this slot
    rt->fanin_refcount[slot] = 0;
    rt->fanout_refcount[slot] = 0;
    rt->task_state[slot] = PTO_TASK_PENDING;
    
    rt->active_task_count++;
    rt->total_tasks_scheduled++;
    
    DEBUG_PRINT("[PTO Runtime] Allocated task %d (slot %d): %s (buf=%d B, reuse=%d B, is_cube=%d)\n", 
           task_id, slot, func_name, buffer_bytes, reuse_bytes, is_cube);
    
    return task_id;
}

void pto_task_set_cycle_func(PTORuntime* rt, int32_t task_id, CycleCostFunc cycle_func) {
    if (!rt || task_id < 0 || task_id >= rt->next_task_id) return;
    rt->pend_task[PTO_TASK_SLOT(task_id)].cycle_func = cycle_func;
}

void pto_task_add_input(PTORuntime* rt, int32_t task_id,
                        void* tensor, int64_t row_off, int64_t col_off,
                        int64_t rows, int64_t cols) {
    if (task_id < 0 || task_id >= rt->next_task_id) {
        fprintf(stderr, "[PTO Runtime] ERROR: Invalid task_id %d\n", task_id);
        return;
    }
    
    PendingTask* task = &rt->pend_task[PTO_TASK_SLOT(task_id)];
    
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
    
    // Look up producer in TensorMap and wire dependencies.
    pthread_mutex_lock(&rt->task_mutex);
    int32_t producer_id = pto_tensormap_lookup(rt, &region);

    if (producer_id >= 0 && producer_id != task_id) {
        PendingTask* producer = &rt->pend_task[PTO_TASK_SLOT(producer_id)];
        if (!producer->is_active || producer->task_id != producer_id || producer->is_consumed) {
            // Should be filtered by TensorMap validity, but guard anyway.
            producer_id = -1;
        }
    }

    if (producer_id >= 0 && producer_id != task_id) {
        // Add producer to this task's fanin list
        int32_t fanin_node = pto_dep_list_alloc_one_locked(rt);
        if (fanin_node != 0) {
            rt->dep_list_pool[fanin_node].task_id = producer_id;
            rt->dep_list_pool[fanin_node].next_offset = task->fanin_head;
            task->fanin_head = fanin_node;
            task->fanin_count++;
        }

        // Add this task to producer's fanout list and increment total refcount
        PendingTask* producer = &rt->pend_task[PTO_TASK_SLOT(producer_id)];
        int32_t fanout_node = pto_dep_list_alloc_one_locked(rt);
        if (fanout_node != 0) {
            rt->dep_list_pool[fanout_node].task_id = task_id;
            pto_task_fanout_lock(producer);
            rt->dep_list_pool[fanout_node].next_offset = producer->fanout_head;
            producer->fanout_head = fanout_node;
            producer->fanout_consumer_count++;
            producer->fanout_count++;  // +1 consumer reference
            pto_task_fanout_unlock(producer);
        }

        // If producer already complete, dependency is immediately satisfied.
        int32_t task_slot = PTO_TASK_SLOT(task_id);
        if (producer->is_complete) {
            rt->fanin_refcount[task_slot]++;
        }

        DEBUG_PRINT("[PTO Runtime] Task %d depends on task %d (tensor=%p, offset=[%lld,%lld])\n",
               task_id, producer_id, tensor, (long long)row_off, (long long)col_off);
    } else {
        DEBUG_PRINT("[PTO Runtime] Task %d input (tensor=%p, offset=[%lld,%lld]) - no producer\n",
               task_id, tensor, (long long)row_off, (long long)col_off);
    }

    pthread_mutex_unlock(&rt->task_mutex);
}

void pto_task_add_output(PTORuntime* rt, int32_t task_id,
                         void* tensor, int64_t row_off, int64_t col_off,
                         int64_t rows, int64_t cols) {
    if (task_id < 0 || task_id >= rt->next_task_id) {
        fprintf(stderr, "[PTO Runtime] ERROR: Invalid task_id %d\n", task_id);
        return;
    }
    
    PendingTask* task = &rt->pend_task[PTO_TASK_SLOT(task_id)];
    
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

    // Record output arg index; output regions are registered in TensorMap on submit.
    if (task->num_outputs < PTO_MAX_ARGS) {
        task->output_arg_index[task->num_outputs++] = task->num_args - 1;
    }
    
    DEBUG_PRINT("[PTO Runtime] Task %d output (tensor=%p, offset=[%lld,%lld], shape=[%lld,%lld])\n",
           task_id, tensor, (long long)row_off, (long long)col_off,
           (long long)rows, (long long)cols);
}

static inline int32_t pto_align_up_i32(int32_t v, int32_t a);
static void* pto_heap_alloc_locked(PTORuntime* rt, int32_t size_bytes);

bool pto_task_prepare_submit(PTORuntime* rt, int32_t task_id) {
    if (!rt || task_id < 0 || task_id >= rt->next_task_id) return false;

    pthread_mutex_lock(&rt->task_mutex);
    int32_t slot = PTO_TASK_SLOT(task_id);
    PendingTask* task = &rt->pend_task[slot];
    if (!task->is_active || task->task_id != task_id) {
        pthread_mutex_unlock(&rt->task_mutex);
        return false;
    }
    if (task->is_submitted) {
        bool ready = (rt->task_state[slot] == PTO_TASK_READY);
        pthread_mutex_unlock(&rt->task_mutex);
        return ready;
    }

    // Allocate packed output buffer for outputs with NULL base pointer.
    int64_t total_bytes = 0;
    int32_t needs_alloc = 0;
    for (int i = 0; i < task->num_outputs; i++) {
        int arg_idx = task->output_arg_index[i];
        if (arg_idx < 0 || arg_idx >= task->num_args) continue;
        TaskArg* arg = &task->args[arg_idx];
        if (arg->region.raw_tensor == NULL) {
            if (arg->region.row_offset != 0 || arg->region.col_offset != 0) {
                fprintf(stderr, "[PTO Runtime] ERROR: output alloc requires zero offsets (task %d)\n", task_id);
                pthread_mutex_unlock(&rt->task_mutex);
                return false;
            }
            int64_t bytes = arg->region.rows * arg->region.cols * (int64_t)sizeof(float);
            bytes = pto_align_up_i32((int32_t)bytes, 64);
            total_bytes += bytes;
            needs_alloc++;
        }
    }

    if (needs_alloc > 0) {
        void* base = pto_heap_alloc_locked(rt, (int32_t)total_bytes);
        if (!base) {
            fprintf(stderr, "[PTO Runtime] ERROR: heap alloc failed for task %d (%lld bytes)\n",
                    task_id, (long long)total_bytes);
            pthread_mutex_unlock(&rt->task_mutex);
            return false;
        }
        task->packed_buffer_base = base;
        task->packed_buffer_end = (uint8_t*)base + (size_t)total_bytes;

        int32_t off = 0;
        for (int i = 0; i < task->num_outputs; i++) {
            int arg_idx = task->output_arg_index[i];
            TaskArg* arg = &task->args[arg_idx];
            if (arg->region.raw_tensor == NULL) {
                int32_t bytes = (int32_t)(arg->region.rows * arg->region.cols * (int64_t)sizeof(float));
                bytes = pto_align_up_i32(bytes, 64);
                arg->region.raw_tensor = (uint8_t*)base + off;
                task->output_offsets[i] = off;
                off += bytes;
            } else {
                task->output_offsets[i] = -1;
            }
        }
    }

    // Register outputs in TensorMap
    for (int i = 0; i < task->num_outputs; i++) {
        int arg_idx = task->output_arg_index[i];
        if (arg_idx < 0 || arg_idx >= task->num_args) continue;
        TaskArg* arg = &task->args[arg_idx];
        pto_tensormap_insert(rt, &arg->region, task_id);
    }

    // Snapshot dep list pool for tail advancement.
    task->dep_pool_end = rt->dep_list_top;

    task->is_submitted = true;

    // If all dependencies are satisfied, task is ready.
    if (rt->fanin_refcount[slot] == task->fanin_count) {
        rt->task_state[slot] = PTO_TASK_READY;
        pthread_mutex_unlock(&rt->task_mutex);
        return true;
    }

    rt->task_state[slot] = PTO_TASK_PENDING;
    pthread_mutex_unlock(&rt->task_mutex);
    return false;
}

// =============================================================================
// Cycle Trace Recording Implementation (Platform Independent)
// =============================================================================

void pto_trace_init(int32_t num_workers) {
    if (pto_global_trace) {
        free(pto_global_trace);
    }
    pto_global_trace = (CycleTrace*)calloc(1, sizeof(CycleTrace));
    if (!pto_global_trace) return;
    
    pto_global_trace->count = 0;
    pto_global_trace->num_workers = num_workers > 0 ? num_workers : 1;
    pto_global_trace->num_vector_workers = 0;
    pto_global_trace->num_cube_workers = 0;
    pto_global_trace->enabled = true;
    
    // Initialize per-worker cycle counters
    for (int i = 0; i < PTO_MAX_WORKERS; i++) {
        pto_global_trace->per_worker_cycle[i] = 0;
    }
}

void pto_trace_init_dual(int32_t num_vector_workers, int32_t num_cube_workers) {
    int32_t total = num_vector_workers + num_cube_workers;
    pto_trace_init(total);
    if (pto_global_trace) {
        pto_global_trace->num_vector_workers = num_vector_workers;
        pto_global_trace->num_cube_workers = num_cube_workers;
    }
}

void pto_trace_record(int32_t worker_id, const char* func_name, int64_t cycle_cost) {
    if (!pto_global_trace || !pto_global_trace->enabled) return;
    if (pto_global_trace->count >= PTO_MAX_TRACE_ENTRIES) return;
    if (worker_id < 0 || worker_id >= PTO_MAX_WORKERS) return;
    
    int idx = pto_global_trace->count++;
    CycleTraceEntry* entry = &pto_global_trace->entries[idx];
    
    // Copy function name
    strncpy(entry->func_name, func_name ? func_name : "unknown", PTO_MAX_FUNC_NAME_LEN - 1);
    entry->func_name[PTO_MAX_FUNC_NAME_LEN - 1] = '\0';
    
    entry->worker_id = worker_id;
    entry->start_cycle = pto_global_trace->per_worker_cycle[worker_id];
    entry->end_cycle = entry->start_cycle + cycle_cost;
    
    // Update worker cycle counter
    pto_global_trace->per_worker_cycle[worker_id] = entry->end_cycle;
}

void pto_trace_record_with_time(int32_t worker_id, const char* func_name, 
                                 int64_t start_cycle, int64_t end_cycle) {
    if (!pto_global_trace || !pto_global_trace->enabled) return;
    if (pto_global_trace->count >= PTO_MAX_TRACE_ENTRIES) return;
    if (worker_id < 0 || worker_id >= PTO_MAX_WORKERS) return;
    
    int idx = pto_global_trace->count++;
    CycleTraceEntry* entry = &pto_global_trace->entries[idx];
    
    // Copy function name
    strncpy(entry->func_name, func_name ? func_name : "unknown", PTO_MAX_FUNC_NAME_LEN - 1);
    entry->func_name[PTO_MAX_FUNC_NAME_LEN - 1] = '\0';
    
    entry->worker_id = worker_id;
    entry->start_cycle = start_cycle;
    entry->end_cycle = end_cycle;
    
    // Update worker cycle counter to the end of this task
    pto_global_trace->per_worker_cycle[worker_id] = end_cycle;
}

int64_t pto_trace_get_cycle(int32_t worker_id) {
    if (!pto_global_trace) return 0;
    if (worker_id < 0 || worker_id >= PTO_MAX_WORKERS) return 0;
    return pto_global_trace->per_worker_cycle[worker_id];
}

void pto_trace_cleanup(void) {
    if (pto_global_trace) {
        free(pto_global_trace);
        pto_global_trace = NULL;
    }
}

char* pto_trace_to_chrome_json(void) {
    if (!pto_global_trace) return NULL;
    
    int32_t num_vec = pto_global_trace->num_vector_workers;
    int32_t num_cube = pto_global_trace->num_cube_workers;
    bool dual_mode = (num_vec > 0 && num_cube > 0);
    
    // Estimate output size (generous allocation)
    // Add extra space for thread name metadata if dual mode
    size_t metadata_size = dual_mode ? (num_vec + num_cube) * 128 : 0;
    size_t buf_size = 2048 + pto_global_trace->count * 256 + metadata_size;
    char* buf = (char*)malloc(buf_size);
    if (!buf) return NULL;
    
    char* ptr = buf;
    ptr += sprintf(ptr, "{\n  \"traceEvents\": [\n");
    
    // Add thread name metadata events for dual-queue mode
    if (dual_mode) {
        // Vector workers: pid=0, tid=0 to num_vec-1
        for (int i = 0; i < num_vec; i++) {
            ptr += sprintf(ptr, "    {\"name\": \"thread_name\", \"ph\": \"M\", \"pid\": 0, \"tid\": %d, "
                           "\"args\": {\"name\": \"Vector-%d\"}},\n", i, i);
        }
        // Cube workers: pid=1, tid=0 to num_cube-1
        for (int i = 0; i < num_cube; i++) {
            ptr += sprintf(ptr, "    {\"name\": \"thread_name\", \"ph\": \"M\", \"pid\": 1, \"tid\": %d, "
                           "\"args\": {\"name\": \"Cube-%d\"}},\n", i, i);
        }
        // Process name metadata
        ptr += sprintf(ptr, "    {\"name\": \"process_name\", \"ph\": \"M\", \"pid\": 0, "
                       "\"args\": {\"name\": \"Vector Workers (%d)\"}},\n", num_vec);
        ptr += sprintf(ptr, "    {\"name\": \"process_name\", \"ph\": \"M\", \"pid\": 1, "
                       "\"args\": {\"name\": \"Cube Workers (%d)\"}},\n", num_cube);
    }
    
    for (int i = 0; i < pto_global_trace->count; i++) {
        CycleTraceEntry* e = &pto_global_trace->entries[i];
        int64_t duration = e->end_cycle - e->start_cycle;
        
        // Determine pid and tid based on worker type
        int pid = 0;
        int tid = e->worker_id;
        
        if (dual_mode && e->worker_id >= num_vec) {
            // Cube worker (worker_id >= num_vector_workers)
            pid = 1;
            tid = e->worker_id - num_vec;
        }
        
        // Chrome Tracing format (duration event)
        ptr += sprintf(ptr, "    {\"name\": \"%s\", \"cat\": \"task\", \"ph\": \"X\", "
                       "\"ts\": %lld, \"dur\": %lld, \"pid\": %d, \"tid\": %d}%s\n",
                       e->func_name,
                       (long long)(e->start_cycle),     // timestamp in microseconds (we use cycles)
                       (long long)duration,              // duration
                       pid, tid,
                       (i < pto_global_trace->count - 1) ? "," : "");
    }
    
    ptr += sprintf(ptr, "  ],\n");
    ptr += sprintf(ptr, "  \"displayTimeUnit\": \"ns\",\n");
    ptr += sprintf(ptr, "  \"metadata\": {\n");
    ptr += sprintf(ptr, "    \"num_workers\": %d,\n", pto_global_trace->num_workers);
    if (dual_mode) {
        ptr += sprintf(ptr, "    \"num_vector_workers\": %d,\n", num_vec);
        ptr += sprintf(ptr, "    \"num_cube_workers\": %d,\n", num_cube);
    }
    ptr += sprintf(ptr, "    \"total_entries\": %d\n", pto_global_trace->count);
    ptr += sprintf(ptr, "  }\n");
    ptr += sprintf(ptr, "}\n");
    
    return buf;
}

void pto_trace_write_json(const char* filename) {
    char* json = pto_trace_to_chrome_json();
    if (!json) {
        fprintf(stderr, "Error: Failed to generate trace JSON\n");
        return;
    }
    
    FILE* f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Error: Failed to open %s for writing\n", filename);
        free(json);
        return;
    }
    
    fputs(json, f);
    fclose(f);
    free(json);
    
    printf("Trace written to: %s\n", filename);
    printf("  Open in Chrome: chrome://tracing and load the file\n");
}

void pto_trace_print_summary(void) {
    if (!pto_global_trace) {
        printf("Trace: not initialized\n");
        return;
    }
    
    printf("\n=== Cycle Trace Summary ===\n");
    printf("Total entries: %d\n", pto_global_trace->count);
    printf("Num workers: %d\n", pto_global_trace->num_workers);
    
    // Per-worker statistics
    int64_t max_cycle = 0;
    for (int w = 0; w < pto_global_trace->num_workers; w++) {
        int64_t cycle = pto_global_trace->per_worker_cycle[w];
        printf("  Worker %d: %lld cycles\n", w, (long long)cycle);
        if (cycle > max_cycle) max_cycle = cycle;
    }
    printf("Max cycle (makespan): %lld\n", (long long)max_cycle);
    
    // Function breakdown
    printf("\nFunction breakdown:\n");
    
    // Simple aggregation (could be made more efficient with hash table)
    typedef struct { char name[PTO_MAX_FUNC_NAME_LEN]; int64_t total_cycles; int count; } FuncStats;
    FuncStats stats[100];
    int num_stats = 0;
    
    for (int i = 0; i < pto_global_trace->count; i++) {
        CycleTraceEntry* e = &pto_global_trace->entries[i];
        int64_t dur = e->end_cycle - e->start_cycle;
        
        // Find or create entry
        int found = -1;
        for (int s = 0; s < num_stats; s++) {
            if (strcmp(stats[s].name, e->func_name) == 0) {
                found = s;
                break;
            }
        }
        
        if (found >= 0) {
            stats[found].total_cycles += dur;
            stats[found].count++;
        } else if (num_stats < 100) {
            strncpy(stats[num_stats].name, e->func_name, PTO_MAX_FUNC_NAME_LEN - 1);
            stats[num_stats].total_cycles = dur;
            stats[num_stats].count = 1;
            num_stats++;
        }
    }
    
    for (int s = 0; s < num_stats; s++) {
        printf("  %s: %lld cycles (%d calls)\n", 
               stats[s].name, (long long)stats[s].total_cycles, stats[s].count);
    }
    printf("===========================\n\n");
}

// =============================================================================
// Debug Dump Implementation (Platform Independent)
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
    
    // Task Table - only dump tasks within current window
    fprintf(fp, "================================================================================\n");
    fprintf(fp, "TASK TABLE (sliding window, size=%d)\n", PTO_TASK_WINDOW_SIZE);
    fprintf(fp, "================================================================================\n\n");
    
    // Determine dump range (limited by window)
    int32_t dump_start = rt->last_task_alive;
    int32_t dump_end = rt->next_task_id;
    int32_t dump_count = dump_end - dump_start;
    if (dump_count > PTO_TASK_WINDOW_SIZE) {
        dump_count = PTO_TASK_WINDOW_SIZE;
        dump_start = dump_end - PTO_TASK_WINDOW_SIZE;
    }
    
    fprintf(fp, "  Window: tasks %d to %d (%d tasks)\n\n", dump_start, dump_end - 1, dump_count);
    
    for (int32_t i = dump_start; i < dump_end; i++) {
        PendingTask* task = &rt->pend_task[PTO_TASK_SLOT(i)];
        if (!task->is_active || task->task_id != i) continue;
        
        fprintf(fp, "--------------------------------------------------------------------------------\n");
        fprintf(fp, "TASK %d (slot %d)\n", task->task_id, PTO_TASK_SLOT(i));
        fprintf(fp, "--------------------------------------------------------------------------------\n");
        fprintf(fp, "  Function:     %s\n", task->func_name ? task->func_name : "(null)");
        fprintf(fp, "  Func Ptr:     %p\n", task->func_ptr);
        fprintf(fp, "  Is Active:    %s\n", task->is_active ? "true" : "false");
        fprintf(fp, "  Is Submitted: %s\n", task->is_submitted ? "true" : "false");
        fprintf(fp, "  Is Complete:  %s\n", task->is_complete ? "true" : "false");
        fprintf(fp, "  Is Consumed:  %s\n", task->is_consumed ? "true" : "false");
        fprintf(fp, "  Scope Depth:  %d\n", task->scope_depth);
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
        {
            int32_t slot = PTO_TASK_SLOT(i);
            fprintf(fp, "    fanin_count     = %d\n", task->fanin_count);
            fprintf(fp, "    fanin_refcount  = %d\n", rt->fanin_refcount[slot]);
            fprintf(fp, "    fanin_remaining = %d\n", task->fanin_count - rt->fanin_refcount[slot]);
        }
        fprintf(fp, "\n");
        
        // Fanout list
        fprintf(fp, "  FANOUT LIST (consumers that depend on this task)\n");
        fprintf(fp, "  ------------------------------------------------\n");
        {
            int32_t slot = PTO_TASK_SLOT(i);
            fprintf(fp, "    fanout_total      = %d\n", task->fanout_count);
            fprintf(fp, "    fanout_consumers  = %d\n", task->fanout_consumer_count);
            fprintf(fp, "    fanout_refcount   = %d\n", rt->fanout_refcount[slot]);
            fprintf(fp, "    Consumers:\n");
            int32_t off = task->fanout_head;
            int32_t shown = 0;
            while (off != 0 && shown < task->fanout_consumer_count) {
                int32_t consumer_id = rt->dep_list_pool[off].task_id;
                PendingTask* consumer = &rt->pend_task[PTO_TASK_SLOT(consumer_id)];
                fprintf(fp, "      -> Task %d (%s)\n", consumer_id,
                        (consumer->is_active && consumer->func_name) ? consumer->func_name : "(null)");
                off = rt->dep_list_pool[off].next_offset;
                shown++;
            }
            if (task->fanout_consumer_count == 0) {
                fprintf(fp, "      (none)\n");
            }
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
        int32_t off = rt->tensor_map.buckets[i];
        while (off >= 0) {
            TensorMapEntry* entry = &rt->tensor_map.entry_pool[off];
            if (entry->producer_task_id >= rt->last_task_alive) {
                fprintf(fp, "  [bucket %d] tensor=%p, offset=[%lld,%lld], shape=[%lld,%lld] -> producer: Task %d\n",
                        i,
                        entry->region.raw_tensor,
                        (long long)entry->region.row_offset,
                        (long long)entry->region.col_offset,
                        (long long)entry->region.rows,
                        (long long)entry->region.cols,
                        entry->producer_task_id);
                tensor_count++;
            }
            off = entry->next_in_bucket;
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
    for (int32_t i = dump_start; i < dump_end; i++) {
        PendingTask* task = &rt->pend_task[PTO_TASK_SLOT(i)];
        if (!task->is_active || task->task_id != i) continue;
        
        // Status indicator
        int32_t slot = PTO_TASK_SLOT(i);
        const char* status = (rt->task_state[slot] == PTO_TASK_CONSUMED) ? "[CONSUMED]" :
                             (rt->task_state[slot] == PTO_TASK_COMPLETED) ? "[DONE]" :
                             (rt->task_state[slot] == PTO_TASK_READY) ? "[READY]" :
                             "[WAIT]";
        
        fprintf(fp, "  Task %d (%s) %s\n", i, 
                task->func_name ? task->func_name : "?", status);
        
        int32_t off = task->fanout_head;
        int32_t shown = 0;
        while (off != 0 && shown < task->fanout_consumer_count) {
            int32_t consumer_id = rt->dep_list_pool[off].task_id;
            PendingTask* consumer = &rt->pend_task[PTO_TASK_SLOT(consumer_id)];
            fprintf(fp, "    └──> Task %d (%s)\n", consumer_id,
                    (consumer->is_active && consumer->func_name) ? consumer->func_name : "?");
            off = rt->dep_list_pool[off].next_offset;
            shown++;
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
    
    printf("================================================================================\n");
    printf("PTO RUNTIME DUMP\n");
    printf("================================================================================\n\n");
    
    printf("SUMMARY\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("  Total tasks scheduled:  %lld\n", (long long)rt->total_tasks_scheduled);
    printf("  Total tasks completed:  %lld\n", (long long)rt->total_tasks_completed);
    printf("  Active task count:      %d\n", rt->active_task_count);
    printf("  Next task ID:           %d\n", rt->next_task_id);
    printf("  Last task alive:        %d\n", rt->last_task_alive);
    printf("  Ready queue size:       %d\n", rt->ready_count);
    printf("\n");
    
    // Determine dump range (limited by window)
    int32_t dump_start = rt->last_task_alive;
    int32_t dump_end = rt->next_task_id;
    int32_t dump_count = dump_end - dump_start;
    if (dump_count > PTO_TASK_WINDOW_SIZE) {
        dump_count = PTO_TASK_WINDOW_SIZE;
        dump_start = dump_end - PTO_TASK_WINDOW_SIZE;
    }
    
    printf("TASK TABLE (window: %d to %d, %d tasks)\n", dump_start, dump_end - 1, dump_count);
    printf("--------------------------------------------------------------------------------\n");
    for (int32_t i = dump_start; i < dump_end; i++) {
        PendingTask* task = &rt->pend_task[PTO_TASK_SLOT(i)];
        if (!task->is_active || task->task_id != i) continue;
        int32_t slot = PTO_TASK_SLOT(i);
        const char* status = (rt->task_state[slot] == PTO_TASK_CONSUMED) ? "CONSUMED" :
                             (rt->task_state[slot] == PTO_TASK_COMPLETED) ? "DONE" :
                             (rt->task_state[slot] == PTO_TASK_READY) ? "READY" :
                             "WAIT";
        
        int32_t remaining = task->fanin_count - rt->fanin_refcount[slot];
        printf("  Task %d: %-20s [%s] fanin_rem=%d fanout_cons=%d buf=%.1fKB\n",
               i, task->func_name ? task->func_name : "?", status,
               remaining, task->fanout_consumer_count,
               task->buffer_size_with_reuse / 1024.0);
    }
    printf("\n");
    
    return 0;
}

// =============================================================================
// Cycle-Accurate Simulation Implementation
// =============================================================================

// A2A3 Core Simulator Integration (optional)
// When A2A3_CORE_SIM_AVAILABLE is defined and liba2a3_core.a is linked,
// cycle costs are computed by the cycle-accurate core model.
// Otherwise, heuristic estimates are used.

#ifdef A2A3_CORE_SIM_AVAILABLE
#include "runtime_a2a3_sim/core_model/a2a3_sim_integration.h"
static bool g_core_sim_initialized = false;
#endif

// Heuristic-based cycle cost estimation (fallback)
static int64_t pto_estimate_cycle_cost_heuristic(const char* func_name) {
    if (!func_name) return 10;
    
    // MatMul operations (Cube Engine) - most expensive
    if (strstr(func_name, "matmul") || strstr(func_name, "gemm") || 
        strstr(func_name, "MATMUL") || strstr(func_name, "GEMM")) {
        return 50;  // Base matmul cost per tile
    }
    
    // Reduction operations
    if (strstr(func_name, "rowsum") || strstr(func_name, "rowmax") ||
        strstr(func_name, "colsum") || strstr(func_name, "colmax") ||
        strstr(func_name, "rowmin") || strstr(func_name, "colmin")) {
        return 20;
    }
    
    // Broadcast/expand operations
    if (strstr(func_name, "expand") || strstr(func_name, "broadcast")) {
        return 16;
    }
    
    // Activation functions
    if (strstr(func_name, "silu") || strstr(func_name, "gelu") ||
        strstr(func_name, "relu") || strstr(func_name, "sigmoid") ||
        strstr(func_name, "tanh")) {
        return 20;
    }
    
    // Softmax components
    if (strstr(func_name, "softmax") || strstr(func_name, "exp")) {
        return 24;
    }
    
    // Normalization
    if (strstr(func_name, "norm") || strstr(func_name, "rsqrt") ||
        strstr(func_name, "rmsnorm") || strstr(func_name, "layernorm")) {
        return 30;
    }
    
    // RoPE (rotary position embedding)
    if (strstr(func_name, "rope") || strstr(func_name, "rotary")) {
        return 40;
    }
    
    // Attention components
    if (strstr(func_name, "attention") || strstr(func_name, "score")) {
        return 60;
    }
    
    // Memory operations
    if (strstr(func_name, "load") || strstr(func_name, "store") ||
        strstr(func_name, "LOAD") || strstr(func_name, "STORE")) {
        return 100;  // GM access is expensive
    }
    
    // Basic element-wise ops
    if (strstr(func_name, "add") || strstr(func_name, "sub") ||
        strstr(func_name, "mul") || strstr(func_name, "div")) {
        return 8;
    }
    
    // Default
    return 10;
}

// Main cycle cost estimation function
// Uses core simulator when available, otherwise falls back to heuristics
int64_t pto_estimate_cycle_cost(const char* func_name) {
#ifdef A2A3_CORE_SIM_AVAILABLE
    // Initialize core simulator on first use
    if (!g_core_sim_initialized) {
        a2a3_sim_init();
        g_core_sim_initialized = true;
    }
    // Use core simulator for cycle estimation
    bool is_cube = (func_name && (strstr(func_name, "matmul") || strstr(func_name, "gemm") ||
                                   strstr(func_name, "MATMUL") || strstr(func_name, "GEMM")));
    return a2a3_sim_get_task_cycles(func_name, is_cube, 32 * 128);
#else
    // Fallback to heuristic estimation
    return pto_estimate_cycle_cost_heuristic(func_name);
#endif
}

// Cleanup function for core simulator (call during runtime shutdown)
void pto_cleanup_core_sim(void) {
#ifdef A2A3_CORE_SIM_AVAILABLE
    if (g_core_sim_initialized) {
        a2a3_sim_cleanup();
        g_core_sim_initialized = false;
    }
#endif
}

void pto_simulate_all(PTORuntime* rt) {
    if (!rt) return;
    
    // Use platform-configured number of workers
    // For A2A3: num_vector_workers + num_cube_workers (typically 48 + 24 = 72)
    // For other platforms: num_workers or default to 4
    int NUM_WORKERS = rt->num_workers;
    if (NUM_WORKERS <= 0) {
        // Fallback: check if A2A3 dual-queue mode with separate counts
        if (rt->dual_queue_mode && (rt->num_vector_workers > 0 || rt->num_cube_workers > 0)) {
            NUM_WORKERS = rt->num_vector_workers + rt->num_cube_workers;
        } else {
            NUM_WORKERS = 4;  // Default fallback
        }
    }
    
    // Safety: cap to PTO_MAX_WORKERS to prevent buffer overflow
    if (NUM_WORKERS > PTO_MAX_WORKERS) {
        fprintf(stderr, "[PTO Simulator] WARNING: Requested %d workers exceeds PTO_MAX_WORKERS (%d), capping\n",
                NUM_WORKERS, PTO_MAX_WORKERS);
        NUM_WORKERS = PTO_MAX_WORKERS;
    }
    
    printf("\n[PTO Simulator] ======== Starting Cycle-Accurate Simulation ========\n");
    printf("  Total tasks: %lld\n", (long long)rt->total_tasks_scheduled);
    if (rt->dual_queue_mode) {
        printf("  Workers: %d (%d vector + %d cube)\n", 
               NUM_WORKERS, rt->num_vector_workers, rt->num_cube_workers);
    } else {
        printf("  Workers: %d\n", NUM_WORKERS);
    }
    
    int64_t worker_cycles[PTO_MAX_WORKERS] = {0};
    int tasks_per_worker[PTO_MAX_WORKERS] = {0};
    int worker_round_robin = 0;
    
    // Process tasks in dependency order (within window)
    // For simulation, we assume all tasks are within the window (single-shot simulation)
    int32_t sim_start = rt->last_task_alive;
    int32_t sim_end = rt->next_task_id;
    int32_t total_to_simulate = sim_end - sim_start;
    
    if (total_to_simulate > PTO_TASK_WINDOW_SIZE) {
        fprintf(stderr, "[PTO Simulator] WARNING: %d tasks exceed window size %d, simulating last %d\n",
                total_to_simulate, PTO_TASK_WINDOW_SIZE, PTO_TASK_WINDOW_SIZE);
        sim_start = sim_end - PTO_TASK_WINDOW_SIZE;
        total_to_simulate = PTO_TASK_WINDOW_SIZE;
    }

    // Local dependency state for simulation
    int32_t* remaining = (int32_t*)calloc((size_t)total_to_simulate, sizeof(int32_t));
    if (!remaining) {
        fprintf(stderr, "[PTO Simulator] ERROR: Failed to allocate remaining[]\n");
        return;
    }
    for (int32_t task_id = sim_start; task_id < sim_end; task_id++) {
        PendingTask* task = &rt->pend_task[PTO_TASK_SLOT(task_id)];
        if (!task->is_active || task->task_id != task_id) continue;
        remaining[task_id - sim_start] = task->fanin_count;
        task->is_complete = false;
        task->end_cycle = 0;
        task->earliest_start_cycle = 0;
    }
    
    int tasks_completed = 0;
    int max_iterations = total_to_simulate * 2;  // Safety limit
    int iteration = 0;
    
    while (tasks_completed < total_to_simulate && iteration < max_iterations) {
        iteration++;
        bool made_progress = false;
        
        // Find a ready task (remaining deps == 0 and not complete)
        for (int32_t task_id = sim_start; task_id < sim_end; task_id++) {
            PendingTask* task = &rt->pend_task[PTO_TASK_SLOT(task_id)];
            
            if (!task->is_active || task->task_id != task_id) continue;
            if (task->is_complete) continue;
            if (remaining[task_id - sim_start] > 0) continue;
            
            // Found a ready task - simulate its execution
            const char* func_name = task->func_name ? task->func_name : "unknown";
            
            // Estimate cycle cost
            int64_t cycle_cost = task->cycle_func 
                ? task->cycle_func(NULL, 0) 
                : pto_estimate_cycle_cost(func_name);
            
            // Assign to worker based on task type (dual-queue mode) or simple load balancing
            int best_worker = 0;
            if (rt->dual_queue_mode && rt->num_vector_workers > 0 && rt->num_cube_workers > 0) {
                // Dual-queue mode: respect is_cube flag
                int worker_start, worker_end;
                if (task->is_cube) {
                    // Cube task → assign to cube workers only
                    worker_start = rt->num_vector_workers;
                    worker_end = NUM_WORKERS;
                } else {
                    // Vector task → assign to vector workers only
                    worker_start = 0;
                    worker_end = rt->num_vector_workers;
                }
                
                // Find worker with lowest cycle count within the appropriate range
                best_worker = worker_start;
                for (int w = worker_start + 1; w < worker_end; w++) {
                    if (worker_cycles[w] < worker_cycles[best_worker]) {
                        best_worker = w;
                    }
                }
            } else {
                // Single-queue mode: simple load balancing across all workers
                for (int w = 1; w < NUM_WORKERS; w++) {
                    if (worker_cycles[w] < worker_cycles[best_worker]) {
                        best_worker = w;
                    }
                }
            }
            
            // Calculate actual start time (max of worker availability and task's earliest start)
            int64_t task_earliest = task->earliest_start_cycle;
            int64_t actual_start = (worker_cycles[best_worker] > task_earliest) 
                                  ? worker_cycles[best_worker] 
                                  : task_earliest;
            int64_t actual_end = actual_start + cycle_cost;
            
            // Update worker state
            worker_cycles[best_worker] = actual_end;
            tasks_per_worker[best_worker]++;
            
            // Record trace
            if (pto_global_trace) {
                pto_trace_record_with_time(best_worker, func_name, actual_start, actual_end);
            }
            
            // Update task state
            task->end_cycle = actual_end;
            task->is_complete = true;
            tasks_completed++;
            made_progress = true;
            
            // Update dependencies - reduce remaining deps of dependent tasks
            int32_t off = task->fanout_head;
            int32_t seen = 0;
            while (off != 0 && seen < task->fanout_consumer_count) {
                int32_t dep_id = rt->dep_list_pool[off].task_id;
                if (dep_id >= sim_start && dep_id < sim_end) {
                    PendingTask* dep_task = &rt->pend_task[PTO_TASK_SLOT(dep_id)];
                    if (dep_task->is_active && dep_task->task_id == dep_id) {
                        int32_t idx = dep_id - sim_start;
                        if (remaining[idx] > 0) remaining[idx]--;
                        if (actual_end > dep_task->earliest_start_cycle) {
                            dep_task->earliest_start_cycle = actual_end;
                        }
                    }
                }
                off = rt->dep_list_pool[off].next_offset;
                seen++;
            }
            
            DEBUG_PRINT("[Sim] Task %d: %s on Worker %d, cycles=[%lld, %lld], cost=%lld\n",
                   task_id, func_name, best_worker, 
                   (long long)actual_start, (long long)actual_end, (long long)cycle_cost);
        }
        
        if (!made_progress) {
            // Check if there are incomplete tasks with non-zero fanin (deadlock)
            int waiting_tasks = 0;
            for (int32_t i = sim_start; i < sim_end; i++) {
                if (!rt->pend_task[PTO_TASK_SLOT(i)].is_complete) {
                    waiting_tasks++;
                }
            }
            if (waiting_tasks > 0) {
                fprintf(stderr, "[PTO Simulator] WARNING: %d tasks waiting with dependencies - possible deadlock\n", 
                        waiting_tasks);
            }
            break;
        }
    }
    
    // Calculate statistics
    int64_t max_cycle = 0;
    for (int w = 0; w < NUM_WORKERS; w++) {
        if (worker_cycles[w] > max_cycle) {
            max_cycle = worker_cycles[w];
        }
    }
    
    printf("\n[PTO Simulator] ======== Simulation Complete ========\n");
    printf("  Tasks simulated: %d / %d\n", tasks_completed, total_to_simulate);
    printf("  Makespan (total cycles): %lld\n", (long long)max_cycle);
    
    // Count active workers (those that executed at least one task)
    int active_workers = 0;
    for (int w = 0; w < NUM_WORKERS; w++) {
        if (tasks_per_worker[w] > 0) active_workers++;
    }
    
    printf("  Active workers: %d / %d\n", active_workers, NUM_WORKERS);
    printf("  Worker utilization (top workers):\n");
    
    // Only show workers that did work (limit output for large worker counts)
    int shown = 0;
    int max_to_show = (active_workers > 20) ? 10 : active_workers;
    for (int w = 0; w < NUM_WORKERS && shown < max_to_show; w++) {
        if (tasks_per_worker[w] > 0) {
            double util = max_cycle > 0 ? (100.0 * worker_cycles[w] / max_cycle) : 0;
            printf("    Worker %d: %lld cycles (%d tasks, %.1f%% utilization)\n",
                   w, (long long)worker_cycles[w], tasks_per_worker[w], util);
            shown++;
        }
    }
    if (active_workers > max_to_show) {
        printf("    ... and %d more workers\n", active_workers - max_to_show);
    }
    printf("\n");
    
    rt->total_tasks_completed = tasks_completed;
    free(remaining);
}
