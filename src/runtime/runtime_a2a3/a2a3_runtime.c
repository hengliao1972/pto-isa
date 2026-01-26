/**
 * PTO Runtime - Ascend A2/A3 Runtime API Implementation
 * 
 * This file implements the public A2A3 Runtime API defined in a2a3_runtime_api.h.
 * It integrates the SO loader, orchestration, dependency resolution, and worker
 * threads to provide a complete runtime system.
 */

#define _POSIX_C_SOURCE 199309L

#include "a2a3_runtime_api.h"
#include "host/a2a3_so_loader.h"
#include "host/a2a3_host.h"
#include "core/a2a3_core_worker.h"
#include "orchestration/a2a3_orchestration.h"
#include "../pto_runtime_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// =============================================================================
// Internal State
// =============================================================================

static PTORuntime* g_runtime = NULL;
static A2A3RuntimeConfig g_config;
static A2A3OrchFunc g_orch_func_ptr = NULL;
static bool g_initialized = false;
static A2A3RuntimeStats g_stats;

// Error messages
static const char* g_error_messages[] = {
    "Success",                           // 0
    "Invalid configuration",             // -1
    "Failed to load .so file",           // -2
    "Function not found",                // -3
    "Memory allocation failed",          // -4
    "Thread creation failed",            // -5
    "Runtime not initialized",           // -6
    "Runtime already initialized",       // -7
};

// =============================================================================
// Error Handling
// =============================================================================

const char* a2a3_runtime_error_string(int error_code) {
    if (error_code > 0 || error_code < -7) {
        return "Unknown error";
    }
    return g_error_messages[-error_code];
}

// =============================================================================
// Runtime Lifecycle
// =============================================================================

int a2a3_runtime_init(A2A3RuntimeConfig* config) {
    if (g_initialized) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: Runtime already initialized\n");
        return A2A3_ERROR_ALREADY_INIT;
    }
    
    if (!config) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: NULL config\n");
        return A2A3_ERROR_INVALID_CONFIG;
    }
    
    // Copy configuration
    memcpy(&g_config, config, sizeof(A2A3RuntimeConfig));
    
    // Apply defaults
    if (g_config.num_orch_threads < 1) g_config.num_orch_threads = A2A3_DEFAULT_ORCH_THREADS;
    if (g_config.num_dep_threads < 1) g_config.num_dep_threads = A2A3_DEFAULT_DEP_THREADS;
    if (g_config.num_aiv_workers < 1) g_config.num_aiv_workers = A2A3_DEFAULT_AIV_WORKERS;
    if (g_config.num_aic_workers < 1) g_config.num_aic_workers = A2A3_DEFAULT_AIC_WORKERS;
    
    printf("[A2A3 Runtime] ================================================\n");
    printf("[A2A3 Runtime] Initializing Ascend A2/A3 Runtime\n");
    printf("[A2A3 Runtime]   Orchestration threads: %d\n", g_config.num_orch_threads);
    printf("[A2A3 Runtime]   Dependency threads:    %d\n", g_config.num_dep_threads);
    printf("[A2A3 Runtime]   AIV workers:           %d\n", g_config.num_aiv_workers);
    printf("[A2A3 Runtime]   AIC workers:           %d\n", g_config.num_aic_workers);
    printf("[A2A3 Runtime] ================================================\n");
    
    // Initialize SO loader
    a2a3_so_loader_init();
    
    // Load orchestration function if path provided
    if (g_config.orchestration_so_path) {
        g_orch_func_ptr = a2a3_load_orchestration(g_config.orchestration_so_path, 
                                                   g_config.orchestration_func_name);
        if (!g_orch_func_ptr) {
            fprintf(stderr, "[A2A3 Runtime] ERROR: Failed to load orchestration from %s\n",
                    g_config.orchestration_so_path);
            a2a3_so_loader_cleanup();
            return A2A3_ERROR_SO_LOAD_FAILED;
        }
    }
    
    // Load InCore AIV functions
    if (g_config.incore_aiv_dir) {
        int count = a2a3_load_incore_dir(g_config.incore_aiv_dir, false);
        if (count < 0) {
            fprintf(stderr, "[A2A3 Runtime] WARNING: Failed to load AIV functions from %s\n",
                    g_config.incore_aiv_dir);
        }
    }
    
    // Load InCore AIC functions
    if (g_config.incore_aic_dir) {
        int count = a2a3_load_incore_dir(g_config.incore_aic_dir, true);
        if (count < 0) {
            fprintf(stderr, "[A2A3 Runtime] WARNING: Failed to load AIC functions from %s\n",
                    g_config.incore_aic_dir);
        }
    }
    
    // Allocate runtime context
    g_runtime = (PTORuntime*)malloc(sizeof(PTORuntime));
    if (!g_runtime) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: Failed to allocate runtime\n");
        a2a3_so_loader_cleanup();
        return A2A3_ERROR_MEMORY_ALLOC;
    }
    
    // Initialize runtime
    pto_runtime_init(g_runtime);
    a2a3_orch_init(g_runtime, g_config.num_aiv_workers, g_config.num_aic_workers);
    
    g_runtime->num_workers = g_config.num_aiv_workers + g_config.num_aic_workers;
    g_runtime->num_vector_workers = g_config.num_aiv_workers;
    g_runtime->num_cube_workers = g_config.num_aic_workers;
    g_runtime->num_dep_resolvers = g_config.num_dep_threads;
    g_runtime->shutdown_requested = false;
    g_runtime->execution_started = false;
    g_runtime->orchestration_complete = false;
    
    // Clear stats
    memset(&g_stats, 0, sizeof(g_stats));
    g_stats.num_incore_funcs_loaded = a2a3_get_incore_count();
    
    g_initialized = true;
    
    printf("[A2A3 Runtime] Initialization complete\n");
    printf("[A2A3 Runtime]   Loaded %d InCore functions\n", g_stats.num_incore_funcs_loaded);
    
    return A2A3_SUCCESS;
}

int a2a3_runtime_execute(void* user_data) {
    printf("[A2A3 Runtime] === a2a3_runtime_execute() called ===\n");
    fflush(stdout);
    
    if (!g_initialized || !g_runtime) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: Runtime not initialized\n");
        return A2A3_ERROR_NOT_INITIALIZED;
    }
    
    printf("[A2A3 Runtime] Step 1: Checking orchestration function...\n");
    fflush(stdout);
    
    if (!g_orch_func_ptr) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: No orchestration function loaded\n");
        return A2A3_ERROR_FUNC_NOT_FOUND;
    }
    
    printf("[A2A3 Runtime] Step 2: Orchestration function OK, starting execution...\n");
    fflush(stdout);
    
    // =========================================================================
    // DEBUG_ORCHESTRATION mode: Only run orchestration, skip task execution
    // =========================================================================
    if (g_config.debug_orchestration_only) {
        printf("[A2A3 Runtime] *** DEBUG_ORCHESTRATION MODE ***\n");
        printf("[A2A3 Runtime] Running orchestration function only (no task execution)\n");
        fflush(stdout);
        
        struct timespec start_time, end_time;
        clock_gettime(CLOCK_MONOTONIC, &start_time);
        
        // Call orchestration function directly (not in separate thread)
        void* orch_user_data = user_data ? user_data : g_config.user_data;
        printf("[A2A3 Runtime] Calling orchestration function...\n");
        fflush(stdout);
        
        g_orch_func_ptr(g_runtime, orch_user_data);
        
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        
        // Mark orchestration as complete
        g_runtime->orchestration_complete = true;
        
        // Collect stats
        g_stats.total_tasks_scheduled = g_runtime->total_tasks_scheduled;
        g_stats.total_tasks_completed = 0;  // No tasks executed in debug mode
        g_stats.total_execution_time_ms = 
            (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
            (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;
        
        printf("\n");
        printf("=======================================================\n");
        printf("  DEBUG_ORCHESTRATION Results\n");
        printf("=======================================================\n");
        printf("  Number of Tasks submitted by Orchestration function: %lld\n", 
               (long long)g_stats.total_tasks_scheduled);
        printf("  Orchestration time: %.3f ms\n", g_stats.total_execution_time_ms);
        if (g_stats.total_execution_time_ms > 0) {
            printf("  Task submission rate: %.2f tasks/ms\n", 
                   g_stats.total_tasks_scheduled / g_stats.total_execution_time_ms);
        }
        printf("=======================================================\n");
        printf("\n");
        fflush(stdout);
        
        return A2A3_SUCCESS;
    }
    
    // =========================================================================
    // Normal execution mode
    // =========================================================================
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    // =========================================================================
    // 1. Spawn AIV (Vector) workers
    // =========================================================================
    printf("[A2A3 Runtime] Step 3: Spawning %d AIV workers...\n", g_config.num_aiv_workers);
    fflush(stdout);
    for (int i = 0; i < g_config.num_aiv_workers; i++) {
        A2A3WorkerContext* ctx = (A2A3WorkerContext*)malloc(sizeof(A2A3WorkerContext));
        if (!ctx) {
            fprintf(stderr, "[A2A3 Runtime] ERROR: Failed to allocate AIV worker context\n");
            g_runtime->shutdown_requested = true;
            return A2A3_ERROR_MEMORY_ALLOC;
        }
        ctx->rt = g_runtime;
        ctx->worker_id = i;
        ctx->is_cube_worker = false;
        
        if (pthread_create(&g_runtime->workers[i], NULL, a2a3_vector_worker_func, ctx) != 0) {
            fprintf(stderr, "[A2A3 Runtime] ERROR: Failed to create AIV worker %d\n", i);
            free(ctx);
            g_runtime->shutdown_requested = true;
            return A2A3_ERROR_THREAD_CREATE;
        }
    }
    
    printf("[A2A3 Runtime] Step 4: AIV workers spawned successfully.\n");
    fflush(stdout);
    
    // =========================================================================
    // 2. Spawn AIC (Cube) workers
    // =========================================================================
    printf("[A2A3 Runtime] Step 5: Spawning %d AIC workers...\n", g_config.num_aic_workers);
    fflush(stdout);
    for (int i = 0; i < g_config.num_aic_workers; i++) {
        int worker_idx = g_config.num_aiv_workers + i;
        A2A3WorkerContext* ctx = (A2A3WorkerContext*)malloc(sizeof(A2A3WorkerContext));
        if (!ctx) {
            fprintf(stderr, "[A2A3 Runtime] ERROR: Failed to allocate AIC worker context\n");
            g_runtime->shutdown_requested = true;
            return A2A3_ERROR_MEMORY_ALLOC;
        }
        ctx->rt = g_runtime;
        ctx->worker_id = worker_idx;
        ctx->is_cube_worker = true;
        
        if (pthread_create(&g_runtime->workers[worker_idx], NULL, a2a3_cube_worker_func, ctx) != 0) {
            fprintf(stderr, "[A2A3 Runtime] ERROR: Failed to create AIC worker %d\n", i);
            free(ctx);
            g_runtime->shutdown_requested = true;
            return A2A3_ERROR_THREAD_CREATE;
        }
    }
    
    printf("[A2A3 Runtime] Step 6: AIC workers spawned successfully.\n");
    fflush(stdout);
    
    // =========================================================================
    // 3. Spawn Dependency Resolver threads
    // =========================================================================
    printf("[A2A3 Runtime] Step 7: Spawning %d dependency resolver threads...\n", g_config.num_dep_threads);
    fflush(stdout);
    for (int i = 0; i < g_config.num_dep_threads; i++) {
        A2A3DepResolverContext* ctx = (A2A3DepResolverContext*)malloc(sizeof(A2A3DepResolverContext));
        if (!ctx) {
            fprintf(stderr, "[A2A3 Runtime] ERROR: Failed to allocate dep resolver context\n");
            g_runtime->shutdown_requested = true;
            return A2A3_ERROR_MEMORY_ALLOC;
        }
        ctx->rt = g_runtime;
        ctx->thread_id = i;
        
        if (pthread_create(&g_runtime->dep_resolvers[i], NULL, a2a3_dep_resolver_func, ctx) != 0) {
            fprintf(stderr, "[A2A3 Runtime] ERROR: Failed to create dep resolver %d\n", i);
            free(ctx);
            g_runtime->shutdown_requested = true;
            return A2A3_ERROR_THREAD_CREATE;
        }
    }
    
    printf("[A2A3 Runtime] Step 8: Dependency resolver threads spawned successfully.\n");
    fflush(stdout);
    
    // =========================================================================
    // 4. Spawn Orchestration thread
    // =========================================================================
    printf("[A2A3 Runtime] Step 9: Spawning orchestration thread...\n");
    fflush(stdout);
    A2A3OrchContext* orch_ctx = (A2A3OrchContext*)malloc(sizeof(A2A3OrchContext));
    if (!orch_ctx) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: Failed to allocate orch context\n");
        g_runtime->shutdown_requested = true;
        return A2A3_ERROR_MEMORY_ALLOC;
    }
    orch_ctx->rt = g_runtime;
    orch_ctx->orch_func = (void*)g_orch_func_ptr;
    orch_ctx->user_data = user_data ? user_data : g_config.user_data;
    
    if (pthread_create(&g_runtime->orch_thread, NULL, a2a3_orch_thread_func, orch_ctx) != 0) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: Failed to create orchestration thread\n");
        free(orch_ctx);
        g_runtime->shutdown_requested = true;
        return A2A3_ERROR_THREAD_CREATE;
    }
    
    printf("[A2A3 Runtime] Step 10: Orchestration thread spawned successfully.\n");
    fflush(stdout);
    
    // =========================================================================
    // 5. Enable Execution
    // =========================================================================
    // CRITICAL: Set execution_started = true so workers can start processing tasks!
    // Without this, workers will wait forever because can_execute = false.
    printf("[A2A3 Runtime] Step 11: Enabling task execution...\n");
    fflush(stdout);
    
    pthread_mutex_lock(&g_runtime->queue_mutex);
    g_runtime->execution_started = true;
    // Wake up all workers that might be waiting
    pthread_cond_broadcast(&g_runtime->vector_queue_not_empty);
    pthread_cond_broadcast(&g_runtime->cube_queue_not_empty);
    pthread_mutex_unlock(&g_runtime->queue_mutex);
    
    printf("[A2A3 Runtime] Step 12: Task execution enabled.\n");
    fflush(stdout);
    
    // =========================================================================
    // 6. Wait for completion
    // =========================================================================
    printf("[A2A3 Runtime] Step 13: All threads started. Waiting for execution to complete...\n");
    fflush(stdout);
    
    struct timespec poll_interval = {0, 10000000};  // 10ms
    int64_t last_reported = 0;
    
    while (1) {
        pthread_mutex_lock(&g_runtime->task_mutex);
        bool all_done = (g_runtime->orchestration_complete && 
                        g_runtime->total_tasks_completed >= g_runtime->total_tasks_scheduled);
        int64_t completed = g_runtime->total_tasks_completed;
        int64_t scheduled = g_runtime->total_tasks_scheduled;
        pthread_mutex_unlock(&g_runtime->task_mutex);
        
        if (all_done && scheduled > 0) {
            printf("[A2A3 Runtime] All %lld tasks completed!\n", (long long)completed);
            break;
        }
        
        // Progress reporting
        if (completed > last_reported + 1000) {
            if (scheduled > 0) {
                printf("[A2A3 Runtime] Progress: %lld / %lld (%.1f%%)\n",
                       (long long)completed, (long long)scheduled,
                       100.0 * completed / scheduled);
            }
            last_reported = completed;
        }
        
        nanosleep(&poll_interval, NULL);
    }
    
    // =========================================================================
    // 7. Shutdown threads
    // =========================================================================
    printf("[A2A3 Runtime] Step 14: Shutting down threads...\n");
    fflush(stdout);
    g_runtime->shutdown_requested = true;
    
    // Wake up all waiting threads
    pthread_mutex_lock(&g_runtime->queue_mutex);
    pthread_cond_broadcast(&g_runtime->vector_queue_not_empty);
    pthread_cond_broadcast(&g_runtime->cube_queue_not_empty);
    pthread_mutex_unlock(&g_runtime->queue_mutex);
    
    // Wait for orchestration thread
    pthread_join(g_runtime->orch_thread, NULL);
    
    // Wait for dep resolvers
    for (int i = 0; i < g_config.num_dep_threads; i++) {
        pthread_join(g_runtime->dep_resolvers[i], NULL);
    }
    
    // Wait for workers
    int total_workers = g_config.num_aiv_workers + g_config.num_aic_workers;
    for (int i = 0; i < total_workers; i++) {
        pthread_join(g_runtime->workers[i], NULL);
    }
    
    // =========================================================================
    // 8. Collect stats
    // =========================================================================
    printf("[A2A3 Runtime] Step 15: Collecting statistics...\n");
    fflush(stdout);
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    
    g_stats.total_tasks_scheduled = g_runtime->total_tasks_scheduled;
    g_stats.total_tasks_completed = g_runtime->total_tasks_completed;
    g_stats.total_execution_time_ms = 
        (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
        (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;
    
    // Count tasks by type (approximation based on queue stats)
    g_stats.aiv_tasks_executed = g_runtime->vector_ready_count;
    g_stats.aic_tasks_executed = g_runtime->cube_ready_count;
    
    printf("[A2A3 Runtime] ================================================\n");
    printf("[A2A3 Runtime] Execution Complete\n");
    printf("[A2A3 Runtime]   Total tasks:      %lld\n", (long long)g_stats.total_tasks_scheduled);
    printf("[A2A3 Runtime]   Completed:        %lld\n", (long long)g_stats.total_tasks_completed);
    printf("[A2A3 Runtime]   Execution time:   %.2f ms\n", g_stats.total_execution_time_ms);
    if (g_stats.total_execution_time_ms > 0) {
        printf("[A2A3 Runtime]   Throughput:       %.2f tasks/ms\n", 
               g_stats.total_tasks_completed / g_stats.total_execution_time_ms);
    }
    printf("[A2A3 Runtime] ================================================\n");
    
    return A2A3_SUCCESS;
}

void a2a3_runtime_finalize(void) {
    if (!g_initialized) return;
    
    printf("[A2A3 Runtime] Finalizing...\n");
    
    // Cleanup SO loader
    a2a3_so_loader_cleanup();
    
    // Cleanup runtime
    if (g_runtime) {
        pto_runtime_shutdown(g_runtime);
        free(g_runtime);
        g_runtime = NULL;
    }
    
    g_orch_func_ptr = NULL;
    g_initialized = false;
    
    printf("[A2A3 Runtime] Finalized\n");
}

// =============================================================================
// Memory Management
// =============================================================================

void* a2a3_runtime_malloc(size_t size_bytes) {
    if (!g_initialized) {
        fprintf(stderr, "[A2A3 Runtime] ERROR: Runtime not initialized\n");
        return NULL;
    }
    return a2a3_host_malloc(size_bytes);
}

void a2a3_runtime_free(void* ptr) {
    a2a3_host_free(ptr);
}

int a2a3_runtime_copy_to_device(void* dst_device, const void* src_host, size_t size_bytes) {
    if (!g_initialized) {
        return A2A3_ERROR_NOT_INITIALIZED;
    }
    return a2a3_host_copy_to_device(dst_device, src_host, size_bytes);
}

int a2a3_runtime_copy_from_device(void* dst_host, const void* src_device, size_t size_bytes) {
    if (!g_initialized) {
        return A2A3_ERROR_NOT_INITIALIZED;
    }
    return a2a3_host_copy_from_device(dst_host, src_device, size_bytes);
}

// =============================================================================
// InCore Function Registry
// =============================================================================

int a2a3_runtime_register_incore(const char* func_name, A2A3InCoreFunc func_ptr, bool is_cube) {
    return a2a3_register_incore(func_name, func_ptr, is_cube);
}

A2A3InCoreFunc a2a3_runtime_lookup_incore(const char* func_name) {
    return a2a3_lookup_incore(func_name);
}

// =============================================================================
// Statistics
// =============================================================================

void a2a3_runtime_get_stats(A2A3RuntimeStats* stats) {
    if (stats) {
        memcpy(stats, &g_stats, sizeof(A2A3RuntimeStats));
    }
}

void a2a3_runtime_print_stats(void) {
    printf("\n");
    printf("A2A3 Runtime Statistics\n");
    printf("========================\n");
    printf("Tasks Scheduled:     %lld\n", (long long)g_stats.total_tasks_scheduled);
    printf("Tasks Completed:     %lld\n", (long long)g_stats.total_tasks_completed);
    printf("AIV Tasks:           %lld\n", (long long)g_stats.aiv_tasks_executed);
    printf("AIC Tasks:           %lld\n", (long long)g_stats.aic_tasks_executed);
    printf("Execution Time:      %.2f ms\n", g_stats.total_execution_time_ms);
    printf("InCore Functions:    %d\n", g_stats.num_incore_funcs_loaded);
    printf("\n");
}

bool a2a3_runtime_is_initialized(void) {
    return g_initialized;
}
