/**
 * BGEMM Test for Runtime2 (Multi-Threaded)
 * 
 * Tests the runtime2 system with a batched GEMM pattern:
 * For each batch:
 *   For each tile (m, n):
 *     For k in K:
 *       gemm_tile: P[m,n] = A[m,k] * B[k,n]
 *       tile_add:  C[m,n] += P[m,n]
 * 
 * Usage:
 *   ./test_bgemm_runtime2 [batch] [m] [n] [k] [window] [cube_workers] [vector_workers]
 * 
 * Examples:
 *   ./test_bgemm_runtime2 8 8 8 8 16384           # 8192 tasks, 4 cube + 4 vector
 *   ./test_bgemm_runtime2 8 8 8 8 16384 24 48     # 8192 tasks, 24 cube + 48 vector (A2A3)
 * 
 * Set task_window_size smaller than total_tasks to trigger flow control.
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../pto_runtime2.h"
#include "../pto_runtime2_threaded.h"
#include "../pto_runtime2_types.h"

// Test configuration
#define TEST_BATCH    4
#define TEST_M_TILES  4
#define TEST_N_TILES  4
#define TEST_K_TILES  4

// Default worker counts
#define DEFAULT_CUBE_WORKERS   4
#define DEFAULT_VECTOR_WORKERS 4

// =============================================================================
// BGEMM Orchestration Parameters
// =============================================================================

typedef struct {
    int batch;
    int m_tiles;
    int n_tiles;
    int k_tiles;
    float* A;
    float* B;
    float* C;
    float* P;
    int task_count;
} BgemmParams;

// =============================================================================
// BGEMM Orchestration Function (for multi-threaded mode)
// =============================================================================

static void bgemm_orchestration(PTO2Runtime* rt, void* arg) {
    BgemmParams* p = (BgemmParams*)arg;
    
    p->task_count = 0;
    
    for (int b = 0; b < p->batch; b++) {
        pto2_rt_scope_begin(rt);  // Batch scope
        
        for (int m = 0; m < p->m_tiles; m++) {
            for (int n = 0; n < p->n_tiles; n++) {
                pto2_rt_scope_begin(rt);  // Tile scope
                
                for (int k = 0; k < p->k_tiles; k++) {
                    // Task indices for dependency tracking
                    int a_idx = b * (p->m_tiles * p->k_tiles) + m * p->k_tiles + k;
                    int b_idx = b * (p->k_tiles * p->n_tiles) + k * p->n_tiles + n;
                    int c_idx = b * (p->m_tiles * p->n_tiles) + m * p->n_tiles + n;
                    
                    // gemm_tile: P = A * B (Cube operation)
                    PTO2TaskParam gemm_params[3] = {
                        PTO2_INPUT(p->A, a_idx, 128),
                        PTO2_INPUT(p->B, b_idx, 128),
                        PTO2_OUTPUT(p->P, c_idx, 128)
                    };
                    pto2_rt_submit_task(rt, 0, PTO2_WORKER_CUBE, NULL,
                                        "gemm_tile", gemm_params, 3);
                    p->task_count++;
                    
                    // tile_add: C += P (Vector operation)
                    PTO2TaskParam add_params[3] = {
                        PTO2_INPUT(p->C, c_idx, 128),
                        PTO2_INPUT(p->P, c_idx, 128),
                        PTO2_OUTPUT(p->C, c_idx, 128)
                    };
                    pto2_rt_submit_task(rt, 1, PTO2_WORKER_VECTOR, NULL,
                                        "tile_add", add_params, 3);
                    p->task_count++;
                }
                
                pto2_rt_scope_end(rt);  // End tile scope
            }
        }
        
        pto2_rt_scope_end(rt);  // End batch scope
    }
}

// =============================================================================
// Multi-Threaded Test
// =============================================================================

static int run_multi_threaded_test(int batch, int m_tiles, int n_tiles, int k_tiles, 
                                    int task_window_size, int cube_workers, int vector_workers) {
    int total_tasks = batch * m_tiles * n_tiles * k_tiles * 2;
    
    printf("=== BGEMM Runtime2 Test (Multi-Threaded) ===\n");
    printf("Configuration:\n");
    printf("  Batch:    %d\n", batch);
    printf("  M tiles:  %d\n", m_tiles);
    printf("  N tiles:  %d\n", n_tiles);
    printf("  K tiles:  %d\n", k_tiles);
    printf("  Total tasks: %d\n", total_tasks);
    printf("  Task window: %d\n", task_window_size);
    printf("  CUBE workers:   %d\n", cube_workers);
    printf("  VECTOR workers: %d\n", vector_workers);
    
    if (task_window_size < total_tasks) {
        printf("  *** FLOW CONTROL EXPECTED (window < tasks) ***\n");
    }
    printf("\n");
    
    // Create threaded runtime in simulation mode with custom task_window_size
    PTO2RuntimeThreaded* rt = pto2_runtime_create_threaded_custom(
        cube_workers, vector_workers, true,
        task_window_size, PTO2_HEAP_SIZE, PTO2_DEP_LIST_POOL_SIZE);
    
    if (!rt) {
        fprintf(stderr, "Failed to create threaded runtime\n");
        return 1;
    }
    
    // Allocate dummy tensors
    float* A = (float*)calloc(1024 * 1024, sizeof(float));
    float* B = (float*)calloc(1024 * 1024, sizeof(float));
    float* C = (float*)calloc(1024 * 1024, sizeof(float));
    float* P = (float*)calloc(1024 * 1024, sizeof(float));
    
    // Setup parameters
    BgemmParams params = {
        .batch = batch,
        .m_tiles = m_tiles,
        .n_tiles = n_tiles,
        .k_tiles = k_tiles,
        .A = A,
        .B = B,
        .C = C,
        .P = P,
        .task_count = 0
    };
    
    // Measure total time
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    printf("Running multi-threaded execution...\n");
    
    // Run with multi-threading (orchestrator in separate thread)
    pto2_runtime_run_threaded(rt, bgemm_orchestration, &params);
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double total_time_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                           (end.tv_nsec - start.tv_nsec) / 1000000.0;
    
    int64_t total_cycles = pto2_runtime_get_total_cycles(rt);
    
    // Print summary
    printf("\n=== Summary ===\n");
    printf("  Tasks:        %d\n", params.task_count);
    printf("  Total time:   %.3f ms\n", total_time_ms);
    printf("  Throughput:   %.2f tasks/ms\n", params.task_count / total_time_ms);
    printf("  Sim cycles:   %lld\n", (long long)total_cycles);
    
    // Print threaded stats
    pto2_runtime_print_threaded_stats(rt);
    
    // Write trace
    pto2_runtime_write_trace(rt, "bgemm_runtime2_threaded_trace.json");
    
    // Cleanup
    pto2_runtime_destroy_threaded(rt);
    free(A);
    free(B);
    free(C);
    free(P);
    
    printf("\n=== Test Complete ===\n");
    return 0;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    int batch = TEST_BATCH;
    int m_tiles = TEST_M_TILES;
    int n_tiles = TEST_N_TILES;
    int k_tiles = TEST_K_TILES;
    int task_window_size = PTO2_TASK_WINDOW_SIZE;
    int cube_workers = DEFAULT_CUBE_WORKERS;
    int vector_workers = DEFAULT_VECTOR_WORKERS;
    
    // Parse optional args: batch m n k window cube_workers vector_workers
    if (argc > 1) batch = atoi(argv[1]);
    if (argc > 2) m_tiles = atoi(argv[2]);
    if (argc > 3) n_tiles = atoi(argv[3]);
    if (argc > 4) k_tiles = atoi(argv[4]);
    if (argc > 5) task_window_size = atoi(argv[5]);
    if (argc > 6) cube_workers = atoi(argv[6]);
    if (argc > 7) vector_workers = atoi(argv[7]);
    
    // Ensure task_window_size is power of 2
    int tw = 1;
    while (tw < task_window_size) tw *= 2;
    task_window_size = tw;
    
    return run_multi_threaded_test(batch, m_tiles, n_tiles, k_tiles, task_window_size,
                                   cube_workers, vector_workers);
}
