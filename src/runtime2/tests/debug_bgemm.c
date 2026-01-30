#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "../pto_runtime2.h"
#include "../pto_runtime2_threaded.h"
#include "../pto_runtime2_types.h"

int main() {
    printf("=== Debug BGEMM Test ===\n");
    
    // Small test: 2 batches, 2x2 tiles, 2 k
    // Total: 2*2*2*2*2 = 32 tasks
    int batch = 2, m = 2, n = 2, k = 2;
    int total_tasks = batch * m * n * k * 2;
    int window = 64;
    
    printf("Total tasks: %d, Window: %d\n\n", total_tasks, window);
    
    PTO2RuntimeThreaded* rt = pto2_runtime_create_threaded_custom(
        2, 2, true, window, PTO2_HEAP_SIZE, PTO2_DEP_LIST_POOL_SIZE);
    
    if (!rt) {
        fprintf(stderr, "Failed to create runtime\n");
        return 1;
    }
    
    float* A = calloc(1024, sizeof(float));
    float* B = calloc(1024, sizeof(float));
    float* C = calloc(1024, sizeof(float));
    float* P = calloc(1024, sizeof(float));
    
    printf("Submitting tasks...\n");
    
    // Submit tasks synchronously (not threaded for debugging)
    PTO2Runtime* base = (PTO2Runtime*)rt;
    int task_count = 0;
    
    for (int b = 0; b < batch; b++) {
        pto2_rt_scope_begin(base);
        for (int mi = 0; mi < m; mi++) {
            for (int ni = 0; ni < n; ni++) {
                pto2_rt_scope_begin(base);
                for (int ki = 0; ki < k; ki++) {
                    int idx = b*m*n + mi*n + ni;
                    
                    PTO2TaskParam gemm_params[3] = {
                        PTO2_INPUT(A, idx, 128),
                        PTO2_INPUT(B, idx, 128),
                        PTO2_OUTPUT(P, idx, 128)
                    };
                    pto2_rt_submit_task(base, 0, PTO2_WORKER_CUBE, NULL,
                                        "gemm_tile", gemm_params, 3);
                    task_count++;
                    printf("  Submitted gemm task %d\n", task_count-1);
                    
                    PTO2TaskParam add_params[3] = {
                        PTO2_INPUT(C, idx, 128),
                        PTO2_INPUT(P, idx, 128),
                        PTO2_OUTPUT(C, idx, 128)
                    };
                    pto2_rt_submit_task(base, 1, PTO2_WORKER_VECTOR, NULL,
                                        "tile_add", add_params, 3);
                    task_count++;
                    printf("  Submitted add task %d\n", task_count-1);
                }
                pto2_rt_scope_end(base);
            }
        }
        pto2_rt_scope_end(base);
    }
    
    printf("\nSubmitted %d tasks total\n", task_count);
    printf("current_task_index: %d\n", base->sm_handle->header->current_task_index);
    
    // Now start threads
    printf("\nStarting threads...\n");
    pto2_runtime_start_threads(rt);
    
    // Wait a bit
    printf("Waiting 2 seconds...\n");
    sleep(2);
    
    // Check status
    printf("\nStatus after 2s:\n");
    printf("  last_task_alive: %d\n", base->sm_handle->header->last_task_alive);
    printf("  orchestrator_done: %d\n", base->sm_handle->header->orchestrator_done);
    
    // Mark done
    pto2_rt_orchestration_done(base);
    rt->thread_ctx.orchestrator_done = true;
    printf("\nMarked orchestration done\n");
    
    // Wait more
    sleep(2);
    printf("\nStatus after 4s:\n");
    printf("  last_task_alive: %d\n", base->sm_handle->header->last_task_alive);
    
    // Stop
    pto2_runtime_stop_threads(rt);
    pto2_runtime_destroy_threaded(rt);
    
    free(A); free(B); free(C); free(P);
    
    printf("\n=== Done ===\n");
    return 0;
}
