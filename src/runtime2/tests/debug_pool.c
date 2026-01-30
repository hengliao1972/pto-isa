#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "../pto_runtime2.h"
#include "../pto_runtime2_threaded.h"
#include "../pto_runtime2_types.h"
#include "../pto_scheduler.h"
#include "../pto_orchestrator.h"

int main() {
    // Use default pool size
    PTO2RuntimeThreaded* rt = pto2_runtime_create_threaded_custom(4, 4, true, 16384, 64*1024*1024, 65536);
    PTO2Runtime* base = (PTO2Runtime*)rt;
    PTO2DepListPool* dep_pool = &base->orchestrator.dep_pool;
    
    printf("Dep pool: capacity=%d, top=%d\n", dep_pool->capacity, dep_pool->top);
    
    float* A = calloc(1024*1024, sizeof(float));
    float* B = calloc(1024*1024, sizeof(float));
    float* C = calloc(1024*1024, sizeof(float));
    float* P = calloc(1024*1024, sizeof(float));
    
    // Submit tasks and monitor pool
    int tasks = 0;
    for (int batch = 0; batch < 8; batch++) {
        pto2_rt_scope_begin(base);
        for (int m = 0; m < 8; m++) {
            for (int n = 0; n < 8; n++) {
                pto2_rt_scope_begin(base);
                for (int k = 0; k < 8; k++) {
                    int a_idx = batch * 64 + m * 8 + k;
                    int b_idx = batch * 64 + k * 8 + n;
                    int c_idx = batch * 64 + m * 8 + n;
                    
                    PTO2TaskParam gemm_params[3] = {
                        PTO2_INPUT(A, a_idx, 128),
                        PTO2_INPUT(B, b_idx, 128),
                        PTO2_OUTPUT(P, c_idx, 128)
                    };
                    pto2_rt_submit_task(base, 0, PTO2_WORKER_CUBE, NULL, "gemm", gemm_params, 3);
                    tasks++;
                    
                    PTO2TaskParam add_params[3] = {
                        PTO2_INPUT(C, c_idx, 128),
                        PTO2_INPUT(P, c_idx, 128),
                        PTO2_OUTPUT(C, c_idx, 128)
                    };
                    pto2_rt_submit_task(base, 1, PTO2_WORKER_VECTOR, NULL, "add", add_params, 3);
                    tasks++;
                }
                pto2_rt_scope_end(base);
            }
        }
        pto2_rt_scope_end(base);
        
        printf("After batch %d: tasks=%d, dep_pool.top=%d (%.1f%% used)\n",
               batch, tasks, dep_pool->top, 100.0 * dep_pool->top / dep_pool->capacity);
        
        if (dep_pool->top >= dep_pool->capacity) {
            printf("  WARNING: Pool wrapped around!\n");
        }
    }
    
    printf("\nFinal: tasks=%d, dep_pool.top=%d, capacity=%d\n",
           tasks, dep_pool->top, dep_pool->capacity);
    
    pto2_runtime_destroy_threaded(rt);
    free(A); free(B); free(C); free(P);
    return 0;
}
