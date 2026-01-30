#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "../pto_runtime2.h"
#include "../pto_runtime2_threaded.h"
#include "../pto_runtime2_types.h"

volatile int submitted = 0;

void orch_func(PTO2Runtime* rt, void* arg) {
    float* A = calloc(1024, sizeof(float));
    float* P = calloc(1024, sizeof(float));
    
    // Submit 4096 tasks
    for (int b = 0; b < 8; b++) {
        pto2_rt_scope_begin(rt);
        for (int m = 0; m < 8; m++) {
            for (int n = 0; n < 8; n++) {
                pto2_rt_scope_begin(rt);
                for (int k = 0; k < 4; k++) {
                    int idx = b*64 + m*8 + n;
                    
                    PTO2TaskParam p[3] = {
                        PTO2_INPUT(A, idx, 128),
                        PTO2_INPUT(A, idx, 128),
                        PTO2_OUTPUT(P, idx, 128)
                    };
                    pto2_rt_submit_task(rt, 0, PTO2_WORKER_CUBE, NULL, "gemm", p, 3);
                    submitted++;
                    
                    PTO2TaskParam p2[3] = {
                        PTO2_INPUT(P, idx, 128),
                        PTO2_INPUT(P, idx, 128),
                        PTO2_OUTPUT(P, idx, 128)
                    };
                    pto2_rt_submit_task(rt, 1, PTO2_WORKER_VECTOR, NULL, "add", p2, 3);
                    submitted++;
                    
                    if (submitted % 512 == 0) {
                        printf("Submitted %d tasks, last_alive=%d\n", 
                               submitted, rt->sm_handle->header->last_task_alive);
                    }
                }
                pto2_rt_scope_end(rt);
            }
        }
        pto2_rt_scope_end(rt);
        printf("Batch %d done, submitted=%d, last_alive=%d\n", 
               b, submitted, rt->sm_handle->header->last_task_alive);
    }
    
    free(A); free(P);
}

int main() {
    printf("Creating runtime with window=8192...\n");
    PTO2RuntimeThreaded* rt = pto2_runtime_create_threaded_custom(
        4, 4, true, 8192, PTO2_HEAP_SIZE, PTO2_DEP_LIST_POOL_SIZE);
    
    printf("Running threaded...\n");
    pto2_runtime_run_threaded(rt, orch_func, NULL);
    
    printf("Done! tasks_consumed=%lld\n", 
           (long long)rt->base.scheduler.tasks_consumed);
    
    pto2_runtime_destroy_threaded(rt);
    return 0;
}
