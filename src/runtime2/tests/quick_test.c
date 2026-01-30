#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "../pto_runtime2.h"
#include "../pto_runtime2_threaded.h"

int main() {
    printf("Creating runtime with window=4096...\n"); fflush(stdout);
    PTO2RuntimeThreaded* rt = pto2_runtime_create_threaded_custom(4, 4, true, 4096, 64*1024*1024, 65536);
    if (!rt) {
        printf("Failed to create runtime!\n");
        return 1;
    }
    printf("Runtime created. Window=%d, mask=%d\n", 
           rt->base.sm_handle->header->task_window_size,
           rt->base.scheduler.task_window_mask);
    fflush(stdout);
    
    PTO2Runtime* base = (PTO2Runtime*)rt;
    float* A = calloc(1024, sizeof(float));
    float* P = calloc(1024, sizeof(float));
    
    printf("Starting threads...\n"); fflush(stdout);
    pto2_runtime_start_threads(rt);
    
    // 4x4x4x4 = 512 tasks
    printf("Submitting 512 tasks...\n"); fflush(stdout);
    for (int b = 0; b < 4; b++) {
        pto2_rt_scope_begin(base);
        for (int m = 0; m < 4; m++) {
            for (int n = 0; n < 4; n++) {
                pto2_rt_scope_begin(base);
                for (int k = 0; k < 4; k++) {
                    int idx = b*16 + m*4 + n;
                    PTO2TaskParam gemm_p[3] = {
                        PTO2_INPUT(A, idx, 128),
                        PTO2_INPUT(A, idx, 128),
                        PTO2_OUTPUT(P, idx, 128)
                    };
                    pto2_rt_submit_task(base, 0, PTO2_WORKER_CUBE, NULL, "gemm", gemm_p, 3);
                    
                    PTO2TaskParam add_p[3] = {
                        PTO2_INPUT(P, idx, 128),
                        PTO2_INPUT(P, idx, 128),
                        PTO2_OUTPUT(P, idx, 128)
                    };
                    pto2_rt_submit_task(base, 1, PTO2_WORKER_VECTOR, NULL, "add", add_p, 3);
                }
                pto2_rt_scope_end(base);
            }
        }
        pto2_rt_scope_end(base);
        printf("Batch %d done\n", b); fflush(stdout);
    }
    
    printf("Marking done...\n"); fflush(stdout);
    pto2_rt_orchestration_done(base);
    rt->thread_ctx.orchestrator_done = true;
    
    printf("Waiting for completion...\n"); fflush(stdout);
    for (int i = 0; i < 10; i++) {
        usleep(500000);
        printf("  consumed=%lld\n", (long long)rt->base.scheduler.tasks_consumed);
        fflush(stdout);
        if (rt->base.scheduler.tasks_consumed >= 512) break;
    }
    
    printf("Stopping threads...\n"); fflush(stdout);
    pto2_runtime_stop_threads(rt);
    pto2_runtime_destroy_threaded(rt);
    free(A); free(P);
    printf("Done!\n");
    return 0;
}
