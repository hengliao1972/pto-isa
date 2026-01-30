#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "../pto_runtime2.h"
#include "../pto_runtime2_threaded.h"
#include "../pto_runtime2_types.h"
#include "../pto_scheduler.h"

int main() {
    printf("Creating runtime...\n"); fflush(stdout);
    PTO2RuntimeThreaded* rt = pto2_runtime_create_threaded(4, 4, true);
    if (!rt) {
        printf("Failed to create runtime!\n");
        return 1;
    }
    printf("Runtime created. Window=%d\n", rt->base.sm_handle->header->task_window_size);
    fflush(stdout);
    
    PTO2Runtime* base = (PTO2Runtime*)rt;
    float* A = calloc(8192, sizeof(float));
    float* P = calloc(8192, sizeof(float));
    
    printf("Starting threads...\n"); fflush(stdout);
    pto2_runtime_start_threads(rt);
    
    // 8x8x8x8 = 8192 tasks
    printf("Submitting 8192 tasks...\n"); fflush(stdout);
    int count = 0;
    for (int b = 0; b < 8; b++) {
        pto2_rt_scope_begin(base);
        for (int m = 0; m < 8; m++) {
            for (int n = 0; n < 8; n++) {
                pto2_rt_scope_begin(base);
                for (int k = 0; k < 8; k++) {
                    int idx = b*512 + m*64 + n*8 + k;
                    
                    PTO2TaskParam gemm_p[3] = {
                        PTO2_INPUT(A, idx, 128),
                        PTO2_INPUT(A, idx, 128),
                        PTO2_OUTPUT(P, idx, 128)
                    };
                    pto2_rt_submit_task(base, 0, PTO2_WORKER_CUBE, NULL, "gemm_tile", gemm_p, 3);
                    count++;
                    
                    PTO2TaskParam add_p[3] = {
                        PTO2_INPUT(P, idx, 128),
                        PTO2_INPUT(P, idx, 128),
                        PTO2_OUTPUT(P, idx, 128)
                    };
                    pto2_rt_submit_task(base, 1, PTO2_WORKER_VECTOR, NULL, "tile_add", add_p, 3);
                    count++;
                }
                pto2_rt_scope_end(base);
            }
        }
        pto2_rt_scope_end(base);
        printf("  Batch %d done (%d tasks)\n", b, count); fflush(stdout);
    }
    
    printf("Marking done...\n"); fflush(stdout);
    pto2_rt_orchestration_done(base);
    rt->thread_ctx.orchestrator_done = true;
    
    // Wait
    printf("Waiting...\n"); fflush(stdout);
    for (int i = 0; i < 20; i++) {
        usleep(500000);
        int64_t consumed = rt->base.scheduler.tasks_consumed;
        printf("  %ds: consumed=%lld/8192\n", (i+1)/2, (long long)consumed);
        fflush(stdout);
        if (consumed >= 8192) break;
    }
    
    printf("Stopping...\n"); fflush(stdout);
    pto2_runtime_stop_threads(rt);
    pto2_runtime_destroy_threaded(rt);
    free(A); free(P);
    printf("Done!\n");
    return 0;
}
