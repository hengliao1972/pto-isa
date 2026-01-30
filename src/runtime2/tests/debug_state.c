#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "../pto_runtime2.h"
#include "../pto_runtime2_threaded.h"
#include "../pto_runtime2_types.h"
#include "../pto_scheduler.h"

int main() {
    PTO2RuntimeThreaded* rt = pto2_runtime_create_threaded(4, 4, true);
    PTO2Runtime* base = (PTO2Runtime*)rt;
    PTO2SchedulerState* sched = &rt->base.scheduler;
    
    float* A = calloc(8192, sizeof(float));
    float* P = calloc(8192, sizeof(float));
    
    pto2_runtime_start_threads(rt);
    
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
                    
                    PTO2TaskParam add_p[3] = {
                        PTO2_INPUT(P, idx, 128),
                        PTO2_INPUT(P, idx, 128),
                        PTO2_OUTPUT(P, idx, 128)
                    };
                    pto2_rt_submit_task(base, 1, PTO2_WORKER_VECTOR, NULL, "tile_add", add_p, 3);
                }
                pto2_rt_scope_end(base);
            }
        }
        pto2_rt_scope_end(base);
    }
    
    pto2_rt_orchestration_done(base);
    rt->thread_ctx.orchestrator_done = true;
    
    usleep(3000000);
    
    // Check all task states
    int state_counts[5] = {0};
    for (int i = 0; i < 8192; i++) {
        int slot = PTO2_TASK_SLOT(i);
        int state = sched->task_state[slot];
        if (state >= 0 && state < 5) state_counts[state]++;
    }
    
    printf("Task state distribution:\n");
    printf("  0 (UNKNOWN): %d\n", state_counts[0]);
    printf("  1 (PENDING): %d\n", state_counts[1]);
    printf("  2 (READY):   %d\n", state_counts[2]);
    printf("  3 (COMPLETED): %d\n", state_counts[3]);
    printf("  4 (CONSUMED): %d\n", state_counts[4]);
    printf("\nTotal: completed=%lld, consumed=%lld\n",
           (long long)sched->tasks_completed, (long long)sched->tasks_consumed);
    
    pto2_runtime_stop_threads(rt);
    pto2_runtime_destroy_threaded(rt);
    free(A); free(P);
    return 0;
}
