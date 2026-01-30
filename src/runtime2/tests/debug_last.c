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
    
    printf("consumed=%lld/8192, last_alive=%d\n",
           (long long)sched->tasks_consumed, sched->last_task_alive);
    
    // Find ALL stuck tasks
    printf("\nAll stuck tasks:\n");
    for (int i = 0; i < 8192; i++) {
        int slot = PTO2_TASK_SLOT(i);
        if (sched->task_state[slot] != PTO2_TASK_CONSUMED) {
            PTO2TaskDescriptor* task = pto2_sm_get_task(base->sm_handle, i);
            printf("  Task %d: state=%d, fanout=%d, refcount=%d, scope=%d, completed=%d\n",
                   i, sched->task_state[slot], task->fanout_count,
                   sched->fanout_refcount[slot], task->scope_depth,
                   sched->tasks_completed >= 8192);
        }
    }
    
    pto2_runtime_stop_threads(rt);
    pto2_runtime_destroy_threaded(rt);
    free(A); free(P);
    return 0;
}
