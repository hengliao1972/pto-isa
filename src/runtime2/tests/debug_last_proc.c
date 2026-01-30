#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "../pto_runtime2.h"
#include "../pto_runtime2_threaded.h"
#include "../pto_runtime2_types.h"
#include "../pto_scheduler.h"

int main() {
    PTO2RuntimeThreaded* rt = pto2_runtime_create_threaded_custom(4, 4, true, 16384, 64*1024*1024, 65536);
    PTO2Runtime* base = (PTO2Runtime*)rt;
    PTO2SchedulerState* sched = &rt->base.scheduler;
    
    float* A = calloc(1024*1024, sizeof(float));
    float* B = calloc(1024*1024, sizeof(float));
    float* C = calloc(1024*1024, sizeof(float));
    float* P = calloc(1024*1024, sizeof(float));
    
    pto2_runtime_start_threads(rt);
    
    // Submit 8x8x8x8 = 8192 tasks
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
                    pto2_rt_submit_task(base, 0, PTO2_WORKER_CUBE, NULL, "gemm_tile", gemm_params, 3);
                    
                    PTO2TaskParam add_params[3] = {
                        PTO2_INPUT(C, c_idx, 128),
                        PTO2_INPUT(P, c_idx, 128),
                        PTO2_OUTPUT(C, c_idx, 128)
                    };
                    pto2_rt_submit_task(base, 1, PTO2_WORKER_VECTOR, NULL, "tile_add", add_params, 3);
                }
                pto2_rt_scope_end(base);
            }
        }
        pto2_rt_scope_end(base);
    }
    
    pto2_rt_orchestration_done(base);
    rt->thread_ctx.orchestrator_done = true;
    
    usleep(5000000);  // Wait 5s
    
    // Check task states around the boundary
    printf("Task states around last_alive=%d:\n", sched->last_task_alive);
    for (int i = sched->last_task_alive - 2; i <= sched->last_task_alive + 5; i++) {
        if (i < 0 || i >= 8192) continue;
        int slot = pto2_task_slot(sched, i);
        PTO2TaskDescriptor* task = pto2_sm_get_task(base->sm_handle, i);
        printf("  Task %d: state=%d, fanout=%d, refcount=%d\n",
               i, sched->task_state[slot], task->fanout_count, sched->fanout_refcount[slot]);
    }
    
    // Count tasks in each state
    int state_counts[5] = {0};
    for (int i = 0; i < 8192; i++) {
        int slot = pto2_task_slot(sched, i);
        int st = sched->task_state[slot];
        if (st >= 0 && st < 5) state_counts[st]++;
    }
    printf("\nState distribution:\n");
    printf("  UNKNOWN(0): %d\n", state_counts[0]);
    printf("  PENDING(1): %d\n", state_counts[1]);
    printf("  READY(2):   %d\n", state_counts[2]);
    printf("  COMPLETED(3): %d\n", state_counts[3]);
    printf("  CONSUMED(4): %d\n", state_counts[4]);
    
    printf("\ncompleted=%lld, consumed=%lld\n", 
           (long long)sched->tasks_completed, (long long)sched->tasks_consumed);
    
    pto2_runtime_stop_threads(rt);
    pto2_runtime_destroy_threaded(rt);
    free(A); free(B); free(C); free(P);
    return 0;
}
