#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "../pto_runtime2.h"
#include "../pto_runtime2_threaded.h"
#include "../pto_runtime2_types.h"
#include "../pto_scheduler.h"

PTO2RuntimeThreaded* g_rt;

int main() {
    g_rt = pto2_runtime_create_threaded_custom(
        4, 4, true, 8192, PTO2_HEAP_SIZE, PTO2_DEP_LIST_POOL_SIZE);
    
    PTO2Runtime* rt = (PTO2Runtime*)g_rt;
    PTO2SchedulerState* sched = &g_rt->base.scheduler;
    
    float* A = calloc(4096, sizeof(float));
    float* P = calloc(4096, sizeof(float));
    
    pto2_runtime_start_threads(g_rt);
    
    // 2 batches x 500 = 1000
    printf("Submitting 1000 tasks (2 batches x 500)...\n");
    for (int b = 0; b < 2; b++) {
        pto2_rt_scope_begin(rt);
        for (int i = 0; i < 500; i++) {
            int idx = b * 500 + i;
            PTO2TaskParam p[3] = {
                PTO2_INPUT(A, idx, 128),
                PTO2_INPUT(A, idx, 128),
                PTO2_OUTPUT(P, idx, 128)
            };
            pto2_rt_submit_task(rt, 0, PTO2_WORKER_CUBE, NULL, "task", p, 3);
        }
        pto2_rt_scope_end(rt);
        printf("  Batch %d done\n", b);
    }
    
    pto2_rt_orchestration_done(rt);
    g_rt->thread_ctx.orchestrator_done = true;
    
    usleep(2000000);  // Wait 2s
    
    printf("\nResult: completed=%lld, consumed=%lld, last_alive=%d\n",
           (long long)sched->tasks_completed, (long long)sched->tasks_consumed,
           sched->last_task_alive);
    
    if (sched->last_task_alive < 1000) {
        int tid = sched->last_task_alive;
        int slot = tid & (8192-1);
        PTO2TaskDescriptor* task = pto2_sm_get_task(rt->sm_handle, tid);
        printf("\nStuck task %d: fanout=%d, refcount=%d, scope_depth=%d\n",
               tid, task->fanout_count, sched->fanout_refcount[slot], task->scope_depth);
    }
    
    pto2_runtime_stop_threads(g_rt);
    pto2_runtime_destroy_threaded(g_rt);
    free(A); free(P);
    return 0;
}
