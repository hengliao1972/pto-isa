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
    
    int batches = 8;
    int per_batch = 500;
    int total = batches * per_batch;
    
    printf("Submitting %d tasks (%d batches x %d)...\n", total, batches, per_batch);
    for (int b = 0; b < batches; b++) {
        pto2_rt_scope_begin(rt);
        for (int i = 0; i < per_batch; i++) {
            int idx = b * per_batch + i;
            PTO2TaskParam p[3] = {
                PTO2_INPUT(A, idx, 128),
                PTO2_INPUT(A, idx, 128),
                PTO2_OUTPUT(P, idx, 128)
            };
            pto2_rt_submit_task(rt, 0, PTO2_WORKER_CUBE, NULL, "task", p, 3);
        }
        pto2_rt_scope_end(rt);
    }
    printf("All batches submitted\n");
    
    pto2_rt_orchestration_done(rt);
    g_rt->thread_ctx.orchestrator_done = true;
    
    // Wait up to 10s
    for (int i = 0; i < 20; i++) {
        usleep(500000);
        if (sched->tasks_consumed >= total) break;
        if (i % 4 == 3) {
            printf("  %ds: consumed=%lld/%d\n", (i+1)/2, (long long)sched->tasks_consumed, total);
        }
    }
    
    printf("\nResult: completed=%lld, consumed=%lld, last_alive=%d\n",
           (long long)sched->tasks_completed, (long long)sched->tasks_consumed,
           sched->last_task_alive);
    
    if (sched->last_task_alive < total) {
        int tid = sched->last_task_alive;
        int slot = tid & (8192-1);
        PTO2TaskDescriptor* task = pto2_sm_get_task(rt->sm_handle, tid);
        printf("Stuck task %d: fanout=%d, refcount=%d\n",
               tid, task->fanout_count, sched->fanout_refcount[slot]);
    }
    
    pto2_runtime_stop_threads(g_rt);
    pto2_runtime_destroy_threaded(g_rt);
    free(A); free(P);
    return 0;
}
