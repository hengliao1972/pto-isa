#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "../pto_runtime2.h"
#include "../pto_runtime2_threaded.h"
#include "../pto_runtime2_types.h"
#include "../pto_scheduler.h"

PTO2RuntimeThreaded* g_rt;

int main() {
    // Use smaller window to see if slot collision causes the bug
    g_rt = pto2_runtime_create_threaded_custom(
        4, 4, true, 1024, PTO2_HEAP_SIZE, PTO2_DEP_LIST_POOL_SIZE);
    
    PTO2Runtime* rt = (PTO2Runtime*)g_rt;
    float* A = calloc(4096, sizeof(float));
    float* P = calloc(4096, sizeof(float));
    
    printf("Window=1024, 8 batches x 500 tasks = 4000 tasks\n");
    pto2_runtime_start_threads(g_rt);
    
    PTO2SchedulerState* sched = &g_rt->base.scheduler;
    
    for (int b = 0; b < 8; b++) {
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
        printf("Batch %d: submitted=%d, consumed=%lld\n", 
               b, (b+1)*500, (long long)sched->tasks_consumed);
    }
    
    pto2_rt_orchestration_done(rt);
    g_rt->thread_ctx.orchestrator_done = true;
    
    // Wait
    for (int i = 0; i < 10; i++) {
        usleep(500000);
        if (sched->tasks_consumed >= 4000) break;
        printf("  %dms: consumed=%lld\n", (i+1)*500, (long long)sched->tasks_consumed);
    }
    
    printf("\nFinal: completed=%lld, consumed=%lld, last_alive=%d\n",
           (long long)sched->tasks_completed, (long long)sched->tasks_consumed,
           sched->last_task_alive);
    
    // Check first stuck task
    if (sched->last_task_alive < 4000) {
        int i = sched->last_task_alive;
        int slot = i & (1024-1);
        PTO2TaskDescriptor* task = pto2_sm_get_task(rt->sm_handle, i);
        printf("First stuck: Task %d (slot %d): state=%d, fanout_count=%d, refcount=%d\n",
               i, slot, sched->task_state[slot], task->fanout_count, 
               sched->fanout_refcount[slot]);
    }
    
    pto2_runtime_stop_threads(g_rt);
    pto2_runtime_destroy_threaded(g_rt);
    free(A); free(P);
    return 0;
}
