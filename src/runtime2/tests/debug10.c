#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "../pto_runtime2.h"
#include "../pto_runtime2_threaded.h"
#include "../pto_runtime2_types.h"
#include "../pto_scheduler.h"

PTO2RuntimeThreaded* g_rt;

void orch_func(PTO2Runtime* rt, void* arg) {
    float* A = calloc(1024, sizeof(float));
    float* P = calloc(1024, sizeof(float));
    
    // 1000 tasks, 10 batches
    for (int b = 0; b < 10; b++) {
        pto2_rt_scope_begin(rt);
        for (int i = 0; i < 100; i++) {
            int idx = b * 100 + i;
            PTO2TaskParam p[3] = {
                PTO2_INPUT(A, idx % 1024, 128),
                PTO2_INPUT(A, idx % 1024, 128),
                PTO2_OUTPUT(P, idx % 1024, 128)
            };
            pto2_rt_submit_task(rt, 0, PTO2_WORKER_CUBE, NULL, "task", p, 3);
        }
        pto2_rt_scope_end(rt);
    }
    
    free(A); free(P);
}

int main() {
    g_rt = pto2_runtime_create_threaded_custom(
        4, 4, true, 2048, PTO2_HEAP_SIZE, PTO2_DEP_LIST_POOL_SIZE);
    
    pto2_runtime_start_threads(g_rt);
    orch_func((PTO2Runtime*)g_rt, NULL);
    pto2_rt_orchestration_done((PTO2Runtime*)g_rt);
    g_rt->thread_ctx.orchestrator_done = true;
    
    // Wait
    for (int i = 0; i < 10; i++) {
        usleep(500000);
        PTO2SchedulerState* sched = &g_rt->base.scheduler;
        if (sched->tasks_consumed >= 1000) break;
    }
    
    PTO2SchedulerState* sched = &g_rt->base.scheduler;
    printf("completed=%lld, consumed=%lld, last_alive=%d\n",
           (long long)sched->tasks_completed,
           (long long)sched->tasks_consumed,
           sched->last_task_alive);
    
    // Check stuck tasks
    int stuck = 0;
    for (int i = 0; i < 1000; i++) {
        int slot = i & (2048-1);
        if (sched->task_state[slot] != PTO2_TASK_CONSUMED) {
            if (stuck < 5) {
                PTO2TaskDescriptor* task = pto2_sm_get_task(g_rt->base.sm_handle, i);
                printf("  Task %d: state=%d, fanout=%d, refcount=%d\n",
                       i, sched->task_state[slot], task->fanout_count, 
                       sched->fanout_refcount[slot]);
            }
            stuck++;
        }
    }
    printf("Total stuck: %d\n", stuck);
    
    pto2_runtime_stop_threads(g_rt);
    pto2_runtime_destroy_threaded(g_rt);
    return 0;
}
