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
        4, 4, true, 256, PTO2_HEAP_SIZE, PTO2_DEP_LIST_POOL_SIZE);
    
    PTO2Runtime* rt = (PTO2Runtime*)g_rt;
    float* A = calloc(100, sizeof(float));
    float* P = calloc(100, sizeof(float));
    
    printf("Starting threads...\n");
    pto2_runtime_start_threads(g_rt);
    
    PTO2SchedulerState* sched = &g_rt->base.scheduler;
    
    // 2 batches, 5 tasks each
    for (int b = 0; b < 2; b++) {
        printf("\n=== Batch %d ===\n", b);
        pto2_rt_scope_begin(rt);
        for (int i = 0; i < 5; i++) {
            int idx = b * 5 + i;
            PTO2TaskParam p[3] = {
                PTO2_INPUT(A, idx, 128),
                PTO2_INPUT(A, idx, 128),
                PTO2_OUTPUT(P, idx, 128)
            };
            pto2_rt_submit_task(rt, 0, PTO2_WORKER_CUBE, NULL, "task", p, 3);
        }
        
        printf("Before scope_end (batch %d):\n", b);
        for (int i = b*5; i < (b+1)*5; i++) {
            int slot = i & (256-1);
            PTO2TaskDescriptor* task = pto2_sm_get_task(rt->sm_handle, i);
            printf("  Task %d: fanout_count=%d, fanout_refcount=%d\n",
                   i, task->fanout_count, sched->fanout_refcount[slot]);
        }
        
        pto2_rt_scope_end(rt);
        
        printf("After scope_end (batch %d):\n", b);
        for (int i = b*5; i < (b+1)*5; i++) {
            int slot = i & (256-1);
            PTO2TaskDescriptor* task = pto2_sm_get_task(rt->sm_handle, i);
            printf("  Task %d: fanout_count=%d, fanout_refcount=%d\n",
                   i, task->fanout_count, sched->fanout_refcount[slot]);
        }
    }
    
    pto2_rt_orchestration_done(rt);
    g_rt->thread_ctx.orchestrator_done = true;
    
    usleep(1000000);
    
    printf("\n=== Final state ===\n");
    printf("completed=%lld, consumed=%lld, last_alive=%d\n",
           (long long)sched->tasks_completed, (long long)sched->tasks_consumed,
           sched->last_task_alive);
    for (int i = 0; i < 10; i++) {
        int slot = i & (256-1);
        PTO2TaskDescriptor* task = pto2_sm_get_task(rt->sm_handle, i);
        printf("  Task %d: state=%d, fanout_count=%d, fanout_refcount=%d\n",
               i, sched->task_state[slot], task->fanout_count, sched->fanout_refcount[slot]);
    }
    
    pto2_runtime_stop_threads(g_rt);
    pto2_runtime_destroy_threaded(g_rt);
    free(A); free(P);
    return 0;
}
