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
    
    // 简化测试：2 batches x 100 tasks = 200
    printf("Submitting 200 tasks (2 batches x 100)...\n");
    for (int b = 0; b < 2; b++) {
        pto2_rt_scope_begin(rt);
        for (int i = 0; i < 100; i++) {
            int idx = b * 100 + i;
            PTO2TaskParam p[3] = {
                PTO2_INPUT(A, idx, 128),
                PTO2_INPUT(A, idx, 128),
                PTO2_OUTPUT(P, idx, 128)
            };
            pto2_rt_submit_task(rt, 0, PTO2_WORKER_CUBE, NULL, "task", p, 3);
        }
        pto2_rt_scope_end(rt);
        printf("  Batch %d: scope_end called\n", b);
    }
    
    pto2_rt_orchestration_done(rt);
    g_rt->thread_ctx.orchestrator_done = true;
    
    usleep(1000000);  // Wait 1s
    
    printf("\nResult:\n");
    printf("  completed=%lld, consumed=%lld, last_alive=%d\n",
           (long long)sched->tasks_completed, (long long)sched->tasks_consumed,
           sched->last_task_alive);
    
    // Find first stuck task
    if (sched->last_task_alive < 200) {
        int task_id = sched->last_task_alive;
        int slot = task_id & (8192-1);
        PTO2TaskDescriptor* task = pto2_sm_get_task(rt->sm_handle, task_id);
        printf("\nFirst stuck: Task %d\n", task_id);
        printf("  state=%d, fanout_count=%d, fanout_refcount=%d\n",
               sched->task_state[slot], task->fanout_count, sched->fanout_refcount[slot]);
        printf("  scope_depth=%d, fanin_count=%d\n", task->scope_depth, task->fanin_count);
        
        // Check surrounding tasks
        printf("\nSurrounding tasks:\n");
        for (int i = task_id - 2; i <= task_id + 2 && i < 200; i++) {
            if (i < 0) continue;
            int s = i & (8192-1);
            PTO2TaskDescriptor* t = pto2_sm_get_task(rt->sm_handle, i);
            printf("  Task %d: state=%d, fanout=%d, refcount=%d\n",
                   i, sched->task_state[s], t->fanout_count, sched->fanout_refcount[s]);
        }
    }
    
    pto2_runtime_stop_threads(g_rt);
    pto2_runtime_destroy_threaded(g_rt);
    free(A); free(P);
    return 0;
}
