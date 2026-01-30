#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "../pto_runtime2.h"
#include "../pto_runtime2_threaded.h"
#include "../pto_runtime2_types.h"
#include "../pto_scheduler.h"

PTO2RuntimeThreaded* g_rt;

void orch_func(PTO2Runtime* rt, void* arg) {
    float* A = calloc(4096, sizeof(float));
    float* P = calloc(4096, sizeof(float));
    
    // 4000 tasks
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
        printf("Batch %d done\n", b);
    }
    
    free(A); free(P);
}

int main() {
    g_rt = pto2_runtime_create_threaded_custom(
        4, 4, true, 8192, PTO2_HEAP_SIZE, PTO2_DEP_LIST_POOL_SIZE);
    
    printf("Starting 4000 tasks...\n");
    pto2_runtime_start_threads(g_rt);
    orch_func((PTO2Runtime*)g_rt, NULL);
    pto2_rt_orchestration_done((PTO2Runtime*)g_rt);
    g_rt->thread_ctx.orchestrator_done = true;
    
    // Wait
    for (int i = 0; i < 20; i++) {
        usleep(500000);
        PTO2SchedulerState* sched = &g_rt->base.scheduler;
        printf("  %dms: consumed=%lld\n", (i+1)*500, (long long)sched->tasks_consumed);
        if (sched->tasks_consumed >= 4000) break;
    }
    
    PTO2SchedulerState* sched = &g_rt->base.scheduler;
    printf("Final: completed=%lld, consumed=%lld\n",
           (long long)sched->tasks_completed,
           (long long)sched->tasks_consumed);
    
    pto2_runtime_stop_threads(g_rt);
    pto2_runtime_destroy_threaded(g_rt);
    return 0;
}
