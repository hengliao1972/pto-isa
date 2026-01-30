#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "../pto_runtime2.h"
#include "../pto_runtime2_threaded.h"
#include "../pto_runtime2_types.h"

PTO2RuntimeThreaded* g_rt;

void orch_func(PTO2Runtime* rt, void* arg) {
    float* A = calloc(1024, sizeof(float));
    float* P = calloc(1024, sizeof(float));
    
    // 4096 tasks
    for (int b = 0; b < 8; b++) {
        pto2_rt_scope_begin(rt);
        for (int i = 0; i < 512; i++) {
            int idx = b * 512 + i;
            PTO2TaskParam p[3] = {
                PTO2_INPUT(A, idx % 1024, 128),
                PTO2_INPUT(A, idx % 1024, 128),
                PTO2_OUTPUT(P, idx % 1024, 128)
            };
            pto2_rt_submit_task(rt, 0, PTO2_WORKER_CUBE, NULL, "task", p, 3);
        }
        pto2_rt_scope_end(rt);
        printf("Batch %d: submitted=%d\n", b, (b+1)*512);
        fflush(stdout);
    }
    
    printf("Orchestration done (4096 tasks)\n"); fflush(stdout);
    free(A); free(P);
}

int main() {
    g_rt = pto2_runtime_create_threaded_custom(
        4, 4, true, 8192, PTO2_HEAP_SIZE, PTO2_DEP_LIST_POOL_SIZE);
    
    printf("Starting 4096 tasks with window=8192...\n");
    
    pto2_runtime_start_threads(g_rt);
    orch_func((PTO2Runtime*)g_rt, NULL);
    pto2_rt_orchestration_done((PTO2Runtime*)g_rt);
    g_rt->thread_ctx.orchestrator_done = true;
    
    printf("Waiting for completion...\n"); fflush(stdout);
    for (int i = 0; i < 20; i++) {
        usleep(500000);
        int64_t completed = g_rt->base.scheduler.tasks_completed;
        int64_t consumed = g_rt->base.scheduler.tasks_consumed;
        printf("  %dms: completed=%lld, consumed=%lld\n", (i+1)*500, 
               (long long)completed, (long long)consumed);
        fflush(stdout);
        if (consumed >= 4096) break;
    }
    
    pto2_runtime_stop_threads(g_rt);
    printf("Final: completed=%lld, consumed=%lld\n", 
           (long long)g_rt->base.scheduler.tasks_completed,
           (long long)g_rt->base.scheduler.tasks_consumed);
    pto2_runtime_destroy_threaded(g_rt);
    return 0;
}
