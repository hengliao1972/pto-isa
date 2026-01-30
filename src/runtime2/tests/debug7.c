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
    
    for (int b = 0; b < 2; b++) {
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
        printf("Batch %d: last_alive=%d, completed=%lld, consumed=%lld\n", 
               b, rt->sm_handle->header->last_task_alive,
               (long long)g_rt->base.scheduler.tasks_completed,
               (long long)g_rt->base.scheduler.tasks_consumed);
        fflush(stdout);
    }
    
    printf("Orchestration done\n"); fflush(stdout);
    free(A); free(P);
}

int main() {
    g_rt = pto2_runtime_create_threaded_custom(
        4, 4, true, 512, PTO2_HEAP_SIZE, PTO2_DEP_LIST_POOL_SIZE);
    
    printf("Starting...\n");
    
    // Start threads
    pto2_runtime_start_threads(g_rt);
    
    // Run orchestration
    orch_func((PTO2Runtime*)g_rt, NULL);
    
    // Mark done
    pto2_rt_orchestration_done((PTO2Runtime*)g_rt);
    g_rt->thread_ctx.orchestrator_done = true;
    
    // Wait a bit
    printf("Waiting...\n"); fflush(stdout);
    for (int i = 0; i < 6; i++) {
        usleep(500000);
        printf("  After %dms: last_alive=%d, completed=%lld, consumed=%lld\n",
               (i+1)*500, g_rt->base.sm_handle->header->last_task_alive,
               (long long)g_rt->base.scheduler.tasks_completed,
               (long long)g_rt->base.scheduler.tasks_consumed);
        fflush(stdout);
    }
    
    pto2_runtime_stop_threads(g_rt);
    pto2_runtime_destroy_threaded(g_rt);
    return 0;
}
