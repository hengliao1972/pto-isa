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
    PTO2ThreadContext* ctx = &g_rt->thread_ctx;
    PTO2SchedulerState* sched = &g_rt->base.scheduler;
    
    float* A = calloc(4096, sizeof(float));
    float* P = calloc(4096, sizeof(float));
    
    printf("Starting threads...\n");
    pto2_runtime_start_threads(g_rt);
    
    printf("Submitting 4000 tasks (8 batches x 500)...\n");
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
        printf("  Batch %d done\n", b);
    }
    
    pto2_rt_orchestration_done(rt);
    ctx->orchestrator_done = true;
    printf("Orchestration done\n\n");
    
    // Monitor progress
    printf("Monitoring (window=8192):\n");
    int prev_consumed = 0;
    for (int i = 0; i < 20; i++) {
        usleep(200000);  // 200ms
        int64_t completed = sched->tasks_completed;
        int64_t consumed = sched->tasks_consumed;
        int32_t last_alive = sched->last_task_alive;
        
        printf("  %dms: completed=%lld, consumed=%lld, last_alive=%d",
               (i+1)*200, (long long)completed, (long long)consumed, last_alive);
        
        // Check if stuck
        if (consumed == prev_consumed && consumed < 4000 && completed == 4000) {
            // Check first non-consumed task
            int slot = last_alive & (8192-1);
            PTO2TaskDescriptor* task = pto2_sm_get_task(rt->sm_handle, last_alive);
            printf(" | STUCK task %d: state=%d, fanout=%d, refcount=%d",
                   last_alive, sched->task_state[slot], 
                   task->fanout_count, sched->fanout_refcount[slot]);
        }
        printf("\n");
        
        prev_consumed = consumed;
        if (consumed >= 4000) break;
    }
    
    // Worker stats
    printf("\nWorker stats:\n");
    for (int w = 0; w < ctx->num_workers; w++) {
        printf("  worker[%d]: executed=%lld\n", w, (long long)ctx->workers[w].tasks_executed);
    }
    
    pto2_runtime_stop_threads(g_rt);
    pto2_runtime_destroy_threaded(g_rt);
    free(A); free(P);
    return 0;
}
