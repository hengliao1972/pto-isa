#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "../pto_runtime2.h"
#include "../pto_runtime2_threaded.h"
#include "../pto_runtime2_types.h"
#include "../pto_scheduler.h"

// Simple test: just 16 tasks in a chain
int main() {
    printf("Simple test: 16 tasks\n");
    
    PTO2RuntimeThreaded* rt = pto2_runtime_create_threaded_custom(2, 2, true, 1024, 1024*1024, 4096);
    PTO2Runtime* base = (PTO2Runtime*)rt;
    PTO2SchedulerState* sched = &rt->base.scheduler;
    
    float* P = calloc(1024, sizeof(float));
    
    // Submit 8 pairs: gemm -> add
    pto2_rt_scope_begin(base);
    for (int k = 0; k < 8; k++) {
        // gemm: no input deps, output P[k]
        PTO2TaskParam gemm_p[3] = {
            PTO2_INPUT(P, k, 64),
            PTO2_INPUT(P, k, 64),
            PTO2_OUTPUT(P, k, 64)
        };
        pto2_rt_submit_task(base, 0, PTO2_WORKER_CUBE, NULL, "gemm", gemm_p, 3);
        
        // add: input from gemm (P[k]), output P[k]
        PTO2TaskParam add_p[3] = {
            PTO2_INPUT(P, k, 64),
            PTO2_INPUT(P, k, 64),
            PTO2_OUTPUT(P, k, 64)
        };
        pto2_rt_submit_task(base, 1, PTO2_WORKER_VECTOR, NULL, "add", add_p, 3);
    }
    pto2_rt_scope_end(base);
    
    printf("Submitted 16 tasks\n");
    
    // Check task descriptors
    printf("\nTask descriptors:\n");
    for (int i = 0; i < 16; i++) {
        PTO2TaskDescriptor* t = pto2_sm_get_task(base->sm_handle, i);
        printf("  Task %d (%s): fanin=%d, fanout=%d\n",
               i, t->func_name, t->fanin_count, t->fanout_count);
    }
    
    // Start threads
    printf("\nStarting threads...\n");
    pto2_runtime_start_threads(rt);
    
    pto2_rt_orchestration_done(base);
    rt->thread_ctx.orchestrator_done = true;
    
    // Monitor for 5 seconds
    for (int sec = 0; sec < 10; sec++) {
        usleep(500000);  // 0.5s
        
        int64_t completed = sched->tasks_completed;
        int64_t consumed = sched->tasks_consumed;
        
        printf("t=%.1fs: completed=%lld, consumed=%lld\n",
               (sec + 1) * 0.5, (long long)completed, (long long)consumed);
        
        if (completed == 16 && consumed == 16) {
            printf("SUCCESS: All 16 tasks done!\n");
            break;
        }
        
        // If stuck, show details
        if (sec >= 4 && completed < 16) {
            printf("\nSTUCK! Details:\n");
            for (int i = 0; i < 16; i++) {
                int slot = pto2_task_slot(sched, i);
                PTO2TaskDescriptor* t = pto2_sm_get_task(base->sm_handle, i);
                printf("  Task %d: state=%d, fanin=%d, fanin_refcount=%d\n",
                       i, sched->task_state[slot], t->fanin_count, sched->fanin_refcount[slot]);
            }
            break;
        }
    }
    
    pto2_runtime_stop_threads(rt);
    pto2_runtime_destroy_threaded(rt);
    free(P);
    return 0;
}
