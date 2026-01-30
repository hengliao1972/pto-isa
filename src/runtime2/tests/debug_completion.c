#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "../pto_runtime2.h"
#include "../pto_runtime2_threaded.h"
#include "../pto_runtime2_types.h"
#include "../pto_scheduler.h"
#include "../pto_worker.h"

// Just run a small test and manually check
int main() {
    PTO2RuntimeThreaded* rt = pto2_runtime_create_threaded_custom(1, 1, true, 1024, 1024*1024, 4096);
    PTO2Runtime* base = (PTO2Runtime*)rt;
    PTO2SchedulerState* sched = &rt->base.scheduler;
    
    float* A = calloc(1024, sizeof(float));
    float* P = calloc(1024, sizeof(float));
    
    // Submit just 4 tasks: gemm0, add1, gemm2, add3
    pto2_rt_scope_begin(base);
    
    // Task 0: gemm -> P[0]
    PTO2TaskParam p0[3] = { PTO2_INPUT(A, 0, 64), PTO2_INPUT(A, 0, 64), PTO2_OUTPUT(P, 0, 64) };
    pto2_rt_submit_task(base, 0, PTO2_WORKER_CUBE, NULL, "gemm0", p0, 3);
    
    // Task 1: add <- P[0] -> P[0]
    PTO2TaskParam p1[3] = { PTO2_INPUT(P, 0, 64), PTO2_INPUT(P, 0, 64), PTO2_OUTPUT(P, 0, 64) };
    pto2_rt_submit_task(base, 1, PTO2_WORKER_VECTOR, NULL, "add1", p1, 3);
    
    // Task 2: gemm -> P[1]
    PTO2TaskParam p2[3] = { PTO2_INPUT(A, 1, 64), PTO2_INPUT(A, 1, 64), PTO2_OUTPUT(P, 1, 64) };
    pto2_rt_submit_task(base, 0, PTO2_WORKER_CUBE, NULL, "gemm2", p2, 3);
    
    // Task 3: add <- P[1], P[0] -> P[1]
    PTO2TaskParam p3[3] = { PTO2_INPUT(P, 1, 64), PTO2_INPUT(P, 0, 64), PTO2_OUTPUT(P, 1, 64) };
    pto2_rt_submit_task(base, 1, PTO2_WORKER_VECTOR, NULL, "add3", p3, 3);
    
    pto2_rt_scope_end(base);
    
    printf("Submitted 4 tasks\n");
    printf("\nTask descriptors:\n");
    for (int i = 0; i < 4; i++) {
        PTO2TaskDescriptor* t = pto2_sm_get_task(base->sm_handle, i);
        printf("  Task %d (%s): fanin=%d, fanin_head=%d, fanout=%d, fanout_head=%d\n",
               i, t->func_name, t->fanin_count, t->fanin_head, t->fanout_count, t->fanout_head);
    }
    
    // Start threads and wait
    printf("\nStarting threads...\n");
    pto2_runtime_start_threads(rt);
    
    pto2_rt_orchestration_done(base);
    rt->thread_ctx.orchestrator_done = true;
    
    usleep(3000000);  // Wait 3s
    
    printf("\nFinal state:\n");
    for (int i = 0; i < 4; i++) {
        int slot = pto2_task_slot(sched, i);
        PTO2TaskDescriptor* t = pto2_sm_get_task(base->sm_handle, i);
        printf("  Task %d (%s): state=%d, fanin_count=%d, fanin_refcount=%d\n",
               i, t->func_name, sched->task_state[slot], t->fanin_count, sched->fanin_refcount[slot]);
    }
    
    printf("\ncompleted=%lld, consumed=%lld\n",
           (long long)sched->tasks_completed, (long long)sched->tasks_consumed);
    
    pto2_runtime_stop_threads(rt);
    pto2_runtime_destroy_threaded(rt);
    free(A); free(P);
    return 0;
}
