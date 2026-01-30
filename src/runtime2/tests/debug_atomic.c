#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "../pto_runtime2.h"
#include "../pto_runtime2_threaded.h"
#include "../pto_runtime2_types.h"
#include "../pto_scheduler.h"
#include "../pto_orchestrator.h"

// Add a trace counter for fanin updates
volatile int fanin_updates = 0;

int main() {
    PTO2RuntimeThreaded* rt = pto2_runtime_create_threaded_custom(4, 4, true, 16384, 64*1024*1024, 65536);
    PTO2Runtime* base = (PTO2Runtime*)rt;
    PTO2SchedulerState* sched = &rt->base.scheduler;
    
    float* A = calloc(1024*1024, sizeof(float));
    float* B = calloc(1024*1024, sizeof(float));
    float* C = calloc(1024*1024, sizeof(float));
    float* P = calloc(1024*1024, sizeof(float));
    
    pto2_runtime_start_threads(rt);
    
    for (int batch = 0; batch < 8; batch++) {
        pto2_rt_scope_begin(base);
        for (int m = 0; m < 8; m++) {
            for (int n = 0; n < 8; n++) {
                pto2_rt_scope_begin(base);
                for (int k = 0; k < 8; k++) {
                    int a_idx = batch * 64 + m * 8 + k;
                    int b_idx = batch * 64 + k * 8 + n;
                    int c_idx = batch * 64 + m * 8 + n;
                    
                    PTO2TaskParam gemm_params[3] = {
                        PTO2_INPUT(A, a_idx, 128),
                        PTO2_INPUT(B, b_idx, 128),
                        PTO2_OUTPUT(P, c_idx, 128)
                    };
                    pto2_rt_submit_task(base, 0, PTO2_WORKER_CUBE, NULL, "gemm", gemm_params, 3);
                    
                    PTO2TaskParam add_params[3] = {
                        PTO2_INPUT(C, c_idx, 128),
                        PTO2_INPUT(P, c_idx, 128),
                        PTO2_OUTPUT(C, c_idx, 128)
                    };
                    pto2_rt_submit_task(base, 1, PTO2_WORKER_VECTOR, NULL, "add", add_params, 3);
                }
                pto2_rt_scope_end(base);
            }
        }
        pto2_rt_scope_end(base);
    }
    
    pto2_rt_orchestration_done(base);
    rt->thread_ctx.orchestrator_done = true;
    
    usleep(8000000);
    
    // Count fanin updates that should have happened
    // Each gemm has 1 consumer (add), each add may have 0-2 consumers
    // Total fanin edges = sum of all tasks' fanin_count
    int total_fanin_edges = 0;
    int total_fanin_refcount = 0;
    
    for (int i = 0; i < 8192; i++) {
        PTO2TaskDescriptor* t = pto2_sm_get_task(base->sm_handle, i);
        int slot = pto2_task_slot(sched, i);
        total_fanin_edges += t->fanin_count;
        total_fanin_refcount += sched->fanin_refcount[slot];
    }
    
    printf("Total fanin edges: %d\n", total_fanin_edges);
    printf("Total fanin_refcount sum: %d\n", total_fanin_refcount);
    printf("Difference (missing updates): %d\n", total_fanin_edges - total_fanin_refcount);
    printf("\ncompleted=%lld, consumed=%lld\n",
           (long long)sched->tasks_completed, (long long)sched->tasks_consumed);
    
    // Check how many tasks have incorrect refcount
    int mismatched = 0;
    for (int i = 0; i < 8192; i++) {
        PTO2TaskDescriptor* t = pto2_sm_get_task(base->sm_handle, i);
        int slot = pto2_task_slot(sched, i);
        if (sched->task_state[slot] < 3) {  // Not completed
            if (sched->fanin_refcount[slot] != t->fanin_count) {
                mismatched++;
            }
        }
    }
    printf("Tasks with fanin_refcount mismatch: %d\n", mismatched);
    
    pto2_runtime_stop_threads(rt);
    pto2_runtime_destroy_threaded(rt);
    free(A); free(B); free(C); free(P);
    return 0;
}
