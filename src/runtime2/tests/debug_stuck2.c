#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "../pto_runtime2.h"
#include "../pto_runtime2_threaded.h"
#include "../pto_runtime2_types.h"
#include "../pto_scheduler.h"
#include "../pto_orchestrator.h"

void bgemm_orch(PTO2Runtime* rt, void* arg);

int main() {
    printf("Creating runtime with window=16384...\n");
    PTO2RuntimeThreaded* rt = pto2_runtime_create_threaded_custom(4, 4, true, 16384, 64*1024*1024, 65536);
    PTO2Runtime* base = (PTO2Runtime*)rt;
    PTO2SchedulerState* sched = &rt->base.scheduler;
    
    float* A = calloc(1024*1024, sizeof(float));
    float* B = calloc(1024*1024, sizeof(float));
    float* C = calloc(1024*1024, sizeof(float));
    float* P = calloc(1024*1024, sizeof(float));
    
    int params[5] = {8, 8, 8, 8, 0};  // 8192 tasks
    params[4] = (int)(size_t)A;  // Hack to pass pointers
    
    printf("Running threaded...\n");
    pto2_runtime_start_threads(rt);
    
    // Submit tasks
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
                    pto2_rt_submit_task(base, 0, PTO2_WORKER_CUBE, NULL, "gemm_tile", gemm_params, 3);
                    
                    PTO2TaskParam add_params[3] = {
                        PTO2_INPUT(C, c_idx, 128),
                        PTO2_INPUT(P, c_idx, 128),
                        PTO2_OUTPUT(C, c_idx, 128)
                    };
                    pto2_rt_submit_task(base, 1, PTO2_WORKER_VECTOR, NULL, "tile_add", add_params, 3);
                }
                pto2_rt_scope_end(base);
            }
        }
        pto2_rt_scope_end(base);
    }
    
    pto2_rt_orchestration_done(base);
    rt->thread_ctx.orchestrator_done = true;
    
    // Wait and check
    for (int i = 0; i < 30; i++) {
        usleep(1000000);
        int64_t completed = sched->tasks_completed;
        int64_t consumed = sched->tasks_consumed;
        printf("t=%ds: completed=%lld, consumed=%lld, last_alive=%d\n",
               i+1, (long long)completed, (long long)consumed, sched->last_task_alive);
        
        if (completed == consumed && completed == 8192) {
            printf("All done!\n");
            break;
        }
        
        // If stuck, find stuck tasks
        if (completed != consumed && i >= 5) {
            printf("\nFinding stuck tasks (first 5):\n");
            int found = 0;
            for (int tid = sched->last_task_alive; tid < 8192 && found < 5; tid++) {
                int slot = pto2_task_slot(sched, tid);
                if (sched->task_state[slot] == PTO2_TASK_COMPLETED) {
                    PTO2TaskDescriptor* task = pto2_sm_get_task(base->sm_handle, tid);
                    printf("  Task %d: state=COMPLETED, fanout=%d, refcount=%d, scope=%d\n",
                           tid, task->fanout_count, sched->fanout_refcount[slot], task->scope_depth);
                    found++;
                }
            }
            break;
        }
    }
    
    pto2_runtime_stop_threads(rt);
    pto2_runtime_destroy_threaded(rt);
    free(A); free(B); free(C); free(P);
    return 0;
}
