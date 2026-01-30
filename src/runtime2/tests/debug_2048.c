#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "../pto_runtime2.h"
#include "../pto_runtime2_threaded.h"
#include "../pto_runtime2_types.h"
#include "../pto_scheduler.h"
#include "../pto_orchestrator.h"

int main() {
    PTO2RuntimeThreaded* rt = pto2_runtime_create_threaded(4, 4, true);
    PTO2Runtime* base = (PTO2Runtime*)rt;
    PTO2SchedulerState* sched = &rt->base.scheduler;
    
    float* A = calloc(8192, sizeof(float));
    float* P = calloc(8192, sizeof(float));
    
    pto2_runtime_start_threads(rt);
    
    for (int b = 0; b < 8; b++) {
        pto2_rt_scope_begin(base);
        for (int m = 0; m < 8; m++) {
            for (int n = 0; n < 8; n++) {
                pto2_rt_scope_begin(base);
                for (int k = 0; k < 8; k++) {
                    int idx = b*512 + m*64 + n*8 + k;
                    PTO2TaskParam gemm_p[3] = {
                        PTO2_INPUT(A, idx, 128),
                        PTO2_INPUT(A, idx, 128),
                        PTO2_OUTPUT(P, idx, 128)
                    };
                    pto2_rt_submit_task(base, 0, PTO2_WORKER_CUBE, NULL, "gemm_tile", gemm_p, 3);
                    
                    PTO2TaskParam add_p[3] = {
                        PTO2_INPUT(P, idx, 128),
                        PTO2_INPUT(P, idx, 128),
                        PTO2_OUTPUT(P, idx, 128)
                    };
                    pto2_rt_submit_task(base, 1, PTO2_WORKER_VECTOR, NULL, "tile_add", add_p, 3);
                }
                pto2_rt_scope_end(base);
            }
        }
        pto2_rt_scope_end(base);
    }
    
    pto2_rt_orchestration_done(base);
    rt->thread_ctx.orchestrator_done = true;
    
    usleep(3000000);
    
    // Check task 2048 and 2049
    for (int i = 2048; i <= 2049; i++) {
        int slot = PTO2_TASK_SLOT(i);
        PTO2TaskDescriptor* task = pto2_sm_get_task(base->sm_handle, i);
        
        printf("Task %d:\n", i);
        printf("  state=%d, fanout=%d, refcount=%d, scope=%d\n",
               sched->task_state[slot], task->fanout_count,
               sched->fanout_refcount[slot], task->scope_depth);
        printf("  fanin_count=%d, fanout_head=%d\n", 
               task->fanin_count, task->fanout_head);
        
        // Print fanout list
        if (task->fanout_head > 0) {
            printf("  fanout tasks: ");
            int head = task->fanout_head;
            while (head > 0) {
                PTO2DepListEntry* e = pto2_dep_pool_get(&base->orchestrator.dep_pool, head);
                if (!e) break;
                printf("%d ", e->task_id);
                head = e->next_offset;
            }
            printf("\n");
        }
        printf("\n");
    }
    
    pto2_runtime_stop_threads(rt);
    pto2_runtime_destroy_threaded(rt);
    free(A); free(P);
    return 0;
}
