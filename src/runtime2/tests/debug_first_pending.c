#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "../pto_runtime2.h"
#include "../pto_runtime2_threaded.h"
#include "../pto_runtime2_types.h"
#include "../pto_scheduler.h"
#include "../pto_orchestrator.h"

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
    
    // Find first PENDING task with refcount=0
    for (int i = 0; i < 8192; i++) {
        int slot = pto2_task_slot(sched, i);
        PTO2TaskDescriptor* t = pto2_sm_get_task(base->sm_handle, i);
        
        if (sched->task_state[slot] == 0 && sched->fanin_refcount[slot] == 0) {
            printf("First PENDING with refcount=0: Task %d\n", i);
            printf("  fanin_count=%d, fanin_head=%d\n\n", t->fanin_count, t->fanin_head);
            
            // Show producers
            int head = t->fanin_head;
            while (head > 0) {
                PTO2DepListEntry* e = pto2_dep_pool_get(&base->orchestrator.dep_pool, head);
                if (!e) break;
                
                int pid = e->task_id;
                int pslot = pto2_task_slot(sched, pid);
                PTO2TaskDescriptor* producer = pto2_sm_get_task(base->sm_handle, pid);
                
                printf("Producer %d (%s):\n", pid, producer->func_name);
                printf("  state=%d, fanout_head=%d, fanout_count=%d, fanout_refcount=%d\n",
                       sched->task_state[pslot], producer->fanout_head,
                       producer->fanout_count, sched->fanout_refcount[pslot]);
                
                // Check if producer has consumer in fanout list
                int fhead = producer->fanout_head;
                printf("  Fanout list: ");
                while (fhead > 0) {
                    PTO2DepListEntry* fe = pto2_dep_pool_get(&base->orchestrator.dep_pool, fhead);
                    if (!fe) break;
                    printf("%d ", fe->task_id);
                    fhead = fe->next_offset;
                }
                printf("\n\n");
                
                head = e->next_offset;
            }
            break;
        }
    }
    
    pto2_runtime_stop_threads(rt);
    pto2_runtime_destroy_threaded(rt);
    free(A); free(B); free(C); free(P);
    return 0;
}
