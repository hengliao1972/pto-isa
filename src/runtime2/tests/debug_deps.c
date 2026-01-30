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
    
    // Submit 8x8x8x8 = 8192 tasks
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
    
    usleep(5000000);  // Wait 5s
    
    // Find stuck tasks (state=0 with producers completed)
    printf("Finding stuck consumers:\n");
    int found = 0;
    for (int i = 0; i < 8192 && found < 3; i++) {
        int slot = pto2_task_slot(sched, i);
        if (sched->task_state[slot] == 0) {  // PENDING
            PTO2TaskDescriptor* t = pto2_sm_get_task(base->sm_handle, i);
            printf("\nTask %d (%s): state=0, fanin=%d, fanin_refcount=%d\n",
                   i, t->func_name, t->fanin_count, sched->fanin_refcount[slot]);
            
            // Show producers
            printf("  Producers (from fanin_head=%d):\n", t->fanin_head);
            int head = t->fanin_head;
            while (head > 0) {
                PTO2DepListEntry* e = pto2_dep_pool_get(&base->orchestrator.dep_pool, head);
                if (!e) break;
                int pid = e->task_id;
                int pslot = pto2_task_slot(sched, pid);
                PTO2TaskDescriptor* producer = pto2_sm_get_task(base->sm_handle, pid);
                printf("    Producer %d (%s): state=%d, fanout_head=%d\n",
                       pid, producer->func_name, sched->task_state[pslot], producer->fanout_head);
                
                // Check producer's fanout
                printf("      Fanout consumers: ");
                int fhead = producer->fanout_head;
                while (fhead > 0) {
                    PTO2DepListEntry* fe = pto2_dep_pool_get(&base->orchestrator.dep_pool, fhead);
                    if (!fe) break;
                    printf("%d ", fe->task_id);
                    fhead = fe->next_offset;
                }
                printf("\n");
                
                head = e->next_offset;
            }
            found++;
        }
    }
    
    pto2_runtime_stop_threads(rt);
    pto2_runtime_destroy_threaded(rt);
    free(A); free(B); free(C); free(P);
    return 0;
}
