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
    
    usleep(5000000);
    
    int last = sched->last_task_alive;
    printf("last_task_alive=%d\n\n", last);
    
    // Trace the dependency chain from last_task_alive
    printf("Dependency chain starting from Task %d:\n", last);
    int current = last;
    for (int depth = 0; depth < 20 && current < 8192; depth++) {
        int slot = pto2_task_slot(sched, current);
        PTO2TaskDescriptor* t = pto2_sm_get_task(base->sm_handle, current);
        
        printf("  Task %d (%s): state=%d, fanin=%d, fanin_refcount=%d, fanout=%d, fanout_refcount=%d\n",
               current, t->func_name, sched->task_state[slot], 
               t->fanin_count, sched->fanin_refcount[slot],
               t->fanout_count, sched->fanout_refcount[slot]);
        
        if (sched->task_state[slot] == 4) {  // CONSUMED
            current++;
            continue;
        }
        
        // If not consumed, show why
        if (sched->task_state[slot] == 3) {  // COMPLETED
            printf("    -> COMPLETED but not CONSUMED: fanout_refcount(%d) < fanout_count(%d)\n",
                   sched->fanout_refcount[slot], t->fanout_count);
        } else if (sched->task_state[slot] == 0) {  // PENDING
            printf("    -> PENDING: fanin_refcount(%d) < fanin_count(%d)\n",
                   sched->fanin_refcount[slot], t->fanin_count);
            
            // Show which producer hasn't completed
            int head = t->fanin_head;
            while (head > 0) {
                PTO2DepListEntry* e = pto2_dep_pool_get(&base->orchestrator.dep_pool, head);
                if (!e) break;
                int pid = e->task_id;
                int pslot = pto2_task_slot(sched, pid);
                printf("    -> Producer %d: state=%d\n", pid, sched->task_state[pslot]);
                head = e->next_offset;
            }
        }
        
        current++;
    }
    
    pto2_runtime_stop_threads(rt);
    pto2_runtime_destroy_threaded(rt);
    free(A); free(B); free(C); free(P);
    return 0;
}
