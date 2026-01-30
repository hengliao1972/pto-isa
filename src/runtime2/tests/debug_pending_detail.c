#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "../pto_runtime2.h"
#include "../pto_runtime2_threaded.h"
#include "../pto_scheduler.h"
#include "../pto_ring_buffer.h"

int main() {
    PTO2RuntimeThreaded* rt = pto2_runtime_create_threaded_custom(4, 4, true, 16384, 64*1024*1024, 65536);
    PTO2Runtime* base = (PTO2Runtime*)rt;
    PTO2SchedulerState* sched = &rt->base.scheduler;
    PTO2DepListPool* dep_pool = &rt->base.orchestrator.dep_pool;
    
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
                        PTO2_INPUT(A, a_idx, 128), PTO2_INPUT(B, b_idx, 128), PTO2_OUTPUT(P, c_idx, 128)
                    };
                    pto2_rt_submit_task(base, 0, PTO2_WORKER_CUBE, NULL, "gemm", gemm_params, 3);
                    
                    PTO2TaskParam add_params[3] = {
                        PTO2_INPUT(C, c_idx, 128), PTO2_INPUT(P, c_idx, 128), PTO2_OUTPUT(C, c_idx, 128)
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
    
    printf("=== PENDING 任务详情 ===\n");
    for (int i = 0; i < 8192; i++) {
        int slot = pto2_task_slot(sched, i);
        if (sched->task_state[slot] == PTO2_TASK_PENDING) {
            PTO2TaskDescriptor* t = pto2_sm_get_task(base->sm_handle, i);
            printf("\n[Task %d] %s: PENDING\n", i, t->func_name);
            printf("  fanin_count (descriptor): %d\n", t->fanin_count);
            printf("  fanin_refcount (sched):   %d\n", sched->fanin_refcount[slot]);
            printf("  fanin_head:               %d\n", t->fanin_head);
            
            // List producers
            int32_t cur = t->fanin_head;
            int prod_idx = 0;
            while (cur > 0) {
                PTO2DepListEntry* e = pto2_dep_pool_get(dep_pool, cur);
                if (!e) break;
                int prod_id = e->task_id;
                int prod_slot = pto2_task_slot(sched, prod_id);
                PTO2TaskDescriptor* prod = pto2_sm_get_task(base->sm_handle, prod_id);
                const char* states[] = {"PENDING", "READY", "RUNNING", "COMPLETED", "CONSUMED"};
                printf("  Producer[%d]: Task %d (%s) state=%s\n", 
                       prod_idx++, prod_id, prod->func_name, states[sched->task_state[prod_slot]]);
                cur = e->next_offset;
            }
        }
    }
    
    pto2_runtime_stop_threads(rt);
    pto2_runtime_destroy_threaded(rt);
    free(A); free(B); free(C); free(P);
    return 0;
}
