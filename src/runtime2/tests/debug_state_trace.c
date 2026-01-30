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
    
    printf("\n=== 分析卡住的 PENDING 任务 ===\n");
    int pending_count = 0;
    for (int i = 0; i < 8192 && pending_count < 5; i++) {
        int slot = pto2_task_slot(sched, i);
        if (sched->task_state[slot] == PTO2_TASK_PENDING) {
            pending_count++;
            PTO2TaskDescriptor* t = pto2_sm_get_task(base->sm_handle, i);
            printf("\n[PENDING Task %d] fanin=%d/%d\n", i, sched->fanin_refcount[slot], t->fanin_count);
            
            // 检查每个 producer
            int32_t cur = t->fanin_head;
            while (cur > 0) {
                PTO2DepListEntry* e = pto2_dep_pool_get(dep_pool, cur);
                if (!e) break;
                int prod_id = e->task_id;
                int prod_slot = pto2_task_slot(sched, prod_id);
                PTO2TaskDescriptor* prod = pto2_sm_get_task(base->sm_handle, prod_id);
                
                const char* state_name[] = {"PENDING", "READY", "RUNNING", "COMPLETED", "CONSUMED"};
                printf("  -> Producer %d: state=%s, fanout_count=%d, scope_depth=%d\n",
                       prod_id, state_name[sched->task_state[prod_slot]], 
                       prod->fanout_count, prod->scope_depth);
                       
                cur = e->next_offset;
            }
        }
    }
    
    printf("\n\n=== 分析卡住的 COMPLETED 任务 ===\n");
    int completed_count = 0;
    for (int i = 0; i < 8192 && completed_count < 5; i++) {
        int slot = pto2_task_slot(sched, i);
        if (sched->task_state[slot] == PTO2_TASK_COMPLETED) {
            completed_count++;
            PTO2TaskDescriptor* t = pto2_sm_get_task(base->sm_handle, i);
            printf("[COMPLETED Task %d] fanout=%d/%d, scope_depth=%d\n", 
                   i, sched->fanout_refcount[slot], t->fanout_count, t->scope_depth);
        }
    }
    
    printf("\n=== 结果: completed=%lld, consumed=%lld ===\n",
           (long long)sched->tasks_completed, (long long)sched->tasks_consumed);
    
    pto2_runtime_stop_threads(rt);
    pto2_runtime_destroy_threaded(rt);
    free(A); free(B); free(C); free(P);
    return 0;
}
