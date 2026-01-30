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
    
    printf("\n=== 分析卡住的任务 ===\n");
    
    for (int i = 0; i < 8192; i++) {
        int slot = pto2_task_slot(sched, i);
        PTO2TaskState state = sched->task_state[slot];
        if (state == PTO2_TASK_PENDING) {  // 只看 PENDING 的
            PTO2TaskDescriptor* t = pto2_sm_get_task(base->sm_handle, i);
            printf("\n[Task %d] %s: state=PENDING, fanin=%d/%d\n",
                   i, t->func_name, sched->fanin_refcount[slot], t->fanin_count);
            
            // 检查它的 fanin 依赖
            int32_t current = t->fanin_head;
            while (current > 0) {
                PTO2DepListEntry* entry = pto2_dep_pool_get(dep_pool, current);
                if (!entry) break;
                int32_t dep_id = entry->task_id;
                int32_t dep_slot = pto2_task_slot(sched, dep_id);
                PTO2TaskDescriptor* dep = pto2_sm_get_task(base->sm_handle, dep_id);
                printf("  -> depends on Task %d (%s): state=%d, fanout=%d/%d\n",
                       dep_id, dep->func_name, sched->task_state[dep_slot],
                       sched->fanout_refcount[dep_slot], dep->fanout_count);
                
                // 检查依赖任务的 fanout 列表是否包含当前任务
                int found = 0;
                int32_t fo_cur = dep->fanout_head;
                while (fo_cur > 0) {
                    PTO2DepListEntry* fo_entry = pto2_dep_pool_get(dep_pool, fo_cur);
                    if (!fo_entry) break;
                    if (fo_entry->task_id == i) { found = 1; break; }
                    fo_cur = fo_entry->next_offset;
                }
                printf("     (found in fanout list: %s)\n", found ? "YES" : "NO");
                
                current = entry->next_offset;
            }
        }
    }
    
    pto2_runtime_stop_threads(rt);
    pto2_runtime_destroy_threaded(rt);
    free(A); free(B); free(C); free(P);
    return 0;
}
