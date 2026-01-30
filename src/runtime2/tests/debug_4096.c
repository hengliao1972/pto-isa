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
    
    // Check tasks 4096 and 4097 immediately after submission
    printf("After submission:\n");
    PTO2TaskDescriptor* t4096 = pto2_sm_get_task(base->sm_handle, 4096);
    PTO2TaskDescriptor* t4097 = pto2_sm_get_task(base->sm_handle, 4097);
    
    printf("  Task 4096: fanout_head=%d, fanout_count=%d\n", t4096->fanout_head, t4096->fanout_count);
    printf("  Task 4097: fanin_head=%d, fanin_count=%d\n", t4097->fanin_head, t4097->fanin_count);
    
    // Check fanout list of 4096
    printf("\n  Task 4096 fanout list:\n");
    int head = t4096->fanout_head;
    while (head > 0) {
        PTO2DepListEntry* e = pto2_dep_pool_get(&base->orchestrator.dep_pool, head);
        if (!e) { printf("    [NULL entry]\n"); break; }
        printf("    offset=%d: consumer=%d\n", head, e->task_id);
        head = e->next_offset;
    }
    
    // Check fanin list of 4097
    printf("\n  Task 4097 fanin list:\n");
    head = t4097->fanin_head;
    while (head > 0) {
        PTO2DepListEntry* e = pto2_dep_pool_get(&base->orchestrator.dep_pool, head);
        if (!e) { printf("    [NULL entry]\n"); break; }
        printf("    offset=%d: producer=%d\n", head, e->task_id);
        head = e->next_offset;
    }
    
    pto2_rt_orchestration_done(base);
    rt->thread_ctx.orchestrator_done = true;
    
    usleep(5000000);
    
    printf("\nAfter execution:\n");
    int slot_4096 = pto2_task_slot(sched, 4096);
    int slot_4097 = pto2_task_slot(sched, 4097);
    printf("  Task 4096: state=%d, fanout_refcount=%d\n", sched->task_state[slot_4096], sched->fanout_refcount[slot_4096]);
    printf("  Task 4097: state=%d, fanin_refcount=%d\n", sched->task_state[slot_4097], sched->fanin_refcount[slot_4097]);
    
    pto2_runtime_stop_threads(rt);
    pto2_runtime_destroy_threaded(rt);
    free(A); free(B); free(C); free(P);
    return 0;
}
