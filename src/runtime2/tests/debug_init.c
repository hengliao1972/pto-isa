#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "../pto_runtime2.h"
#include "../pto_runtime2_threaded.h"
#include "../pto_runtime2_types.h"
#include "../pto_scheduler.h"

int main() {
    PTO2RuntimeThreaded* rt = pto2_runtime_create_threaded_custom(4, 4, true, 16384, 64*1024*1024, 65536);
    PTO2Runtime* base = (PTO2Runtime*)rt;
    PTO2SchedulerState* sched = &rt->base.scheduler;
    
    float* A = calloc(1024, sizeof(float));
    float* P = calloc(1024, sizeof(float));
    
    // Don't start scheduler thread, manually test initialization
    
    // Submit just 16 tasks
    pto2_rt_scope_begin(base);
    for (int k = 0; k < 8; k++) {
        PTO2TaskParam gemm_params[3] = {
            PTO2_INPUT(A, k, 128),
            PTO2_INPUT(A, k, 128),
            PTO2_OUTPUT(P, k, 128)
        };
        pto2_rt_submit_task(base, 0, PTO2_WORKER_CUBE, NULL, "gemm", gemm_params, 3);
        
        PTO2TaskParam add_params[3] = {
            PTO2_INPUT(P, k, 128),
            PTO2_INPUT(P, k, 128),
            PTO2_OUTPUT(P, k, 128)
        };
        pto2_rt_submit_task(base, 1, PTO2_WORKER_VECTOR, NULL, "add", add_params, 3);
    }
    pto2_rt_scope_end(base);
    
    printf("Submitted 16 tasks, current_task_index=%d\n", 
           base->sm_handle->header->current_task_index);
    
    // Check task descriptors
    printf("\nTask descriptors:\n");
    for (int i = 0; i < 16; i++) {
        PTO2TaskDescriptor* task = pto2_sm_get_task(base->sm_handle, i);
        int slot = pto2_task_slot(sched, i);
        printf("  Task %d (slot %d): task_id=%d, is_active=%d, fanin=%d, fanout=%d, scope=%d\n",
               i, slot, task->task_id, task->is_active, task->fanin_count, 
               task->fanout_count, task->scope_depth);
    }
    
    // Manually initialize tasks in scheduler
    printf("\nManually initializing tasks in scheduler:\n");
    for (int i = 0; i < 16; i++) {
        PTO2TaskDescriptor* task = pto2_sm_get_task(base->sm_handle, i);
        int slot = pto2_task_slot(sched, i);
        
        sched->task_state[slot] = PTO2_TASK_PENDING;
        sched->fanin_refcount[slot] = 0;
        
        if (task->fanin_count == 0) {
            sched->task_state[slot] = PTO2_TASK_READY;
        }
        
        printf("  Task %d: state=%d\n", i, sched->task_state[slot]);
    }
    
    pto2_runtime_destroy_threaded(rt);
    free(A); free(P);
    return 0;
}
