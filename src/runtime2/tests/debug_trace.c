#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "../pto_runtime2.h"
#include "../pto_runtime2_threaded.h"
#include "../pto_runtime2_types.h"
#include "../pto_scheduler.h"
#include "../pto_orchestrator.h"

// Monitor a specific task
#define MONITOR_TASK 4035

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
    
    // Monitor Task MONITOR_TASK periodically
    int monitor_slot = pto2_task_slot(sched, MONITOR_TASK);
    PTO2TaskDescriptor* monitor_task = pto2_sm_get_task(base->sm_handle, MONITOR_TASK);
    
    printf("Monitoring Task %d:\n", MONITOR_TASK);
    printf("  fanin_count=%d, fanin_head=%d\n", monitor_task->fanin_count, monitor_task->fanin_head);
    
    // Show its producers
    printf("  Producers: ");
    int head = monitor_task->fanin_head;
    while (head > 0) {
        PTO2DepListEntry* e = pto2_dep_pool_get(&base->orchestrator.dep_pool, head);
        if (!e) break;
        printf("%d ", e->task_id);
        head = e->next_offset;
    }
    printf("\n\n");
    
    // Poll state
    for (int i = 0; i < 20; i++) {
        usleep(500000);  // 0.5s
        
        int state = __atomic_load_n(&sched->task_state[monitor_slot], __ATOMIC_ACQUIRE);
        int refcount = __atomic_load_n(&sched->fanin_refcount[monitor_slot], __ATOMIC_ACQUIRE);
        
        printf("t=%.1fs: state=%d, fanin_refcount=%d\n", (i+1)*0.5, state, refcount);
        
        if (state >= 3) {  // COMPLETED or CONSUMED
            printf("  Task reached state %d!\n", state);
            break;
        }
        
        if (i >= 5 && state == 0) {
            printf("  Task stuck in PENDING!\n");
            // Show producer states
            head = monitor_task->fanin_head;
            while (head > 0) {
                PTO2DepListEntry* e = pto2_dep_pool_get(&base->orchestrator.dep_pool, head);
                if (!e) break;
                int pid = e->task_id;
                int pslot = pto2_task_slot(sched, pid);
                printf("    Producer %d: state=%d\n", pid, __atomic_load_n(&sched->task_state[pslot], __ATOMIC_ACQUIRE));
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
