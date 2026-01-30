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
                    pto2_rt_submit_task(base, 0, PTO2_WORKER_CUBE, NULL, "gemm_tile", gemm_params, 3);
                    
                    PTO2TaskParam add_params[3] = {
                        PTO2_INPUT(C, c_idx, 128),
                        PTO2_INPUT(P, c_idx, 128),
                        PTO2_OUTPUT(C, c_idx, 128)
                    };
                    pto2_rt_submit_task(base, 1, PTO2_WORKER_VECTOR, NULL, "tile_add", add_params, 3);
                }
                pto2_rt_scope_end(base);
            }
        }
        pto2_rt_scope_end(base);
    }
    
    pto2_rt_orchestration_done(base);
    rt->thread_ctx.orchestrator_done = true;
    
    usleep(5000000);  // Wait 5s
    
    printf("completed=%lld, consumed=%lld\n\n", 
           (long long)sched->tasks_completed, (long long)sched->tasks_consumed);
    
    // Find stuck producer (gemm task that's COMPLETED but not CONSUMED)
    printf("Finding stuck producers and their consumers:\n");
    int found = 0;
    for (int tid = sched->last_task_alive; tid < 8192 && found < 3; tid++) {
        int slot = pto2_task_slot(sched, tid);
        if (sched->task_state[slot] == PTO2_TASK_COMPLETED) {
            PTO2TaskDescriptor* task = pto2_sm_get_task(base->sm_handle, tid);
            printf("\nStuck producer Task %d (%s):\n", tid, task->func_name);
            printf("  fanout_count=%d, fanout_refcount=%d, scope_depth=%d\n",
                   task->fanout_count, sched->fanout_refcount[slot], task->scope_depth);
            
            // Find consumers
            printf("  Consumers (from fanout_head=%d):\n", task->fanout_head);
            int head = task->fanout_head;
            while (head > 0) {
                PTO2DepListEntry* e = pto2_dep_pool_get(&base->orchestrator.dep_pool, head);
                if (!e) break;
                int consumer_id = e->task_id;
                int consumer_slot = pto2_task_slot(sched, consumer_id);
                PTO2TaskDescriptor* consumer = pto2_sm_get_task(base->sm_handle, consumer_id);
                printf("    Consumer %d (%s): state=%d, fanin_count=%d, fanin_head=%d\n",
                       consumer_id, consumer->func_name, sched->task_state[consumer_slot],
                       consumer->fanin_count, consumer->fanin_head);
                
                // Check if consumer's fanin contains this producer
                int fanin = consumer->fanin_head;
                printf("      Fanin producers: ");
                while (fanin > 0) {
                    PTO2DepListEntry* fe = pto2_dep_pool_get(&base->orchestrator.dep_pool, fanin);
                    if (!fe) break;
                    printf("%d ", fe->task_id);
                    fanin = fe->next_offset;
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
