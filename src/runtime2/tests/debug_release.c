#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "../pto_runtime2.h"
#include "../pto_runtime2_threaded.h"
#include "../pto_runtime2_types.h"
#include "../pto_scheduler.h"

// Override release_producer to track calls
static int release_count[8192] = {0};

void my_release_producer(PTO2SchedulerState* sched, int32_t producer_id) {
    int32_t slot = PTO2_TASK_SLOT(producer_id);
    release_count[slot]++;
    if (slot == 0) {
        printf("  release_producer called for task %d (slot 0), total calls: %d\n", 
               producer_id, release_count[0]);
    }
    pto2_scheduler_release_producer(sched, producer_id);
}

PTO2RuntimeThreaded* g_rt;

int main() {
    g_rt = pto2_runtime_create_threaded_custom(
        4, 4, true, 8192, PTO2_HEAP_SIZE, PTO2_DEP_LIST_POOL_SIZE);
    
    PTO2Runtime* rt = (PTO2Runtime*)g_rt;
    PTO2SchedulerState* sched = &g_rt->base.scheduler;
    
    float* A = calloc(4096, sizeof(float));
    float* P = calloc(4096, sizeof(float));
    
    // NOT starting threads - just check scope_end behavior
    
    printf("Testing which scope_end calls release_producer on Task 0\n\n");
    
    for (int b = 0; b < 8; b++) {
        printf("=== Batch %d ===\n", b);
        pto2_rt_scope_begin(rt);
        
        for (int i = 0; i < 500; i++) {
            int idx = b * 500 + i;
            PTO2TaskParam p[3] = {
                PTO2_INPUT(A, idx, 128),
                PTO2_INPUT(A, idx, 128),
                PTO2_OUTPUT(P, idx, 128)
            };
            pto2_rt_submit_task(rt, 0, PTO2_WORKER_CUBE, NULL, "task", p, 3);
        }
        
        printf("  Calling scope_end (releases tasks [%d, %d))...\n", 
               b*500, (b+1)*500);
        printf("  Task 0 refcount before: %d\n", sched->fanout_refcount[0]);
        
        pto2_rt_scope_end(rt);
        
        printf("  Task 0 refcount after:  %d\n", sched->fanout_refcount[0]);
        printf("  Total release calls to slot 0: %d\n\n", release_count[0]);
    }
    
    pto2_runtime_destroy_threaded(g_rt);
    free(A); free(P);
    return 0;
}
