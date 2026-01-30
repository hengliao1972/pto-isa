#include <stdio.h>
#include <stdlib.h>
#include "../pto_runtime2.h"
#include "../pto_runtime2_threaded.h"
#include "../pto_runtime2_types.h"
#include "../pto_orchestrator.h"

PTO2RuntimeThreaded* g_rt;

int main() {
    g_rt = pto2_runtime_create_threaded_custom(
        4, 4, true, 8192, PTO2_HEAP_SIZE, PTO2_DEP_LIST_POOL_SIZE);
    
    PTO2Runtime* rt = (PTO2Runtime*)g_rt;
    PTO2OrchestratorState* orch = &rt->orchestrator;
    
    float* A = calloc(4096, sizeof(float));
    float* P = calloc(4096, sizeof(float));
    
    // NOT starting threads - just check scope behavior
    
    printf("Testing scope begin/end positions\n\n");
    
    for (int b = 0; b < 8; b++) {
        printf("=== Batch %d ===\n", b);
        
        // Record state before scope_begin
        int32_t current_before = orch->task_ring.current_index;
        
        pto2_rt_scope_begin(rt);
        int32_t scope_begin_pos = orch->scope_stack[orch->scope_stack_top];
        printf("  scope_begin: current=%d, stack_pos=%d\n", current_before, scope_begin_pos);
        
        for (int i = 0; i < 500; i++) {
            int idx = b * 500 + i;
            PTO2TaskParam p[3] = {
                PTO2_INPUT(A, idx, 128),
                PTO2_INPUT(A, idx, 128),
                PTO2_OUTPUT(P, idx, 128)
            };
            pto2_rt_submit_task(rt, 0, PTO2_WORKER_CUBE, NULL, "task", p, 3);
        }
        
        int32_t current_after = orch->task_ring.current_index;
        printf("  After submit: current=%d (submitted %d tasks)\n", 
               current_after, current_after - scope_begin_pos);
        
        // scope_end will call pto2_scheduler_on_scope_end(sched, begin_pos, end_pos)
        // Check what range that covers
        printf("  scope_end will release: [%d, %d)\n", scope_begin_pos, current_after);
        
        pto2_rt_scope_end(rt);
        printf("\n");
    }
    
    pto2_runtime_destroy_threaded(g_rt);
    free(A); free(P);
    return 0;
}
