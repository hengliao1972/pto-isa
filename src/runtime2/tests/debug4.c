#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "../pto_runtime2.h"
#include "../pto_runtime2_threaded.h"
#include "../pto_runtime2_types.h"

void orch_func(PTO2Runtime* rt, void* arg) {
    printf("Orch: Starting...\n"); fflush(stdout);
    
    float* A = calloc(1024, sizeof(float));
    float* P = calloc(1024, sizeof(float));
    
    for (int i = 0; i < 100; i++) {
        PTO2TaskParam p[3] = {
            PTO2_INPUT(A, i, 128),
            PTO2_INPUT(A, i, 128),
            PTO2_OUTPUT(P, i, 128)
        };
        pto2_rt_submit_task(rt, 0, PTO2_WORKER_CUBE, NULL, "task", p, 3);
        
        if (i % 10 == 0) {
            printf("Orch: Submitted %d tasks\n", i+1); fflush(stdout);
        }
    }
    
    printf("Orch: Done submitting\n"); fflush(stdout);
    free(A); free(P);
}

int main() {
    printf("Creating runtime...\n"); fflush(stdout);
    PTO2RuntimeThreaded* rt = pto2_runtime_create_threaded_custom(
        4, 4, true, 256, PTO2_HEAP_SIZE, PTO2_DEP_LIST_POOL_SIZE);
    
    printf("Starting threads...\n"); fflush(stdout);
    pto2_runtime_run_threaded(rt, orch_func, NULL);
    
    printf("Done! consumed=%lld\n", (long long)rt->base.scheduler.tasks_consumed);
    pto2_runtime_destroy_threaded(rt);
    return 0;
}
