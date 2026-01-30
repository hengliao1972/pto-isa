#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include "../pto_runtime2.h"
#include "../pto_runtime2_threaded.h"
#include "../pto_runtime2_types.h"
#include "../pto_scheduler.h"

PTO2RuntimeThreaded* g_rt;

void orch_func(PTO2Runtime* rt, void* arg) {
    printf("[Orch] Starting orchestration...\n"); fflush(stdout);
    
    float* A = calloc(100, sizeof(float));
    float* P = calloc(100, sizeof(float));
    
    pto2_rt_scope_begin(rt);
    for (int i = 0; i < 10; i++) {
        PTO2TaskParam p[3] = {
            PTO2_INPUT(A, i, 128),
            PTO2_INPUT(A, i, 128),
            PTO2_OUTPUT(P, i, 128)
        };
        pto2_rt_submit_task(rt, 0, PTO2_WORKER_CUBE, NULL, "task", p, 3);
        printf("[Orch] Submitted task %d\n", i); fflush(stdout);
    }
    pto2_rt_scope_end(rt);
    
    printf("[Orch] Done submitting\n"); fflush(stdout);
    free(A); free(P);
}

int main() {
    printf("Creating runtime...\n"); fflush(stdout);
    g_rt = pto2_runtime_create_threaded_custom(
        2, 2, true, 256, PTO2_HEAP_SIZE, PTO2_DEP_LIST_POOL_SIZE);
    
    PTO2ThreadContext* ctx = &g_rt->thread_ctx;
    
    printf("Thread context:\n");
    printf("  num_workers: %d\n", ctx->num_workers);
    printf("  num_cube_workers: %d\n", ctx->num_cube_workers);
    printf("  num_vector_workers: %d\n", ctx->num_vector_workers);
    printf("  scheduler_running: %d\n", ctx->scheduler_running);
    printf("  shutdown: %d\n", ctx->shutdown);
    fflush(stdout);
    
    printf("\nStarting threads...\n"); fflush(stdout);
    pto2_runtime_start_threads(g_rt);
    
    printf("After start_threads:\n");
    printf("  scheduler_running: %d\n", ctx->scheduler_running);
    for (int i = 0; i < ctx->num_workers; i++) {
        printf("  worker[%d].thread: %lu, shutdown: %d\n", 
               i, (unsigned long)ctx->workers[i].thread,
               ctx->workers[i].shutdown);
    }
    fflush(stdout);
    
    // Run orchestration
    printf("\nRunning orchestration...\n"); fflush(stdout);
    orch_func((PTO2Runtime*)g_rt, NULL);
    
    // Mark done
    pto2_rt_orchestration_done((PTO2Runtime*)g_rt);
    ctx->orchestrator_done = true;
    
    printf("\nWaiting 2s...\n"); fflush(stdout);
    for (int i = 0; i < 4; i++) {
        usleep(500000);
        PTO2SchedulerState* sched = &g_rt->base.scheduler;
        printf("  %dms: completed=%lld, consumed=%lld, last_alive=%d\n",
               (i+1)*500, (long long)sched->tasks_completed,
               (long long)sched->tasks_consumed, sched->last_task_alive);
        
        // Check worker stats
        for (int w = 0; w < ctx->num_workers; w++) {
            printf("    worker[%d]: tasks_executed=%lld\n", 
                   w, (long long)ctx->workers[w].tasks_executed);
        }
        fflush(stdout);
    }
    
    pto2_runtime_stop_threads(g_rt);
    pto2_runtime_destroy_threaded(g_rt);
    return 0;
}
