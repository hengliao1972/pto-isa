#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "../pto_runtime2.h"
#include "../pto_runtime2_threaded.h"
#include "../pto_scheduler.h"

int main() {
    PTO2RuntimeThreaded* rt = pto2_runtime_create_threaded_custom(4, 4, true, 16384, 64*1024*1024, 65536);
    PTO2ThreadContext* ctx = &rt->thread_ctx;
    
    printf("Completion queue capacity: %d\n", ctx->completion_queue.capacity);
    printf("Task window size: %d\n", rt->base.scheduler.task_window_size);
    
    pto2_runtime_destroy_threaded(rt);
    return 0;
}
