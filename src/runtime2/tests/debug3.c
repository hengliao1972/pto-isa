#include <stdio.h>
#include <stdlib.h>
#include "../pto_runtime2.h"
#include "../pto_runtime2_threaded.h"
#include "../pto_runtime2_types.h"

int main() {
    printf("Step 1: Creating runtime...\n"); fflush(stdout);
    PTO2RuntimeThreaded* rt = pto2_runtime_create_threaded_custom(
        4, 4, true, 8192, PTO2_HEAP_SIZE, PTO2_DEP_LIST_POOL_SIZE);
    
    if (!rt) {
        printf("Failed to create runtime!\n");
        return 1;
    }
    
    printf("Step 2: Runtime created successfully\n"); fflush(stdout);
    printf("  sm_handle: %p\n", (void*)rt->base.sm_handle);
    printf("  task_window_size: %d\n", rt->base.sm_handle->header->task_window_size);
    
    printf("Step 3: Destroying runtime...\n"); fflush(stdout);
    pto2_runtime_destroy_threaded(rt);
    
    printf("Done!\n");
    return 0;
}
