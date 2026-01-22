/**
 * Memory Allocator Implementation
 *
 * This file implements centralized device memory management using the
 * Ascend CANN runtime API.
 */

#include "memoryallocator.h"
#include <runtime/rt.h>
#include <iostream>

int MemoryAllocator::Init(int deviceId) {
    if (initialized_) {
        std::cerr << "Error: MemoryAllocator already initialized\n";
        return -1;
    }

    deviceId_ = deviceId;
    initialized_ = true;
    std::cout << "MemoryAllocator initialized for device " << deviceId << '\n';
    return 0;
}

void* MemoryAllocator::Alloc(size_t size) {
    if (!initialized_) {
        std::cerr << "Error: MemoryAllocator not initialized\n";
        return nullptr;
    }

    void* ptr = nullptr;
    int rc = rtMalloc(&ptr, size, RT_MEMORY_HBM, 0);
    if (rc != 0) {
        std::cerr << "Error: rtMalloc failed: " << rc << " (size=" << size << ")\n";
        return nullptr;
    }

    // Track the pointer
    ptrSet_.insert(ptr);
    return ptr;
}

int MemoryAllocator::Free(void* ptr) {
    if (ptr == nullptr) {
        return 0;
    }

    // Check if we're tracking this pointer
    auto it = ptrSet_.find(ptr);
    if (it == ptrSet_.end()) {
        // Not tracked by us, don't free
        return 0;
    }

    // Free the memory
    int rc = rtFree(ptr);
    if (rc != 0) {
        std::cerr << "Error: rtFree failed: " << rc << '\n';
        return rc;
    }

    // Remove from tracking set
    ptrSet_.erase(it);
    return 0;
}

int MemoryAllocator::Finalize() {
    if (!initialized_) {
        return 0;
    }

    int lastError = 0;
    size_t count = ptrSet_.size();

    if (count > 0) {
        std::cout << "MemoryAllocator: Cleaning up " << count << " remaining allocation(s)\n";
    }

    // Free all remaining tracked pointers
    for (void* ptr : ptrSet_) {
        int rc = rtFree(ptr);
        if (rc != 0) {
            std::cerr << "Error: rtFree failed during Finalize: " << rc << '\n';
            lastError = rc;
        }
    }

    // Clear the set
    ptrSet_.clear();
    initialized_ = false;
    deviceId_ = -1;

    if (count > 0) {
        std::cout << "MemoryAllocator finalized\n";
    }

    return lastError;
}
