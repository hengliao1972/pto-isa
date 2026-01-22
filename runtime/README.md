# PTO Runtime - Graph Execution Framework

ARM64 runtime for executing task dependency graphs on Ascend devices with AICPU and AICore coordination.

## Overview

This runtime enables building and executing computational task graphs on Ascend devices. It provides:
- Task dependency graph management
- Device memory allocation and management
- AICPU and AICore kernel coordination
- Host-device data transfer
- **Python bindings for easy integration**

## Directory Structure

```
runtime/
├── graph/              # Task dependency graph (merged from graph + common)
│   ├── graph.h/cpp         # Graph class implementation
│   ├── handshake.h         # AICPU-AICore handshake protocol
│   └── kernel_args.h       # Kernel argument structures
├── host/               # Host-side runtime (NEW)
│   ├── devicerunner.h/cpp  # Device execution interface
│   └── memoryallocator.h/cpp # Memory management
├── aicpu/              # AICPU kernel implementation
│   ├── graph_executor.cpp  # Task scheduler for AICPU
│   └── device_log.h/cpp    # Device logging utilities
├── aicore/             # AICore kernel implementation
│   └── kernel.cpp          # Task execution kernels (add, mul, etc.)
├── python/             # Python bindings (NEW)
│   ├── bindings.cpp        # pybind11 bindings
│   ├── graphbuilder.py     # Python example
│   ├── CMakeLists.txt      # Python module build config
│   └── README.md           # Python API documentation
├── graphbuilder.cpp    # C++ example application
└── CMakeLists.txt      # Main build configuration
```

## Key Components

### Graph Class ([graph/](graph/))
- Manages task dependency graphs with fixed-size arrays
- Tracks task arguments, dependencies (fanin/fanout), and execution state
- Provides topological ordering for execution

### DeviceRunner ([host/devicerunner.h](host/devicerunner.h))
- Singleton interface for device operations
- Manages AICPU and AICore kernel launching
- Handles memory allocation and data transfer
- Coordinates graph execution workflow

### MemoryAllocator ([host/memoryallocator.h](host/memoryallocator.h))
- Centralized device memory management
- Automatic tracking of allocations
- Prevents memory leaks with automatic cleanup

### AICPU Graph Executor ([aicpu/graph_executor.cpp](aicpu/graph_executor.cpp))
- Task scheduler running on AICPU
- Manages handshake protocol with AICore
- Dispatches ready tasks to AICore cores
- Implements task dependency resolution

### AICore Kernels ([aicore/kernel.cpp](aicore/kernel.cpp))
- Task execution on AICore using PTO ISA
- Implements arithmetic operations (add, mul, etc.)
- Polls handshake buffer for task assignments

## Building

### Prerequisites

- CMake 3.15+
- CANN toolkit (Ascend runtime)
- GCC/G++ with C++17 support
- **Python 3 with development headers (for Python bindings)**
- **pybind11 (automatically fetched if not available)**

### Environment Setup

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
```

### Build C++ Components

```bash
cd runtime
mkdir -p build && cd build
cmake ..
make
```

This produces:
- `graph_executor` - C++ example executable
- `aicpu/libaicpu_graph_kernel.so` - AICPU kernel library
- `aicore/kernel.o` - AICore kernel binary

### Build Python Bindings

```bash
cd runtime
mkdir -p build && cd build
cmake .. -DBUILD_PYTHON_BINDINGS=ON
make
```

This additionally builds:
- `python/pto_runtime.so` - Python module

To disable Python bindings:
```bash
cmake .. -DBUILD_PYTHON_BINDINGS=OFF
```

## Usage

### C++ Example

```bash
cd runtime/build
./graph_executor 9  # Run on device 9
```

Expected output:
```
=== Graph Builder Example ===
...
✓ SUCCESS: All 16384 elements are correct (42.0)
Formula verified: (a + b + 1)(a + b + 2) = (2+3+1)*(2+3+2) = 42
```

### Python Example

See [python/README.md](python/README.md) for detailed Python API documentation.

```bash
cd runtime/build/python
export PYTHONPATH=$(pwd):$PYTHONPATH
python3 ../../python/graphbuilder.py 9
```

Quick Python example:
```python
import numpy as np
import pto_runtime

# Initialize device
runner = pto_runtime.DeviceRunner.get()
runner.init(9, 3, "./aicpu/libaicpu_graph_kernel.so", "./aicore/kernel.o")

# Allocate tensors
dev_a = runner.allocate_tensor(128 * 128 * 4)
dev_b = runner.allocate_tensor(128 * 128 * 4)
dev_c = runner.allocate_tensor(128 * 128 * 4)

# Copy data
a = np.full((128, 128), 2.0, dtype=np.float32)
b = np.full((128, 128), 3.0, dtype=np.float32)
runner.copy_to_device(dev_a, a)
runner.copy_to_device(dev_b, b)

# Build and run graph
graph = pto_runtime.Graph()
t0 = graph.add_task([dev_a, dev_b, dev_c, 128*128], func_id=0)
runner.run(graph)

# Get results
result = np.zeros((128, 128), dtype=np.float32)
runner.copy_from_device(result, dev_c)

# Cleanup
runner.free_tensor(dev_a)
runner.free_tensor(dev_b)
runner.free_tensor(dev_c)
runner.finalize()
```

## Architecture

### Execution Flow

```
1. Host (graphbuilder.cpp or Python):
   ├─ Build task graph (Graph class)
   ├─ Allocate device memory (DeviceRunner)
   ├─ Copy input data to device
   └─ Call DeviceRunner::Run(graph)

2. DeviceRunner::Run():
   ├─ Copy graph to device memory
   ├─ Launch AICPU init kernel (handshake)
   ├─ Launch AICPU main kernel (scheduler)
   ├─ Launch AICore kernel (workers)
   └─ Synchronize streams

3. AICPU (graph_executor.cpp):
   ├─ Handshake with AICore cores
   ├─ Find initially ready tasks (fanin=0)
   ├─ Dispatch tasks to idle AICore cores
   ├─ Wait for task completion
   ├─ Update fanin counters
   └─ Repeat until all tasks done

4. AICore (kernel.cpp):
   ├─ Wait for task assignment
   ├─ Read task arguments
   ├─ Execute kernel (add/mul/etc.)
   ├─ Signal completion
   └─ Repeat until quit signal
```

### Handshake Protocol

Each AICore core has a dedicated handshake buffer:

```c
struct Handshake {
    volatile uint32_t aicpu_ready;  // AICPU→AICore: scheduler ready
    volatile uint32_t aicore_done;  // AICore→AICPU: core ready
    volatile uint64_t task;         // AICPU→AICore: task pointer
    volatile int32_t task_status;   // Task state: 1=busy, 0=done
    volatile int32_t control;       // AICPU→AICore: 1=quit
};
```

## Recent Changes

### Reorganization (Current)

1. **Merged `graph/` and `common/`**
   - Moved `handshake.h` and `kernel_args.h` from `common/` to `graph/`
   - All shared structures now in one location
   - Simplified include paths

2. **Created `host/` directory**
   - Moved `devicerunner.cpp/h` and `memoryallocator.cpp/h`
   - Separates host-side runtime from device kernels
   - Clearer organization

3. **Added Python bindings**
   - New `python/` directory with pybind11 bindings
   - Python API matches C++ interface
   - NumPy integration for efficient data transfer
   - Example Python application (`graphbuilder.py`)

## Logging

Device logs are written to `~/ascend/log/debug/device-<id>/`

Kernel uses `DEV_INFO`, `DEV_DEBUG`, `DEV_WARN`, `DEV_ERROR` macros.

## Notes

- Device ID range: 0-15
- Default device: 9
- Graph supports up to 1024 tasks (configurable via `GRAPH_MAX_TASKS`)
- Memory allocator automatically tracks and frees allocations
- Python bindings require NumPy for array operations

## References

- See [graph/README.md](graph/README.md) for Graph class details
- See [python/README.md](python/README.md) for Python API documentation
- See example: [graphbuilder.cpp](graphbuilder.cpp) or [python/graphbuilder.py](python/graphbuilder.py)
