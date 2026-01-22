# PTO Runtime Python Bindings

Python bindings for the PTO Runtime, enabling graph building and device execution on Ascend devices from Python.

## Building the Bindings

1. Ensure you have Python 3 development headers installed:
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install python3-dev

   # On macOS
   brew install python3
   ```

2. Build the Python module:
   ```bash
   cd runtime
   mkdir -p build && cd build
   cmake .. -DBUILD_PYTHON_BINDINGS=ON
   make
   ```

3. The Python module `pto_runtime` will be built in `runtime/build/python/`

## Using the Bindings

### Setting up the Python path

Add the build directory to your PYTHONPATH:
```bash
export PYTHONPATH=/path/to/runtime/build/python:$PYTHONPATH
```

Or run from the build/python directory:
```bash
cd runtime/build/python
python3 ../../python/graphbuilder.py 9
```

### Example Usage

See [graphbuilder.py](graphbuilder.py) for a complete example. Here's a quick overview:

```python
import numpy as np
import pto_runtime

# Initialize device runner
runner = pto_runtime.DeviceRunner.get()
runner.init(device_id=9, num_cores=3,
            aicpu_so_path="./aicpu/libaicpu_graph_kernel.so",
            aicore_kernel_path="./aicore/kernel.o")

# Allocate device tensors
SIZE = 128 * 128 * 4  # bytes
dev_a = runner.allocate_tensor(SIZE)
dev_b = runner.allocate_tensor(SIZE)
dev_c = runner.allocate_tensor(SIZE)

# Copy data to device
a = np.full((128, 128), 2.0, dtype=np.float32)
b = np.full((128, 128), 3.0, dtype=np.float32)
runner.copy_to_device(dev_a, a)
runner.copy_to_device(dev_b, b)

# Build graph
graph = pto_runtime.Graph()
t0 = graph.add_task([dev_a, dev_b, dev_c, 128*128], func_id=0)
# Add more tasks and dependencies...
graph.add_successor(t0, t1)

# Execute graph
runner.run(graph, launch_aicpu_num=1)

# Get results
result = np.zeros((128, 128), dtype=np.float32)
runner.copy_from_device(result, dev_c)

# Cleanup
runner.free_tensor(dev_a)
runner.free_tensor(dev_b)
runner.free_tensor(dev_c)
runner.finalize()
```

## API Reference

### Graph Class

- `Graph()` - Create a new empty graph
- `add_task(args: list, func_id: int) -> int` - Add a task with arguments and function ID
- `add_successor(from_task: int, to_task: int)` - Add dependency edge
- `get_task_count() -> int` - Get total number of tasks
- `print_graph()` - Print graph structure to stdout

### DeviceRunner Class

- `DeviceRunner.get()` - Get singleton instance (static method)
- `init(device_id, num_cores, aicpu_so_path, aicore_kernel_path) -> int` - Initialize device
- `allocate_tensor(bytes: int) -> int` - Allocate device memory
- `free_tensor(ptr: int)` - Free device memory
- `copy_to_device(dev_ptr: int, host_data: np.ndarray) -> int` - Copy to device
- `copy_from_device(host_data: np.ndarray, dev_ptr: int) -> int` - Copy from device
- `run(graph: Graph, launch_aicpu_num: int = 1) -> int` - Execute graph
- `print_handshake_results()` - Print handshake debugging info
- `finalize() -> int` - Cleanup all resources

## Running the Example

```bash
cd runtime/build/python
python3 ../../python/graphbuilder.py 9
```

Expected output:
```
=== Graph Builder Example (Python) ===
...
✓ SUCCESS: All 16384 elements are correct (42.0)
Formula verified: (a + b + 1)(a + b + 2) = (2+3+1)*(2+3+2) = 42
=== Success ===
```

## Notes

- The bindings use NumPy arrays for efficient data transfer
- Device pointers are represented as Python integers
- Float scalars in task arguments are automatically converted to uint64_t representation
- The module requires the CANN runtime to be installed and `ASCEND_HOME_PATH` to be set
