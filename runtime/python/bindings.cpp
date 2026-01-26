/**
 * Python Bindings for PTO Runtime
 *
 * This file provides Python bindings for the Graph and DeviceRunner classes
 * using pybind11. It enables building and executing task graphs from Python.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../graph/graph.h"
#include "../host/devicerunner.h"

namespace py = pybind11;

PYBIND11_MODULE(pto_runtime, m) {
    m.doc() = "PTO Runtime - Graph building and device execution for Ascend devices";

    py::class_<TaskProfileRecord>(m, "TaskProfileRecord")
        .def_readonly("task_id", &TaskProfileRecord::task_id)
        .def_readonly("func_id", &TaskProfileRecord::func_id)
        .def_readonly("core_type", &TaskProfileRecord::core_type)
        .def_readonly("exec_core_id", &TaskProfileRecord::exec_core_id)
        .def_readonly("exec_core_type", &TaskProfileRecord::exec_core_type)
        .def_readonly("exec_phys_core_id", &TaskProfileRecord::exec_phys_core_id)
        .def_readonly("start_time", &TaskProfileRecord::start_time)
        .def_readonly("end_time", &TaskProfileRecord::end_time)
        .def_readonly("pmu_cnt", &TaskProfileRecord::pmu_cnt);

    // Bind Graph class
    py::class_<Graph>(m, "Graph")
        .def(py::init<>(), "Create a new empty graph")

        .def("add_task",
            [](Graph& self, py::list args, int func_id, int core_type) {
                // Convert Python list to uint64_t array
                std::vector<uint64_t> args_vec;
                for (auto item : args) {
                    if (py::isinstance<py::int_>(item)) {
                        args_vec.push_back(item.cast<uint64_t>());
                    } else if (py::isinstance<py::float_>(item)) {
                        // Handle float scalars by converting to uint64_t bit representation
                        float f = item.cast<float>();
                        union { float f32; uint64_t u64; } converter;
                        converter.f32 = f;
                        args_vec.push_back(converter.u64);
                    } else {
                        throw std::runtime_error("Task arguments must be integers or floats");
                    }
                }
                return self.add_task(args_vec.data(), args_vec.size(), func_id, core_type);
            },
            py::arg("args"), py::arg("func_id"), py::arg("core_type") = 1,
            "Add a task to the graph with the given arguments and function ID.\n"
            "Args:\n"
            "    args: List of arguments (integers or floats)\n"
            "    func_id: Function identifier\n"
            "    core_type: 1=AIC (cube), 2=AIV (vector), 0=any\n"
            "Returns:\n"
            "    Task ID (integer)")

        .def("add_successor", &Graph::add_successor,
            py::arg("from_task"), py::arg("to_task"),
            "Add a dependency edge from from_task to to_task")

        .def("get_task_count", &Graph::get_task_count,
            "Get the total number of tasks in the graph")

        .def("print_graph", &Graph::print_graph,
            "Print the graph structure to stdout");

    // Bind DeviceRunner class
    py::class_<DeviceRunner>(m, "DeviceRunner")
        .def_static("get", &DeviceRunner::Get, py::return_value_policy::reference,
            "Get the singleton DeviceRunner instance")

        .def("init",
            [](DeviceRunner& self, int device_id, int num_cores,
               const std::string& aicpu_so_path,
               const std::string& aicore_kernel_path = "./aicore/kernel.o") {
                return self.Init(device_id, num_cores, aicpu_so_path, aicore_kernel_path);
            },
            py::arg("device_id"), py::arg("num_cores"),
            py::arg("aicpu_so_path"), py::arg("aicore_kernel_path") = "./aicore/kernel.o",
            "Initialize the device and runtime resources.\n"
            "Args:\n"
            "    device_id: Device ID (0-15)\n"
            "    num_cores: Number of cube (AIC) blocks to launch (e.g., 24 on A3). Vector (AIV) subblocks are derived.\n"
            "    aicpu_so_path: Path to AICPU shared object\n"
            "    aicore_kernel_path: Path to AICore kernel binary (optional)\n"
            "Returns:\n"
            "    0 on success, error code on failure")

        .def("compile_and_load_kernel", &DeviceRunner::CompileAndLoadKernel,
            py::arg("func_id"), py::arg("kernel_path"), py::arg("pto_isa_root"),
            "Compile and load a kernel at runtime.\n"
            "Args:\n"
            "    func_id: Function identifier for this kernel\n"
            "    kernel_path: Path to kernel source file (.cpp)\n"
            "    pto_isa_root: Path to PTO-ISA root directory\n"
            "Returns:\n"
            "    0 on success, error code on failure")

        .def("allocate_tensor",
            [](DeviceRunner& self, size_t bytes) -> uintptr_t {
                void* ptr = self.AllocateTensor(bytes);
                return reinterpret_cast<uintptr_t>(ptr);
            },
            py::arg("bytes"),
            "Allocate device tensor memory.\n"
            "Args:\n"
            "    bytes: Size of tensor in bytes\n"
            "Returns:\n"
            "    Device pointer as integer (0 on failure)")

        .def("free_tensor",
            [](DeviceRunner& self, uintptr_t ptr) {
                self.FreeTensor(reinterpret_cast<void*>(ptr));
            },
            py::arg("ptr"),
            "Free device tensor memory.\n"
            "Args:\n"
            "    ptr: Device pointer (as integer)")

        .def("copy_to_device",
            [](DeviceRunner& self, uintptr_t dev_ptr, py::array host_data) {
                py::buffer_info buf = host_data.request();
                size_t bytes = static_cast<size_t>(buf.size) * static_cast<size_t>(buf.itemsize);
                return self.CopyToDevice(reinterpret_cast<void*>(dev_ptr),
                                        buf.ptr, bytes);
            },
            py::arg("dev_ptr"), py::arg("host_data"),
            "Copy data from host numpy array to device.\n"
            "Args:\n"
            "    dev_ptr: Device pointer (as integer)\n"
            "    host_data: NumPy array (any dtype; raw bytes copied)\n"
            "Returns:\n"
            "    0 on success, error code on failure")

        .def("copy_from_device",
            [](DeviceRunner& self, py::array host_data, uintptr_t dev_ptr) {
                py::buffer_info buf = host_data.request();
                size_t bytes = static_cast<size_t>(buf.size) * static_cast<size_t>(buf.itemsize);
                return self.CopyFromDevice(buf.ptr,
                                          reinterpret_cast<void*>(dev_ptr), bytes);
            },
            py::arg("host_data"), py::arg("dev_ptr"),
            "Copy data from device to host numpy array.\n"
            "Args:\n"
            "    host_data: NumPy array to store results (any dtype; raw bytes copied)\n"
            "    dev_ptr: Device pointer (as integer)\n"
            "Returns:\n"
            "    0 on success, error code on failure")

        .def("run", &DeviceRunner::Run,
            py::arg("graph"), py::arg("launch_aicpu_num") = 1,
            "Execute a graph on the device.\n"
            "Args:\n"
            "    graph: Graph object to execute\n"
            "    launch_aicpu_num: Number of AICPU instances\n"
            "Returns:\n"
            "    0 on success, error code on failure")

        .def("set_profile_enabled", &DeviceRunner::SetProfileEnabled, py::arg("enabled"),
            "Enable/disable profile collection (graph copy-back) for subsequent runs.")

        .def("profile_enabled", &DeviceRunner::ProfileEnabled,
            "Return True if profile collection is enabled.")

        .def("has_last_profile", &DeviceRunner::HasLastProfile,
            "Return True if profiling data from the last run is available.")

        .def("get_last_profile", &DeviceRunner::GetLastProfile,
            "Get per-task profiling records from the last run.\n"
            "Returns:\n"
            "    List[TaskProfileRecord]")

        .def("run_task",
            [](DeviceRunner& self, py::list args, int func_id, int launch_aicpu_num = 1) {
                std::vector<uint64_t> args_vec;
                for (auto item : args) {
                    if (py::isinstance<py::int_>(item)) {
                        args_vec.push_back(item.cast<uint64_t>());
                    } else if (py::isinstance<py::float_>(item)) {
                        float f = item.cast<float>();
                        union { float f32; uint64_t u64; } converter;
                        converter.f32 = f;
                        args_vec.push_back(converter.u64);
                    } else {
                        throw std::runtime_error("Task arguments must be integers or floats");
                    }
                }
                return self.RunTask(args_vec, func_id, launch_aicpu_num);
            },
            py::arg("args"), py::arg("func_id"), py::arg("launch_aicpu_num") = 1,
            "Execute a single task on the device.\n"
            "Args:\n"
            "    args: List of arguments (integers or floats)\n"
            "    func_id: Function identifier\n"
            "    launch_aicpu_num: Number of AICPU instances\n"
            "Returns:\n"
            "    0 on success, error code on failure")

        .def("print_handshake_results", &DeviceRunner::PrintHandshakeResults,
            "Print handshake results from device")

        .def("finalize", &DeviceRunner::Finalize,
            "Cleanup all resources.\n"
            "Returns:\n"
            "    0 on success, error code on failure");

    // Version info
    m.attr("__version__") = "1.0.0";
}
