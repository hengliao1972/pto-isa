from __future__ import annotations

import ctypes
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Optional, Union


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _ensure_ref_runtime_python_on_path() -> None:
    repo_root = _repo_root()
    p = repo_root / "ref_runtime" / "python"
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


_LIB: Optional[ctypes.CDLL] = None
_LIB_PATH: Optional[Path] = None
_COMPILER: Optional[Any] = None
_AICPU_BINARY: Optional[bytes] = None
_AICORE_BINARY: Optional[bytes] = None


def _get_binary_compiler() -> Any:
    global _COMPILER
    if _COMPILER is not None:
        return _COMPILER

    _ensure_ref_runtime_python_on_path()
    try:
        from binary_compiler import BinaryCompiler  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "pto_runtime: failed to import ref_runtime/python/binary_compiler.py. "
            "Ensure repo root is on PYTHONPATH."
        ) from exc

    _COMPILER = BinaryCompiler()
    return _COMPILER


def _default_include_dirs() -> list[str]:
    repo_root = _repo_root()
    return [
        str(repo_root / "include"),
        str(repo_root / "ref_runtime" / "src" / "runtime"),
        str(repo_root / "ref_runtime" / "src" / "runtime" / "graph"),
    ]


def _read_binary_blob(spec: Union[None, str, os.PathLike[str], bytes, bytearray, memoryview]) -> bytes:
    if spec is None:
        raise ValueError("binary spec is None")
    if isinstance(spec, (bytes, bytearray, memoryview)):
        return bytes(spec)
    return Path(spec).read_bytes()


def _ensure_device_binaries() -> tuple[bytes, bytes]:
    global _AICPU_BINARY, _AICORE_BINARY
    if _AICPU_BINARY is not None and _AICORE_BINARY is not None:
        return _AICPU_BINARY, _AICORE_BINARY

    compiler = _get_binary_compiler()
    include_dirs = _default_include_dirs()
    graph_sources = [str(_repo_root() / "ref_runtime" / "src" / "runtime" / "graph")]

    try:
        aicore_binary = compiler.compile("aicore", include_dirs, graph_sources)
        aicpu_binary = compiler.compile("aicpu", include_dirs, graph_sources)
    except Exception as exc:  # pragma: no cover
        where = os.environ.get("ASCEND_HOME_PATH", "").strip()
        hint = (
            "Ensure CANN is installed and `ASCEND_HOME_PATH` is set "
            "(e.g. `source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash`)."
        )
        if where:
            hint += f" (ASCEND_HOME_PATH={where})"
        raise RuntimeError(f"pto_runtime: failed to build device runtime binaries. {hint}") from exc

    _AICPU_BINARY = bytes(aicpu_binary)
    _AICORE_BINARY = bytes(aicore_binary)
    return _AICPU_BINARY, _AICORE_BINARY


def _load_lib() -> ctypes.CDLL:
    global _LIB, _LIB_PATH
    if _LIB is not None:
        return _LIB

    try:
        compiler = _get_binary_compiler()
        include_dirs = _default_include_dirs()
        source_dirs = [
            str(_repo_root() / "ref_runtime" / "src" / "runtime" / "graph"),
            str(_repo_root() / "ref_runtime" / "src" / "runtime" / "host"),
        ]
        host_binary = compiler.compile("host", include_dirs, source_dirs)
    except Exception as exc:  # pragma: no cover
        where = os.environ.get("ASCEND_HOME_PATH", "").strip()
        hint = (
            "Ensure CANN is installed and `ASCEND_HOME_PATH` is set "
            "(e.g. `source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash`)."
        )
        if where:
            hint += f" (ASCEND_HOME_PATH={where})"
        raise ImportError(f"pto_runtime: failed to build host runtime. {hint}") from exc

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".so", prefix="pto_host_runtime_")
    tmp.write(host_binary)
    tmp.flush()
    tmp.close()
    _LIB_PATH = Path(tmp.name)

    lib = ctypes.CDLL(str(_LIB_PATH))
    _bind_ctypes_signatures(lib)
    _LIB = lib
    return lib


def _bind_ctypes_signatures(lib: ctypes.CDLL) -> None:
    c_int = ctypes.c_int
    c_size_t = ctypes.c_size_t
    c_void_p = ctypes.c_void_p
    c_uint8 = ctypes.c_uint8
    c_uint32 = ctypes.c_uint32
    c_uint64 = ctypes.c_uint64
    c_char_p = ctypes.c_char_p

    lib.Graph_Create.argtypes = []
    lib.Graph_Create.restype = c_void_p
    lib.Graph_Destroy.argtypes = [c_void_p]
    lib.Graph_Destroy.restype = c_int
    lib.Graph_AddTask.argtypes = [c_void_p, ctypes.POINTER(c_uint64), c_int, c_int, c_int]
    lib.Graph_AddTask.restype = c_int
    lib.Graph_AddSuccessor.argtypes = [c_void_p, c_int, c_int]
    lib.Graph_AddSuccessor.restype = c_int
    lib.Graph_GetTaskCount.argtypes = [c_void_p]
    lib.Graph_GetTaskCount.restype = c_int

    lib.DeviceRunner_Init.argtypes = [
        c_int,
        ctypes.POINTER(c_uint8),
        c_size_t,
        ctypes.POINTER(c_uint8),
        c_size_t,
        c_char_p,
    ]
    lib.DeviceRunner_Init.restype = c_int

    lib.DeviceRunner_AllocateTensor.argtypes = [c_size_t]
    lib.DeviceRunner_AllocateTensor.restype = c_void_p
    lib.DeviceRunner_FreeTensor.argtypes = [c_void_p]
    lib.DeviceRunner_FreeTensor.restype = None
    lib.DeviceRunner_CopyToDevice.argtypes = [c_void_p, c_void_p, c_size_t]
    lib.DeviceRunner_CopyToDevice.restype = c_int
    lib.DeviceRunner_CopyFromDevice.argtypes = [c_void_p, c_void_p, c_size_t]
    lib.DeviceRunner_CopyFromDevice.restype = c_int

    lib.DeviceRunner_Run.argtypes = [c_void_p, c_int, c_int]
    lib.DeviceRunner_Run.restype = c_int
    lib.DeviceRunner_PrintHandshakeResults.argtypes = [c_void_p]
    lib.DeviceRunner_PrintHandshakeResults.restype = None

    lib.DeviceRunner_SetProfileEnabled.argtypes = [c_int]
    lib.DeviceRunner_SetProfileEnabled.restype = c_int
    lib.DeviceRunner_ProfileEnabled.argtypes = []
    lib.DeviceRunner_ProfileEnabled.restype = c_int
    lib.DeviceRunner_HasLastProfile.argtypes = []
    lib.DeviceRunner_HasLastProfile.restype = c_int

    class _PtoTaskProfileRecord(ctypes.Structure):
        _fields_ = [
            ("task_id", c_int),
            ("func_id", c_int),
            ("core_type", c_int),
            ("exec_core_id", c_uint32),
            ("exec_core_type", c_uint32),
            ("exec_phys_core_id", c_uint32),
            ("start_time", c_uint64),
            ("end_time", c_uint64),
            ("pmu_cnt", c_uint32 * 8),
        ]

    lib._PtoTaskProfileRecord = _PtoTaskProfileRecord  # type: ignore[attr-defined]
    lib.DeviceRunner_GetLastProfile.argtypes = [ctypes.POINTER(_PtoTaskProfileRecord), c_int]
    lib.DeviceRunner_GetLastProfile.restype = c_int

    lib.DeviceRunner_CompileAndLoadKernel.argtypes = [c_int, c_char_p, c_int]
    lib.DeviceRunner_CompileAndLoadKernel.restype = c_int

    lib.DeviceRunner_Finalize.argtypes = []
    lib.DeviceRunner_Finalize.restype = c_int


@dataclass(frozen=True)
class TaskProfileRecord:
    task_id: int
    func_id: int
    core_type: int
    exec_core_id: int
    exec_core_type: int
    exec_phys_core_id: int
    start_time: int
    end_time: int
    pmu_cnt: tuple[int, ...]


class Graph:
    def __init__(self) -> None:
        lib = _load_lib()
        handle = lib.Graph_Create()
        if not handle:
            raise RuntimeError("Graph_Create failed")
        self._handle = ctypes.c_void_p(handle)

    @property
    def _ptr(self) -> ctypes.c_void_p:
        return self._handle

    def add_task(self, args: list[Any], *, func_id: int, core_type: int = 1) -> int:
        lib = _load_lib()
        c_uint64 = ctypes.c_uint64

        packed: list[int] = []
        for item in args:
            if isinstance(item, bool):
                packed.append(int(item))
            elif isinstance(item, int):
                packed.append(int(item) & 0xFFFFFFFFFFFFFFFF)
            elif isinstance(item, float):
                import struct

                bits = struct.unpack("<I", struct.pack("<f", float(item)))[0]
                packed.append(int(bits))
            else:
                raise TypeError(f"unsupported task arg type: {type(item)}")

        arr = (c_uint64 * len(packed))(*[c_uint64(x) for x in packed])
        return int(lib.Graph_AddTask(self._ptr, arr, int(len(packed)), int(func_id), int(core_type)))

    def add_successor(self, from_task: int, to_task: int) -> None:
        lib = _load_lib()
        rc = int(lib.Graph_AddSuccessor(self._ptr, int(from_task), int(to_task)))
        if rc != 0:
            raise RuntimeError(f"Graph_AddSuccessor failed: rc={rc}")

    def get_task_count(self) -> int:
        lib = _load_lib()
        return int(lib.Graph_GetTaskCount(self._ptr))

    def __del__(self) -> None:  # pragma: no cover
        try:
            lib = _load_lib()
            if getattr(self, "_handle", None):
                lib.Graph_Destroy(self._ptr)
        except Exception:
            pass


class DeviceRunner:
    _instance: ClassVar["DeviceRunner" | None] = None

    def __init__(self) -> None:
        self._cube_blocks: int = 0
        self._initialized: bool = False

    @classmethod
    def get(cls) -> "DeviceRunner":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def init(
        self,
        device_id: int,
        num_cores: int,
        aicpu_so_path: Union[None, str, os.PathLike[str], bytes, bytearray, memoryview] = None,
        aicore_kernel_path: Union[None, str, os.PathLike[str], bytes, bytearray, memoryview] = None,
        pto_isa_root: str | None = None,
    ) -> int:
        lib = _load_lib()

        if pto_isa_root is None:
            pto_isa_root = os.fspath(_repo_root())

        # Allow passing pre-built file paths, raw bytes, or None (compile via BinaryCompiler).
        if aicpu_so_path is None or aicore_kernel_path is None:
            aicpu_default, aicore_default = _ensure_device_binaries()
            if aicpu_so_path is None:
                aicpu_so_path = aicpu_default
            if aicore_kernel_path is None:
                aicore_kernel_path = aicore_default

        aicpu_bytes = _read_binary_blob(aicpu_so_path)
        aicore_bytes = _read_binary_blob(aicore_kernel_path)

        aicpu_buf = (ctypes.c_uint8 * len(aicpu_bytes)).from_buffer_copy(aicpu_bytes)
        aicore_buf = (ctypes.c_uint8 * len(aicore_bytes)).from_buffer_copy(aicore_bytes)

        rc = int(
            lib.DeviceRunner_Init(
                int(device_id),
                aicpu_buf,
                ctypes.c_size_t(len(aicpu_bytes)),
                aicore_buf,
                ctypes.c_size_t(len(aicore_bytes)),
                ctypes.c_char_p(os.fsencode(pto_isa_root)),
            )
        )
        if rc == 0:
            self._cube_blocks = int(num_cores)
            self._initialized = True
        return rc

    def compile_and_load_kernel(self, func_id: int, kernel_path: str, pto_isa_root: str | None = None, core_type: int = 0) -> int:
        lib = _load_lib()
        _ = pto_isa_root  # pto_isa_root is configured at init() time
        return int(
            lib.DeviceRunner_CompileAndLoadKernel(
                int(func_id),
                ctypes.c_char_p(os.fsencode(kernel_path)),
                int(core_type),
            )
        )

    def allocate_tensor(self, bytes: int) -> int:
        lib = _load_lib()
        ptr = lib.DeviceRunner_AllocateTensor(ctypes.c_size_t(int(bytes)))
        return int(ctypes.c_void_p(ptr).value or 0)

    def free_tensor(self, ptr: int) -> None:
        lib = _load_lib()
        lib.DeviceRunner_FreeTensor(ctypes.c_void_p(int(ptr)))

    def copy_to_device(self, dev_ptr: int, host_data: Any) -> int:
        lib = _load_lib()
        import numpy as np

        if not isinstance(host_data, np.ndarray):
            raise TypeError("copy_to_device expects a NumPy array")
        if not host_data.flags["C_CONTIGUOUS"]:
            host_data = np.ascontiguousarray(host_data)
        return int(
            lib.DeviceRunner_CopyToDevice(
                ctypes.c_void_p(int(dev_ptr)),
                ctypes.c_void_p(int(host_data.ctypes.data)),
                ctypes.c_size_t(int(host_data.nbytes)),
            )
        )

    def copy_from_device(self, host_data: Any, dev_ptr: int) -> int:
        lib = _load_lib()
        import numpy as np

        if not isinstance(host_data, np.ndarray):
            raise TypeError("copy_from_device expects a NumPy array")
        if not host_data.flags["C_CONTIGUOUS"]:
            raise ValueError("copy_from_device requires a C-contiguous NumPy array")
        return int(
            lib.DeviceRunner_CopyFromDevice(
                ctypes.c_void_p(int(host_data.ctypes.data)),
                ctypes.c_void_p(int(dev_ptr)),
                ctypes.c_size_t(int(host_data.nbytes)),
            )
        )

    def set_profile_enabled(self, enabled: bool) -> None:
        lib = _load_lib()
        rc = int(lib.DeviceRunner_SetProfileEnabled(1 if enabled else 0))
        if rc != 0:
            raise RuntimeError(f"DeviceRunner_SetProfileEnabled failed: rc={rc}")

    def profile_enabled(self) -> bool:
        lib = _load_lib()
        return bool(int(lib.DeviceRunner_ProfileEnabled()) != 0)

    def has_last_profile(self) -> bool:
        lib = _load_lib()
        return bool(int(lib.DeviceRunner_HasLastProfile()) != 0)

    def get_last_profile(self) -> list[TaskProfileRecord]:
        lib = _load_lib()
        rec_t = getattr(lib, "_PtoTaskProfileRecord")  # type: ignore[attr-defined]
        n = int(lib.DeviceRunner_GetLastProfile(None, 0))
        if n <= 0:
            return []
        buf = (rec_t * n)()
        m = int(lib.DeviceRunner_GetLastProfile(buf, int(n)))
        if m < 0:
            raise RuntimeError(f"DeviceRunner_GetLastProfile failed: rc={m}")
        out: list[TaskProfileRecord] = []
        for i in range(int(m)):
            r = buf[i]
            out.append(
                TaskProfileRecord(
                    task_id=int(r.task_id),
                    func_id=int(r.func_id),
                    core_type=int(r.core_type),
                    exec_core_id=int(r.exec_core_id),
                    exec_core_type=int(r.exec_core_type),
                    exec_phys_core_id=int(r.exec_phys_core_id),
                    start_time=int(r.start_time),
                    end_time=int(r.end_time),
                    pmu_cnt=tuple(int(x) for x in r.pmu_cnt),
                )
            )
        return out

    def run(self, graph: Graph, launch_aicpu_num: int = 1) -> int:
        lib = _load_lib()
        if not self._initialized:
            raise RuntimeError("DeviceRunner not initialized; call init() first")
        if self._cube_blocks <= 0:
            raise RuntimeError("invalid num_cores; expected cube block count > 0")
        total_workers = int(self._cube_blocks) * 3
        return int(lib.DeviceRunner_Run(graph._ptr, int(total_workers), int(launch_aicpu_num)))

    def print_handshake_results(self, graph: Graph) -> None:
        lib = _load_lib()
        lib.DeviceRunner_PrintHandshakeResults(graph._ptr)

    def finalize(self) -> int:
        lib = _load_lib()
        self._initialized = False
        self._cube_blocks = 0
        return int(lib.DeviceRunner_Finalize())
