from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    # `pto/` lives at repo root.
    return Path(__file__).resolve().parents[1]


def default_ptoas() -> Path:
    return repo_root() / "bin" / "ptoas"


@dataclass(frozen=True)
class PtoasConfig:
    ptoas: Path = field(default_factory=default_ptoas)
    memory_model: str = "MEMORY_BASE"
    enable_insert_sync: bool = True
    rewrite_unified_abi: bool = True
    repo_root: Path = field(default_factory=repo_root)
    timeout_s: float | None = None
    log_path: Path | None = None
    print_cmd: bool = False


def _postprocess_cce_cpp(*, cpp_path: Path, cfg: PtoasConfig) -> None:
    """
    Adapt llvm-project `ptoas` output for the runtime's AICore dynamic dispatch.

    Fixes:
    - Inject `#define MEMORY_BASE` / `#define REGISTER_BASE` for PTO headers.
    - Rewrite entry signature to unified ABI: `AICORE void f(__gm__ int64_t* args)`.
      This matches `UnifiedKernelFunc` in `ref_runtime/src/platform/a2a3/aicore/kernel.cpp`.
    """
    text = cpp_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # 1) Ensure memory model define exists before PTO headers.
    if cfg.memory_model not in ("MEMORY_BASE", "REGISTER_BASE"):
        raise ValueError("memory_model must be MEMORY_BASE or REGISTER_BASE")

    injected: list[str] = []
    injected.append("#if defined(__CCE__)")
    injected.append(f"#define {cfg.memory_model}")
    injected.append("#endif")
    injected.append('#include "kernel_operator.h"')
    injected.append("#include <cstdint>")

    # Insert before the first include (or at top if none).
    insert_at = 0
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("#include"):
            insert_at = i
            break
    lines[insert_at:insert_at] = injected

    if not cfg.rewrite_unified_abi:
        cpp_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return

    # 2) Rewrite the first AICORE kernel signature to unified ABI.
    import re

    sig_re = re.compile(r"^\s*__global__\s+AICORE\s+void\s+(\w+)\s*\(([^)]*)\)\s*\{\s*$")
    for i, ln in enumerate(lines):
        m = sig_re.match(ln)
        if not m:
            continue
        func_name = m.group(1)
        params = [p.strip() for p in m.group(2).split(",") if p.strip()]

        parsed: list[tuple[str, str]] = []
        for p in params:
            # Split "type name" on the last space.
            if " " not in p:
                raise RuntimeError(f"ptoas cpp: cannot parse param: {p!r}")
            ty, name = p.rsplit(" ", 1)
            parsed.append((ty.strip(), name.strip()))

        # Replace signature line.
        lines[i] = f'extern "C" AICORE void {func_name}(__gm__ int64_t* args) {{'

        # Insert arg unpacking (use original var names so body stays unchanged).
        unpack: list[str] = []
        for arg_i, (ty, name) in enumerate(parsed):
            if "*" in ty:
                unpack.append(f"  {ty} {name} = ({ty})(args[{arg_i}]);")
            else:
                unpack.append(f"  {ty} {name} = ({ty})args[{arg_i}];")
        lines[i + 1 : i + 1] = unpack
        break
    else:
        raise RuntimeError("ptoas cpp: did not find a '__global__ AICORE void ...' entry to rewrite")

    cpp_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def compile_pto_to_cce_cpp(*, pto_path: Path, out_cpp: Path, cfg: PtoasConfig) -> None:
    out_cpp.parent.mkdir(parents=True, exist_ok=True)
    args: list[str] = [os.fspath(cfg.ptoas)]
    if cfg.enable_insert_sync:
        args += ["--enable-insert-sync"]
    args += ["-o", os.fspath(out_cpp), os.fspath(pto_path)]
    if cfg.print_cmd:
        print("pto.runtime: running:", " ".join(args), flush=True)

    stdout = None
    stderr = None
    log_f = None
    if cfg.log_path is not None:
        cfg.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(cfg.log_path, "w", encoding="utf-8")
        stdout = log_f
        stderr = subprocess.STDOUT

    try:
        subprocess.run(
            args,
            cwd=os.fspath(cfg.repo_root),
            check=True,
            timeout=cfg.timeout_s,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
    except subprocess.TimeoutExpired as exc:
        where = f" (log: {cfg.log_path})" if cfg.log_path is not None else ""
        raise RuntimeError(f"ptoas timed out after {cfg.timeout_s}s{where}") from exc
    except subprocess.CalledProcessError as exc:
        where = f" (log: {cfg.log_path})" if cfg.log_path is not None else ""
        raise RuntimeError(f"ptoas failed with rc={exc.returncode}{where}") from exc
    finally:
        if log_f is not None:
            log_f.close()

    _postprocess_cce_cpp(cpp_path=out_cpp, cfg=cfg)


def _write_pto_text(*, pto_text: str, out_pto: Path) -> Path:
    out_pto.parent.mkdir(parents=True, exist_ok=True)
    out_pto.write_text(pto_text, encoding="utf-8")
    return out_pto


def compile_and_load_kernel_from_pto(
    *,
    runner: Any,
    func_id: int,
    pto: Path | str | Any,
    out_dir: Path | None = None,
    pto_isa_root: Path | None = None,
    ptoas_cfg: PtoasConfig | None = None,
) -> Path:
    """
    Compile a PTO-AS program to CCE C++ via `ptoas`, then `compile_and_load_kernel(...)` via runtime.

    `pto` may be:
    - a `.pto` file path
    - PTO-AS text
    - a `KernelSpec`-like object with `.pto` (string) attribute
    """
    if out_dir is None:
        out_dir = Path(tempfile.mkdtemp(prefix="pto_runtime_"))
    if pto_isa_root is None:
        pto_isa_root = repo_root()
    if ptoas_cfg is None:
        ptoas_cfg = PtoasConfig()

    # Resolve PTO input.
    pto_path: Path
    if isinstance(pto, Path):
        pto_path = pto
    elif isinstance(pto, str):
        maybe_path = Path(pto)
        if maybe_path.exists():
            pto_path = maybe_path
        else:
            pto_path = _write_pto_text(pto_text=pto, out_pto=out_dir / f"kernel_{func_id}.pto")
    else:
        pto_text = getattr(pto, "pto", None)
        if not isinstance(pto_text, str):
            raise TypeError("pto must be a Path, PTO-AS text, or an object with a .pto string")
        pto_path = _write_pto_text(pto_text=pto_text, out_pto=out_dir / f"kernel_{func_id}.pto")

    out_cpp = out_dir / f"kernel_{func_id}.cpp"
    compile_pto_to_cce_cpp(pto_path=pto_path, out_cpp=out_cpp, cfg=ptoas_cfg)

    rc = int(runner.compile_and_load_kernel(int(func_id), os.fspath(out_cpp), os.fspath(pto_isa_root)))
    if rc != 0:
        raise RuntimeError(f"runtime compile_and_load_kernel failed (func_id={func_id}, rc={rc})")
    return out_cpp
