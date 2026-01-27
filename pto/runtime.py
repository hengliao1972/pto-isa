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


def default_ascend_home() -> Path:
    p = os.environ.get("ASCEND_HOME_PATH", "").strip()
    if p:
        return Path(p)
    return Path.home() / "Ascend" / "ascend-toolkit" / "latest"


@dataclass(frozen=True)
class PtoasConfig:
    ptoas: Path = field(default_factory=default_ptoas)
    target: str = "npu"
    arch: str = "dav-c220-cube"
    memory_model: str = "MEMORY_BASE"
    kernel_abi: str = "mpmd"
    kernel_name: str | None = None
    insert_events: bool = True
    assign_tile_addrs: bool = True
    ascend_home: Path = field(default_factory=default_ascend_home)
    repo_root: Path = field(default_factory=repo_root)
    timeout_s: float | None = None
    log_path: Path | None = None
    print_cmd: bool = False


def compile_pto_to_cce_cpp(*, pto_path: Path, out_cpp: Path, cfg: PtoasConfig) -> None:
    out_cpp.parent.mkdir(parents=True, exist_ok=True)
    args: list[str] = [os.fspath(cfg.ptoas)]
    args += [f"--target={cfg.target}"]
    args += [f"--arch={cfg.arch}"]
    args += [f"--memory-model={cfg.memory_model}"]
    args += [f"--kernel-abi={cfg.kernel_abi}"]
    args += [f"--ascend-home={cfg.ascend_home}"]
    args += [f"--repo-root={cfg.repo_root}"]
    if cfg.kernel_name:
        args += [f"--kernel-name={cfg.kernel_name}"]
    if cfg.insert_events:
        args += ["--insert-events"]
    else:
        args += ["--no-insert-events"]
    if cfg.assign_tile_addrs:
        args += ["--assign-tile-addrs"]
    args += ["-o", os.fspath(out_cpp)]
    args += [os.fspath(pto_path)]
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
