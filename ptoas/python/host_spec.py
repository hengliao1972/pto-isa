from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal


TensorRole = Literal["in", "out", "inout"]


@dataclass(frozen=True)
class HostTensorArg:
    # Argument order is implied by list position (matches %argN).
    dtype: str
    shape: tuple[int, int]
    role: TensorRole = "in"
    # Optional view metadata; used by host array generator for non-ND tensors (e.g. DN).
    # When omitted, callers should assume a contiguous ND buffer with default row-major strides.
    layout: str = "ND"
    stride: tuple[int, int] | None = None


@dataclass(frozen=True)
class HostSpec:
    """
    Host-side metadata embedded in a `.pto` file for cross-platform testing/running.

    This is intentionally small and JSON-serializable so the same `.pto` can drive:
      - CPU compilation + run (reference)
      - NPU compilation + run, compared against CPU
    """

    args: tuple[HostTensorArg, ...]
    seed: int = 0
    block_dim: int = 1
    kernel_name: str = "pto_kernel"

    def to_dict(self) -> dict:
        return {
            "kernel_name": str(self.kernel_name),
            "seed": int(self.seed),
            "block_dim": int(self.block_dim),
            "args": [
                {
                    "dtype": str(a.dtype),
                    "shape": [int(a.shape[0]), int(a.shape[1])],
                    "role": a.role,
                    "layout": str(a.layout),
                    "stride": ([int(a.stride[0]), int(a.stride[1])] if a.stride is not None else None),
                }
                for a in self.args
            ],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    def output_indices(self) -> list[int]:
        return [i for i, a in enumerate(self.args) if a.role in ("out", "inout")]


_BEGIN = "// PTO_HOST_SPEC_BEGIN v1"
_END = "// PTO_HOST_SPEC_END"


def encode_host_spec(spec: HostSpec) -> str:
    payload = {
        "kernel_name": spec.kernel_name,
        "seed": int(spec.seed),
        "block_dim": int(spec.block_dim),
        "args": [
            {
                "dtype": a.dtype,
                "shape": [int(a.shape[0]), int(a.shape[1])],
                "role": a.role,
                "layout": str(a.layout),
                "stride": ([int(a.stride[0]), int(a.stride[1])] if a.stride is not None else None),
            }
            for a in spec.args
        ],
    }
    # Keep this readable in diffs by formatting with 2-space indents.
    body = json.dumps(payload, sort_keys=True, indent=2)
    lines = [_BEGIN] + [f"// {ln}" for ln in body.splitlines()] + [_END, ""]
    return "\n".join(lines)


def prepend_host_spec_to_pto(*, pto: str, spec: HostSpec) -> str:
    return encode_host_spec(spec) + pto.lstrip()


def parse_host_spec_from_pto(pto: str) -> HostSpec | None:
    lines = pto.splitlines()
    try:
        i0 = next(i for i, ln in enumerate(lines) if ln.strip() == _BEGIN)
        i1 = next(i for i, ln in enumerate(lines) if i > i0 and ln.strip() == _END)
    except StopIteration:
        return None

    json_lines: list[str] = []
    for ln in lines[i0 + 1 : i1]:
        s = ln.strip()
        if not s.startswith("//"):
            continue
        s = s[2:].lstrip()
        json_lines.append(s)
    payload = json.loads("\n".join(json_lines))

    args: list[HostTensorArg] = []
    for a in payload.get("args", []):
        shape = a["shape"]
        stride = a.get("stride", None)
        stride2 = None
        if stride is not None:
            stride2 = (int(stride[0]), int(stride[1]))
        args.append(
            HostTensorArg(
                dtype=a["dtype"],
                shape=(int(shape[0]), int(shape[1])),
                role=a.get("role", "in"),
                layout=str(a.get("layout", "ND")),
                stride=stride2,
            )
        )

    return HostSpec(
        args=tuple(args),
        seed=int(payload.get("seed", 0)),
        block_dim=int(payload.get("block_dim", 1)),
        kernel_name=str(payload.get("kernel_name", "pto_kernel")),
    )


def infer_host_spec_from_pto(*, pto: str) -> HostSpec:
    """
    Best-effort inference for older `.pto` files that don't embed a host spec.

    Best-effort inference for `.pto` files produced by the Python frontend.

    This implementation supports the common patterns emitted by `ast_frontend`:
      - `arith.constant <int> : index` for shape/stride operands
      - 2D views:
          `%x = pto.make_tensor_view %argN, shape=[%cH,%cW] strides=[%cS0,%cS1] : !pto.tensor_view<2xDT>`
      - 5D views (Ascend A2/A3 convention for ND/NZ conversions):
          `%x = pto.make_tensor_view %argN, shape=[1,1,1,%cH,%cW] strides=[1,1,1,%cS0,%cS1] : !pto.tensor_view<5xDT>`
    """
    const_pat = re.compile(r"^(%[a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*arith\.constant\s+(-?\d+)\s*:\s*index\s*$")
    consts: dict[str, int] = {}
    for ln in pto.splitlines():
        m = const_pat.match(ln.strip())
        if not m:
            continue
        consts[m.group(1)] = int(m.group(2))

    pat = re.compile(
        r"^(%[a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*pto\.make_tensor_view\s+%arg(\d+)\s*,\s*"
        r"shape\s*=\s*\[(.*?)\]\s*"
        r"strides\s*=\s*\[(.*?)\]\s*:\s*"
        r"!pto\.tensor_view<(\d+)x([^>]+)>\s*$"
    )
    found: dict[int, HostTensorArg] = {}
    for m in pat.finditer(pto):
        idx = int(m.group(2))
        shape_ops = [s.strip() for s in m.group(3).split(",") if s.strip()]
        stride_ops = [s.strip() for s in m.group(4).split(",") if s.strip()]
        rank = int(m.group(5))
        dt = m.group(6).strip()

        if len(shape_ops) != rank or len(stride_ops) != rank:
            raise ValueError("failed to infer host args from .pto (shape/stride count does not match tensor_view rank)")

        # Map both 2D and 5D forms to a 2D host shape/stride.
        if rank == 2:
            h = consts.get(shape_ops[0])
            w = consts.get(shape_ops[1])
            s0 = consts.get(stride_ops[0])
            s1 = consts.get(stride_ops[1])
        elif rank == 5:
            h = consts.get(shape_ops[3])
            w = consts.get(shape_ops[4])
            s0 = consts.get(stride_ops[3])
            s1 = consts.get(stride_ops[4])
        else:
            raise ValueError(f"failed to infer host args from .pto (unsupported tensor_view rank={rank})")

        if h is None or w is None or s0 is None or s1 is None:
            raise ValueError("failed to infer host args from .pto (non-constant shape/stride in pto.make_tensor_view)")
        stride = None if (s0 == w and s1 == 1) else (s0, s1)
        found[idx] = HostTensorArg(dtype=dt, shape=(h, w), role="in", layout="ND", stride=stride)
    if not found:
        raise ValueError("failed to infer host args from .pto (no pto.make_tensor_view %argN found)")

    args: list[HostTensorArg] = []
    for i in sorted(found):
        args.append(found[i])
    if args:
        args[-1] = HostTensorArg(dtype=args[-1].dtype, shape=args[-1].shape, role="out")
    return HostSpec(args=tuple(args), seed=0, block_dim=1, kernel_name="pto_kernel")
