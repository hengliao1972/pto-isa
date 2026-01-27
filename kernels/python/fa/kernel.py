from __future__ import annotations

from pto import PTO, KernelSpec, compile_kernel_spec


def fa16():
    # Minimal "FA" placeholder kernel for validating the end-to-end flow.
    pto = PTO("fa16")
    pto.prologue()

    # Signature (tensor args):
    #   arg0: q f16[16,16]
    #   arg1: k f16[16,16]
    #   arg2: v f16[16,16]
    #   arg3: out f16[16,16]
    #
    # Compute:
    #   out = q + k + v
    q = pto.tensor(dtype="f16", shape=(16, 16), role="in")
    k = pto.tensor(dtype="f16", shape=(16, 16), role="in")
    v = pto.tensor(dtype="f16", shape=(16, 16), role="in")
    out = pto.tensor(dtype="f16", shape=(16, 16), role="out")

    tq = pto.vec(dtype="f16", shape=(16, 16))
    tk = pto.vec(dtype="f16", shape=(16, 16))
    tv = pto.vec(dtype="f16", shape=(16, 16))
    to = pto.vec(dtype="f16", shape=(16, 16))

    tq = pto.load(q)
    tk = pto.load(k)
    to = pto.add(tq, tk)
    tv = pto.load(v)
    to = pto.add(to, tv)
    pto.store(out, to)

    pto.epilogue()
    return pto.program()


def make_fa16_kernel(*, target: str) -> KernelSpec:
    if target not in ("cpu", "npu"):
        raise ValueError("target must be cpu|npu")
    return compile_kernel_spec(fa16)


def make_fa16_pto(*, target: str) -> str:
    return make_fa16_kernel(target=target).pto
