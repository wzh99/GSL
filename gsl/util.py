from typing import Any

from tvm import ir, tir


def cvt_ir_value(val) -> Any:
    if isinstance(val, (tir.IntImm, tir.FloatImm, tir.StringImm)):
        return val.value
    elif isinstance(val, ir.Array):
        return [cvt_ir_value(e) for e in val]
    else:
        return val
