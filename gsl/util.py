from typing import Any, List

from tvm import ir, tir, relay

default_font_name = ''


def _set_default_font_name():
    import sys
    global default_font_name
    sys_name = sys.platform
    if sys_name == 'darwin':
        default_font_name = 'Latin Modern Mono'
    elif sys_name == 'win32':
        default_font_name = 'LM Mono 12 Regular'


def cvt_ir_value(val) -> Any:
    if isinstance(val, (tir.IntImm, tir.FloatImm, tir.StringImm)):
        return val.value
    elif isinstance(val, ir.Array):
        return [cvt_ir_value(e) for e in val]
    else:
        return val


def get_shared_attr(expr: relay.Expr, name: str):
    expr_ty = expr.checked_type
    if not isinstance(expr_ty, ir.TensorType):
        raise ValueError(
            'Cannot get attribute from an expression not of tensor type.'
        )
    if name == 'shape':
        return expr_ty.concrete_shape
    elif name == 'dtype':
        return expr_ty.dtype
    else:
        raise RuntimeError('Unreachable.')


def get_expr_pred(expr: relay.Expr) -> List[relay.Expr]:
    if isinstance(expr, relay.Call):
        return list(expr.args)
    elif isinstance(expr, relay.Tuple):
        return list(expr.fields)
    elif isinstance(expr, relay.TupleGetItem):
        return [expr.tuple_value]
    else:
        return []


_set_default_font_name()
