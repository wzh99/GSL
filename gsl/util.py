from typing import Any, List

from tvm import ir, tir, relay, runtime

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
    elif isinstance(val, runtime.String):
        return str(val)
    elif isinstance(val, ir.Array):
        return tuple([cvt_ir_value(e) for e in val])
    elif isinstance(val, runtime.NDArray):
        return val.asnumpy()
    else:
        return val


def get_tensor_type_attr(ty: ir.TensorType, name: str):
    if name == 'shape':
        return ty.concrete_shape
    elif name == 'ndim':
        return len(ty.concrete_shape)
    elif name == 'dtype':
        return ty.dtype
    else:
        raise RuntimeError(
            'Unknown attribute {} for tensor type.'.format(name)
        )


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
