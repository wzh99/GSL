from typing import Dict, List, Any, Optional, Callable

import numpy as np
from tvm import relay, transform, ir, tir

from . import default_dtype, util


@relay.transform.function_pass(opt_level=0)
class ParamFoldPass:
    def __init__(self, params: Dict[str, np.ndarray]):
        self.params = params.copy()

    def transform_function(self, fn: relay.Function, _mod: ir.IRModule,
                           _ctx: transform.PassContext) -> relay.Function:
        new_body = _ParamFolder(self.params).visit(fn.body)
        return relay.Function(relay.analysis.free_vars(new_body), new_body)

    def __call__(self, mod: ir.IRModule) -> ir.IRModule: ...


class _ParamFolder(relay.ExprMutator):
    def __init__(self, params: Dict[str, np.ndarray]):
        super().__init__()
        self.params = params
        self.next_idx = 1

    def visit(self, expr: relay.Expr):
        if self.memo_map.__contains__(expr):
            return self.memo_map[expr]
        ret = super().visit(expr)
        self.memo_map[expr] = ret
        return ret

    def visit_call(self, call: relay.Call):
        # Fold arguments
        call = super().visit_call(call)

        # Fold parameters
        op_name = call.op.name
        try:
            if _direct_mapped.__contains__(op_name):  # mapped to numpy APIs
                args = self._get_values(call.args)
                attrs = self._cvt_attrs(call.attrs)
                np_func = eval('np.{}'.format(op_name))
                return self._add_param(np_func(*args, **attrs))
            elif _eval_funcs.__contains__(op_name):
                args = self._get_values(call.args)
                attrs = self._cvt_attrs(call.attrs)
                return self._add_param(_eval_funcs[op_name](args, attrs))
            elif op_name == 'matrix_set_diag':
                # In this project, we assume the first input of `matrix_set_diag` is always zero.
                # This way, the semantic is similar to `np.diag`.
                data = self._get_values(call.args[1:2])[0]
                return self._add_param(np.diag(data))
            elif op_name == 'nn.batch_matmul':
                args = self._get_values(call.args)
                return self._add_param(
                    np.matmul(args[0], np.transpose(args[1], axes=(0, 2, 1)))
                )
            else:
                return call
        except _FoldException:
            return call

    def _get_values(self, args: List[relay.Expr]) -> List[np.ndarray]:
        values = []
        for a in args:
            if isinstance(a, relay.Constant):
                values.append(np.array(a.data.asnumpy(), dtype=default_dtype))
            elif isinstance(a, relay.Var) and self.params.__contains__(a.name_hint):
                values.append(self.params[a.name_hint])
            else:
                raise _FoldException()
        return values

    @classmethod
    def _cvt_attrs(cls, attrs: Optional[ir.Attrs]) -> Dict[str, Any]:
        if attrs is None or len(attrs.keys()) == 0:
            return {}
        else:
            return dict([(name, cls._cvt_value(util.cvt_ir_value(attrs[name])))
                         for name in attrs.keys()])

    @classmethod
    def _cvt_value(cls, val) -> Any:
        if isinstance(val, (tir.IntImm, tir.FloatImm, tir.StringImm)):
            return val.value
        elif isinstance(val, ir.Array):
            return [cls._cvt_value(e) for e in val]
        else:
            return val

    def _add_param(self, value: np.ndarray) -> relay.Var:
        name = self._next_param_name()
        self.params[name] = value
        return relay.var(name, shape=value.shape, dtype=str(value.dtype))

    def _next_param_name(self) -> str:
        while True:
            name = '_param_{}'.format(self.next_idx)
            self.next_idx += 1
            if not self.params.__contains__(name):
                return name


class _FoldException(Exception):
    pass


_direct_mapped = {
    'negative',
    'add',
    'subtract',
    'multiply',
    'divide',
    'abs',
    'exp',
    'sqrt',
    'zeros',
    'reshape',
    'transpose',
}

_eval_funcs: Dict[str, Callable[[List[np.ndarray], Dict[str, Any]], np.ndarray]] = {
    'expand_dims': lambda args, attrs: np.expand_dims(args[0], attrs['axis']),
}
