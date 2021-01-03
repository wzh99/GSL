from typing import Dict, List, Any, Optional

import numpy as np
from tvm import relay, transform, ir, tir

from . import dtype


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
        try:
            ret = super().visit(expr)
        except _FoldException:
            ret = expr
        self.memo_map[expr] = ret
        return ret

    def visit_call(self, call: relay.Call):
        # Fold arguments
        call = super().visit_call(call)

        # Fold parameters
        op_name = call.op.name
        if _api_mapped.__contains__(op_name):  # mapped to numpy APIs
            # Collect values and attributes from relay call node
            args = self._collect_values(call.args)
            attrs = self._cvt_attrs(call.attrs)
            np_func = eval('np.{}'.format(op_name))
            return self._add_param(np_func(*args, **attrs))
        else:
            return call

    def _collect_values(self, args: List[relay.Expr]) -> List[np.ndarray]:
        values = []
        for a in args:
            if isinstance(a, relay.Constant):
                values.append(np.array(a.data.asnumpy(), dtype=dtype))
            elif isinstance(a, relay.Var):
                values.append(self.params[a.name_hint])
            else:
                raise _FoldException()
        return values

    def _cvt_attrs(self, attrs: Optional[ir.Attrs]) -> Dict[str, Any]:
        if attrs is None:
            return {}
        else:
            return dict([(name, self._cvt_value(attrs[name])) for name in attrs.keys()])

    def _cvt_value(self, val) -> Any:
        if isinstance(val, tir.IntImm):
            return int(val)
        elif isinstance(val, ir.Array):
            return [self._cvt_value(e) for e in val]
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


_api_mapped = {
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
