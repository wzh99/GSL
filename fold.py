from typing import Dict, List

import numpy as np
from tvm import relay


class ParamFolder(relay.ExprMutator):
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
        name = call.op.name
        if _direct_eval.__contains__(name):  # directly mapped to numpy functions
            if not self._all_avail(call.args):
                return call
            func = eval('np.{}'.format(name))
            args = self._collect_values(call.args)
            return self._add_param(func(*args))
        elif _with_attrs.__contains__(name):  # evaluate with attributes
            if not self._all_avail(call.args):
                return call
            args = self._collect_values(call.args)
            res = _with_attrs[name](args, call.attrs)
            return self._add_param(res)
        else:
            return call

    @staticmethod
    def _all_avail(args: List[relay.Expr]) -> bool:
        for a in args:
            if not isinstance(a, (relay.Constant, relay.Var)):
                return False

    def _collect_values(self, args: List[relay.Expr]) -> List[np.ndarray]:
        values = []
        for a in args:
            if isinstance(a, relay.Constant):
                values.append(np.array(a.data))
            elif isinstance(a, relay.Var):
                values.append(self.params[a.name_hint])
            else:
                raise RuntimeError('Cannot collect value.')
        return values

    def _add_param(self, val: np.ndarray) -> relay.Var:
        name = self._next_param_name()
        self.params[name] = val
        return relay.var(name, shape=val.shape, dtype=str(val.dtype))

    def _next_param_name(self) -> str:
        while True:
            name = '_param_{}'.format(self.next_idx)
            self.next_idx += 1
            if not self.params.__contains__(name):
                return name


_direct_eval = {'negative', 'add', 'subtract', 'multiply', 'divide', 'abs', 'exp', 'sqrt'}

_with_attrs = {
    'zeros': lambda args, attrs: np.zeros(attrs['shape'], attrs['dtype']),
    'reshape': lambda args, attrs: np.reshape(args[0], attrs['newshape']),
    'transpose': lambda args, attrs: np.transpose(args[0], attrs['axes']),
    'expand_dims': lambda args, attrs: np.expand_dims(args[0], attrs['axis']),
}
