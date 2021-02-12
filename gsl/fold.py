from typing import Dict, List, Any, Optional, Callable

import numpy as np
from tvm import relay, transform, ir

from . import util
from .work import Workload


def fold(wl: Workload) -> Workload:
    """
    Pre-compute graph nodes whose operands are already available in parameter set.
    The folding is performed recursively in reverse post-order, and the result will be stored as
    new parameter. Folded parameters will be filtered removed from parameter set.

    :param wl: Workload object whose parameters need folding.
    :return: Workload object after folding.
    """
    # Fold parameters
    fold_pass = _FoldFuncPass(wl.params)
    mod = fold_pass(wl.mod)
    new_wl = Workload(mod, fold_pass.params, name=wl.name)

    # Filter out unused parameters
    param_names = set([p.name_hint for p in mod['main'].params])
    used_params = dict()
    for name, val in new_wl.params.items():
        if param_names.__contains__(name):
            used_params[name] = val
    new_wl.params = used_params

    return new_wl


@relay.transform.function_pass(opt_level=0)
class _FoldFuncPass:
    def __init__(self, params: Dict[str, np.ndarray]):
        self.params = params.copy()

    def transform_function(self, fn: relay.Function, _mod: ir.IRModule,
                           _ctx: transform.PassContext) -> relay.Function:
        new_body = _FoldMutator(self.params).visit(fn.body)
        return relay.Function(relay.analysis.free_vars(new_body), new_body)

    def __call__(self, mod: ir.IRModule) -> ir.IRModule: ...


class _FoldMutator(relay.ExprMutator):
    def __init__(self, params: Dict[str, np.ndarray]):
        super().__init__()
        self.params = params
        self.next_idx = 1

    def visit(self, expr: relay.Expr):
        if expr in self.memo_map:
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
            if op_name in _direct_mapped:  # mapped to numpy APIs
                args = self._get_values(call.args)
                attrs = self._cvt_attrs(call.attrs)
                np_func = eval('np.{}'.format(op_name))
                return self._add_param(np_func(*args, **attrs))
            elif op_name in _eval_funcs:
                args = self._get_values(call.args)
                attrs = self._cvt_attrs(call.attrs)
                return self._add_param(_eval_funcs[op_name](args, attrs))
            elif op_name == 'concatenate':
                args = self._get_values(call.args[0].fields)
                attrs = self._cvt_attrs(call.attrs)
                return self._add_param(np.concatenate(args, axis=attrs['axis']))
            else:
                return call
        except _FoldException:
            return call

    def _get_values(self, args: List[relay.Expr]) -> List[np.ndarray]:
        values = []
        for a in args:
            if isinstance(a, relay.Constant):
                values.append(np.array(a.data.asnumpy(), dtype=a.checked_type.dtype))
            elif isinstance(a, relay.Var) and a.name_hint in self.params:
                values.append(self.params[a.name_hint])
            else:
                raise _FoldException()
        return values

    @classmethod
    def _cvt_attrs(cls, attrs: Optional[ir.Attrs]) -> Dict[str, Any]:
        if (not isinstance(attrs, ir.Attrs)) or len(attrs.keys()) == 0:
            return {}
        else:
            return dict([(name, util.cvt_ir_value(attrs[name])) for name in attrs.keys()])

    def _add_param(self, value: np.ndarray) -> relay.Var:
        name = self._next_param_name()
        self.params[name] = value
        return relay.var(name, shape=value.shape, dtype=str(value.dtype))

    def _next_param_name(self) -> str:
        while True:
            name = '_param_{}'.format(self.next_idx)
            self.next_idx += 1
            if name not in self.params:
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
    'ones',
    'reshape',
    'transpose',
}

_eval_funcs: Dict[str, Callable[[List[np.ndarray], Dict[str, Any]], np.ndarray]] = {
    'expand_dims':
        lambda args, attrs: _expand_dims(args[0], attrs['axis'], attrs['num_newaxis']),
    'cast':
        lambda args, attrs: args[0].astype(attrs['dtype']),
    'matrix_set_diag':
        lambda args, attrs: _matrix_set_diag(args[0], args[1]),
    'nn.pad':
        lambda args, attrs: np.pad(args[0], attrs['pad_width']),
    'nn.batch_matmul':
        lambda args, _: np.matmul(args[0], np.transpose(args[1], axes=(0, 2, 1))),
}


def _expand_dims(data: np.ndarray, axis: int, num_newaxis: int) -> np.ndarray:
    for i in range(num_newaxis):
        data = np.expand_dims(data, axis + i)
    return data


def _matrix_set_diag(data: np.ndarray, diagonal: np.ndarray) -> np.ndarray:
    data = data.copy()
    data[np.diag_indices_from(data)] = diagonal
    return data
