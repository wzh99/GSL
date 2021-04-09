from typing import Dict, Any

from tvm import relay

from . import attr, pat, util
from .attr import Attr, Env
from .pat import Pattern

PatExprMap = Dict[pat.Pattern, relay.Expr]
ExprTypeMap = Dict[relay.Expr, relay.Type]


class AttrEvaluator(attr.AttrVisitor[Env, Any]):
    def __init__(self, pat_to_expr: PatExprMap, ty_map: ExprTypeMap):
        self.pat_to_expr = pat_to_expr
        self.ty_map = ty_map

    def visit(self, a: Attr, env: Env) -> Any:
        return util.cvt_ir_value(super().visit(a, env))

    def visit_any(self, a: attr.Any, env: Env):
        return None

    def visit_const(self, const: attr.Const, env: Env):
        return const.value

    def visit_getattr(self, get_attr: attr.GetAttr, env: Env):
        # Get actual expression from map
        p = get_attr.pat
        if isinstance(p, pat.GetInst):  # map template to instance
            p = eval_get_inst(p, self.pat_to_expr, self.ty_map, env)
        name = get_attr.name

        # Handle variadic pattern
        if isinstance(p, pat.Variadic) and name == 'length':
            return len(p)

        # Get expression from mapping
        if p not in self.pat_to_expr:
            raise RuntimeError('Pattern not matched before.')
        expr = self.pat_to_expr[p]

        # Get attribute from expression
        if name in Pattern.tensor_attrs:
            return self._get_tensor_attr(expr, name)
        elif isinstance(p, pat.Call) and expr.attrs is not None and name in expr.attrs.keys():
            return expr.attrs[name]
        elif isinstance(p, pat.GetItem) and name == 'index':
            return expr.index
        else:
            raise RuntimeError('Cannot get attribute from expression.')

    def _get_tensor_attr(self, expr: relay.Expr, name: str):
        if expr not in self.ty_map:
            raise RuntimeError(
                'Type of expression is not available.'
            )
        ty = self.ty_map[expr]
        if not isinstance(ty, relay.TensorType):
            raise RuntimeError(
                'Expression is not of tensor type.'
            )
        return util.get_tensor_type_attr(ty, name)

    def visit_range(self, ran_attr: attr.Range, env: Env):
        stop = self.visit(ran_attr.stop, env)
        start = self.visit(ran_attr.start, env)
        step = self.visit(ran_attr.step, env)
        if start is None:
            ran_val = range(stop)
        elif step is None:
            ran_val = range(start, stop)
        else:
            ran_val = range(start, stop, step)
        return tuple(ran_val)

    def visit_tuple(self, tup_attr: attr.Tuple, env: Env):
        return tuple([self.visit(f, env) for f in tup_attr.fields])

    def visit_tuple_len(self, tuple_len: attr.TupleLen, env: Env):
        return len(self.visit(tuple_len.tup, env))

    def visit_getitem(self, getitem: attr.GetItem, env: Env):
        return self.visit(getitem.tup, env)[self.visit(getitem.index, env)]

    def visit_slice(self, slc: attr.Slice, env: Env):
        start = self.visit(slc.start, env)
        stop = self.visit(slc.stop, env)
        step = self.visit(slc.step, env)
        return slice(start, stop, step)

    def visit_getslice(self, getslice: attr.GetSlice, env: Env):
        tup = self.visit(getslice.tup, env)
        slc = self.visit_slice(getslice.slc, env)
        return tup[slc]

    def visit_unary(self, unary: attr.Unary, env: Env):
        v = self.visit(unary.attr, env)
        v_ty = v.__class__
        uop = unary.op
        op_func = attr.Unary.eval_funcs[uop]
        if v_ty not in op_func:
            raise RuntimeError(
                'Operator \'{}\' not defined for type {}'.format(
                    uop.value, v_ty
                )
            )
        return op_func[v_ty](v)

    def visit_binary(self, binary: attr.Binary, env: Env):
        lv, rv = self.visit(binary.lhs, env), self.visit(binary.rhs, env)
        ty_tup = (lv.__class__, rv.__class__)
        bin_op = binary.op
        op_func = attr.Binary.eval_func[bin_op]
        if ty_tup not in op_func:
            raise RuntimeError(
                'Operator \'{}\' not defined for type ({}, {})'.format(
                    bin_op.value, ty_tup[0], ty_tup[1]
                )
            )
        return op_func[ty_tup](lv, rv)

    def visit_cond(self, cond: attr.Cond, env: Env):
        pv = self.visit(cond.pred, env)
        if not isinstance(pv, bool):
            raise RuntimeError(
                'Predicate of condition cannot be evaluated to a boolean value.'
            )
        return self.visit(cond.then_br, env) if pv else self.visit(cond.else_br, env)

    def visit_symbol(self, sym: attr.Symbol, env: Env):
        val = env[sym]
        if val is None:
            raise RuntimeError(
                'Symbol \'{}\' not found in environment.'.format(sym)
            )
        return val

    def visit_variadic(self, var: attr.Variadic, env: Env):
        # Check if length is provided
        if var.len is None:
            raise RuntimeError(
                'Cannot evaluate variadic attribute whose length is not specified.'
            )
        length = self.visit(var.len, env)

        # Evaluate fields
        fields = []
        for i in range(length):
            new_env = env if var.len is None else env + (var.index, i)
            fields.append(self.visit(var.field, new_env))

        return fields

    def visit_map(self, m: attr.Map, env: Env):
        tup = self.visit(m.tup, env)
        return tuple(map(lambda e: self.visit(m.body, env + (m.sym, e)), tup))

    def visit_reduce_indexed(self, red: attr.ReduceIndexed, env: Env):
        length = self.visit(red.len, env)
        result = self.visit(red.init, env)
        for i in range(length):
            elem = self.visit(red.elem, env + (red.index, i))
            result = self._try_reduce(red.op, result, elem)
        return result

    def visit_reduce_tuple(self, red: attr.ReduceTuple, env: Env):
        tup = self.visit(red.tup, env)
        result = self.visit(red.init, env)
        for elem in tup:
            result = self._try_reduce(red.op, result, elem)
        return result

    @classmethod
    def _try_reduce(cls, op: attr.BinaryOp, prev: Any, elem: Any) -> Any:
        ty_tup = (prev.__class__, elem.__class__)
        func_map = attr.Binary.eval_func[op]
        if ty_tup not in func_map:
            raise RuntimeError(
                'Cannot reduce values of type ({}, {})'.format(ty_tup[0], ty_tup[1])
            )
        return func_map[ty_tup](prev, elem)


def eval_get_inst(get_inst: pat.GetInst, pat_to_expr: PatExprMap, ty_map: ExprTypeMap, env: Env) \
        -> pat.Pattern:
    idx = AttrEvaluator(pat_to_expr, ty_map).visit(get_inst.idx, env)
    return get_inst.var.get_inst(idx, get_inst.tpl)
