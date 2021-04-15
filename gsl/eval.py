from typing import Dict, Any, Optional, List, Set

from tvm import relay

from . import attr, pat, util
from .attr import Attr, Env
from .pat import Pattern


class PatExprMap:
    def __init__(self):
        self._map: Dict[Pattern, relay.Expr] = dict()
        self._pat: List[Pattern] = []
        self._expr: Set[relay.Expr] = set()

    def __len__(self):
        return len(self._pat)

    def __contains__(self, item: Pattern):
        return item in self._map

    def __getitem__(self, item: Pattern):
        return self._map[item]

    def __setitem__(self, key: Pattern, value: relay.Expr):
        assert key not in self._map
        self._map[key] = value
        self._pat.append(key)
        self._expr.add(value)

    def items(self):
        return self._map.items()

    def has_expr(self, expr: relay.Expr):
        return expr in self._expr

    def record(self):
        return self.MapRecord(self)

    def _rollback(self, n: int):
        assert n <= len(self._pat)
        for p in self._pat[n:]:
            self._expr.remove(self._map[p])
            del self._map[p]
        self._pat = self._pat[:n]

    class MapRecord:
        def __init__(self, m: 'PatExprMap'):
            self._map = m
            self._len = len(m)

        def restore(self):
            self._map._rollback(self._len)


ExprTypeMap = Dict[relay.Expr, relay.Type]
EvalHistory = Dict[Attr, Any]


class AttrEvaluator(attr.AttrVisitor[Env, Any]):
    def __init__(self, pat_to_expr: PatExprMap, ty_map: ExprTypeMap,
                 eval_his: Optional[EvalHistory] = None):
        self.pat_to_expr = pat_to_expr
        self.ty_map = ty_map
        self.eval_his = eval_his

    def visit(self, a: Attr, env: Env) -> Any:
        if (self.eval_his is not None) and (not a.has_free_sym) and (a in self.eval_his):
            return self.eval_his[a]
        else:
            val = util.cvt_ir_value(super().visit(a, env))
            if (self.eval_his is not None) and (not a.has_free_sym):
                self.eval_his[a] = val
            return val

    def visit_none(self, n: attr.NoneAttr, env: Env):
        return None

    def visit_const(self, const: attr.Const, env: Env):
        return const.value_

    def visit_getattr(self, get_attr: attr.GetAttr, env: Env):
        # Get actual expression from map
        p = get_attr.pat_
        if isinstance(p, pat.GetInst):  # map template to instance
            p = eval_get_inst(p, self.pat_to_expr, self.ty_map, env)
        name = get_attr.name_

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
        elif isinstance(p, pat.Const) and name == 'value':
            return expr.data
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
        stop = self.visit(ran_attr.stop_, env)
        start = self.visit(ran_attr.start_, env)
        step = self.visit(ran_attr.step_, env)
        if start is None:
            ran_val = range(stop)
        elif step is None:
            ran_val = range(start, stop)
        else:
            ran_val = range(start, stop, step)
        return tuple(ran_val)

    def visit_tuple(self, tup_attr: attr.Tuple, env: Env):
        return tuple([self.visit(f, env) for f in tup_attr.fields_])

    def visit_tuple_len(self, tuple_len: attr.TupleLen, env: Env):
        return len(self.visit(tuple_len.tup_, env))

    def visit_getitem(self, getitem: attr.GetItem, env: Env):
        return self.visit(getitem.tup_, env)[self.visit(getitem.index_, env)]

    def visit_slice(self, slc: attr.Slice, env: Env):
        start = self.visit(slc.start_, env)
        stop = self.visit(slc.stop_, env)
        step = self.visit(slc.step_, env)
        return slice(start, stop, step)

    def visit_getslice(self, getslice: attr.GetSlice, env: Env):
        tup = self.visit(getslice.tup_, env)
        slc = self.visit_slice(getslice.slc_, env)
        return tup[slc]

    def visit_reverse(self, rev: attr.Reverse, env: Env):
        tup = self.visit(rev.tup_, env)
        return tuple(reversed(tup))

    def visit_unary(self, unary: attr.Unary, env: Env):
        v = self.visit(unary.attr_, env)
        v_ty = v.__class__
        uop = unary.op_
        op_func = attr.Unary.eval_funcs[uop]
        if v_ty not in op_func:
            raise RuntimeError(
                'Operator \'{}\' not defined for type {}'.format(
                    uop.value, v_ty
                )
            )
        return op_func[v_ty](v)

    def visit_binary(self, binary: attr.Binary, env: Env):
        # Evaluate both sides
        lv, rv = self.visit(binary.lhs_, env), self.visit(binary.rhs_, env)

        # Handle == and !=
        if binary.op_ == attr.BinaryOp.EQ:
            return lv == rv
        elif binary.op_ == attr.BinaryOp.NE:
            return lv != rv

        # Evaluate rest ops
        ty_tup = (lv.__class__, rv.__class__)
        bin_op = binary.op_
        op_func = attr.Binary.eval_func[bin_op]
        if ty_tup not in op_func:
            raise RuntimeError(
                'Operator \'{}\' not defined for type ({}, {})'.format(
                    bin_op.value, ty_tup[0], ty_tup[1]
                )
            )
        return op_func[ty_tup](lv, rv)

    def visit_cond(self, cond: attr.Cond, env: Env):
        pv = self.visit(cond.pred_, env)
        if not isinstance(pv, bool):
            raise RuntimeError(
                'Predicate of condition cannot be evaluated to a boolean value.'
            )
        return self.visit(cond.then_br_, env) if pv else self.visit(cond.else_br_, env)

    def visit_match(self, match: attr.Match, env: Env):
        alt = match.alt_
        if alt.matched_idx_ is None:
            raise RuntimeError(
                'None of the alternative pattern is matched.'
            )
        return self.visit(match.clauses_[alt.matched_idx_], env)

    def visit_layout_remap(self, remap: attr.LayoutRemap, env: Env):
        src = self.visit(remap.src_, env)
        if not isinstance(src, str):
            raise RuntimeError(
                'Source layout is not a string.'
            )
        tgt = self.visit(remap.tgt_, env)
        if not isinstance(tgt, str):
            raise RuntimeError(
                'Target layout is not a string.'
            )
        indices: List[int] = []
        for c in tgt:
            idx = src.find(c)
            if idx == -1:
                raise RuntimeError(
                    'Cannot convert layout from \'{}\' to \'{}\'.'.format(src, tgt)
                )
            indices.append(idx)
        return tuple(indices)

    def visit_symbol(self, sym: attr.Symbol, env: Env):
        val = env[sym]
        if val is None:
            raise RuntimeError(
                'Symbol \'{}\' not found in environment.'.format(sym)
            )
        return val

    def visit_variadic(self, var: attr.Variadic, env: Env):
        # Check if length is provided
        if var.len_ is None:
            raise RuntimeError(
                'Cannot evaluate variadic attribute whose length is not specified.'
            )
        length = self.visit(var.len_, env)

        # Evaluate fields
        fields = []
        for i in range(length):
            fields.append(self.visit(var.field_, env + (var.index_, i)))

        return fields

    def visit_in(self, in_tup: attr.In, env: Env):
        return self.visit(in_tup.val_, env) in self.visit(in_tup.tup_, env)

    def visit_map(self, m: attr.Map, env: Env):
        tup = self.visit(m.tup_, env)
        return tuple(map(lambda e: self.visit(m.body_, env + (m.sym_, e)), tup))

    def visit_zip(self, z: attr.Zip, env: Env):
        return tuple(zip(*[self.visit(tup, env) for tup in z.tuples_]))

    def visit_reduce_indexed(self, red: attr.ReduceIndexed, env: Env):
        length = self.visit(red.len_, env)
        result = self.visit(red.init_, env)
        for i in range(length):
            elem = self.visit(red.elem_, env + (red.index_, i))
            result = self._try_reduce(red.op_, result, elem)
        return result

    def visit_reduce_tuple(self, red: attr.ReduceTuple, env: Env):
        tup = self.visit(red.tup_, env)
        result = self.visit(red.init, env)
        for elem in tup:
            result = self._try_reduce(red.op_, result, elem)
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


def eval_get_inst(get_inst: pat.GetInst, pat_to_expr: PatExprMap, ty_map: ExprTypeMap, env: Env,
                  eval_his: Optional[EvalHistory] = None) \
        -> pat.Pattern:
    idx = AttrEvaluator(pat_to_expr, ty_map, eval_his).visit(get_inst.idx_, env)
    return get_inst.var_.get_inst(idx, get_inst.tpl_)
