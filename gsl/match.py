import numpy as np
from tvm import relay, ir

from . import attr, pat, spec, util
from .attr import Attr, Env
from .eval import PatExprMap, ExprTypeMap, AttrEvaluator, eval_get_inst


class Matcher:
    def __init__(self, pat_to_expr: PatExprMap, ty_map: ExprTypeMap):
        self.pat_to_expr = pat_to_expr
        self.ty_map = ty_map

    def match(self, p: pat.Pattern, expr: relay.Expr, env: Env) -> bool:
        # Already matched, use history record
        if p in self.pat_to_expr:
            return self.pat_to_expr[p] == expr

        # Reject if the expression has been matched with another node
        if p.injective and self.pat_to_expr.has_expr(expr):
            return False

        # Try matching according to pattern node type
        if isinstance(p, pat.Wildcard):
            res = True
        elif isinstance(p, pat.Variable):
            res = self.match_var(p, expr, env)
        elif isinstance(p, pat.Const):
            res = self.match_const(p, expr, env)
        elif isinstance(p, pat.Op):
            res = self.match_op(p, expr, env)
        elif isinstance(p, pat.Call):
            res = self.match_call(p, expr, env)
        elif isinstance(p, pat.Tuple):
            res = self.match_tuple(p, expr, env)
        elif isinstance(p, pat.GetItem):
            res = self.match_getitem(p, expr, env)
        elif isinstance(p, pat.Alt):
            res = self.match_alt(p, expr, env)
        elif isinstance(p, pat.Variadic):
            res = self.match_variadic(p, expr, env)
        elif isinstance(p, pat.GetInst):
            res = self.match_get_inst(p, expr, env)
        else:
            res = False

        # Add to record if matched
        if res and not isinstance(p, pat.ConcreteOp):
            self.pat_to_expr[p] = expr
        return res

    def match_var(self, var: pat.Variable, expr: relay.Expr, env: Env) -> bool:
        # Match variable node
        if not isinstance(expr, relay.Var):
            return False

        # Match attributes
        ty = expr.checked_type
        for n, a in var.attrs.items():
            if not self.match_attr(a, util.get_tensor_type_attr(ty, n), env):
                return False

        return True

    def match_const(self, const: pat.Const, expr: relay.Expr, env: Env) -> bool:
        # Match constant node
        if not isinstance(expr, relay.Constant):
            return False

        # Match value if provided
        if const.val_ is None:
            return True
        expr_val = expr.data.asnumpy()
        if isinstance(const.val_, np.ndarray):
            if not np.array_equal(const.val_, expr_val):
                return False
        if isinstance(const.val_, attr.Attr):
            pat_val = AttrEvaluator(self.pat_to_expr, self.ty_map).visit(const.val_, env)
            if not np.array_equal(pat_val, expr_val):
                return False

        return True

    def match_call(self, call: pat.Call, expr: relay.Expr, env: Env) -> bool:
        # Match call node
        if not isinstance(expr, relay.Call):
            return False

        # Match op
        if not self.match(call.op, expr.op, env):
            return False

        # Match arguments
        # Arguments must be matched before attributes, because attribute matching may depend on
        # match result of arguments.
        if len(call.args) != len(expr.args):
            return False
        for pat_arg, expr_arg in zip(call.args, expr.args):
            if not self.match(pat_arg, expr_arg, env):
                return False

        # Match attributes
        for n, a in call.attrs.items():
            if (expr.attrs is None) or (n not in expr.attrs.keys()):
                raise RuntimeError(
                    'Attribute \'{}\' not found in op \'{}\'.'.format(n, expr.op.name)
                )
            if not self.match_attr(a, expr.attrs[n], env):
                return False

        return True

    @classmethod
    def match_op(cls, pat_op: pat.Op, expr_op: relay.Expr, _env: Env) -> bool:
        if not isinstance(expr_op, ir.Op):
            return False
        if isinstance(pat_op, pat.ConcreteOp):
            return pat_op.name == expr_op.name
        elif isinstance(pat_op, pat.OpWithTrait):
            return spec.match_trait(expr_op.name, pat_op.trait)
        else:
            raise RuntimeError('Unreachable.')

    def match_tuple(self, tup: pat.Tuple, expr: relay.Expr, env: Env) -> bool:
        # Match tuple node
        if not isinstance(expr, relay.Tuple):
            return False

        # Check number of fields
        if len(tup.fields) != len(expr.fields):
            return False

        # Match fields
        for pat_f, expr_f in zip(tup.fields, expr.fields):
            if not self.match(pat_f, expr_f, env):
                return False

        return True

    def match_getitem(self, getitem: pat.GetItem, expr: relay.Expr, env: Env) -> bool:
        if not isinstance(expr, relay.TupleGetItem):
            return False
        if not self.match(getitem.tup, expr.tuple_value, env):
            return False
        return self.match_attr(getitem.idx, expr.index, env)

    def match_alt(self, alt: pat.Alt, expr: relay.Expr, env: Env) -> bool:
        rec = self.pat_to_expr.record()
        for idx in range(len(alt.pats)):
            if self.match(alt.pats[idx], expr, env):
                alt.matched_idx = idx
                return True
            else:
                rec.restore()
        return False

    def match_variadic(self, var: pat.Variadic, expr: relay.Expr, env: Env) -> bool:
        # Matches tuple node
        if not isinstance(expr, relay.Tuple):
            return False

        # Match length if provided
        if var.len is not None:
            length = AttrEvaluator(self.pat_to_expr, self.ty_map).visit(var.len, env)
            if length != len(expr.fields):
                return False

        # Match tuple fields
        for i in range(len(expr.fields)):
            expr_f = expr.fields[i]
            new_env = env
            if var.index is not None:
                new_env = env + (var.index, i)
            pat_f = var.instantiate()
            if not self.match(pat_f, expr_f, new_env):
                return False

        return True

    def match_get_inst(self, get_inst: pat.GetInst, expr: relay.Expr, env: Env) -> bool:
        p = eval_get_inst(get_inst, self.pat_to_expr, self.ty_map, env)
        return self.match(p, expr, env)

    def match_attr(self, pat_attr: Attr, expr_attr, env: Env) -> bool:
        # Get expression attribute value
        expr_val = util.cvt_ir_value(expr_attr)

        # Match attribute according to attribute type
        if isinstance(pat_attr, attr.Variadic):
            # Variadic can only match sequential collection
            if not isinstance(expr_val, (tuple, list)):
                return False

            # Check length if provided
            if pat_attr.len is not None:
                length = AttrEvaluator(self.pat_to_expr, self.ty_map).visit(pat_attr.len, env)
                if length != len(expr_val):
                    return False

            # Match fields
            for i in range(len(expr_val)):
                new_env = env if pat_attr.index is None \
                    else Env(prev=env, symbol=pat_attr.index, value=i)
                if not self.match_attr(pat_attr.field, expr_val[i], new_env):
                    return False

            return True
        else:
            pat_val = AttrEvaluator(self.pat_to_expr, self.ty_map).visit(pat_attr, env)
            return self._match_val(pat_val, expr_val)

    @classmethod
    def _match_val(cls, pat_val, expr_val) -> bool:
        if pat_val is None:
            return True  # `None` matches any value
        elif isinstance(pat_val, (int, float, str)):
            return pat_val == expr_val
        elif isinstance(pat_val, (tuple, list)) and isinstance(expr_val, (tuple, list)):
            if len(pat_val) != len(expr_val):
                return False
            for p, e in zip(pat_val, expr_val):
                if not cls._match_val(p, e):
                    return False
            return True
        else:
            return False
