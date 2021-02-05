from . import util
from .eval import *


class Matcher:
    def __init__(self, pat_to_expr: PatExprMap):
        self.pat_to_expr = pat_to_expr
        self.expr_matched = set(pat_to_expr.values())

    def match(self, pat: Pattern, expr: relay.Expr, env: Env) -> bool:
        # Already matched, use history record
        if pat in self.pat_to_expr:
            return self.pat_to_expr[pat] == expr

        # Reject if the expression has been matched with another node
        if expr in self.expr_matched:
            return False

        # Try matching according to pattern node type
        if isinstance(pat, Wildcard):
            res = True
        elif isinstance(pat, Var):
            res = self.match_var(pat, expr, env)
        elif isinstance(pat, Const):
            res = self.match_const(pat, expr, env)
        elif isinstance(pat, Call):
            res = self.match_call(pat, expr, env)
        elif isinstance(pat, Tup):
            res = self.match_tuple(pat, expr, env)
        elif isinstance(pat, GetItem):
            res = self.match_getitem(pat, expr, env)
        elif isinstance(pat, Variadic):
            res = self.match_variadic(pat, expr, env)
        elif isinstance(pat, GetInstance):
            res = self.match_get_inst(pat, expr, env)
        else:
            res = False

        # Add to record if matched
        if res:
            self.pat_to_expr[pat] = expr
            self.expr_matched.add(expr)
        return res

    def match_var(self, var: Var, expr: relay.Expr, env: Env) -> bool:
        # Match variable node
        if not isinstance(expr, relay.Var):
            return False

        # Match attributes
        for name, attr in var.attrs.items():
            if not self._match_attr(attr, AttrEvaluator.get_expr_attr(expr, name), env):
                return False

        return True

    def match_const(self, const: Const, expr: relay.Expr, env: Env) -> bool:
        # Match constant node
        if not isinstance(expr, relay.Constant):
            return False

        # Match value if provided
        expr_val = expr.data.asnumpy()
        if isinstance(const.value, np.ndarray):
            if not np.array_equal(const.value, expr_val):
                return False
        if isinstance(const.value, Attr):
            pat_val = AttrEvaluator(self.pat_to_expr).visit(const.value, env)
            if not np.array_equal(pat_val, expr_val):
                return False

        return True

    def match_call(self, call: Call, expr: relay.Expr, env: Env) -> bool:
        # Match call node
        if not isinstance(expr, relay.Call):
            return False

        # Match op
        # If op matches, the number of arguments also matches
        if not self.match_op(call.op, expr.op):
            return False

        # Match arguments
        # Arguments must be matched before attributes, because attribute matching may depend on
        # match result of arguments.
        for pat_arg, expr_arg in zip(call.args, expr.args):
            if not self.match(pat_arg, expr_arg, env):
                return False

        # Match attributes
        for name, attr in call.attrs.items():
            if (expr.attrs is None) or (name not in expr.attrs.keys()):
                raise RuntimeError(
                    'Attribute \'{}\' not found in op \'{}\'.'.format(name, expr.op.name)
                )
            if not self._match_attr(attr, expr.attrs[name], env):
                return False

        return True

    def match_op(self, pat_op: Op, expr_op: ir.Op) -> bool:
        if isinstance(pat_op, ConcreteOp):
            return pat_op.name == expr_op.name
        elif isinstance(pat_op, OpWithFlag):
            matched = spec.match_flag(expr_op.name, pat_op.flag)
            if matched:
                self.pat_to_expr[pat_op] = expr_op
            return matched
        else:
            raise RuntimeError('Unreachable.')

    def _match_attr(self, pat_attr: Attr, expr_attr, env: Env) -> bool:
        pat_val = AttrEvaluator(self.pat_to_expr).visit(pat_attr, env)
        pat_val = util.cvt_ir_value(pat_val)
        expr_val = util.cvt_ir_value(expr_attr)
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

    def match_tuple(self, tup: Tup, expr: relay.Expr, env: Env) -> bool:
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

    def match_getitem(self, getitem: GetItem, expr: relay.Expr, env: Env) -> bool:
        if not isinstance(expr, relay.TupleGetItem):
            return False
        if not self.match(getitem.tup, expr.tuple_value, env):
            return False
        idx = AttrEvaluator(self.pat_to_expr).visit(getitem.index, env)
        return idx == expr.index

    def match_variadic(self, var: Variadic, expr: relay.Expr, env: Env) -> bool:
        pass

    def match_get_inst(self, get_inst: GetInstance, expr: relay.Expr, env: Env) -> bool:
        pass
