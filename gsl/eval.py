from tvm import ir, relay

from . import util
from .pat import *

PatExprMap = Dict[Pattern, relay.Expr]


class AttrEvaluator(AttrVisitor[Env]):
    def __init__(self, pat_to_expr: PatExprMap):
        self.pat_to_expr = pat_to_expr

    def visit(self, attr: Attr, env: Env) -> Any:
        return util.cvt_ir_value(super().visit(attr, env))

    def visit_any(self, a: AnyAttr, env: Env):
        return None

    def visit_const(self, const: ConstAttr, env: Env):
        return const.value

    def visit_getattr(self, get_attr: GetAttr, env: Env):
        # Get actual expression from map
        pat = get_attr.pat
        if isinstance(pat, GetInstance):  # map template to instance
            pat = eval_get_inst(pat, self.pat_to_expr, env)
        name = get_attr.name

        # Handle variadic pattern
        if isinstance(pat, Variadic) and name == 'length':
            return len(pat)

        # Get expression from mapping
        if pat not in self.pat_to_expr:
            raise RuntimeError(
                'Pattern not matched before.'
            )
        expr = self.pat_to_expr[pat]

        # Get attribute from expression
        if name in Pattern.shared_attrs:
            return util.get_shared_attr(expr, name)
        elif isinstance(pat, Call):
            if (expr.attrs is None) or (name not in expr.attrs.keys()):
                raise RuntimeError(
                    'Attribute \'{}\' not found in op \'{}\'.'.format(name, expr.op.name)
                )
            return expr.attrs[name]
        elif isinstance(pat, Variadic) and name == 'length':
            return len(pat)
        else:
            raise RuntimeError('Unreachable.')

    def visit_tuple(self, tup_attr: TupleAttr, env: Env):
        return tuple([self.visit(f, env) for f in tup_attr.fields])

    def visit_getitem(self, getitem: GetItemAttr, env: Env):
        return self.visit(getitem.seq, env)[self.visit(getitem.index, env)]

    def visit_binary(self, binary: BinaryAttr, env: Env):
        lv, rv = self.visit(binary.lhs, env), self.visit(binary.rhs, env)
        ty_tup = (lv.__class__, rv.__class__)
        bin_op = binary.op
        op_func = BinaryAttr.eval_func[bin_op]
        if ty_tup not in op_func:
            raise RuntimeError(
                'Operator \'{}\' not defined for type ({}, {})'.format(
                    bin_op.value, ty_tup[0], ty_tup[1]
                )
            )
        return op_func[ty_tup](lv, rv)

    def visit_symbol(self, sym: Symbol, env: Env):
        val = env[sym]
        if val is None:
            raise RuntimeError('Symbol \'{}\' not found in environment.'.format(sym))
        return val

    def visit_variadic(self, var: VariadicAttr, env: Env):
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
            fields.append(self.visit(var.attr, new_env))

        return fields

    def visit_sum(self, s: Sum, env: Env):
        length = self.visit(s.len, env)
        result = 0
        for i in range(length):
            result += self.visit(s.attr, env + (s.index, i))
        return result


def eval_get_inst(get_inst: GetInstance, pat_to_expr: PatExprMap, env: Env) -> Pattern:
    idx = AttrEvaluator(pat_to_expr).visit(get_inst.index, env)
    return get_inst.var.get_instance(idx, get_inst.t)
