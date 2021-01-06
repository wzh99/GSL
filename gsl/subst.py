from inspect import signature, Parameter
from typing import Optional, Set

from tvm import ir, transform
from tvm.relay import dataflow_pattern as dfp

from . import util
from .fold import ParamFoldPass
from .graph import *
from .work import Workload


class Substitution:
    """
    Represents a graph substitution rule.
    """

    def __init__(self, src: Node, tgt: Node):
        """
        Constructor.
        :param src: Source graph pattern.
        :param tgt: Target graph pattern.
        """
        # Check source and target
        src_checker = _SrcPatChecker()
        src_checker.visit(src)
        _TgtPatChecker(set(src_checker.visited.keys()), src_checker.wildcard_vars).visit(tgt)

        # Create expression rewriter
        self.rewriter = _ExprRewriter(src, tgt)

    def __call__(self, wl: Workload, fold_param: bool = True, new_name: Optional[str] = None) \
            -> Workload:
        """
        Apply substitution to workload.
        :param wl: Workload whose graph is to be altered.
        :param fold_param: whether to pre-compute nodes whose operands are already available.
        :return New workload after application of substitution rule.
        """

        # Keep original name if new name is not provided
        if new_name is None:
            new_name = wl.name

        # Apply substitution to graph
        mod = _SubstFuncPass(self.rewriter)(wl.mod)
        if fold_param:
            fold_pass = ParamFoldPass(wl.params)
            mod = fold_pass(mod)
            new_wl = Workload(mod, fold_pass.params, name=new_name)
        else:
            new_wl = Workload(mod, wl.params, name=new_name)

        # Filter out unused parameters
        param_names = set([p.name_hint for p in mod['main'].params])
        used_params = dict()
        for new_name, val in new_wl.params.items():
            if param_names.__contains__(new_name):
                used_params[new_name] = val
        new_wl.params = used_params

        return new_wl


class _SrcPatChecker(NodeVisitor):
    def __init__(self):
        super().__init__()
        self.wildcard_vars: Set[Union[Wildcard, Var]] = set()

    def visit_wildcard(self, wildcard: Wildcard) -> Any:
        self.wildcard_vars.add(wildcard)

    def visit_var(self, var: Var) -> Any:
        self.wildcard_vars.add(var)

    def visit_const(self, const: Const) -> Any:
        if isinstance(const.value, AttrExpr):
            raise TypeError(
                'Constant node in source graph cannot store an attribute expression.'
            )

    def visit_call(self, call: Call) -> Any:
        super().visit_call(call)
        for v in call.attrs.values():
            _SrcAttrChecker(self.visited).visit(v)


class _SrcAttrChecker(AttrVisitor):
    def __init__(self, visited: Dict[Node, Any]):
        self.visited = visited

    def visit_get_attr(self, get_attr: GetAttr):
        if not self.visited.__contains__(get_attr.node):
            raise AttributeError(
                'Attribute in source pattern refers to undefined node.'
            )


class _TgtPatChecker(NodeVisitor):
    def __init__(self, src_nodes: Set[Node], wildcard_vars: Set[Union[Wildcard, Var]]):
        super().__init__()
        self.src_nodes = src_nodes
        self.wildcard_vars = wildcard_vars

    def visit_wildcard(self, wildcard: Wildcard) -> Any:
        if not self.wildcard_vars.__contains__(wildcard):
            raise ValueError(
                'Target pattern contains wildcard node not defined in source graph.'
            )

    def visit_var(self, var: Var) -> Any:
        if not self.wildcard_vars.__contains__(var):
            raise ValueError(
                'Target pattern contains variable node not defined in source graph.'
            )

    def visit_const(self, const: Const) -> Any:
        if const.value is None:
            raise ValueError(
                'Constant node in target pattern must contain a value.'
            )

    def visit_call(self, call: Call) -> Any:
        # Visit arguments
        super().visit_call(call)

        # Check if all non-default attributes are provided
        func = op.get_func(call.op)
        num_input = op.num_inputs[func]
        required = set()
        for name, param in list(signature(func).parameters.items())[num_input:]:
            if param.default == Parameter.empty:
                required.add(name)
        required.difference_update(call.attrs.keys())
        if len(required) != 0:
            raise AttributeError('Required attributes {} are not provided for op \'{}\'.'
                                 .format(tuple(required), call.op))

        # Check if all attribute expressions only contain reference to nodes in source graph
        for a in call.attrs.values():
            _TgtAttrChecker(self.src_nodes).visit(a)


class _TgtAttrChecker(AttrVisitor):
    def __init__(self, src_nodes: Set[Node]):
        self.src_nodes = src_nodes

    def visit_get_attr(self, get_attr: GetAttr):
        if not self.src_nodes.__contains__(get_attr.node):
            raise AttributeError(
                'Attribute in target pattern refers to node not defined in source graph.'
            )


@relay.transform.function_pass(opt_level=0)
class _SubstFuncPass:
    def __init__(self, rewriter):
        self.rewriter = rewriter

    def transform_function(self, fn: relay.Function, _mod: ir.IRModule,
                           _ctx: transform.PassContext) -> relay.Function:
        new_body = self.rewriter.rewrite(fn.body)
        return relay.Function(relay.analysis.free_vars(new_body), new_body)

    def __call__(self, mod: ir.IRModule) -> ir.IRModule: ...


class _ExprRewriter(dfp.DFPatternCallback):
    def __init__(self, src: Node, tgt: Node):
        # Initialize fields
        super().__init__()
        self.src = src
        self.tgt = tgt

        # Create source pattern for matching
        pat_creator = _PatternCreator()
        self.pattern = pat_creator.visit(src)
        self.gsl_to_dfp = pat_creator.visited

    def callback(self, pre: relay.Expr, post: relay.Expr, node_map: ir.Map) -> relay.Expr:
        # Map graph pattern nodes to real graph expressions
        gsl_to_expr = dict([(gsl_node, node_map[dfp_node][0])
                            for gsl_node, dfp_node in self.gsl_to_dfp.items()])

        # Further check source graph with additional constraints
        try:
            _SrcGraphMatcher(gsl_to_expr).visit(self.src)
        except _SrcMismatchException:
            return pre  # failed to match, don't modify graph

        # Build target graph
        return _GraphBuilder(gsl_to_expr).visit(self.tgt)


class _PatternCreator(NodeVisitor):
    def visit_wildcard(self, wildcard: Wildcard) -> dfp.DFPattern:
        return dfp.wildcard()

    def visit_var(self, var: Var) -> dfp.DFPattern:
        return dfp.is_var()

    def visit_const(self, const: Const) -> dfp.DFPattern:
        return dfp.is_constant()

    def visit_call(self, call: Call) -> dfp.DFPattern:
        super().visit_call(call)
        args = [self.visited[node] for node in call.args]
        return dfp.is_op(call.op)(*args)

    # noinspection PyTypeChecker
    def visit_tuple(self, tup: Tuple) -> dfp.DFPattern:
        super().visit_tuple(tup)
        fields = [self.visited[node] for node in tup.fields]
        return dfp.is_tuple(fields)

    def visit_getitem(self, getitem: GetItem) -> dfp.DFPattern:
        super().visit_getitem(getitem)
        return dfp.is_tuple_get_item(self.visited[getitem.tup], index=getitem.index)


class _SrcMismatchException(Exception):
    pass


class _SrcGraphMatcher(NodeVisitor):
    def __init__(self, gsl_to_expr: Dict[Node, relay.Expr]):
        super().__init__()
        self.gsl_to_expr = gsl_to_expr

    def visit_const(self, const: Const) -> Any:
        expr = self.gsl_to_expr[const]
        if isinstance(const.value, np.ndarray) and \
                (not np.array_equal(const.value, expr.data.asnumpy())):
            raise _SrcMismatchException()

    def visit_call(self, call: Call) -> Any:
        super().visit_call(call)
        expr = self.gsl_to_expr[call]
        for name, attr in call.attrs.items():
            if not self._attr_equal(expr.attrs[name], attr):
                raise _SrcMismatchException()

    def _attr_equal(self, ir_attr, pat_attr: AttrExpr) -> bool:
        ir_val = util.cvt_ir_value(ir_attr)
        pat_val = AttrEvaluator(self.gsl_to_expr).visit(pat_attr)
        if isinstance(ir_val, (int, float, str)):
            return ir_val == pat_val
        elif isinstance(ir_val, list):
            return ir_val == list(pat_val)
        else:
            return False


class _GraphBuilder(NodeVisitor):
    def __init__(self, gsl_to_expr: Dict[Node, relay.Expr]):
        super().__init__()
        self.gsl_to_expr = gsl_to_expr

    def visit_wildcard(self, wildcard: Wildcard) -> Any:
        return self.gsl_to_expr[wildcard]

    def visit_var(self, var: Var) -> Any:
        return self.gsl_to_expr[var]

    def visit_const(self, const: Const) -> Any:
        if isinstance(const.value, np.ndarray):
            value = const.value
        elif isinstance(const.value, AttrExpr):
            value = AttrEvaluator(self.gsl_to_expr).visit(const.value)
        else:
            raise RuntimeError('Impossible case.')
        return relay.const(value)

    def visit_call(self, call: Call) -> Any:
        args = [self.visit(a) for a in call.args]
        attrs = dict([(name, AttrEvaluator(self.gsl_to_expr).visit(attr))
                      for name, attr in call.attrs.items()])
        func = op.get_func(call.op)
        return func(*args, **attrs)

    def visit_tuple(self, tup: Tuple) -> Any:
        return relay.Tuple([self.visit(f) for f in tup.fields])

    def visit_getitem(self, getitem: GetItem) -> Any:
        return self.visit(getitem.tup)[getitem.index]


class ExprRewriter:
    def __init__(self, src_pats: List[Node], tgt_pats: List[Node]):
        if len(src_pats) != len(tgt_pats):
            raise ValueError('Number of source and target patterns does not match.')
        self.src_pats = src_pats
        self.tgt_pats = tgt_pats

    def rewrite(self, expr: relay.Expr) -> relay.Expr:
        while True:
            # Find matched expressions with patterns
            pat_to_expr: Dict[Node, relay.Expr] = dict()
            src_matched = []
            for src_pat in self.src_pats:
                src_expr = self._find_match(src_pat, expr, pat_to_expr, src_matched)
                if src_expr is None:
                    return expr  # even one subgraph is not found, exit immediately
                else:
                    src_matched.append(src_expr)

            # Generate target expressions and map source to them
            tgt_expr = [_RelayBuilder(pat_to_expr).visit(tgt) for tgt in self.tgt_pats]
            expr_map = dict(zip(src_matched, tgt_expr))

            # Rewrite expression
            expr = _RewriteMutator(expr_map).visit(expr)

    def _find_match(self, pat: Node, expr: relay.Expr, pat_to_expr: Dict[Node, relay.Expr],
                    src_matched: List[relay.Expr]) -> Optional[relay.Expr]:
        class StackElem:
            def __init__(self, e: relay.Expr, count: int):
                self.expr = e
                self.count = count

        stack: List[StackElem] = [StackElem(expr, 0)]
        visited: Set[relay.Expr] = {expr}

        def update(e: relay.Expr):
            if not visited.__contains__(e):
                stack.append(StackElem(e, 0))
                visited.add(e)

        while len(stack) > 0:
            # Pop an element from stack
            elem = stack.pop()

            if elem.count == 0:
                # Add children to stack if this expression is visited for the first time
                stack.append(StackElem(elem.expr, 1))
                if isinstance(elem.expr, relay.Call):
                    for a in reversed(elem.expr.args):
                        update(a)
                elif isinstance(elem.expr, relay.Tuple):
                    for f in reversed(elem.expr.fields):
                        update(f)
                elif isinstance(elem.expr, relay.TupleGetItem):
                    update(elem.expr.tuple_value)
            else:
                # Match pattern with this expression if it has been visited once
                if src_matched.__contains__(elem.expr):
                    continue  # matched expression cannot be matched again
                matcher = _ExprMatcher(pat_to_expr.copy())
                res = matcher.match(pat, elem.expr)
                if res:
                    pat_to_expr.update(matcher.pat_to_expr)  # update map if matches
                    return elem.expr

        return None


class _RelayBuilder(NodeVisitor):
    def __init__(self, pat_to_expr: Dict[Node, relay.Expr]):
        super().__init__()
        self.visited = pat_to_expr

    def visit_wildcard(self, wildcard: Wildcard) -> Any:
        return self.visited[wildcard]

    def visit_var(self, var: Var) -> Any:
        return self.visited[var]

    def visit_const(self, const: Const) -> Any:
        if isinstance(const.value, np.ndarray):
            value = const.value
        elif isinstance(const.value, AttrExpr):
            value = AttrEvaluator(self.visited).visit(const.value)
        else:
            raise RuntimeError('Impossible case.')
        return relay.const(value)

    def visit_call(self, call: Call) -> Any:
        args = [self.visit(a) for a in call.args]
        attrs = dict([(name, AttrEvaluator(self.visited).visit(attr))
                      for name, attr in call.attrs.items()])
        func = op.get_func(call.op)
        return func(*args, **attrs)

    def visit_tuple(self, tup: Tuple) -> Any:
        return relay.Tuple([self.visit(f) for f in tup.fields])

    def visit_getitem(self, getitem: GetItem) -> Any:
        return self.visit(getitem.tup)[getitem.index]


class _RewriteMutator(relay.ExprMutator):
    def __init__(self, expr_map: Dict[relay.Expr, relay.Expr]):
        super().__init__()
        self.expr_map = expr_map

    def visit(self, expr: relay.Expr):
        if self.expr_map.__contains__(expr):
            return self.expr_map[expr]
        return super().visit(expr)


class _ExprMatcher:
    def __init__(self, pat_to_expr: Dict[Node, relay.Expr]):
        self.pat_to_expr = pat_to_expr

    def match(self, pat: Node, expr: relay.Expr) -> bool:
        if self.pat_to_expr.__contains__(pat) and self.pat_to_expr[pat] != expr:
            return False
        if isinstance(pat, Wildcard):
            res = True
        elif isinstance(pat, Var):
            res = self.match_var(pat, expr)
        elif isinstance(pat, Const):
            res = self.match_const(pat, expr)
        elif isinstance(pat, Call):
            res = self.match_call(pat, expr)
        elif isinstance(pat, Tuple):
            res = self.match_tuple(pat, expr)
        elif isinstance(pat, GetItem):
            res = self.match_getitem(pat, expr)
        else:
            res = False
        if res:
            self.pat_to_expr[pat] = expr
        return res

    @classmethod
    def match_var(cls, _var: Var, expr: relay.Expr) -> bool:
        return isinstance(expr, relay.Var)

    @classmethod
    def match_const(cls, const: Const, expr: relay.Expr) -> bool:
        # Match constant node
        if not isinstance(expr, relay.Constant):
            return False

        # Match value if provided
        if isinstance(const.value, np.ndarray) and \
                (not np.array_equal(const.value, expr.data.asnumpy())):
            return False

        return True

    def match_call(self, call: Call, expr: relay.Expr) -> bool:
        # Match call node
        if not isinstance(expr, relay.Call):
            return False

        # Match op
        # If op matches, the number of arguments also matches
        if call.op != expr.op.name:
            return False

        # Match attributes
        for name, attr in call.attrs.items():
            if not self._attr_equal(attr, expr.attrs[name]):
                return False

        # Match arguments
        for pat_arg, expr_arg in zip(call.args, expr.args):
            if not self.match(pat_arg, expr_arg):
                return False

        return True

    def _attr_equal(self, pat_attr: AttrExpr, expr_attr) -> bool:
        ir_val = util.cvt_ir_value(expr_attr)
        pat_val = AttrEvaluator(self.pat_to_expr).visit(pat_attr)
        if isinstance(ir_val, (int, float, str)):
            return ir_val == pat_val
        elif isinstance(ir_val, list):
            return ir_val == list(pat_val)
        else:
            return False

    def match_tuple(self, tup: Tuple, expr: relay.Expr) -> bool:
        # Match tuple node
        if not isinstance(expr, relay.Tuple):
            return False

        # Check number of fields
        if len(tup.fields) != len(expr.fields):
            return False

        # Match fields
        for pat_f, expr_f in zip(tup.fields, expr.fields):
            if not self.match(pat_f, expr_f):
                return False

        return True

    def match_getitem(self, getitem: GetItem, expr: relay.Expr) -> bool:
        if not isinstance(expr, relay.TupleGetItem):
            return False
        return getitem.index == expr.index and self.match(getitem.tup, expr.tuple_value)
