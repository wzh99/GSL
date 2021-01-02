from inspect import signature, Parameter
from typing import Set

from tvm import relay, ir, transform
from tvm.relay import dataflow_pattern as dfp

from graph import *
from work import Workload


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

    def __call__(self, wl: Workload, fold_param: bool = True) -> Workload:
        """
        Apply substitution to workload.
        :param wl: Workload whose graph is to be altered.
        :param fold_param: whether to pre-compute nodes whose operands are already available.
        :return New workload after application of substitution rule.
        """
        # Apply substitution to graph
        new_mod = _SubstFuncPass(self.rewriter)(wl.mod)
        new_wl = Workload(new_mod, wl.params)

        # Filter out unused parameters
        param_names = set([p.name_hint for p in new_mod['main'].params])
        used_params = dict()
        for name, val in new_wl.params.items():
            if param_names.__contains__(name):
                used_params[name] = val
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
                'Attribute in target pattern refers to nodes is not defined in source graph.'
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
        except _SrcNotMatchException:
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
    def visit_tuple(self, tp: Tuple) -> dfp.DFPattern:
        super().visit_tuple(tp)
        fields = [self.visited[node] for node in tp.fields]
        return dfp.is_tuple(fields)

    def visit_getitem(self, getitem: GetItem) -> dfp.DFPattern:
        super().visit_getitem(getitem)
        return dfp.is_tuple_get_item(self.visited[getitem.tup], index=getitem.index)


class _SrcNotMatchException(Exception):
    pass


class _SrcGraphMatcher(NodeVisitor):
    def __init__(self, gsl_to_expr: Dict[Node, relay.Expr]):
        super().__init__()
        self.gsl_to_expr = gsl_to_expr

    def visit_const(self, const: Const) -> Any:
        expr = self.gsl_to_expr[const]
        if isinstance(const.value, np.ndarray) and \
                (not np.array_equal(const.value, np.array(expr.value))):
            raise _SrcNotMatchException()

    def visit_call(self, call: Call) -> Any:
        expr = self.gsl_to_expr[call]
        for name, attr in call.attrs.items():
            if expr.attrs[name] != _AttrEvaluator(self.gsl_to_expr).visit(attr):
                raise _SrcNotMatchException()


class _AttrEvaluator(AttrVisitor):
    def __init__(self, gsl_to_expr: Dict[Node, relay.Expr]):
        self.gsl_to_expr = gsl_to_expr

    def visit_const(self, const: ConstAttr):
        return const.value

    def visit_get_attr(self, get_attr: GetAttr):
        node = get_attr.node
        name = get_attr.name
        expr = self.gsl_to_expr[get_attr.node]
        if isinstance(node, Call):
            return expr.attrs[get_attr.name]
        elif isinstance(node, Var):
            if name == 'shape':
                return expr.type_annotation.concrete_shape
            elif name == 'dtype':
                return expr.type_annotation.dtype
            else:
                raise RuntimeError('Impossible case.')
        else:
            raise RuntimeError('Impossible case.')

    def visit_list(self, list_attr: ListAttr):
        return [self.visit(f) for f in list_attr.fields]

    def visit_tuple(self, tup_attr: TupleAttr):
        return tuple([self.visit(f) for f in tup_attr.fields])

    def visit_getitem(self, getitem: GetItemAttr):
        return self.visit(getitem.seq)[getitem.index]

    def visit_binary(self, binary: BinaryExpr):
        raise NotImplementedError()


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
            value = _AttrEvaluator(self.gsl_to_expr).visit(const.value)
        else:
            raise RuntimeError('Impossible case.')
        return relay.const(value)

    def visit_call(self, call: Call) -> Any:
        args = [self.visit(a) for a in call.args]
        attrs = dict([(name, _AttrEvaluator(self.gsl_to_expr).visit(attr))
                      for name, attr in call.attrs.items()])
        func = op.get_func(call.op)
        return func(*args, **attrs)

    def visit_tuple(self, tp: Tuple) -> Any:
        return relay.Tuple([self.visit(f) for f in tp.fields])

    def visit_getitem(self, getitem: GetItem) -> Any:
        return relay.TupleGetItem(self.visit(getitem.tup), getitem.index)
