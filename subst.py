from inspect import signature, Parameter
from types import FunctionType
from typing import Set

from tvm import relay, ir, transform
from tvm.relay import dataflow_pattern as dfp

import op
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
        _TgtPatChecker(src_checker.wildcard_vars).visit(tgt)

        # Create expression rewriter
        self.rewriter = _ExprRewriter(src, tgt)

    def apply(self, wl: Workload, fold_param: bool = True) -> Workload:
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

    def visit_call(self, call: Call) -> Any:
        # Visit arguments
        super().visit_call(call)

        # Check call node
        func = op.name_to_func(call.op)
        _check_call(func, call)


def _check_call(func: FunctionType, call: Call):
    # Check number of inputs
    num_input = op.num_inputs[func]
    if num_input != len(call.args):
        raise ValueError(
            'Expect {} input tensor(s), got {}.'.format(num_input, len(call.args))
        )

    # Check whether names of specified attributes actually exist
    func_attrib = list(signature(func).parameters.keys())[num_input:]
    for name in call.attrib.keys():
        if not func_attrib.__contains__(name):
            raise AttributeError('Unknown attribute name \'{}\'.'.format(name))


class _TgtPatChecker(NodeVisitor):
    def __init__(self, wildcard_vars: Set[Union[Wildcard, Var]]):
        super().__init__()
        self.wildcard_vars = wildcard_vars

    def visit_wildcard(self, wildcard: Wildcard) -> Any:
        if not self.wildcard_vars.__contains__(wildcard):
            raise ValueError(
                'Target graph contains wildcard node not defined in source graph.'
            )

    def visit_var(self, var: Var) -> Any:
        if not self.wildcard_vars.__contains__(var):
            raise ValueError(
                'Target graph contains variable node not defined in source graph.'
            )

    def visit_call(self, call: Call) -> Any:
        # Visit arguments
        super().visit_call(call)

        # Check call node
        func = op.name_to_func(call.op)
        _check_call(func, call)

        # Check whether all non-default attributes are provided
        num_input = op.num_inputs[func]
        required = set()
        for name, param in list(signature(func).parameters.items())[num_input:]:
            if param.default == Parameter.empty:
                required.add(name)
        required.difference_update(call.attrib.keys())
        if len(required) != 0:
            raise AttributeError('Attributes {} are not provided for \'{}\'.'.format(
                tuple(required), call.op
            ))


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
        # Map GSL pattern nodes to computation graph expressions
        gsl_to_expr = dict([(gsl_node, node_map[dfp_node][0])
                            for gsl_node, dfp_node in self.gsl_to_dfp.items()])

        # Check whether source graph satisfies constraints of source pattern

        return pre


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


@relay.transform.function_pass(opt_level=0)
class _SubstFuncPass:
    def __init__(self, rewriter: _ExprRewriter):
        self.rewriter = rewriter

    def transform_function(self, fn: relay.Function, _mod: ir.IRModule,
                           _ctx: transform.PassContext) -> relay.Function:
        new_body = self.rewriter.rewrite(fn.body)
        return relay.Function(relay.analysis.free_vars(new_body), new_body)

    def __call__(self, mod: ir.IRModule) -> ir.IRModule: ...


class _GraphBuilder(NodeVisitor):
    def __init__(self, gsl_to_expr: Dict[Node, relay.Expr]):
        super().__init__()
        self.gsl_to_expr = gsl_to_expr
