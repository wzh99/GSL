from inspect import signature, Parameter
from typing import Set

from tvm import relay, transform, ir

from . import fold
from .pat import *
from .rewrite import ExprRewriter
from .work import Workload


class Substitution:
    """
    Represents a graph substitution rule.
    """

    def __init__(self, src_pats: Union[Pattern, List[Pattern]],
                 tgt_pats: Union[Pattern, List[Pattern]]):
        """
        Constructor.

        :param src_pats: A single source pattern, or a list of source patterns. The patterns should
            not be used as source nor target patterns of other substitutions.
        :param tgt_pats: A single target pattern, or a list of target patterns. The patterns can
            only be used in source patterns of current substitution. Number of target patterns must
            exactly match source patterns, with i-th target pattern mapped to i-th source pattern.
        """
        # Convert source and target patterns to lists if necessary
        if isinstance(src_pats, Pattern):
            src_pats = [src_pats]
        if isinstance(tgt_pats, Pattern):
            tgt_pats = [tgt_pats]

        # Check if number of source and target pattern matches
        if len(src_pats) != len(tgt_pats):
            raise ValueError(
                'Numbers of source and target patterns do not match.'
            )

        # Check source patterns
        src_nodes: Set[Pattern] = set()
        for i in range(len(src_pats)):
            # Check if output nodes have no successors
            src = src_pats[i]
            if len(src.succ) != 0:
                raise ValueError('Source output node cannot have successors.')

            # Check pattern graph
            src_checker = _SrcPatChecker(src_nodes, i)
            src_checker.visit(src, Env())

            # Check if it is connected to the whole subgraph
            cur_visited = set(src_checker.visited.keys())
            shared = src_nodes.intersection(cur_visited)
            if len(src_nodes) != 0 and len(shared) == 0:
                raise ValueError(
                    'Source pattern {} is not connected to union of previous ones.'.format(i)
                )

            # Update source node set
            src_nodes.update(cur_visited)

        # Check target patterns
        tgt_checker = _TgtPatChecker(src_nodes)
        for tgt in tgt_pats:
            tgt_checker.visit(tgt, Env())

        # Store source and target patterns
        self.src_pats = src_pats
        self.tgt_pats = tgt_pats

    def __call__(self, wl: Workload, fast_mode: bool = False, fold_params: bool = True,
                 new_name: Optional[str] = None) -> Workload:
        """
        Apply substitution to workload.

        :param wl: Workload whose graph is to be altered.
        :param fast_mode: Whether to use fast substitution algorithm for single output pattern.
        In this mode, unmatched successors of interior nodes will not be checked.
        :param fold_params: Whether to pre-compute nodes whose operands are already available in
            parameters.
        :return: New workload after applying the substitution.
        """

        # Keep original name if new name is not provided
        if new_name is None:
            new_name = wl.name

        # Apply substitution to graph
        rewriter = ExprRewriter(self.src_pats, self.tgt_pats, fast_mode)
        mod = _SubstFuncPass(rewriter)(wl.mod)
        new_wl = Workload(mod, wl.params, name=new_name)
        if fold_params:
            new_wl = fold(new_wl)

        return new_wl


class _SrcPatChecker(PatternVisitor[Env]):
    def __init__(self, prev_visited: Set[Pattern], idx: int):
        super().__init__()
        self.prev_visited = prev_visited
        self.idx = idx
        self.attr_checker = _SrcAttrChecker(self)

    def has_visited(self, node: Pattern):
        return node in self.visited or node in self.prev_visited

    def visit(self, node: Pattern, env: Env):
        if (not self.has_visited(node)) and node.is_used:
            raise ValueError(
                'Node in source pattern has been used in other substitutions.'
            )
        super().visit(node, env)
        node.src_idx = self.idx

    def visit_const(self, const: Const, env: Env) -> Any:
        if isinstance(const.value, Attr):
            self.attr_checker.visit(const.value, env)

    def visit_call(self, call: Call, env: Env) -> Any:
        super().visit_call(call, env)

        # Check if all attribute expressions only contain reference to visited nodes
        for a in call.attrs.values():
            self.attr_checker.visit(a, env)


class _SrcAttrChecker(AttrVisitor[Env]):
    def __init__(self, pat_checker: _SrcPatChecker):
        self.checker = pat_checker

    def visit_get_node(self, get_node: GetNodeAttr, env: Env):
        if not self.checker.has_visited(get_node.node):
            raise AttributeError(
                'Attribute in source pattern refers to undefined node.'
            )


class _TgtPatChecker(PatternVisitor[Env]):
    def __init__(self, src_nodes: Set[Pattern]):
        super().__init__()
        self.src_nodes = src_nodes
        self.attr_checker = _TgtAttrChecker(self.src_nodes)

    def visit(self, node: Pattern, env: Env):
        if not (node in self.visited or node in self.src_nodes) \
                and node.in_tgt:
            raise ValueError(
                'Node in target pattern has been used in other substitutions.'
            )
        super().visit(node, env)
        node.in_tgt = True

    def visit_wildcard(self, wildcard: Wildcard, env: Env) -> Any:
        if wildcard not in self.src_nodes:
            raise ValueError(
                'Target pattern contains wildcard node not defined in source graph.'
            )

    def visit_var(self, var: Var, env: Env) -> Any:
        if var not in self.src_nodes:
            raise ValueError(
                'Target pattern contains variable node not defined in source graph.'
            )

    def visit_const(self, const: Const, env: Env) -> Any:
        if const.value is None:
            raise ValueError(
                'Constant node in target pattern must contain a value.'
            )

    def visit_call(self, call: Call, env: Env) -> Any:
        # Visit arguments
        super().visit_call(call, env)

        # Check if all non-default attributes are provided for concrete op
        if isinstance(call.op, ConcreteOp):
            op_name = call.op.name
            func = spec.get_func(op_name)
            num_input = spec.get_num_inputs(op_name)
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
            self.attr_checker.visit(a, env)


class _TgtAttrChecker(AttrVisitor[Env]):
    def __init__(self, src_nodes: Set[Pattern]):
        self.src_nodes = src_nodes

    def visit_get_node(self, get_node: GetNodeAttr, env: Env):
        if get_node.node not in self.src_nodes:
            raise AttributeError(
                'Attribute in target pattern refers to node not defined in source pattern.'
            )


@relay.transform.function_pass(opt_level=0)
class _SubstFuncPass:
    def __init__(self, rewriter: ExprRewriter):
        self.rewriter = rewriter

    def transform_function(self, fn: relay.Function, _mod: ir.IRModule,
                           _ctx: transform.PassContext) -> relay.Function:
        new_body = self.rewriter.rewrite(fn.body)
        return relay.Function(relay.analysis.free_vars(new_body), new_body)

    def __call__(self, mod: ir.IRModule) -> ir.IRModule: ...
