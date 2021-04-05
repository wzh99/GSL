from inspect import signature, Parameter
from typing import Union, List, Set, Optional, Any

from tvm import relay, transform, ir

from . import attr, pat, spec, fold
from .attr import Env, AttrVisitor
from .pat import Pattern, PatternVisitor
from .rewrite import ExprRewriter
from .work import Workload


class Subst:
    """
    Represents a graph substitution rule.
    """

    def __init__(self, src_outs: Union[Pattern, List[Pattern]],
                 tgt_outs: Union[Pattern, List[Pattern]]):
        """
        Constructor.

        :param src_outs: A single source pattern, or a list of source patterns. The patterns should
            not be used as source nor target patterns of other substitutions. If variadic pattern
            is provided, it must be the only pattern.
        :param tgt_outs: A single target pattern, or a list of target patterns. The patterns can
            only be used in source patterns of current substitution. Number of target patterns must
            exactly match source patterns, with i-th target pattern mapped to i-th source pattern.
            If variadic pattern is provided, it must be the only pattern.
        """
        # Convert source and target patterns to lists if necessary
        if isinstance(src_outs, Pattern):
            src_outs = [src_outs]
        if isinstance(tgt_outs, Pattern):
            tgt_outs = [tgt_outs]

        # Check if number of source and target pattern matches
        if len(src_outs) != len(tgt_outs):
            raise ValueError(
                'Numbers of source and target patterns do not match.'
            )

        # Check source and target if variadic pattern is provided
        self.is_var = False
        if any([isinstance(p, pat.Variadic) for p in src_outs]):
            if len(src_outs) != 1:
                raise ValueError(
                    'Variadic must be the only one pattern of source.'
                )
            if len(tgt_outs) != 1 or not isinstance(tgt_outs[0], pat.Variadic):
                raise ValueError(
                    'Variadic must be the only one pattern of target.'
                )
            self.is_var = True

        # Check source patterns
        src_pats: Set[Pattern] = set()
        for i in range(len(src_outs)):
            # Check if output nodes have no successors
            out = src_outs[i]
            if len(out.succ) != 0:
                raise ValueError('Source output node cannot have successors.')

            # Check pattern graph
            src_checker = _SrcPatChecker(src_pats, i)
            src_checker.visit(out, Env())

            # Check if it is connected to the whole subgraph
            cur_visited = set(src_checker.visited.keys())
            shared = src_pats.intersection(cur_visited)
            if len(src_pats) != 0 and len(shared) == 0:
                raise ValueError(
                    'Source pattern {} is not connected to union of previous ones.'.format(i)
                )

            # Update source node set
            src_pats.update(cur_visited)

        # Check connectivity of variadic source
        if self.is_var:
            # noinspection PyTypeChecker
            var: pat.Variadic = src_outs[0]
            non_tpl = src_pats.difference([var], var.templates, var.first)
            if len(non_tpl) == 0:
                raise ValueError(
                    'Variadic source pattern has no common nodes.'
                )

        # Check target patterns
        tgt_checker = _TgtPatChecker(src_pats)
        for tgt in tgt_outs:
            tgt_checker.visit(tgt, Env())

        # Store source and target patterns
        self.src_outs = src_outs
        self.tgt_outs = tgt_outs

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
        rewriter = ExprRewriter(self.src_outs, self.tgt_outs, self.is_var, fast_mode)
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

    def visit(self, p: Pattern, env: Env):
        if (not self.has_visited(p)) and p.is_used:
            raise ValueError(
                'Node in source pattern has been used in other substitutions.'
            )
        super().visit(p, env)
        p.src_idx = self.idx

    def visit_const(self, const: pat.Const, env: Env) -> Any:
        if isinstance(const.value, attr.Attr):
            self.attr_checker.visit(const.value, env)

    def visit_call(self, call: pat.Call, env: Env) -> Any:
        super().visit_call(call, env)

        # Check if all attribute expressions only contain reference to visited nodes
        for a in call.attrs.values():
            self.attr_checker.visit(a, env)

    def visit_getitem(self, getitem: pat.GetItem, env: Env) -> Any:
        super().visit_getitem(getitem, env)
        self.attr_checker.visit(getitem.idx, env)

    def visit_variadic(self, var: pat.Variadic, env: Env) -> Any:
        # Add index to environment
        new_env = env
        if var.index is not None:
            new_env = env + (var.index, True)

        # Check first and template list
        for t in var.first:
            if t is not None:
                if t.check_any(lambda p: p.is_template):
                    raise ValueError(
                        'Pattern as first instance cannot connect to template patterns.'
                    )
                self.visit(t, new_env)
        for t in var.templates:
            self.visit(t, new_env)
        self.visit(var.pat, new_env)

        # Check length
        if var.len is not None:
            self.attr_checker.visit(var.len, env)

    def visit_get_instance(self, get_inst: pat.GetInst, env: Env) -> Any:
        super().visit_get_instance(get_inst, env)
        self.attr_checker.visit(get_inst.idx, env)


class _SrcAttrChecker(AttrVisitor[Env]):
    def __init__(self, pat_checker: _SrcPatChecker):
        self.checker = pat_checker

    def visit_getattr(self, get_attr: attr.GetAttr, env: Env):
        if not self.checker.has_visited(get_attr.pat):
            raise AttributeError(
                'Attribute in source pattern refers to undefined node.'
            )

    def visit_symbol(self, sym: attr.Symbol, env: Env) -> Any:
        if sym not in env:
            raise KeyError('Symbol not found in environment.')

    def visit_variadic(self, var: attr.Variadic, env: Env) -> Any:
        if var.len is not None:
            self.visit(var.len, env)
        new_env = env if var.index is None else env + (var.index, True)
        self.visit(var.attr, new_env)


class _TgtPatChecker(PatternVisitor[Env]):
    def __init__(self, src_nodes: Set[Pattern]):
        super().__init__()
        self.src_nodes = src_nodes
        self.attr_checker = _TgtAttrChecker(self.src_nodes)

    def visit(self, p: Pattern, env: Env):
        if not (p in self.visited or p in self.src_nodes) \
                and p.in_tgt:
            raise ValueError(
                'Node in target pattern has been used in other substitutions.'
            )
        super().visit(p, env)
        p.in_tgt = True

    def visit_wildcard(self, wildcard: pat.Wildcard, env: Env) -> Any:
        if wildcard not in self.src_nodes:
            raise ValueError(
                'Target pattern contains wildcard node not defined in source graph.'
            )

    def visit_var(self, var: pat.Variable, env: Env) -> Any:
        if var not in self.src_nodes:
            raise ValueError(
                'Target pattern contains variable node not defined in source graph.'
            )

    def visit_const(self, const: pat.Const, env: Env) -> Any:
        if const.value is None:
            raise ValueError(
                'Constant node in target pattern must contain a value.'
            )

    def visit_call(self, call: pat.Call, env: Env) -> Any:
        # Visit arguments
        super().visit_call(call, env)

        # Check if all non-default attributes are provided for concrete op
        if isinstance(call.op, pat.ConcreteOp):
            op_name = call.op.name
            api = spec.get_api(op_name)
            num_input = spec.get_num_inputs(op_name)
            required = set()
            for name, param in list(signature(api).parameters.items())[num_input:]:
                if param.default == Parameter.empty:
                    required.add(name)
            required.difference_update(call.attrs.keys())
            if len(required) != 0:
                raise AttributeError(
                    'Required attributes {} are not provided for op \'{}\'.'.format(
                        tuple(required), call.op)
                )

        # Check if all attribute expressions only contain reference to nodes in source graph
        for a in call.attrs.values():
            self.attr_checker.visit(a, env)

    def visit_variadic(self, var: pat.Variadic, env: Env) -> Any:
        # Check length
        if not var.is_output and not var.in_src:
            if var.len is None:
                raise ValueError(
                    'Length is not specified for non-output target pattern.'
                )
            self.attr_checker.visit(var.len, env)

        # Add index to environment and check pattern
        new_env = env
        if var.index is not None:
            new_env = env + (var.index, True)
        self.visit(var.pat, new_env)

        # Check template and first list
        for t in var.templates:
            self.visit(t, new_env)
            if var.has_first(t):
                self.visit(var.tpl_to_fst[t], new_env)

    def visit_get_instance(self, get_inst: pat.GetInst, env: Env) -> Any:
        super().visit_get_instance(get_inst, env)
        self.attr_checker.visit(get_inst.idx, env)


class _TgtAttrChecker(AttrVisitor[Env]):
    def __init__(self, src_nodes: Set[Pattern]):
        self.src_nodes = src_nodes

    def visit_variadic(self, var: attr.Variadic, env: Env) -> Any:
        if var.len is None:
            raise ValueError(
                'Length is not specified for variadic attribute in target pattern.'
            )
        self.visit(var.len, env)
        new_env = env if var.index is None else env + (var.index, True)
        self.visit(var.attr, new_env)


@relay.transform.function_pass(opt_level=0)
class _SubstFuncPass:
    def __init__(self, rewriter: ExprRewriter):
        self.rewriter = rewriter

    def transform_function(self, fn: relay.Function, _mod: ir.IRModule,
                           _ctx: transform.PassContext) -> relay.Function:
        new_body = self.rewriter.rewrite(fn.body)
        return relay.Function(relay.analysis.free_vars(new_body), new_body)

    def __call__(self, mod: ir.IRModule) -> ir.IRModule: ...
