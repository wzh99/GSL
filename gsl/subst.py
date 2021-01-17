import typing as ty
from inspect import signature, Parameter
from typing import Optional, Set

from tvm import transform

from . import util, fold
from .graph import *
from .work import Workload


class Substitution:
    """
    Represents a graph substitution rule.
    """

    def __init__(self, src_pats: Union[Node, List[Node]], tgt_pats: Union[Node, List[Node]]):
        """
        Constructor.

        :param src_pats: A single source pattern, or a list of source patterns.
        :param tgt_pats: A single target pattern, or a list of target patterns. Order of patterns
            in target pattern list must strictly follow the one in source pattern list.
        """
        # Convert source and target patterns to lists if necessary
        if isinstance(src_pats, Node):
            src_pats = [src_pats]
        if isinstance(tgt_pats, Node):
            tgt_pats = [tgt_pats]

        # Check if number of source and target pattern matches
        if len(src_pats) != len(tgt_pats):
            raise ValueError(
                'Numbers of source and target patterns do not match.'
            )

        # Check source patterns
        src_nodes: Set[Node] = set()
        for i in range(len(src_pats)):
            # Check if output nodes have no successors
            src = src_pats[i]
            if len(src.succ) != 0:
                raise ValueError('Source output node cannot have successors.')

            # Check pattern graph
            src_checker = _SrcPatChecker(src_nodes, i)
            src_checker.visit(src)

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
            tgt_checker.visit(tgt)

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
        rewriter = _ExprRewriter(self.src_pats, self.tgt_pats, fast_mode)
        mod = _SubstFuncPass(rewriter)(wl.mod)
        new_wl = Workload(mod, wl.params, name=new_name)
        if fold_params:
            new_wl = fold.fold(new_wl)

        return new_wl


class _SrcPatChecker(NodeVisitor):
    def __init__(self, prev_visited: Set[Node], idx: int):
        super().__init__()
        self.prev_visited = prev_visited
        self.idx = idx
        self.attr_checker = _SrcAttrChecker(self)

    def has_visited(self, node: Node):
        return self.visited.__contains__(node) or self.prev_visited.__contains__(node)

    def visit(self, node: Node):
        if (not self.has_visited(node)) and node.is_used:
            raise ValueError(
                'Node in source pattern has been used in other substitutions.'
            )
        super().visit(node)
        node.src_idx = self.idx

    def visit_const(self, const: Const) -> Any:
        if isinstance(const.value, AttrExpr):
            raise TypeError(
                'Constant node in source graph cannot store an attribute expression.'
            )

    def visit_call(self, call: Call) -> Any:
        super().visit_call(call)

        # Check if all attribute expressions only contain reference to visited nodes
        for a in call.attrs.values():
            self.attr_checker.visit(a)


class _SrcAttrChecker(AttrVisitor):
    def __init__(self, pat_checker: _SrcPatChecker):
        self.checker = pat_checker

    def visit_get_attr(self, get_attr: GetAttr):
        if not self.checker.has_visited(get_attr.node):
            raise AttributeError(
                'Attribute in source pattern refers to undefined node.'
            )


class _TgtPatChecker(NodeVisitor):
    def __init__(self, src_nodes: Set[Node]):
        super().__init__()
        self.src_nodes = src_nodes
        self.attr_checker = _TgtAttrChecker(self.src_nodes)

    def visit(self, node: Node):
        if not (self.visited.__contains__(node) or self.src_nodes.__contains__(node)) \
                and node.in_tgt:
            raise ValueError(
                'Node in target pattern has been used in other substitutions.'
            )
        super().visit(node)
        node.in_tgt = True

    def visit_wildcard(self, wildcard: Wildcard) -> Any:
        if not self.src_nodes.__contains__(wildcard):
            raise ValueError(
                'Target pattern contains wildcard node not defined in source graph.'
            )

    def visit_var(self, var: Var) -> Any:
        if not self.src_nodes.__contains__(var):
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
        func = spec.get_func(call.op)
        num_input = spec.num_inputs[func]
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
            self.attr_checker.visit(a)


class _TgtAttrChecker(AttrVisitor):
    def __init__(self, src_nodes: Set[Node]):
        self.src_nodes = src_nodes

    def visit_get_attr(self, get_attr: GetAttr):
        if not self.src_nodes.__contains__(get_attr.node):
            raise AttributeError(
                'Attribute in target pattern refers to node not defined in source pattern.'
            )


@relay.transform.function_pass(opt_level=0)
class _SubstFuncPass:
    def __init__(self, rewriter):
        self.rewriter: _ExprRewriter = rewriter

    def transform_function(self, fn: relay.Function, _mod: ir.IRModule,
                           _ctx: transform.PassContext) -> relay.Function:
        new_body = self.rewriter.rewrite(fn.body)
        return relay.Function(relay.analysis.free_vars(new_body), new_body)

    def __call__(self, mod: ir.IRModule) -> ir.IRModule: ...


SuccListDict = Dict[relay.Expr, List[relay.Expr]]


class _ExprRewriter:
    def __init__(self, src_pats: List[Node], tgt_pats: List[Node], fast_mode: bool):
        self.src_pats = src_pats
        self.tgt_pats = tgt_pats
        self.fast_mode = fast_mode
        self.history = set()

    def rewrite(self, expr: relay.Expr) -> relay.Expr:
        if self.fast_mode and len(self.src_pats) == 1:
            return _SinglePatRewriter(self).visit(expr)

        # Greedily match all subgraphs
        while True:
            # Build successor list for all expression nodes
            succ_list = self.build_succ_list(expr)

            # Find matched expressions with patterns
            pat_to_expr: Dict[Node, relay.Expr] = dict()
            src_matched = self.find_match(expr, pat_to_expr, succ_list)
            if len(src_matched) == 0:  # even a single match is not found
                return expr

            # Add source patterns to match history
            self.history.update(src_matched)

            # Check whether this subgraph has outgoing edges not described by source patterns
            if not self.check_succ(pat_to_expr, succ_list):
                continue

            # Generate target expressions and map source to them
            tgt_expr = [_RelayBuilder(pat_to_expr).visit(tgt) for tgt in self.tgt_pats]
            expr_map = dict(zip(src_matched, tgt_expr))

            # Rewrite expression
            expr = _RewriteMutator(expr_map).visit(expr)

    def find_match(self, expr: relay.Expr, pat_to_expr: Dict[Node, relay.Expr],
                   succ_list: SuccListDict) -> List[relay.Expr]:
        # Traverse the expression graph
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
            cur_expr = elem.expr

            if elem.count == 0:
                # Add children to stack if this expression is visited for the first time
                stack.append(StackElem(cur_expr, 1))
                for p in reversed(util.get_expr_pred(cur_expr)):
                    update(p)
            else:
                # Do not match this expression if it has been visited this round or is in match
                # history
                if self.history.__contains__(cur_expr):
                    continue

                # Use first pattern to roughly locate the subgraph
                matcher = _ExprMatcher(pat_to_expr.copy())
                res = matcher.match(self.src_pats[0], cur_expr)
                if not res:
                    continue  # even first pattern is not matched, skip this expression
                if len(self.src_pats) == 1:
                    pat_to_expr.update(matcher.pat_to_expr)
                    return [cur_expr]  # work is done for single pattern

                # Match rest of the patterns
                tmp_pat_to_expr = matcher.pat_to_expr
                src_matched = self.match_rest(tmp_pat_to_expr, cur_expr, succ_list)
                if len(src_matched) == 0:
                    continue  # the rest patterns do not match, skip this expression
                pat_to_expr.update(tmp_pat_to_expr)
                return src_matched

        return []

    def match_rest(self, pat_to_expr: Dict[Node, relay.Expr], fst_matched: relay.Expr,
                   succ_map: Dict[relay.Expr, List[relay.Expr]]) -> List[relay.Expr]:
        output_matched = [fst_matched]
        for src_idx in range(len(self.src_pats)):
            # The first pattern is already matched
            if src_idx == 0:
                continue

            # Get output pattern node
            src_pat = self.src_pats[src_idx]

            # Collect matched expression nodes
            expr_matched = set(pat_to_expr.values())

            # Find candidate node-expression pair where the node is connected to matched ones.
            # Since we required the i-th pattern is connected to the union of 0..i-th patterns,
            # there must exist some node that satisfies the condition.
            stack: List[ty.Tuple[Node, relay.Expr]] = []

            def add_succ(pat: Node, expr: relay.Expr):
                if succ_map[expr] is None:
                    return
                for ps in pat.succ:
                    if pat_to_expr.__contains__(ps) or ps.src_idx != src_idx:
                        continue
                    for es in succ_map[expr]:
                        if expr_matched.__contains__(es) or self.history.__contains__(es):
                            continue  # matched expression cannot be matched again
                        stack.append((ps, es))
                        if not isinstance(pat, (Wildcard, Var)):
                            # for non-inputs, only the first successor expression node could match
                            break
                    break  # only need to visit first unmatched successor pattern node

            for p, e in pat_to_expr.items():
                add_succ(p, e)

            # Backtrack until the source pattern is reached
            found = False
            while len(stack) > 0:
                # Pick one pair from queue
                p, e = stack.pop()

                # Add successors to queue if output pattern is not reached
                if p != src_pat:
                    add_succ(p, e)
                    continue

                # Match pattern with expression
                matcher = _ExprMatcher(pat_to_expr.copy())
                res = matcher.match(src_pat, e)
                if not res:
                    continue

                # Update matched nodes and expressions
                pat_to_expr.update(matcher.pat_to_expr)
                output_matched.append(e)
                found = True
                break

            # No match for this pattern, the whole match failed
            if not found:
                return []

        return output_matched

    @staticmethod
    def build_succ_list(expr: relay.Expr) -> SuccListDict:
        succ_visitor = _SuccListBuilder()
        succ_visitor.visit(expr)
        return succ_visitor.succ_list

    @staticmethod
    def check_succ(pat_to_expr: Dict[Node, relay.Expr], succ_list: SuccListDict) -> bool:
        for p, e in pat_to_expr.items():
            # Skip wildcard and variables because they serve as input nodes
            if isinstance(p, (Wildcard, Var)):
                continue

            # Skip output nodes because they can be always be used by their successors
            if p.is_output:
                continue

            # From our matching algorithm, unmatched successor expression nodes could only be last
            # several elements of the expression's successor list. We just need to ensure the
            # pattern and expression node has same number of successors.
            if len(p.succ) != len(succ_list[e]):
                return False

        return True


class _SuccListBuilder(relay.ExprVisitor):
    def __init__(self):
        super().__init__()
        self.succ_list: Dict[relay.Expr, List[relay.Expr]] = dict()

    def visit_call(self, call: relay.Call):
        super().visit_call(call)
        for a in call.args:
            self._add_succ(a, call)

    def visit_tuple(self, tup: relay.Tuple):
        super().visit_tuple(tup)
        for f in tup.fields:
            self._add_succ(f, tup)

    def visit_tuple_getitem(self, t: relay.TupleGetItem):
        super().visit_tuple_getitem(t)
        self._add_succ(t.tuple_value, t)

    def _add_succ(self, pred: relay.Expr, succ: relay.Expr):
        if self.succ_list.__contains__(pred):
            self.succ_list[pred].append(succ)
        else:
            self.succ_list[pred] = [succ]


class _RelayBuilder(NodeVisitor):
    def __init__(self, pat_to_expr: Dict[Node, relay.Expr]):
        super().__init__()
        self.pat_to_expr = pat_to_expr

    def visit(self, node: Node):
        if self.pat_to_expr.__contains__(node):
            return self.pat_to_expr[node]
        else:
            expr = super().visit(node)
            self.pat_to_expr[node] = expr
            return expr

    def visit_const(self, const: Const) -> Any:
        if isinstance(const.value, np.ndarray):
            value = const.value
        elif isinstance(const.value, AttrExpr):
            value = AttrEvaluator(self.pat_to_expr).visit(const.value)
        else:
            raise RuntimeError('Impossible case.')
        return relay.const(value)

    def visit_call(self, call: Call) -> Any:
        args = [self.visit(a) for a in call.args]
        attrs = dict([(name, AttrEvaluator(self.pat_to_expr).visit(attr))
                      for name, attr in call.attrs.items()])
        func = spec.get_func(call.op)
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
        elif self.memo_map.__contains__(expr):
            return self.memo_map[expr]
        else:
            ret = super().visit(expr)
            self.memo_map[expr] = ret
            return ret

    def visit_call(self, call: relay.Call):
        new_args, changed = self._visit_args(call.args)
        if changed:
            return relay.Call(call.op, new_args, call.attrs, call.type_args, call.span)
        else:
            return call

    def visit_tuple(self, tup: relay.Tuple):
        new_fields, changed = self._visit_args(tup.fields)
        if changed:
            return relay.Tuple(new_fields, span=tup.span)
        else:
            return tup

    def visit_tuple_getitem(self, getitem: relay.TupleGetItem):
        new_tup, changed = self._visit_args([getitem.tuple_value])
        if changed:
            return relay.TupleGetItem(new_tup[0], getitem.index)
        else:
            return getitem

    def _visit_args(self, args: List[relay.Expr]):
        changed = False
        new_args = []
        for a in args:
            ret = self.visit(a)
            if ret != a:
                new_args.append(ret)
                changed = True
            else:
                new_args.append(a)
        return new_args, changed


class _SinglePatRewriter(relay.ExprMutator):
    def __init__(self, rewriter: _ExprRewriter):
        super().__init__()
        self.src_pat = rewriter.src_pats[0]
        self.tgt_pat = rewriter.tgt_pats[0]

    def visit(self, expr: relay.Expr):
        # Directly return if it has been visited before
        if self.memo_map.__contains__(expr):
            return self.memo_map[expr]

        # Rewrite predecessors
        new_expr = super().visit(expr)
        self.memo_map[expr] = new_expr

        # Match pattern with this expression
        pat_to_expr: Dict[Node, relay.Expr] = {}
        matcher = _ExprMatcher(pat_to_expr)
        if not matcher.match(self.src_pat, new_expr):
            return new_expr

        # Build new expression
        new_expr = _RelayBuilder(pat_to_expr).visit(self.tgt_pat)
        self.memo_map[expr] = new_expr
        return new_expr


class _ExprMatcher:
    def __init__(self, pat_to_expr: Dict[Node, relay.Expr]):
        self.pat_to_expr = pat_to_expr
        self.expr_matched = set(pat_to_expr.values())

    def match(self, pat: Node, expr: relay.Expr) -> bool:
        # Already matched, use history record
        if self.pat_to_expr.__contains__(pat):
            return self.pat_to_expr[pat] == expr

        # Reject if the expression has been matched with another node
        if self.expr_matched.__contains__(expr):
            return False

        # Try matching according to pattern node type
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

        # Add to record if matched
        if res:
            self.pat_to_expr[pat] = expr
            self.expr_matched.add(expr)
        return res

    def match_var(self, var: Var, expr: relay.Expr) -> bool:
        # Match variable node
        if not isinstance(expr, relay.Var):
            return False

        # Match attributes
        for name, attr in var.attrs.items():
            if not self._match_attr(attr, AttrEvaluator.get_expr_attr(expr, name)):
                return False

        return True

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

        # Match arguments
        # Arguments must be matched before attributes, because attribute matching may depend on
        # match result of arguments.
        for pat_arg, expr_arg in zip(call.args, expr.args):
            if not self.match(pat_arg, expr_arg):
                return False

        # Match attributes
        for name, attr in call.attrs.items():
            if not self._match_attr(attr, expr.attrs[name]):
                return False

        return True

    def _match_attr(self, pat_attr: AttrExpr, expr_attr) -> bool:
        pat_val = AttrEvaluator(self.pat_to_expr).visit(pat_attr)
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
