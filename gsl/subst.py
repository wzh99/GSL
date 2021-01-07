from collections import deque
from inspect import signature, Parameter
from typing import Optional, Set

from tvm import ir, transform

from . import util
from .fold import ParamFoldPass
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
                'Number of source and target patterns do not match.'
            )

        # Check source and target
        src_visited: Dict[Node, Any] = dict()
        src_limit: Set[Union[Wildcard, Var]] = set()
        for src in src_pats:
            _SrcPatChecker(src_visited, src_limit).visit(src)
        for tgt in tgt_pats:
            _TgtPatChecker(src_visited, src_limit).visit(tgt)

        # Create expression rewriter
        self.rewriter = _ExprRewriter(src_pats, tgt_pats)

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
    def __init__(self, src_nodes: Dict[Node, Any], src_limit: Set[Union[Wildcard, Var]]):
        super().__init__()
        # Target pattern cannot contain new wildcard or variable nodes
        self.visited = src_nodes
        self.src_limit = src_limit

    def visit_wildcard(self, wildcard: Wildcard) -> Any:
        self.src_limit.add(wildcard)

    def visit_var(self, var: Var) -> Any:
        self.src_limit.add(var)

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
    def __init__(self, src_nodes: Dict[Node, Any], src_limits: Set[Union[Wildcard, Var]]):
        super().__init__()
        self.src_nodes = src_nodes
        self.src_limits = src_limits

    def visit_wildcard(self, wildcard: Wildcard) -> Any:
        if not self.src_limits.__contains__(wildcard):
            raise ValueError(
                'Target pattern contains wildcard node not defined in source graph.'
            )

    def visit_var(self, var: Var) -> Any:
        if not self.src_limits.__contains__(var):
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
    def __init__(self, src_nodes: Dict[Node, Any]):
        self.src_nodes = src_nodes

    def visit_get_attr(self, get_attr: GetAttr):
        if not self.src_nodes.__contains__(get_attr.node):
            raise AttributeError(
                'Attribute in target pattern refers to node not defined in source graph.'
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


class _ExprRewriter:
    def __init__(self, src_pats: List[Node], tgt_pats: List[Node]):
        self.src_pats = src_pats
        self.tgt_pats = tgt_pats
        self.history = set()

    def rewrite(self, expr: relay.Expr) -> relay.Expr:
        # Clear match history
        self.history.clear()

        # Greedily match all subgraphs
        while True:
            # Find all outgoing edges of call, tuple and getitem in expression
            succ_visitor = _SuccVisitor()
            succ_visitor.visit(expr)
            out_edges = succ_visitor.succ_map

            # Find matched expressions with patterns
            pat_to_expr: Dict[Node, relay.Expr] = dict()
            src_matched = []  # expression matched in this round
            for src_pat in self.src_pats:
                src_expr = self._find_match(src_pat, expr, pat_to_expr, src_matched)
                if src_expr is None:
                    return expr  # even one match is not found, exit immediately
                else:
                    src_matched.append(src_expr)

            # Add source patterns to match history
            self.history.update(src_matched)

            # Check whether this subgraph has outgoing edges not described by source patterns
            if not self._check_succ(self.src_pats, pat_to_expr, out_edges):
                continue

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
            cur_expr = elem.expr

            if elem.count == 0:
                # Add children to stack if this expression is visited for the first time
                stack.append(StackElem(cur_expr, 1))
                if isinstance(elem.expr, relay.Call):
                    for a in reversed(cur_expr.args):
                        update(a)
                elif isinstance(elem.expr, relay.Tuple):
                    for f in reversed(cur_expr.fields):
                        update(f)
                elif isinstance(elem.expr, relay.TupleGetItem):
                    update(cur_expr.tuple_value)
            else:
                # Do not match this expression if it has been visited this round or is in match
                # history
                if src_matched.__contains__(cur_expr) or \
                        self.history.__contains__(cur_expr):
                    continue  # matched expression cannot be matched again
                matcher = _ExprMatcher(pat_to_expr.copy())
                res = matcher.match(pat, cur_expr)
                if res:
                    pat_to_expr.update(matcher.pat_to_expr)  # update map if matches
                    return cur_expr
                else:
                    continue

        return None

    ignore_out_class = (Wildcard, Var, Const)

    @classmethod
    def _check_succ(cls, src_pats: List[Node], pat_to_expr: Dict[Node, relay.Expr],
                    expr_succ: Dict[relay.Expr, List[relay.Expr]]) -> bool:
        # Create set of matched expressions
        matched_expr = set(pat_to_expr.values())

        # Initialize queue for pattern nodes
        queue = deque(src_pats)
        visited: Set[Node] = set(src_pats)

        def update(n: Node):
            if not visited.__contains__(n):
                queue.append(n)
                visited.add(n)

        # Traverse pattern graph
        while len(queue) > 0:
            # Pop a node and add its predecessors to queue
            node = queue.popleft()
            for p in node.pred:
                update(p)

            # Skip if the node is output
            if len(node.succ) == 0:
                continue

            # Skip if the node is a variable, a constant or wildcard
            # These three kinds of nodes will still be available after substitution
            if isinstance(node, cls.ignore_out_class):
                continue

            # Check if matched expression has outgoing edges not described by pattern
            for o in expr_succ[pat_to_expr[node]]:
                if not matched_expr.__contains__(o):
                    return False

        return True


class _SuccVisitor(relay.ExprVisitor):
    def __init__(self):
        super().__init__()
        self.succ_map: Dict[relay.Expr, List[relay.Expr]] = dict()

    def visit_call(self, call: relay.Call):
        super().visit_call(call)
        for a in call.args:
            self._add_edge(a, call)

    def visit_tuple(self, tup: relay.Tuple):
        super().visit_tuple(tup)
        for f in tup.fields:
            self._add_edge(f, tup)

    def visit_tuple_getitem(self, t: relay.TupleGetItem):
        super().visit_tuple_getitem(t)
        self._add_edge(t.tuple_value, t)

    def _add_edge(self, pred: relay.Expr, succ: relay.Expr):
        if isinstance(pred, _ExprRewriter.ignore_out_class):
            return
        if self.succ_map.__contains__(pred):
            self.succ_map[pred].append(succ)
        else:
            self.succ_map[pred] = [succ]


class _RelayBuilder(NodeVisitor):
    def __init__(self, pat_to_expr: Dict[Node, relay.Expr]):
        super().__init__()
        self.pat_to_expr = pat_to_expr

    def visit(self, node: Node):
        if self.pat_to_expr.__contains__(node):
            return self.pat_to_expr[node]
        else:
            return super().visit(node)

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
        elif self.memo_map.__contains__(expr):
            return self.memo_map[expr]
        else:
            ret = super().visit(expr)
            if ret != expr:
                self.memo_map[expr] = ret
                return ret
            else:
                return expr

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
            if not self._attr_eq(attr, expr.attrs[name]):
                return False

        # Match arguments
        for pat_arg, expr_arg in zip(call.args, expr.args):
            if not self.match(pat_arg, expr_arg):
                return False

        return True

    def _attr_eq(self, pat_attr: AttrExpr, expr_attr) -> bool:
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
