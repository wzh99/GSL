from typing import Set

from tvm import ir, relay

from . import util
from .pat import *

PatExprMap = Dict[Pattern, relay.Expr]
SuccListMap = Dict[relay.Expr, List[relay.Expr]]


class ExprRewriter:
    def __init__(self, src_pats: List[Pattern], tgt_pats: List[Pattern], fast_mode: bool):
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
            pat_to_expr: PatExprMap = dict()
            fst_matched = self.match_one(self.src_pats[0], expr, pat_to_expr)
            if fst_matched is None:
                return expr  # even the first output pattern node is not matched
            src_matched = self.match_rest(pat_to_expr, fst_matched, succ_list)
            if len(src_matched) == 0:  # the rest output nodes fail to match
                return expr

            # Add source patterns to match history
            self.history.update(src_matched)

            # Check whether this subgraph has outgoing edges not described by source patterns
            if not self.check_succ(pat_to_expr, succ_list):
                continue

            # Generate target expressions and map source to them
            tgt_expr = [_RelayBuilder(pat_to_expr).visit(tgt, Env()) for tgt in self.tgt_pats]
            expr_map = dict(zip(src_matched, tgt_expr))

            # Rewrite expression
            expr = _RewriteMutator(expr_map).visit(expr)

    def match_one(self, pat: Pattern, expr: relay.Expr, pat_to_expr: PatExprMap) \
            -> Optional[relay.Expr]:
        # Traverse the expression graph
        class StackElem:
            def __init__(self, e: relay.Expr, count: int):
                self.expr = e
                self.count = count

        stack: List[StackElem] = [StackElem(expr, 0)]
        visited: Set[relay.Expr] = {expr}

        def update(e: relay.Expr):
            if e not in visited:
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
                if cur_expr in self.history:
                    continue

                # Use first pattern to roughly locate the subgraph
                matcher = _ExprMatcher(pat_to_expr.copy())
                res = matcher.match(pat, cur_expr, Env())
                if not res:
                    continue  # even first pattern is not matched, skip this expression
                pat_to_expr.update(matcher.pat_to_expr)
                return cur_expr

        return None

    reusable_node = (Wildcard, Var)

    def match_rest(self, pat_to_expr: PatExprMap, fst_matched: relay.Expr,
                   succ_list: SuccListMap) -> List[relay.Expr]:
        output_matched = [fst_matched]
        for src_idx in range(1, len(self.src_pats)):
            # Get output pattern node
            src_pat = self.src_pats[src_idx]

            # Collect matched expression nodes
            expr_matched = set(pat_to_expr.values())

            # Find candidate node-expression pair where the node is connected to matched ones.
            # Since we required the i-th pattern is connected to the union of 0..i-th patterns,
            # there must exist some node that satisfies the condition.
            stack: List[Tuple[Pattern, relay.Expr]] = []

            def add_succ(pat: Pattern, expr: relay.Expr):
                if expr not in succ_list:
                    return
                for ps in pat.succ:
                    if ps in pat_to_expr:
                        continue  # successor pattern already matched
                    if ps.src_idx != src_idx:
                        continue  # not source or not the source pattern concerned
                    for es in succ_list[expr]:
                        if es in expr_matched or es in self.history:
                            continue  # matched expression cannot be matched again
                        stack.append((ps, es))
                        if not isinstance(pat, self.reusable_node):
                            # for non-reusable nodes, only the first successor expression node
                            # could match
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
                res = matcher.match(src_pat, e, Env())
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
    def build_succ_list(expr: relay.Expr) -> SuccListMap:
        succ_visitor = _SuccListBuilder()
        succ_visitor.visit(expr)
        return succ_visitor.succ_list

    @classmethod
    def check_succ(cls, pat_to_expr: PatExprMap, succ_list: SuccListMap) -> bool:
        for p, e in pat_to_expr.items():
            # Skip variables, wildcards and output nodes because they can always be reused
            if isinstance(p, cls.reusable_node) or p.is_output:
                continue

            # From our matching algorithm, unmatched successor expression nodes could only be last
            # several elements of the expression's successor list. We just need to ensure the
            # pattern node and expression node has same number of successors.
            if len(p.src_succ) != len(succ_list[e]):
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
        if pred in self.succ_list:
            self.succ_list[pred].append(succ)
        else:
            self.succ_list[pred] = [succ]


class _RelayBuilder(PatternVisitor[Env]):
    def __init__(self, pat_to_expr: PatExprMap):
        super().__init__()
        self.pat_to_expr = pat_to_expr

    def visit(self, pat: Pattern, env: Env):
        if pat in self.pat_to_expr:
            return self.pat_to_expr[pat]
        else:
            expr = super().visit(pat, env)
            self.pat_to_expr[pat] = expr
            return expr

    def visit_const(self, const: Const, env: Env) -> Any:
        if isinstance(const.value, np.ndarray):
            value = const.value
        elif isinstance(const.value, Attr):
            value = _AttrEvaluator(self.pat_to_expr).visit(const.value, env)
        else:
            raise RuntimeError('Unreachable.')
        return relay.const(value)

    def visit_call(self, call: Call, env: Env) -> Any:
        args = [self.visit(a, env) for a in call.args]
        attrs = dict([(name, _AttrEvaluator(self.pat_to_expr).visit(attr, env))
                      for name, attr in call.attrs.items()])
        op_name = self.visit_op(call.op, env)
        func = spec.get_func(op_name)
        try:
            call_expr = func(*args, **attrs)
        except TypeError:
            raise RuntimeError(
                'Cannot create call expression for op \'{}\' with {} operand(s) and attribute '
                'set {}.'.format(op_name, len(args), attrs))
        return call_expr

    def visit_op(self, op: Op, env: Env) -> Any:
        if isinstance(op, ConcreteOp):
            return op.name
        elif isinstance(op, OpWithFlag):
            return self.pat_to_expr[op].name
        else:
            raise RuntimeError('Unreachable.')

    def visit_tuple(self, tup: Tup, env: Env) -> Any:
        return relay.Tuple([self.visit(f, env) for f in tup.fields])

    def visit_getitem(self, getitem: GetItem, env: Env) -> Any:
        tup = self.visit(getitem.tup, env)
        idx = _AttrEvaluator(self.pat_to_expr).visit(getitem.index, env)
        return tup[idx]


class _RewriteMutator(relay.ExprMutator):
    def __init__(self, expr_map: Dict[relay.Expr, relay.Expr]):
        super().__init__()
        self.expr_map = expr_map

    def visit(self, expr: relay.Expr):
        if expr in self.expr_map:
            return self.expr_map[expr]
        elif expr in self.memo_map:
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
    def __init__(self, rewriter: ExprRewriter):
        super().__init__()
        self.src_pat = rewriter.src_pats[0]
        self.tgt_pat = rewriter.tgt_pats[0]

    def visit(self, expr: relay.Expr):
        # Directly return if it has been visited before
        if expr in self.memo_map:
            return self.memo_map[expr]

        # Rewrite predecessors
        new_expr = super().visit(expr)
        self.memo_map[expr] = new_expr

        # Match pattern with this expression
        pat_to_expr: PatExprMap = {}
        matcher = _ExprMatcher(pat_to_expr)
        if not matcher.match(self.src_pat, new_expr, Env()):
            return new_expr

        # Build new expression
        new_expr = _RelayBuilder(pat_to_expr).visit(self.tgt_pat, Env())
        self.memo_map[expr] = new_expr
        return new_expr


class _ExprMatcher:
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
            if not self._match_attr(attr, _AttrEvaluator.get_expr_attr(expr, name), env):
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
            pat_val = _AttrEvaluator(self.pat_to_expr).visit(const.value, env)
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
        pat_val = _AttrEvaluator(self.pat_to_expr).visit(pat_attr, env)
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
        idx = _AttrEvaluator(self.pat_to_expr).visit(getitem.index, env)
        return idx == expr.index

    def match_variadic(self, var: Variadic, expr: relay.Expr, env: Env) -> bool:
        pass

    def match_get_inst(self, get_inst: GetInstance, expr: relay.Expr, env: Env) -> bool:
        pass


class _AttrEvaluator(AttrVisitor[Env]):
    def __init__(self, pat_to_expr: PatExprMap):
        self.pat_to_expr = pat_to_expr

    @staticmethod
    def get_expr_attr(expr: relay.Expr, name: str):
        expr_ty = expr.checked_type
        if not isinstance(expr_ty, ir.TensorType):
            raise ValueError(
                'Cannot get attribute from an expression not of tensor type.'
            )
        if name == 'shape':
            return expr_ty.concrete_shape
        elif name == 'dtype':
            return expr_ty.dtype
        else:
            raise RuntimeError('Unreachable.')

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
        expr = self.pat_to_expr[pat]

        # Access attribute according to type of node
        if name in Pattern.shared_attrs:
            return self.get_expr_attr(expr, name)
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

    def visit_symbol(self, sym: Symbol, env: Env) -> Any:
        val = env[sym]
        if val is None:
            raise RuntimeError('Symbol \'{}\' not found in environment.'.format(sym))
        return val


def eval_get_inst(get_inst: GetInstance, pat_to_expr: PatExprMap, env: Env) -> Pattern:
    idx = _AttrEvaluator(pat_to_expr).visit(get_inst.index, env)
    return get_inst.var(idx, get_inst.t)
