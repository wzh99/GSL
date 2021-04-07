from typing import Set, List, Dict, Tuple, Optional

import numpy as np
from tvm import relay

from . import pat, spec, util
from .attr import Attr, Env
from .eval import PatExprMap, ExprTypeMap, AttrEvaluator, eval_get_inst
from .match import Matcher
from .pat import Pattern, PatternVisitor

SuccListMap = Dict[relay.Expr, List[relay.Expr]]


class ExprRewriter:
    def __init__(self, src_outs: List[Pattern], tgt_outs: List[Pattern], is_var: bool,
                 fast_mode: bool):
        self.src_outs = src_outs
        self.tgt_outs = tgt_outs
        self.is_var = is_var
        self.fast_mode = fast_mode
        self.ty_map: ExprTypeMap = {}
        self.history = set()

    def rewrite(self, expr: relay.Expr) -> relay.Expr:
        # Build expression type map
        mapper = _TypeMapper()
        mapper.visit(expr)
        self.ty_map = mapper.ty_map

        # Use fast mode if possible
        if self.fast_mode and len(self.src_outs) == 1 and not self.is_var:
            return _SinglePatRewriter(self.src_outs[0], self.tgt_outs[0], self.ty_map).visit(expr)

        # Greedily match all subgraphs
        while True:
            # Build successor list for all expression nodes
            succ_list = self.build_succ_list(expr)

            # Find matched expressions with patterns
            pat_to_expr: PatExprMap = dict()
            if self.is_var:  # match variadic with different procedure
                # noinspection PyTypeChecker
                src_var: pat.Variadic = self.src_outs[0]
                src_matched = self.match_variadic(src_var, expr, pat_to_expr, succ_list)
                if len(src_matched) == 0:
                    break
                elif src_var.min_len is not None and len(src_matched) < src_var.min_len:
                    self.history.update(src_matched)
                    self.clear_pat()
                    continue

            else:
                fst_matched = self.match_one(self.src_outs[0], expr, pat_to_expr)
                if fst_matched is None:
                    break  # even the first output pattern node is not matched
                src_matched = self.match_rest(pat_to_expr, fst_matched, succ_list)
                if len(src_matched) == 0:  # the rest output nodes fail to match
                    break

            # Check whether this subgraph has outgoing edges not described by source patterns
            if not self.check_succ(pat_to_expr, succ_list):
                self.history.update(src_matched)
                self.clear_pat()
                continue

            # Generate target expressions and map source to them
            if self.is_var:
                # noinspection PyTypeChecker
                tgt_expr = self.build_variadic(self.src_outs[0], self.tgt_outs[0], pat_to_expr)
            else:
                tgt_expr = [_RelayBuilder(pat_to_expr, self.ty_map).visit(tgt, Env())
                            for tgt in self.tgt_outs]
            self.history.update(tgt_expr)
            expr_map = dict(zip(src_matched, tgt_expr))

            # Rewrite expression
            expr = _RewriteMutator(expr_map, self.ty_map).visit(expr)

            # Clear instantiated patterns
            self.clear_pat()

        self.clear_pat()
        return expr

    def clear_pat(self):
        for p in self.src_outs:
            p.clear()
        for p in self.tgt_outs:
            p.clear()

    def match_one(self, pattern: Pattern, expr: relay.Expr, pat_to_expr: PatExprMap) \
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
                matcher = Matcher(pat_to_expr.copy(), self.ty_map)
                res = matcher.match(pattern, cur_expr, Env())
                if not res:
                    self.history.add(cur_expr)
                    continue  # even first pattern is not matched, skip this expression
                pat_to_expr.update(matcher.pat_to_expr)
                return cur_expr

        return None

    reusable_pat = (pat.Wildcard, pat.Variable)

    def match_rest(self, pat_to_expr: PatExprMap, fst_matched: relay.Expr,
                   succ_list: SuccListMap) -> List[relay.Expr]:
        # Initialize stack and its operation
        expr_matched = set()
        stack: List[Tuple[Pattern, relay.Expr]] = []

        # Find candidate node-expression pair where the node is connected to matched ones.
        # Since we required the i-th pattern is connected to the union of 0..i-th patterns,
        # there must exist some node that satisfies the condition.
        def add_succ(pattern: Pattern, expr: relay.Expr):
            if expr not in succ_list:
                return
            for ps in pattern.succ:
                if ps in pat_to_expr:
                    continue  # successor pattern already matched
                if ps.src_idx != src_idx:
                    continue  # not source or not the source pattern concerned
                # matched expression cannot be matched again
                cand_es = list(filter(lambda ee: not (ee in expr_matched or ee in self.history),
                                      succ_list[expr]))
                if isinstance(pattern, self.reusable_pat):
                    for es in reversed(cand_es):
                        stack.append((ps, es))
                elif len(cand_es) > 0:
                    stack.append((ps, cand_es[0]))
                break  # only need to visit first unmatched successor pattern node

        # Match each output pattern
        output_matched = [fst_matched]
        for src_idx in range(1, len(self.src_outs)):
            # Get output pattern node
            src_pat = self.src_outs[src_idx]

            # Collect matched expression nodes
            expr_matched.update(pat_to_expr.values())

            # Push pattern-expression pair to stack
            for p, e in pat_to_expr.items():
                add_succ(p, e)

            # Backtrack until the source pattern is reached
            found = False
            while len(stack) > 0:
                # Pick one pair from stack
                p, e = stack.pop()

                # Add successors to stack if output pattern is not reached
                if p != src_pat:
                    add_succ(p, e)
                    continue

                # Match pattern with expression
                matcher = Matcher(pat_to_expr.copy(), self.ty_map)
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
                self.history.add(fst_matched)
                return []

        return output_matched

    def match_variadic(self, src_var: pat.Variadic, expr: relay.Expr, pat_to_expr: PatExprMap,
                       succ_list: SuccListMap) -> List[relay.Expr]:
        # Search for first match of field pattern
        fst_match = self.match_one(src_var.instantiate(), expr, pat_to_expr)
        if fst_match is None:
            return []  # no expression could be a match

        # Find candidate pattern-expression pair
        stack: List[Tuple[Pattern, relay.Expr]] = []
        expr_matched = set()

        def add_succ(pattern: Pattern, ex: relay.Expr):
            if ex not in succ_list:
                return
            if len(pattern.src_succ) == 0:
                return
            p_succ = pattern.src_succ[0]
            cand_es = list(filter(lambda ee: not (ee in expr_matched or ee in self.history),
                                  succ_list[ex]))
            if isinstance(pattern, self.reusable_pat):
                for es in reversed(cand_es):
                    stack.append((p_succ, es))
            elif len(cand_es) > 0:
                stack.append((p_succ, cand_es[0]))

        # Find more matches of variadic pattern
        out_matched = [fst_match]
        while True:
            # Collect all matched expressions
            expr_matched.update(pat_to_expr.values())

            # Push pattern-expression pair to stack
            for p, e in pat_to_expr.items():
                add_succ(p, e)

            # Backtrack to find the initial field pattern
            found = False
            while len(stack) > 0:
                # Pop one pair from stack
                p, e = stack.pop()

                # Push successors to stack if field pattern is not reached
                if p not in src_var.field_inst:
                    add_succ(p, e)
                    continue

                # Try matching field with expression
                env = Env() if src_var.index is None else \
                    Env(symbol=src_var.index, value=len(out_matched))
                matcher = Matcher(pat_to_expr.copy(), self.ty_map)
                result = matcher.match(src_var.instantiate(), e, env)
                if not result:
                    src_var.rollback()
                    continue

                # Add matched expression to record
                pat_to_expr.update(matcher.pat_to_expr)
                out_matched.append(e)
                found = True
                break

            # No more match for this pattern
            if not found:
                return out_matched

    def build_variadic(self, src_var: pat.Variadic, tgt_var: pat.Variadic,
                       pat_to_expr: PatExprMap) -> List[relay.Expr]:
        length = len(src_var)
        tgt_outs: List[relay.Expr] = []
        for i in range(length):
            env = Env() if tgt_var.index is None else Env(symbol=tgt_var.index, value=i)
            inst = tgt_var.instantiate()
            tgt_outs.append(_RelayBuilder(pat_to_expr, self.ty_map).visit(inst, env))
        return tgt_outs

    @staticmethod
    def build_succ_list(expr: relay.Expr) -> SuccListMap:
        succ_visitor = _SuccListBuilder()
        succ_visitor.visit(expr)
        return succ_visitor.succ_list

    @classmethod
    def check_succ(cls, pat_to_expr: PatExprMap, succ_list: SuccListMap) -> bool:
        for p, e in pat_to_expr.items():
            # Skip variables, wildcards and output nodes because they can always be reused
            if isinstance(p, cls.reusable_pat) or p.is_output:
                continue

            # From our matching algorithm, unmatched successor expression nodes could only be last
            # several elements of the expression's successor list. We just need to ensure the
            # pattern node and expression node has same number of successors.
            if len(p.src_succ) != len(succ_list[e]):
                return False

        return True


class _TypeMapper(relay.ExprVisitor):
    def __init__(self):
        super().__init__()
        self.ty_map: ExprTypeMap = {}

    def visit(self, expr: relay.Expr):
        if isinstance(expr, (relay.Constant, relay.Var, relay.Call, relay.Tuple,
                             relay.TupleGetItem)):
            super().visit(expr)
            ty = expr.checked_type
            self.ty_map[expr] = ty


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
    def __init__(self, pat_to_expr: PatExprMap, ty_map: ExprTypeMap):
        super().__init__()
        self.pat_to_expr = pat_to_expr
        self.ty_map = ty_map

    def visit(self, p: Pattern, env: Env) -> relay.Expr:
        if p in self.pat_to_expr:
            return self.pat_to_expr[p]
        elif p.in_src:
            raise RuntimeError(
                'Cannot find matched expression of pattern in source graph.'
            )
        else:
            expr = super().visit(p, env)
            self.pat_to_expr[p] = expr
            return expr

    def visit_const(self, const: pat.Const, env: Env) -> relay.Expr:
        if isinstance(const.value, np.ndarray):
            value = const.value
        elif isinstance(const.value, Attr):
            value = AttrEvaluator(self.pat_to_expr, self.ty_map).visit(const.value, env)
        else:
            raise RuntimeError('Unreachable.')
        return relay.const(value)

    def visit_call(self, call: pat.Call, env: Env) -> relay.Expr:
        args = [self.visit(a, env) for a in call.args]
        attrs = dict([(name, AttrEvaluator(self.pat_to_expr, self.ty_map).visit(attr, env))
                      for name, attr in call.attrs.items()])
        op_name = self.visit_op(call.op, env)
        api = spec.get_api(op_name)
        try:
            call_expr = api(*args, **attrs)
        except TypeError:
            raise RuntimeError(
                'Cannot create call expression for op \'{}\' with {} operand(s) and attribute '
                'set {}.'.format(op_name, len(args), attrs))
        return call_expr

    def visit_op(self, op: pat.Op, env: Env) -> str:
        if isinstance(op, pat.ConcreteOp):
            return op.name
        elif isinstance(op, pat.OpWithTrait):
            return self.pat_to_expr[op].name
        else:
            raise RuntimeError('Unreachable.')

    def visit_tuple(self, tup: pat.Tuple, env: Env) -> relay.Expr:
        return relay.Tuple([self.visit(f, env) for f in tup.fields])

    def visit_getitem(self, getitem: pat.GetItem, env: Env) -> relay.Expr:
        tup = self.visit(getitem.tup, env)
        idx = AttrEvaluator(self.pat_to_expr, self.ty_map).visit(getitem.idx, env)
        return tup[idx]

    def visit_cond(self, cond: pat.Cond, env: Env) -> relay.Expr:
        pred = AttrEvaluator(self.pat_to_expr, self.ty_map).visit(cond.predicate, env)
        return self.visit(cond.then_pat, env) if pred else self.visit(cond.else_pat, env)

    def visit_variadic(self, var: pat.Variadic, env: Env) -> relay.Expr:
        # Evaluate length
        length = AttrEvaluator(self.pat_to_expr, self.ty_map).visit(var.len, env)

        # Create fields
        fields: List[relay.Expr] = []
        for i in range(length):
            new_env = env
            if var.index is not None:
                new_env = env + (var.index, i)
            pat_f = var.instantiate()
            fields.append(self.visit(pat_f, new_env))

        return relay.Tuple(fields)

    def visit_get_instance(self, get_inst: pat.GetInst, env: Env) -> relay.Expr:
        inst = eval_get_inst(get_inst, self.pat_to_expr, self.ty_map, env)
        return self.pat_to_expr[inst]


class _RewriteMutator(relay.ExprMutator):
    def __init__(self, subst_map: Dict[relay.Expr, relay.Expr], ty_map: ExprTypeMap):
        super().__init__()
        self.subst_map = subst_map
        self.ty_map = ty_map

    def visit(self, expr: relay.Expr):
        if expr in self.subst_map:
            new_expr = self.subst_map[expr]
            if expr in self.ty_map:
                self.ty_map[new_expr] = self.ty_map[expr]
                del self.ty_map[expr]
            return new_expr
        elif expr in self.memo_map:
            return self.memo_map[expr]
        else:
            ret = super().visit(expr)
            if expr in self.ty_map:
                self.ty_map[ret] = self.ty_map[expr]
                if ret is not expr:
                    del self.ty_map[expr]
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
    def __init__(self, src: Pattern, tgt: Pattern, ty_map: ExprTypeMap):
        super().__init__()
        self.src = src
        self.tgt = tgt
        self.ty_map = ty_map

    def visit(self, expr: relay.Expr):
        # Directly return if it has been visited before
        if expr in self.memo_map:
            return self.memo_map[expr]

        # Rewrite predecessors
        new_expr = super().visit(expr)

        # Match pattern with this expression
        pat_to_expr: PatExprMap = {}
        matcher = Matcher(pat_to_expr, self.ty_map)
        if matcher.match(self.src, new_expr, Env()):
            new_expr = _RelayBuilder(pat_to_expr, self.ty_map).visit(self.tgt, Env())

        # Map to new expression
        self.memo_map[expr] = new_expr
        if expr in self.ty_map:
            self.ty_map[new_expr] = self.ty_map[expr]

        return new_expr
