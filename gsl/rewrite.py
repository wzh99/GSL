from typing import List, Dict, Tuple, Optional, Set

import numpy as np
from tvm import relay

from . import pat, spec, util
from .attr import Attr, Env
from .eval import PatExprMap, ExprTypeMap, EvalHistory, AttrEvaluator, eval_get_inst
from .match import Matcher, matched_types
from .pat import Pattern, PatternVisitor

SuccListMap = Dict[relay.Expr, List[relay.Expr]]


class ExprRewriter:
    class StackElem:
        def __init__(self, e: relay.Expr, count: int):
            self.expr = e
            self.count = count

    def __init__(self, src_outs: List[Pattern], tgt_outs: List[Pattern], variadic: bool):
        self.src_outs = src_outs
        self.tgt_outs = tgt_outs
        self.variadic = variadic
        self.ty_map: ExprTypeMap = {}
        self.stack: List[ExprRewriter.StackElem] = []
        self.traversed: Set[relay.Expr] = set()
        self.history: Set[relay.Expr] = set()

    def rewrite(self, expr: relay.Expr) -> relay.Expr:
        # Build expression type map
        mapper = _TypeMapper()
        mapper.visit(expr)
        self.ty_map = mapper.ty_map

        # Use single pattern rewriter if possible
        if len(self.src_outs) == 1 and not self.variadic:
            return _SingleRewriter(self.src_outs[0], self.tgt_outs[0], self.ty_map).visit(expr)

        # Build successor list for all expression nodes
        succ_list = self.build_succ_list(expr)

        # Initialize stack
        self.stack.append(self.StackElem(expr, 0))

        # Greedily match all subgraphs
        subst_map: Dict[relay.Expr, relay.Expr] = {}
        while True:
            # Find matched expressions with patterns
            pat_to_expr = PatExprMap()
            if self.variadic:  # match variadic with different procedure
                # noinspection PyTypeChecker
                src_var: pat.Variadic = self.src_outs[0]
                src_matched = self.match_variadic(src_var, pat_to_expr, succ_list)
                if len(src_matched) == 0:
                    break
                elif src_var.min_len_ is not None and len(src_matched) < src_var.min_len_:
                    self.history.update(src_matched)
                    self.clear_pat()
                    continue

            else:
                fst_matched = self.match_one(self.src_outs[0], pat_to_expr)
                if fst_matched is None:
                    break  # even the first output pattern node is not matched
                src_matched = self.match_rest(pat_to_expr, fst_matched, succ_list)
                if len(src_matched) == 0:  # the rest output nodes fail to match
                    self.history.add(fst_matched)
                    self.clear_pat()
                    continue

            # Check whether this subgraph has outgoing edges not described by source patterns
            if not self.check_succ(pat_to_expr, succ_list):
                self.history.update(src_matched)
                self.clear_pat()
                continue

            # Generate target expressions and map source to them
            if self.variadic:
                # noinspection PyTypeChecker
                tgt_expr = self.build_variadic(self.src_outs[0], self.tgt_outs[0], pat_to_expr)
            else:
                eval_his: EvalHistory = {}
                tgt_expr = [_RelayBuilder(pat_to_expr, self.ty_map, eval_his).visit(tgt, Env())
                            for tgt in self.tgt_outs]
            self.history.update(src_matched)
            subst_map.update(zip(src_matched, tgt_expr))

            # Clear temporary data in patterns
            self.clear_pat()

        # Perform substitution
        expr = _RewriteMutator(subst_map, self.ty_map).visit(expr)

        # Clear up and exit
        self.traversed.clear()
        self.history.clear()
        self.clear_pat()

        return expr

    def match_one(self, pattern: Pattern, pat_to_expr: PatExprMap) -> Optional[relay.Expr]:
        # Traverse the expression graph
        while len(self.stack) > 0:
            # Pop an element from stack
            elem = self.stack.pop()
            cur_expr = elem.expr

            if elem.count == 0:
                # Add children to stack if this expression is visited for the first time
                self.stack.append(self.StackElem(cur_expr, 1))
                for ep in reversed(util.get_expr_pred(cur_expr)):
                    self.push_stack(ep)
            else:
                # Do not match this expression if it is in match history
                if cur_expr in self.history:
                    continue
                self.history.add(cur_expr)

                # Use first pattern to roughly locate the subgraph
                rec = pat_to_expr.record()
                matcher = Matcher(pat_to_expr, self.ty_map)
                res = matcher.match(pattern, cur_expr, Env())
                if not res:
                    rec.restore()
                    continue  # even first pattern is not matched, skip this expression
                return cur_expr

        return None

    def push_stack(self, expr: relay.Expr):
        if expr not in self.traversed:
            self.stack.append(ExprRewriter.StackElem(expr, 0))
            self.traversed.add(expr)

    reusable_pat = (pat.Wildcard, pat.Variable)

    def match_rest(self, pat_to_expr: PatExprMap, fst_matched: relay.Expr,
                   succ_list: SuccListMap) -> List[relay.Expr]:
        # Initialize stack and its operation
        stack: List[Tuple[Pattern, relay.Expr]] = []

        # Find candidate node-expression pair where the node is connected to matched ones.
        # Since we required the i-th pattern is connected to the union of 0..i-th patterns,
        # there must exist some node that satisfies the condition.
        def add_succ(pattern: Pattern, expr: relay.Expr):
            if expr not in succ_list:
                return
            for ps in pattern.succ_:
                if ps in pat_to_expr:
                    continue  # successor pattern already matched
                if ps.src_idx_ != src_idx:
                    continue  # not source or not the source pattern concerned
                # matched expression cannot be matched again
                cand_es = list(filter(
                    lambda ee: not (pat_to_expr.has_expr(ee) or ee in self.history),
                    succ_list[expr]
                ))
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
                rec = pat_to_expr.record()
                matcher = Matcher(pat_to_expr, self.ty_map)
                res = matcher.match(src_pat, e, Env())
                if not res:
                    rec.restore()
                    continue

                # Update matched nodes and expressions
                output_matched.append(e)
                found = True
                stack.clear()
                break

            # No match for this pattern, the whole match failed
            if not found:
                self.history.add(fst_matched)
                return []

        return output_matched

    def match_variadic(self, src_var: pat.Variadic, pat_to_expr: PatExprMap,
                       succ_list: SuccListMap) -> List[relay.Expr]:
        # Search for first match of field pattern
        fst_match = self.match_one(src_var.instantiate(), pat_to_expr)
        if fst_match is None:
            return []  # no expression could be a match

        # Find candidate pattern-expression pair
        stack: List[Tuple[Pattern, relay.Expr]] = []

        def add_succ(pattern: Pattern, expression: relay.Expr):
            if expression not in succ_list:
                return
            if len(pattern.src_succ) == 0:
                return
            p_succ = pattern.src_succ[0]
            cand_es = list(filter(
                lambda ee: not (pat_to_expr.has_expr(ee) or ee in self.history),
                succ_list[expression]
            ))
            if isinstance(pattern, self.reusable_pat):
                for es in reversed(cand_es):
                    stack.append((p_succ, es))
            elif len(cand_es) > 0:
                stack.append((p_succ, cand_es[0]))

        # Find more matches of variadic pattern
        out_matched = [fst_match]
        while True:
            # Push pattern-expression pair to stack
            for p, e in pat_to_expr.items():
                add_succ(p, e)

            # Backtrack to find the initial field pattern
            found = False
            while len(stack) > 0:
                # Pop one pair from stack
                p, e = stack.pop()

                # Push successors to stack if field pattern is not reached
                if p not in src_var.field_inst_:
                    add_succ(p, e)
                    continue

                # Try matching field with expression
                env = Env() if src_var.index_ is None else \
                    Env(symbol=src_var.index_, value=len(out_matched))
                rec = pat_to_expr.record()
                matcher = Matcher(pat_to_expr, self.ty_map)
                result = matcher.match(src_var.instantiate(), e, env)
                if not result:
                    rec.restore()
                    src_var.rollback()
                    continue

                # Add matched expression to record
                out_matched.append(e)
                found = True
                stack.clear()

            # No more match for this pattern
            if not found:
                return out_matched

    def build_variadic(self, src_var: pat.Variadic, tgt_var: pat.Variadic,
                       pat_to_expr: PatExprMap) -> List[relay.Expr]:
        length = len(src_var)
        tgt_outs: List[relay.Expr] = []
        eval_his: EvalHistory = {}
        for i in range(length):
            env = Env() if tgt_var.index_ is None else Env(symbol=tgt_var.index_, value=i)
            inst = tgt_var.instantiate()
            tgt_outs.append(_RelayBuilder(pat_to_expr, self.ty_map, eval_his).visit(inst, env))
        return tgt_outs

    def clear_pat(self):
        for p in self.src_outs:
            p.clear()
        for p in self.tgt_outs:
            p.clear()

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
        super().visit(expr)
        if isinstance(expr, matched_types):
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
    def __init__(self, pat_to_expr: PatExprMap, ty_map: ExprTypeMap, eval_his: EvalHistory):
        super().__init__()
        self.pat_to_expr = pat_to_expr
        self.ty_map = ty_map
        self.eval_his = eval_his

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
        if isinstance(const.val_, np.ndarray):
            value = const.val_
        elif isinstance(const.val_, Attr):
            value = self._eval_attr(const.val_, env)
        else:
            raise RuntimeError('Unreachable.')
        dtype = self._eval_attr(const.dtype_, env)
        return relay.const(np.array(value, dtype=dtype))

    def visit_call(self, call: pat.Call, env: Env) -> relay.Expr:
        args = [self.visit(a, env) for a in call.args_]
        attrs = dict([(name, self._eval_attr(attr, env)) for name, attr in call.attrs_.items()])
        op_name = self.visit_op(call.op_, env)
        api = spec.get_api(op_name)
        try:
            call_expr = api(*args, **attrs)
        except TypeError:
            raise RuntimeError(
                'Cannot create call expression for op \'{}\' with {} operand(s) and attribute '
                'set {}.'.format(op_name, len(args), attrs))
        return call_expr

    def visit_op(self, op: pat.Pattern, _env: Env) -> str:
        if isinstance(op, pat.ConcreteOp):
            return op.name_
        else:
            return self.pat_to_expr[op].name

    def visit_tuple(self, tup: pat.Tuple, env: Env) -> relay.Expr:
        return relay.Tuple([self.visit(f, env) for f in tup.fields_])

    def visit_getitem(self, getitem: pat.GetItem, env: Env) -> relay.Expr:
        tup = self.visit(getitem.tup_, env)
        idx = self._eval_attr(getitem.idx_, env)
        return tup[idx]

    def visit_cond(self, cond: pat.Cond, env: Env) -> relay.Expr:
        pred = self._eval_attr(cond.predicate_, env)
        return self.visit(cond.then_pat_, env) if pred else self.visit(cond.else_pat_, env)

    def visit_match(self, match: pat.Match, env: Env) -> relay.Expr:
        alt = match.alt_
        if alt.matched_idx_ is None:
            raise RuntimeError(
                'None of the alternative pattern is matched.'
            )
        return self.visit(match.clauses_[alt.matched_idx_], env)

    def visit_variadic(self, var: pat.Variadic, env: Env) -> relay.Expr:
        # Evaluate length
        length = self._eval_attr(var.len_, env)

        # Create fields
        fields: List[relay.Expr] = []
        for i in range(length):
            new_env = env
            if var.index_ is not None:
                new_env += (var.index_, i)
            pat_f = var.instantiate()
            fields.append(self.visit(pat_f, new_env))

        return relay.Tuple(fields)

    def visit_get_instance(self, get_inst: pat.GetInst, env: Env) -> relay.Expr:
        inst = eval_get_inst(get_inst, self.pat_to_expr, self.ty_map, env, self.eval_his)
        return self.pat_to_expr[inst]

    def _eval_attr(self, attr: Attr, env: Env):
        return AttrEvaluator(self.pat_to_expr, self.ty_map, self.eval_his).visit(attr, env)


class _RewriteMutator(relay.ExprMutator):
    def __init__(self, subst_map: Dict[relay.Expr, relay.Expr], ty_map: ExprTypeMap):
        super().__init__()
        self.subst_map = subst_map
        self.ty_map = ty_map

    def visit(self, expr: relay.Expr):
        if expr in self.subst_map:
            super().visit(expr)
            new_expr = _TgtUpdater(self.memo_map).visit(self.subst_map[expr])
            self.memo_map[expr] = new_expr
        else:
            new_expr = super().visit(expr)
        return new_expr


class _TgtUpdater(relay.ExprMutator):
    def __init__(self, memo_map: Dict[relay.Expr, relay.Expr]):
        super().__init__()
        self.memo_map = memo_map


class _SingleRewriter(relay.ExprMutator):
    def __init__(self, src: Pattern, tgt: Pattern, ty_map: ExprTypeMap):
        super().__init__()
        self.src = src
        self.tgt = tgt
        self.ty_map = ty_map

    def visit(self, pre: relay.Expr):
        # Rewrite predecessors
        mid = super().visit(pre)
        if not isinstance(mid, matched_types):
            return mid
        if pre in self.ty_map:
            self.ty_map[mid] = self.ty_map[pre]

        # Match and rewrite
        pat_to_expr = PatExprMap()
        matcher = Matcher(pat_to_expr, self.ty_map)
        if matcher.match(self.src, mid, Env()):
            post = _RelayBuilder(pat_to_expr, self.ty_map, {}).visit(self.tgt, Env())
            if pre in self.ty_map:
                del self.ty_map[mid]
                self.ty_map[post] = self.ty_map[pre]
            self.memo_map[pre] = post
            return post
        else:
            return mid
