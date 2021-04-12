from functools import reduce
from typing import List, Dict, Callable, Union, Any, Optional, Generic, TypeVar, Set

import numpy as np

from . import attr, spec
from .attr import Attr, Symbol, AttrLike


class Pattern:
    """
    Base class for all pattern graph nodes. This class cannot be directly instantiated.
    """

    NOT_IN_SRC = -1

    def __init__(self):
        self.injective = True
        self.succ: List[Pattern] = []
        self.src_idx = self.NOT_IN_SRC
        self.in_tgt = False
        self.is_tpl = False
        self.free_sym: Set[Symbol] = set()

    @property
    def pred(self) -> List['Pattern']:
        return []

    def update_pred_succ(self):
        for p in self.pred:
            p.succ.append(self)

    @property
    def is_output(self) -> bool:
        return len(self.succ) == 0

    @property
    def in_src(self) -> bool:
        return self.src_idx != self.NOT_IN_SRC

    @property
    def src_succ(self) -> List['Pattern']:
        return list(filter(lambda p: p.in_src and not p.is_tpl, self.succ))

    @property
    def is_used(self) -> bool:
        return self.in_src or self.in_tgt

    def check_all(self, predicate: Callable[['Pattern'], bool]) -> bool:
        if not predicate(self):
            return False
        for p in self.pred:
            if not p.check_all(predicate):
                return False
        return True

    def check_any(self, predicate: Callable[['Pattern'], bool]) -> bool:
        if predicate(self):
            return True
        for p in self.pred:
            if p.check_any(predicate):
                return True
        return False

    tensor_attrs = ['shape', 'dtype', 'ndim']

    @property
    def avail_attrs(self) -> List[str]:
        return []

    def has_attr(self, name: str) -> bool:
        return name in self.avail_attrs

    @property
    def attr_expr(self) -> List[Attr]:
        return []

    @property
    def bounded_sym(self) -> List[Symbol]:
        return []

    @property
    def has_free_sym(self):
        return len(self.free_sym) != 0

    def _update_free_sym(self):
        self.free_sym = reduce(
            set.union, map(lambda p: p.free_sym, self.pred), set()
        ).union(reduce(
            set.union, map(lambda a: a.free_sym, self.attr_expr), set()
        )).difference(self.bounded_sym)

    def __getitem__(self, *item):
        return GetItem(self, item[0])

    def __getattr__(self, name: str) -> attr.GetAttr:
        if not self.has_attr(name):
            raise AttributeError(
                'Attribute {} not found in pattern.'.format(name)
            )
        return attr.GetAttr(self, name)

    def __neg__(self):
        return Call('negative', self)

    def __add__(self, other):
        return Call('add', self, to_pat(other))

    def __radd__(self, other):
        return Call('add', to_pat(other), self)

    def __sub__(self, other):
        return Call('subtract', self, to_pat(other))

    def __rsub__(self, other):
        return Call('subtract', to_pat(other), self)

    def __mul__(self, other):
        return Call('multiply', self, to_pat(other))

    def __rmul__(self, other):
        return Call('multiply', to_pat(other), self)

    def __truediv__(self, other):
        return Call('divide', self, to_pat(other))

    def __rtruediv__(self, other):
        return Call('divide', to_pat(other), self)

    def __contains__(self, sub: 'Pattern') -> bool:
        return self.check_any(lambda p: p is sub)

    def clear(self):
        for p in self.pred:
            p.clear()

    def accept(self, visitor: 'PatternVisitor', arg: 'ArgType'):
        raise NotImplementedError()


class Wildcard(Pattern):
    """
    A wildcard node matches all nodes in graph. Target graph cannot contain wildcard nodes not
    defined in source graph.
    """

    @property
    def avail_attrs(self) -> List[str]:
        return self.tensor_attrs

    def accept(self, visitor: 'PatternVisitor', arg: 'ArgType'):
        return visitor.visit_wildcard(self, arg)


class Variable(Pattern):
    """
    A variable node matches input tensor of the model. Target graph cannot contain variable nodes
    not defined in source graph.
    """

    def __init__(self, shape: Union[tuple, attr.Attr, None] = None,
                 dtype: Union[str, attr.Attr, None] = None,
                 ndim: Union[int, attr.Attr, None] = None):
        super().__init__()

        # Check attributes for variable
        self.attrs: Dict[str, Attr] = {}
        raw_attrs = filter_attrs(dict(zip(
            self.tensor_attrs,
            [shape, dtype, ndim]
        )))
        for n, a in raw_attrs.items():
            self.attrs[n] = attr.to_attr(a)

        self._update_free_sym()

    @property
    def avail_attrs(self) -> List[str]:
        return self.tensor_attrs

    @property
    def attr_expr(self) -> List[Attr]:
        return list(self.attrs.values())

    def accept(self, visitor: 'PatternVisitor', arg: 'ArgType'):
        return visitor.visit_variable(self, arg)


def filter_attrs(attrs: Dict[str, Any]) -> Dict[str, Any]:
    filtered = {}
    for k, v in attrs.items():
        if v is not None:
            filtered[k] = v
    return filtered


ConstValueType = Union[int, float, list, np.ndarray, Attr]


class Const(Pattern):
    """
    A constant nodes stores constants in graph.
    """
    value_class = (int, float, list, np.ndarray, Attr)

    def __init__(self, value: Union[ConstValueType, Attr, None] = None,
                 dtype: AttrLike = None):
        """
        Constructor.

        :param value: In source graph, if the value is provided, it only matches nodes with the
            same value. Otherwise, any constant node will match. If the node also appears in
            target graph, the value will be copied to new graph. New constant nodes can be created
            in target graph. In target graph, the constant nodes can also be specified by an
            attribute expression with respect to nodes in source graph.
        :param dtype: Indicate data type to store constant value. Only effective during rewrite.
        """
        super().__init__()
        if value is None:
            self.val_ = None
        elif isinstance(value, (Attr, np.ndarray)):
            self.val_ = value
        elif isinstance(value, (int, float, list)):
            self.val_ = np.array(value)
        else:
            raise TypeError(
                'Cannot create constant node from value of type {}.'.format(value.__class__)
            )
        self.dtype_ = attr.to_attr(dtype)
        self._update_free_sym()

    @property
    def avail_attrs(self) -> List[str]:
        return self.tensor_attrs + ['value']

    @property
    def attr_expr(self) -> List[Attr]:
        expr = [self.dtype_]
        if isinstance(self.val_, Attr):
            expr.append(self.val_)
        return expr

    def accept(self, visitor: 'PatternVisitor', arg: 'ArgType'):
        return visitor.visit_const(self, arg)


PatternLike = Union[Pattern, ConstValueType]


def to_pat(val: PatternLike) -> Pattern:
    """
    Create a graph pattern node with given value.

    :param val: All types of values that are or can be converted to an graph pattern node.
    :return: Graph pattern node created from given value.
    """
    if isinstance(val, Pattern):
        return val
    elif isinstance(val, Const.value_class):
        return Const(val)
    else:
        raise TypeError('Cannot convert to pattern graph node.')


class Op(Pattern):
    def __str__(self):
        raise NotImplementedError()

    def __call__(self, *args: PatternLike, **raw_attrs: AttrLike):
        return Call(self, *args, **raw_attrs)

    def accept(self, visitor: 'PatternVisitor', arg: 'ArgType'):
        return visitor.visit_op(self, arg)


class ConcreteOp(Op):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def __str__(self):
        return self.name


class OpWithTrait(Op):
    def __init__(self, trait: spec.OpTrait):
        super().__init__()
        self.trait = trait

    def __str__(self):
        return self.trait.name


class Call(Pattern):
    """
    Represents an operator call.
    """

    def __init__(self, op: Union[Pattern, str, spec.OpTrait], *args: PatternLike,
                 **raw_attrs: AttrLike):
        super().__init__()
        self.args = [to_pat(a) for a in args]

        # Convert valid alternatives of Op to node
        if isinstance(op, str):
            op = ConcreteOp(op)
        elif isinstance(op, spec.OpTrait):
            op = OpWithTrait(op)
        self.op: Pattern = op

        # Check number of inputs
        if isinstance(op, ConcreteOp):
            num_input = spec.get_num_inputs(op.name)
            if num_input != len(args):
                raise ValueError(
                    'Expect {} input tensor(s), got {}.'.format(num_input, len(args))
                )

        # Convert raw attribute values to attribute nodes if necessary
        self.attrs = dict([(name, attr.to_attr(val)) for name, val in raw_attrs.items()])

        # Check if specified attributes really exists in op
        if isinstance(op, ConcreteOp):
            attr_names = spec.get_op_attr_names(op.name)
            for name in self.attrs.keys():
                if name not in attr_names:
                    raise AttributeError(
                        'Attribute \'{}\' not found in op \'{}\'.'.format(name, op.name)
                    )

        self._update_free_sym()

    @property
    def pred(self):
        return self.args

    @property
    def avail_attrs(self) -> List[str]:
        return self.tensor_attrs

    @property
    def attr_expr(self) -> List[Attr]:
        return list(self.attrs.values())

    def has_attr(self, name: str) -> bool:
        if name in self.avail_attrs:
            return True
        if isinstance(self.op, ConcreteOp):
            attr_names = spec.get_op_attr_names(self.op.name)
            return name in attr_names
        else:
            return True

    def accept(self, visitor: 'PatternVisitor', arg: 'ArgType'):
        return visitor.visit_call(self, arg)


def same_attr(pat: Pattern, attrs: List[str]) -> Dict[str, attr.Attr]:
    """
    Create a attribute expression dictionary with form `a=p.a` where `a` is attribute name from
    the list, and `p` is a pattern node. This function is especially useful for specifying
    attribute identity constraints between two nodes.

    :param pat: The pattern node from which attributes are accessed.
    :param attrs: List of attribute names.
    :return: Attribute expression dictionary where each entry has form a=p.a.
    """
    return dict([(a, pat.__getattr__(a)) for a in attrs])


class Tuple(Pattern):
    def __init__(self, *raw_fields: PatternLike):
        super().__init__()
        self.fields = [to_pat(f) for f in raw_fields]
        self._update_free_sym()

    @property
    def pred(self):
        return self.fields

    def accept(self, visitor: 'PatternVisitor', arg: 'ArgType'):
        return visitor.visit_tuple(self, arg)


class GetItem(Pattern):
    def __init__(self, tup: Pattern, index: AttrLike = None):
        super().__init__()
        self.tup = tup
        self.idx = attr.to_attr(index)
        self._update_free_sym()

    @property
    def pred(self):
        return [self.tup]

    @property
    def avail_attrs(self) -> List[str]:
        return self.tensor_attrs + ['index']

    @property
    def attr_expr(self) -> List[Attr]:
        return [self.idx]

    def accept(self, visitor: 'PatternVisitor', arg: 'ArgType'):
        return visitor.visit_getitem(self, arg)


class Cond(Pattern):
    """
    Conditionally use one of the patterns.
    This pattern can only be used in rewrite process but not in match.
    """

    def __init__(self, predicate: attr.AttrLike, then_pat: PatternLike, else_pat: PatternLike):
        super().__init__()
        self.predicate = attr.to_attr(predicate)
        self.then_pat = to_pat(then_pat)
        self.else_pat = to_pat(else_pat)
        self._update_free_sym()

    @property
    def pred(self) -> List[Pattern]:
        return [self.then_pat, self.else_pat]

    @property
    def attr_expr(self) -> List[Attr]:
        return [self.predicate]

    def accept(self, visitor: 'PatternVisitor', arg: 'ArgType'):
        return visitor.visit_cond(self, arg)


class Alt(Pattern):
    """
    Try matching expression with given patterns in order until one is matched.
    This pattern can only be used in match process of single pattern.
    """

    def __init__(self, *pats: PatternLike):
        if len(pats) < 2:
            raise ValueError(
                'Must provide at least two alternative patterns.'
            )

        super().__init__()
        self.pats = [to_pat(p) for p in pats]
        self.matched_idx: Optional[int] = None
        self._update_free_sym()

    @property
    def pred(self) -> List['Pattern']:
        return self.pats

    def has_attr(self, name: str) -> bool:
        return all([p.has_attr(name) for p in self.pats])

    def accept(self, visitor: 'PatternVisitor', arg: 'ArgType'):
        return visitor.visit_alt(self, arg)


class Match(Pattern):
    """
    Produce different patterns according to matched alternative pattern.
    This pattern can only be used in rewrite process.
    """

    def __init__(self, alt: Alt, clauses: List[PatternLike]):
        super().__init__()
        if len(clauses) != len(alt.pats):
            raise ValueError(
                'Expect {} clauses, got {}.'.format(len(alt.pats), len(clauses))
            )
        self.alt = alt
        self.clauses = [to_pat(p) for p in clauses]
        self._update_free_sym()

    @property
    def pred(self) -> List['Pattern']:
        return self.clauses + [self.alt]

    def accept(self, visitor: 'PatternVisitor', arg: 'ArgType'):
        return visitor.visit_match(self, arg)


class Variadic(Pattern):
    """
    Matches tuple with arbitrary input, each with the same pattern. It can also be used to specify
    arbitrary number of output nodes, each with the same pattern.
    """

    def __init__(self, field: Pattern,
                 templates: Optional[List[Pattern]] = None,
                 first: Optional[List[Optional[Pattern]]] = None,
                 index: Optional[attr.Symbol] = None,
                 length: Optional[attr.AttrLike] = None,
                 min_len: Optional[int] = None):
        """
        Constructor.

        :param field: The common pattern of tuple fields.
        :param templates: Sub-patterns serve as templates, that is, for each of the expression
            matched, a unique pattern will be created. Note that if at least one sub-pattern is a
            template, the whole pattern must be a template as well.
        :param first: Pattern used in firstly matched expression. Sometimes, there are constraints
            imposed between template patterns. In this situation, pattern used for matching the
            first expression is different from the rest. The i-th pattern in first list will be
            used for the i-th template at first match.
        :param index: Symbol that indicates the index of matched expression.
        :param length: An attribute expression that indicates how long the pattern will be. In
            source pattern, it is optional. If provided, the length of tuple will be checked.
            In target pattern, it is required to specify the length of constructed tuple.
        :param min_len: An integer specifying the minimum number of fields that could be a match
            of this variadic. This only works for variadic source output pattern. In other cases,
            this value will be ignored.
        """
        super().__init__()

        # Initialize field
        self.field = field

        # Check templates
        if templates is not None and len(templates) > 0:  # at least one pattern is a template
            if field not in templates:
                raise ValueError(
                    'Template pattern must be duplicated if any of its sub-pattern should be '
                    'duplicated.'
                )
            if first is not None:
                if len(templates) != len(first):
                    raise ValueError(
                        'Number of first symbols must match number of duplicated symbols.'
                    )
            else:
                first = [None] * len(templates)
        else:
            templates = []
            first = []

        # Map template to first
        self.templates: List[Pattern] = templates
        self.first: List[Optional[Pattern]] = first
        self.tpl_to_fst = dict(zip(templates, first))
        for t, f in self.tpl_to_fst.items():
            if isinstance(t, Variadic):
                raise TypeError(
                    'Variadic cannot be a template pattern.'
                )
            if not field.check_any(lambda p: p is t):
                raise ValueError(
                    'Template is not sub-pattern of field pattern.'
                )
            if f is not None and t.__class__ != f.__class__:
                raise TypeError(
                    'Template and first pattern must be of same type.'
                )
            t.is_tpl = True

        # Initialize index and length
        self.index = index
        self.len: Optional[attr.Attr] = None
        if length is not None:
            self.len = attr.to_attr(length)
        self.min_len = min_len

        # Initialize records during substitution
        self.field_inst: List[Pattern] = []
        self.tpl_inst: List[Dict[Pattern, Pattern]] = []

        self._update_free_sym()

    @property
    def pred(self):
        return [self.field]

    @property
    def avail_attrs(self) -> List[str]:
        return ['length']

    @property
    def attr_expr(self) -> List[Attr]:
        return [] if self.len is None else [self.len]

    @property
    def bounded_sym(self) -> List[Symbol]:
        return [] if self.index is None else [self.index]

    def __call__(self, tpl: Pattern, index: attr.AttrLike):
        if tpl not in self.templates:
            raise ValueError('Pattern is not template of this variadic.')
        return GetInst(self, tpl, attr.to_attr(index))

    def __len__(self) -> int:
        return len(self.field_inst)

    def get_inst(self, idx: int, t: Pattern):
        if idx >= len(self):
            raise RuntimeError(
                'Index {} out of bound {}.'.format(idx, len(self))
            )
        inst_map = self.tpl_inst[idx]
        if t not in inst_map:
            raise RuntimeError(
                'Template pattern not found in instance mapping.'
            )
        return inst_map[t]

    def clear(self):
        super().clear()
        while len(self.field_inst) > 0:
            self.rollback()

    def has_first(self, t: Pattern) -> bool:
        return self.tpl_to_fst[t] is not None

    def instantiate(self) -> Pattern:
        # Instantiate templates
        visitor = _PatInst(self)
        field = visitor.visit(self.field, None)
        inst_map = visitor.map

        # Maps templates to first instances
        if len(self) == 0:
            for tpl in self.templates:
                if self.has_first(tpl):
                    inst_map[tpl] = self.tpl_to_fst[tpl]

        # Add record
        self.field_inst.append(field)
        self.tpl_inst.append(inst_map)

        return field

    def rollback(self):
        self.field_inst.pop()
        inst_map = self.tpl_inst.pop()
        for tpl, inst in inst_map.items():
            if inst not in self.first:
                for p in inst.pred:
                    p.succ.remove(inst)

    def accept(self, visitor: 'PatternVisitor', arg: 'ArgType'):
        return visitor.visit_variadic(self, arg)


class GetInst(Pattern):
    def __init__(self, var: Variadic, tpl: Pattern, index: AttrLike):
        super().__init__()
        self.var = var
        self.tpl = tpl
        self.idx = attr.to_attr(index)
        self._update_free_sym()

    @property
    def avail_attrs(self) -> List[str]:
        return self.tpl.avail_attrs

    @property
    def attr_expr(self) -> List[Attr]:
        return [self.idx]

    def accept(self, visitor: 'PatternVisitor', arg: 'ArgType'):
        return visitor.visit_get_instance(self, arg)


ArgType = TypeVar('ArgType')


class PatternVisitor(Generic[ArgType]):
    def __init__(self):
        self.visited: Dict[Pattern, Any] = dict()

    def visit(self, pat: Pattern, arg: ArgType) -> Any:
        if pat in self.visited:
            return self.visited[pat]
        ret = pat.accept(self, arg)
        self.visited[pat] = ret
        return ret

    def visit_wildcard(self, wildcard: Wildcard, arg: ArgType) -> Any:
        pass

    def visit_variable(self, var: Variable, arg: ArgType) -> Any:
        pass

    def visit_const(self, const: Const, arg: ArgType) -> Any:
        pass

    def visit_op(self, op: Op, arg: ArgType) -> Any:
        pass

    def visit_call(self, call: Call, arg: ArgType) -> Any:
        self.visit(call.op, arg)
        for a in call.args:
            self.visit(a, arg)

    def visit_tuple(self, tup: Tuple, arg: ArgType) -> Any:
        for f in tup.fields:
            self.visit(f, arg)

    def visit_getitem(self, getitem: GetItem, arg: ArgType) -> Any:
        self.visit(getitem.tup, arg)

    def visit_cond(self, cond: Cond, arg: ArgType) -> Any:
        self.visit(cond.then_pat, arg)
        self.visit(cond.else_pat, arg)

    def visit_alt(self, alt: Alt, arg: ArgType) -> Any:
        for p in alt.pats:
            self.visit(p, arg)

    def visit_match(self, match: Match, arg: ArgType) -> Any:
        self.visit(match.alt, arg)
        for c in match.clauses:
            self.visit(c, arg)

    def visit_variadic(self, var: Variadic, arg: ArgType) -> Any:
        self.visit(var.field, arg)

    def visit_get_instance(self, get_inst: GetInst, arg: ArgType) -> Any:
        self.visit(get_inst.var, arg)


class _PatInst(PatternVisitor[None]):
    def __init__(self, var: Variadic):
        super().__init__()
        self.var = var
        self.index = len(var)
        self.map: Dict[Pattern, Pattern] = {}  # template-instance map

    def visit(self, pat: Pattern, arg: None) -> Pattern:
        if pat in self.var.templates:  # current pattern is a template
            if self.index == 0 and self.var.has_first(pat):  # this template has first instance
                inst: Pattern = self.var.tpl_to_fst[pat]
            else:
                # Instantiate template and copy attributes to instance
                inst = super().visit(pat, arg)
                inst.is_tpl = False
                inst.src_idx = pat.src_idx
                inst.in_tgt = pat.in_tgt
                self.map[pat] = inst  # map template to created instance
                inst.update_pred_succ()
            return inst
        else:
            return pat  # not a template, keep it

    def visit_wildcard(self, wildcard: Wildcard, arg: None) -> Pattern:
        return Wildcard()

    def visit_variable(self, var: Variable, arg: None) -> Pattern:
        return Variable(**var.attrs)

    def visit_const(self, const: Const, arg: None) -> Pattern:
        return Const(const.val_)

    def visit_op(self, op: Op, arg: ArgType) -> Any:
        if isinstance(op, ConcreteOp):
            return op
        elif isinstance(op, OpWithTrait):
            return OpWithTrait(op.trait)
        else:
            raise RuntimeError('Unreachable.')

    def visit_call(self, call: Call, arg: None) -> Pattern:
        op = self.visit(call.op, arg)
        args = [self.visit(p, arg) for p in call.args]
        return Call(op, *args, **call.attrs)

    def visit_tuple(self, tup: Tuple, arg: None) -> Pattern:
        return Tuple(*[self.visit(f, arg) for f in tup.fields])

    def visit_getitem(self, getitem: GetItem, arg: None) -> Pattern:
        return GetItem(self.visit(getitem.tup, arg), getitem.idx)

    def visit_cond(self, cond: Cond, arg: None) -> Pattern:
        return Cond(cond.predicate, self.visit(cond.then_pat, arg),
                    self.visit(cond.else_pat, arg))

    def visit_alt(self, alt: Alt, arg: ArgType) -> Pattern:
        return Alt(*[self.visit(p, arg) for p in alt.pats])

    def visit_match(self, match: Match, arg: ArgType) -> Pattern:
        return Match(match.alt, [self.visit(p, arg) for p in match.clauses])

    def visit_variadic(self, var: Variadic, arg: None) -> Pattern:
        raise RuntimeError('Unreachable.')

    def visit_get_instance(self, get_inst: GetInst, arg: None) -> Pattern:
        return GetInst(get_inst.var, get_inst.tpl, get_inst.idx)
