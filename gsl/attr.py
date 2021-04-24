import sys
import typing as ty
from enum import Enum
from typing import Union, Dict, Type, Callable, Generic, TypeVar, Optional, List, Set

AttrPrimType = Union[bool, int, float, str]
AttrValueType = Union[AttrPrimType, tuple, list]


class Attr:
    """
    AST for attribute expression.
    """
    value_class = (bool, int, float, str)

    def __init__(self):
        self.free_sym_: Set[Symbol] = set()
        self.ref_cnt_ = 0

    @property
    def sub_expr(self) -> List['Attr']:
        return []

    @property
    def bounded_sym(self) -> List['Symbol']:
        return []

    @property
    def has_free_sym(self):
        return len(self.free_sym_) != 0

    def _update_free_sym(self):
        for a in self.sub_expr:
            self.free_sym_.update(a.free_sym_)
        self.free_sym_.difference_update(self.bounded_sym)

    def inc_ref_cnt(self):
        self.ref_cnt_ += 1

    def __hash__(self):
        return hash(id(self))

    def __getitem__(self, index: 'AttrLike'):
        if isinstance(index, Slice):
            return GetSlice(self, index)
        else:
            return GetItem(self, index)

    def __neg__(self):
        return Unary(UnaryOp.NEG, self)

    def __invert__(self):
        return Unary(UnaryOp.NOT, self)

    def __add__(self, other: 'AttrLike'):
        return Binary(BinaryOp.ADD, self, other)

    def __radd__(self, other: 'AttrLike'):
        return Binary(BinaryOp.ADD, other, self)

    def __sub__(self, other: 'AttrLike'):
        return Binary(BinaryOp.SUB, self, other)

    def __rsub__(self, other: 'AttrLike'):
        return Binary(BinaryOp.SUB, other, self)

    def __mul__(self, other: 'AttrLike'):
        return Binary(BinaryOp.MUL, self, other)

    def __rmul__(self, other: 'AttrLike'):
        return Binary(BinaryOp.MUL, other, self)

    def __floordiv__(self, other: 'AttrLike'):
        return Binary(BinaryOp.FLOOR_DIV, self, other)

    def __rfloordiv__(self, other: 'AttrLike'):
        return Binary(BinaryOp.FLOOR_DIV, other, self)

    def __mod__(self, other: 'AttrLike'):
        return Binary(BinaryOp.MOD, self, other)

    def __rmod__(self, other: 'AttrLike'):
        return Binary(BinaryOp.MOD, other, self)

    def __eq__(self, other: 'AttrLike'):
        return Binary(BinaryOp.EQ, self, other)

    def __ne__(self, other: 'AttrLike'):
        return Binary(BinaryOp.NE, self, other)

    def __lt__(self, other: 'AttrLike'):
        return Binary(BinaryOp.LT, self, other)

    def __le__(self, other: 'AttrLike'):
        return Binary(BinaryOp.LE, self, other)

    def __gt__(self, other: 'AttrLike'):
        return Binary(BinaryOp.GT, self, other)

    def __ge__(self, other: 'AttrLike'):
        return Binary(BinaryOp.GE, self, other)

    def __and__(self, other: 'AttrLike'):
        return Binary(BinaryOp.AND, self, other)

    def __rand__(self, other: 'AttrLike'):
        return Binary(BinaryOp.AND, other, self)

    def __or__(self, other: 'AttrLike'):
        return Binary(BinaryOp.OR, self, other)

    def __ror__(self, other: 'AttrLike'):
        return Binary(BinaryOp.OR, other, self)

    def max(self, other: 'AttrLike'):
        return Binary(BinaryOp.MAX, self, other)

    def min(self, other: 'AttrLike'):
        return Binary(BinaryOp.MIN, self, other)

    def accept(self, visitor: 'AttrVisitor', arg: 'ArgType'):
        raise NotImplementedError()


AttrLike = Union[Attr, AttrValueType, None]


class NoneAttr(Attr):
    """
    Evaluate to `None`, and matches any attribute.
    """

    def accept(self, visitor: 'AttrVisitor', arg: 'ArgType'):
        return visitor.visit_none(self, arg)


class Const(Attr):
    """
    A compile-time constant attribute value.
    """

    def __init__(self, value: AttrPrimType):
        super().__init__()
        self.value_ = value

    def accept(self, visitor: 'AttrVisitor', arg: 'ArgType'):
        return visitor.visit_const(self, arg)


class GetAttr(Attr):
    """
    Access attribute from a graph node.
    """

    def __init__(self, pat, name: str):
        super().__init__()
        from .pat import Pattern
        self.pat_: Pattern = pat
        self.name_ = name
        self.free_sym_.update(self.pat_.free_sym_)

    def accept(self, visitor: 'AttrVisitor', arg: 'ArgType'):
        return visitor.visit_getattr(self, arg)


class Range(Attr):
    """
    Produces a tuple with elements in given range.
    """

    def __init__(self, stop: AttrLike, start: AttrLike = None, step: AttrLike = None):
        super().__init__()
        self.stop_ = to_attr(stop)
        self.start_ = to_attr(start)
        self.step_ = to_attr(step)
        self._update_free_sym()

    @property
    def sub_expr(self) -> List['Attr']:
        return [self.stop_, self.start_, self.step_]

    def accept(self, visitor: 'AttrVisitor', arg: 'ArgType'):
        return visitor.visit_range(self, arg)


class Tuple(Attr):
    """
    Create a list attribute expression.
    """

    def __init__(self, *fields):
        super().__init__()
        self.fields_ = [to_attr(e) for e in fields]
        self._update_free_sym()

    @property
    def sub_expr(self) -> List['Attr']:
        return self.fields_

    def accept(self, visitor: 'AttrVisitor', arg: 'ArgType'):
        return visitor.visit_tuple(self, arg)


class TupleLen(Attr):
    """
    Get length of a tuple.
    """

    def __init__(self, tup: Attr):
        super().__init__()
        self.tup_ = tup
        self._update_free_sym()

    @property
    def sub_expr(self) -> List['Attr']:
        return [self.tup_]

    def accept(self, visitor: 'AttrVisitor', arg: 'ArgType'):
        return visitor.visit_tuple_len(self, arg)


class GetItem(Attr):
    """
    Get one item from a tuple attribute with given index.
    """

    def __init__(self, tup: Attr, index: AttrLike):
        super().__init__()
        self.tup_ = tup
        self.index_ = to_attr(index)
        self._update_free_sym()

    @property
    def sub_expr(self) -> List['Attr']:
        return [self.tup_, self.index_]

    def accept(self, visitor: 'AttrVisitor', arg: 'ArgType'):
        return visitor.visit_getitem(self, arg)


class Slice(Attr):
    """
    Create a slice attribute.
    """

    def __init__(self, start: AttrLike = None, stop: AttrLike = None, step: AttrLike = None):
        super().__init__()
        self.start_ = to_attr(start)
        self.stop_ = to_attr(stop)
        self.step_ = to_attr(step)
        self._update_free_sym()

    @property
    def sub_expr(self) -> List['Attr']:
        return [self.start_, self.stop_, self.step_]

    def accept(self, visitor: 'AttrVisitor', arg: 'ArgType'):
        return visitor.visit_slice(self, arg)


class GetSlice(Attr):
    """
    Get slice from a tuple.
    """

    def __init__(self, tup: Attr, slc: Slice):
        super().__init__()
        self.tup_ = tup
        self.slc_ = slc
        self._update_free_sym()

    @property
    def sub_expr(self) -> List['Attr']:
        return [self.tup_, self.slc_]

    def accept(self, visitor: 'AttrVisitor', arg: 'ArgType'):
        return visitor.visit_getslice(self, arg)


class In(Attr):
    """
    Check if a value is an element of a tuple.
    """

    def __init__(self, val: AttrLike, tup: AttrLike):
        super().__init__()
        self.val_ = to_attr(val)
        self.tup_ = to_attr(tup)
        self._update_free_sym()

    @property
    def sub_expr(self) -> List['Attr']:
        return [self.val_, self.tup_]

    def accept(self, visitor: 'AttrVisitor', arg: 'ArgType'):
        return visitor.visit_in(self, arg)


class Reverse(Attr):
    def __init__(self, tup: Attr):
        super().__init__()
        self.tup_ = tup

    @property
    def sub_expr(self) -> List['Attr']:
        return [self.tup_]

    def accept(self, visitor: 'AttrVisitor', arg: 'ArgType'):
        return visitor.visit_reverse(self, arg)


def to_attr(val: AttrLike) -> Attr:
    """
    Create an attribute expression with given value.

    :param val: All types of values that are or can be converted to an attribute expression.
    :return: Attribute expression created from given value.
    """
    if val is None:
        return NoneAttr()
    elif isinstance(val, Attr):
        return val
    elif isinstance(val, Attr.value_class):
        return Const(val)
    elif isinstance(val, (tuple, list)):
        return Tuple(*val)
    else:
        raise TypeError(
            'Cannot convert value of type \'{}\' to attribute.'.format(val.__class__)
        )


class UnaryOp(Enum):
    NEG = '-'
    NOT = '~'


class Unary(Attr):
    """
    Unary expression of attribute.
    """

    def __init__(self, uop: UnaryOp, attr: AttrLike):
        super().__init__()
        self.op_ = uop
        self.attr_ = to_attr(attr)
        self._update_free_sym()

    @property
    def sub_expr(self) -> List['Attr']:
        return [self.attr_]

    def accept(self, visitor: 'AttrVisitor', arg: 'ArgType'):
        return visitor.visit_unary(self, arg)

    eval_funcs: Dict[UnaryOp, Dict[Type, Callable[[ty.Any], ty.Any]]] = {
        UnaryOp.NEG: {
            int: int.__neg__,
        },
        UnaryOp.NOT: {
            bool: lambda b: not b,
        },
    }


class BinaryOp(Enum):
    ADD = '+'
    SUB = '-'
    MUL = '*'
    FLOOR_DIV = '//'
    MOD = '%'
    MAX = 'max'
    MIN = 'min'
    EQ = '=='
    NE = '!='
    LT = '<'
    LE = '<='
    GT = '>'
    GE = '>='
    AND = '&'
    OR = '|'


class Binary(Attr):
    """
    Binary expression of attributes..
    """

    def __init__(self, bop: BinaryOp, lhs: AttrLike, rhs: AttrLike):
        super().__init__()
        self.op_ = bop
        self.lhs_ = to_attr(lhs)
        self.rhs_ = to_attr(rhs)
        self._update_free_sym()

    @property
    def sub_expr(self) -> List['Attr']:
        return [self.lhs_, self.rhs_]

    def accept(self, visitor: 'AttrVisitor', arg: 'ArgType'):
        return visitor.visit_binary(self, arg)

    eval_func: Dict[BinaryOp, Dict[ty.Tuple[Type, Type], Callable[[ty.Any, ty.Any], ty.Any]]] = {
        BinaryOp.ADD: {
            (int, int): int.__add__,
            (tuple, tuple): tuple.__add__,
        },
        BinaryOp.SUB: {
            (int, int): int.__sub__,
        },
        BinaryOp.MUL: {
            (int, int): int.__mul__,
        },
        BinaryOp.FLOOR_DIV: {
            (int, int): int.__floordiv__,
        },
        BinaryOp.MOD: {
            (int, int): int.__mod__,
        },
        BinaryOp.MAX: {
            (int, int): max,
        },
        BinaryOp.MIN: {
            (int, int): min,
        },
        BinaryOp.EQ: {
            (int, int): int.__eq__,
            (bool, bool): bool.__eq__,
            (str, str): str.__eq__,
        },
        BinaryOp.NE: {
            (int, int): int.__ne__,
            (bool, bool): bool.__ne__,
            (str, str): str.__ne__,
        },
        BinaryOp.LT: {
            (int, int): int.__lt__,
        },
        BinaryOp.LE: {
            (int, int): int.__le__,
        },
        BinaryOp.GT: {
            (int, int): int.__gt__,
        },
        BinaryOp.GE: {
            (int, int): int.__ge__,
        },
        BinaryOp.AND: {
            (bool, bool): bool.__and__,
        },
        BinaryOp.OR: {
            (bool, bool): bool.__or__,
        },
    }


class Cond(Attr):
    """
    Condition (if-else) attribute expression.
    """

    def __init__(self, pred: Attr, then_br: AttrLike, else_br: AttrLike):
        super().__init__()
        self.pred_ = pred
        self.then_br_ = to_attr(then_br)
        self.else_br_ = to_attr(else_br)
        self._update_free_sym()

    @property
    def sub_expr(self) -> List['Attr']:
        return [self.pred_, self.then_br_, self.else_br_]

    def accept(self, visitor: 'AttrVisitor', arg: 'ArgType'):
        return visitor.visit_cond(self, arg)


class Match(Attr):
    """
    Evaluate different attribute expression according to matched alternative pattern.
    """

    def __init__(self, alt, clauses: List[AttrLike]):
        super().__init__()
        from .pat import Alt
        self.alt_: Alt = alt
        if len(alt.pats_) != len(clauses):
            raise ValueError(
                'Expect {} clauses, got {}.'.format(len(alt.pats_), len(clauses))
            )
        self.clauses_ = [to_attr(a) for a in clauses]
        self._update_free_sym()

    @property
    def sub_expr(self) -> List['Attr']:
        return self.clauses_

    def accept(self, visitor: 'AttrVisitor', arg: 'ArgType'):
        return visitor.visit_match(self, arg)


class LayoutRemap(Attr):
    """
    Produce index tuple that performs layout remapping.
    """

    def __init__(self, src: AttrLike, tgt: AttrLike):
        super().__init__()
        self.src_ = to_attr(src)
        self.tgt_ = to_attr(tgt)
        self._update_free_sym()

    @property
    def sub_expr(self) -> List['Attr']:
        return [self.src_, self.tgt_]

    def accept(self, visitor: 'AttrVisitor', arg: 'ArgType'):
        return visitor.visit_layout_remap(self, arg)


class Symbol(Attr):
    """
    A language symbol which can be mapped to attribute value.
    """

    def __init__(self):
        super().__init__()
        self.free_sym_ = {self}

    def accept(self, visitor: 'AttrVisitor', arg: 'ArgType'):
        return visitor.visit_symbol(self, arg)


class Env:
    """
    Environment, mapping from symbol to attribute value.
    """

    def __init__(self, prev=None, symbol: Optional[Symbol] = None,
                 value: Optional[AttrValueType] = None):
        self.prev: Optional[Env] = prev
        self.symbol = symbol
        self.value = value

    def __add__(self, pair: ty.Tuple[Symbol, AttrValueType]):
        return Env(prev=self, symbol=pair[0], value=pair[1])

    def __getitem__(self, sym: Symbol) -> Optional[AttrValueType]:
        env = self
        while env.symbol is not None:
            if env.symbol is sym:
                return env.value
            else:
                env = env.prev
        return None

    def __contains__(self, sym: Symbol) -> bool:
        return self[sym] is not None


class Variadic(Attr):
    """
    A tuple that can accept any number of fields, each with similar pattern.
    """

    def __init__(self, func: Callable[[Symbol], AttrLike], length: Optional[AttrLike] = None):
        """
        Constructor.

        :param func: How index symbol maps to each tuple field
        :param length: Attribute expression specifying the length of tuple. In source pattern, it
            will be checked if provided. In target pattern, it is required.
        """
        super().__init__()
        self.index_ = Symbol()
        self.field_ = to_attr(func(self.index_))
        self.len_ = to_attr(length)
        self._update_free_sym()

    @property
    def sub_expr(self) -> List['Attr']:
        return [self.field_, self.len_]

    @property
    def bounded_sym(self) -> List['Symbol']:
        return [self.index_]

    def accept(self, visitor: 'AttrVisitor', arg: 'ArgType'):
        return visitor.visit_variadic(self, arg)


class Map(Attr):
    """Map all elements in a tuple to new values"""

    def __init__(self, tup: AttrLike, func: Callable[[Symbol], AttrLike]):
        super().__init__()
        self.tup_ = tup
        self.sym_ = Symbol()
        self.body_ = to_attr(func(self.sym_))
        self._update_free_sym()

    @property
    def sub_expr(self) -> List['Attr']:
        return [self.tup_, self.body_]

    @property
    def bounded_sym(self) -> List['Symbol']:
        return [self.sym_]

    def accept(self, visitor: 'AttrVisitor', arg: 'ArgType'):
        return visitor.visit_map(self, arg)


class Zip(Attr):
    """
    Zip one or more tuples.
    """

    def __init__(self, tuples: List[AttrLike]):
        super().__init__()
        self.tuples_ = [to_attr(tup) for tup in tuples]

    @property
    def sub_expr(self) -> List['Attr']:
        return self.tuples_

    def accept(self, visitor: 'AttrVisitor', arg: 'ArgType'):
        return visitor.visit_zip(self, arg)


reduce_init: Dict[BinaryOp, ty.Any] = {
    BinaryOp.ADD: 0, BinaryOp.MUL: 1, BinaryOp.AND: True, BinaryOp.OR: False,
    BinaryOp.MAX: 0, BinaryOp.MIN: sys.maxsize,
}


class ReduceIndexed(Attr):
    """
    Reduce indexed attribute values with a certain length.
    """

    def __init__(self, op: BinaryOp, func: Callable[[Symbol], AttrLike], length: AttrLike,
                 init: Optional[AttrLike] = None):
        """
        Constructor.

        :param op: Binary operator used for reduction.
        :param func: How index is mapped to each element.
        :param length: Length of reduction.
        :param init: Initial value for reduction
        """
        if op not in reduce_init:
            raise ValueError(
                'Operator \'{}\' cannot be used for reduction.'.format(op.value)
            )

        super().__init__()
        self.op_ = op
        self.index_ = Symbol()
        self.elem_ = to_attr(func(self.index_))
        self.len_ = to_attr(length)
        self.init_ = to_attr(reduce_init[op] if init is None else init)
        self._update_free_sym()

    @property
    def sub_expr(self) -> List['Attr']:
        return [self.elem_, self.len_, self.init_]

    @property
    def bounded_sym(self) -> List['Symbol']:
        return [self.index_]

    def accept(self, visitor: 'AttrVisitor', arg: 'ArgType'):
        return visitor.visit_reduce_indexed(self, arg)


class ReduceTuple(Attr):
    """
    Reduce values in a tuple.
    """

    def __init__(self, op: BinaryOp, tup: AttrLike, init: Optional[AttrLike] = None):
        if op not in reduce_init:
            raise ValueError(
                'Operator \'{}\' cannot be used for reduction.'.format(op.value)
            )

        super().__init__()
        self.op_ = op
        self.tup_ = to_attr(tup)
        self.init = to_attr(reduce_init[op] if init is None else init)
        self._update_free_sym()

    @property
    def sub_expr(self) -> List['Attr']:
        return [self.tup_, self.init]

    def accept(self, visitor: 'AttrVisitor', arg: 'ArgType'):
        return visitor.visit_reduce_tuple(self, arg)


ArgType = TypeVar('ArgType')
RetType = TypeVar('RetType')


class AttrVisitor(Generic[ArgType, RetType]):
    def visit(self, attr: Attr, arg: ArgType) -> RetType:
        return attr.accept(self, arg)

    def visit_none(self, n: NoneAttr, arg: ArgType):
        pass

    def visit_const(self, const: Const, arg: ArgType):
        pass

    def visit_getattr(self, get_attr: GetAttr, arg: ArgType):
        pass

    def visit_range(self, ran: Range, arg: ArgType):
        self.visit(ran.start_, arg)
        self.visit(ran.stop_, arg)
        self.visit(ran.step_, arg)

    def visit_tuple(self, tup_attr: Tuple, arg: ArgType):
        for f in tup_attr.fields_:
            self.visit(f, arg)

    def visit_tuple_len(self, tuple_len: TupleLen, arg: ArgType):
        self.visit(tuple_len.tup_, arg)

    def visit_getitem(self, getitem: GetItem, arg: ArgType):
        self.visit(getitem.tup_, arg)
        self.visit(getitem.index_, arg)

    def visit_slice(self, slc: Slice, arg: ArgType):
        self.visit(slc.start_, arg)
        self.visit(slc.stop_, arg)
        self.visit(slc.step_, arg)

    def visit_getslice(self, getslice: GetSlice, arg: ArgType):
        self.visit(getslice.tup_, arg)
        self.visit(getslice.slc_, arg)

    def visit_in(self, in_tup: In, arg: ArgType):
        self.visit(in_tup.val_, arg)
        self.visit(in_tup.tup_, arg)

    def visit_reverse(self, rev: Reverse, arg: ArgType):
        self.visit(rev.tup_, arg)

    def visit_unary(self, unary: Unary, arg: ArgType):
        self.visit(unary.attr_, arg)

    def visit_binary(self, binary: Binary, arg: ArgType):
        self.visit(binary.lhs_, arg)
        self.visit(binary.rhs_, arg)

    def visit_cond(self, cond: Cond, arg: ArgType):
        self.visit(cond.pred_, arg)
        self.visit(cond.then_br_, arg)
        self.visit(cond.else_br_, arg)

    def visit_match(self, match: Match, arg: ArgType):
        for c in match.clauses_:
            self.visit(c, arg)

    def visit_layout_remap(self, remap: LayoutRemap, arg: ArgType):
        self.visit(remap.src_, arg)
        self.visit(remap.tgt_, arg)

    def visit_symbol(self, sym: Symbol, arg: ArgType):
        pass

    def visit_variadic(self, var: Variadic, arg: ArgType):
        pass

    def visit_map(self, m: Map, arg: ArgType):
        pass

    def visit_zip(self, z: Zip, arg: ArgType):
        for tup in z.tuples_:
            self.visit(tup, arg)

    def visit_reduce_indexed(self, red: ReduceIndexed, arg: ArgType):
        pass

    def visit_reduce_tuple(self, red: ReduceTuple, arg: ArgType):
        self.visit(red.tup_, arg)
        self.visit(red.init, arg)
