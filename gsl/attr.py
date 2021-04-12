import sys
import typing as ty
from enum import Enum
from functools import reduce
from typing import Union, Dict, Type, Callable, Generic, TypeVar, Optional, List, Set

AttrPrimType = Union[bool, int, float, str]
AttrValueType = Union[AttrPrimType, tuple, list]


class Attr:
    """
    AST for attribute expression.
    """
    value_class = (bool, int, float, str)

    def __init__(self):
        self.free_sym: Set[Symbol] = set()

    @property
    def sub_expr(self) -> List['Attr']:
        return []

    @property
    def bounded_sym(self) -> List['Symbol']:
        return []

    @property
    def has_free_sym(self):
        return len(self.free_sym) != 0

    def _update_free_sym(self):
        self.free_sym = reduce(
            set.union, map(lambda a: a.free_sym, self.sub_expr), set()
        ).difference(self.bounded_sym)

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
        self.value = value

    def accept(self, visitor: 'AttrVisitor', arg: 'ArgType'):
        return visitor.visit_const(self, arg)


class GetAttr(Attr):
    """
    Access attribute from a graph node.
    """

    def __init__(self, pat, name: str):
        super().__init__()
        from .pat import Pattern
        self.pat: Pattern = pat
        self.name = name
        self.free_sym.update(self.pat.free_sym)

    def accept(self, visitor: 'AttrVisitor', arg: 'ArgType'):
        return visitor.visit_getattr(self, arg)


class Range(Attr):
    """
    Produces a tuple with elements in given range.
    """

    def __init__(self, stop: AttrLike, start: AttrLike = None, step: AttrLike = None):
        super().__init__()
        self.stop = to_attr(stop)
        self.start = to_attr(start)
        self.step = to_attr(step)
        self._update_free_sym()

    @property
    def sub_expr(self) -> List['Attr']:
        return [self.stop, self.start, self.step]

    def accept(self, visitor: 'AttrVisitor', arg: 'ArgType'):
        return visitor.visit_range(self, arg)


class Tuple(Attr):
    """
    Create a list attribute expression.
    """

    def __init__(self, *fields):
        super().__init__()
        self.fields = [to_attr(e) for e in fields]
        self._update_free_sym()

    @property
    def sub_expr(self) -> List['Attr']:
        return self.fields

    def accept(self, visitor: 'AttrVisitor', arg: 'ArgType'):
        return visitor.visit_tuple(self, arg)


class TupleLen(Attr):
    """
    Get length of a tuple.
    """

    def __init__(self, tup: Attr):
        super().__init__()
        self.tup = tup
        self._update_free_sym()

    @property
    def sub_expr(self) -> List['Attr']:
        return [self.tup]

    def accept(self, visitor: 'AttrVisitor', arg: 'ArgType'):
        return visitor.visit_tuple_len(self, arg)


class GetItem(Attr):
    """
    Get one item from a tuple attribute with given index.
    """

    def __init__(self, tup: Attr, index: AttrLike):
        super().__init__()
        self.tup = tup
        self.index = to_attr(index)
        self._update_free_sym()

    @property
    def sub_expr(self) -> List['Attr']:
        return [self.tup, self.index]

    def accept(self, visitor: 'AttrVisitor', arg: 'ArgType'):
        return visitor.visit_getitem(self, arg)


class Slice(Attr):
    """
    Create a slice attribute.
    """

    def __init__(self, start: AttrLike = None, stop: AttrLike = None, step: AttrLike = None):
        super().__init__()
        self.start = to_attr(start)
        self.stop = to_attr(stop)
        self.step = to_attr(step)
        self._update_free_sym()

    @property
    def sub_expr(self) -> List['Attr']:
        return [self.start, self.stop, self.step]

    def accept(self, visitor: 'AttrVisitor', arg: 'ArgType'):
        return visitor.visit_slice(self, arg)


class GetSlice(Attr):
    """
    Get slice from a tuple.
    """

    def __init__(self, tup: Attr, slc: Slice):
        super().__init__()
        self.tup = tup
        self.slc = slc
        self._update_free_sym()

    @property
    def sub_expr(self) -> List['Attr']:
        return [self.tup, self.slc]

    def accept(self, visitor: 'AttrVisitor', arg: 'ArgType'):
        return visitor.visit_getslice(self, arg)


class Reverse(Attr):
    def __init__(self, tup: Attr):
        super().__init__()
        self.tup = tup

    @property
    def sub_expr(self) -> List['Attr']:
        return [self.tup]

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
        self.op = uop
        self.attr = to_attr(attr)
        self._update_free_sym()

    @property
    def sub_expr(self) -> List['Attr']:
        return [self.attr]

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
        self.op = bop
        self.lhs = to_attr(lhs)
        self.rhs = to_attr(rhs)
        self._update_free_sym()

    @property
    def sub_expr(self) -> List['Attr']:
        return [self.lhs, self.rhs]

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
        self.pred = pred
        self.then_br = to_attr(then_br)
        self.else_br = to_attr(else_br)
        self._update_free_sym()

    @property
    def sub_expr(self) -> List['Attr']:
        return [self.pred, self.then_br, self.else_br]

    def accept(self, visitor: 'AttrVisitor', arg: 'ArgType'):
        return visitor.visit_cond(self, arg)


class Match(Attr):
    """
    Evaluate different attribute expression according to matched alternative pattern.
    """

    def __init__(self, alt, clauses: List[AttrLike]):
        super().__init__()
        from .pat import Alt
        self.alt: Alt = alt
        if len(alt.pats) != len(clauses):
            raise ValueError(
                'Expect {} clauses, got {}.'.format(len(alt.pats), len(clauses))
            )
        self.clauses = [to_attr(a) for a in clauses]
        self._update_free_sym()

    @property
    def sub_expr(self) -> List['Attr']:
        return self.clauses

    def accept(self, visitor: 'AttrVisitor', arg: 'ArgType'):
        return visitor.visit_match(self, arg)


class Symbol(Attr):
    """
    A language symbol which can be mapped to attribute value.
    """

    def __init__(self):
        super().__init__()
        self.free_sym = {self}

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
        self.index = Symbol()
        self.field = to_attr(func(self.index))
        self.len = to_attr(length)
        self._update_free_sym()

    @property
    def sub_expr(self) -> List['Attr']:
        return [self.field, self.len]

    @property
    def bounded_sym(self) -> List['Symbol']:
        return [self.index]

    def accept(self, visitor: 'AttrVisitor', arg: 'ArgType'):
        return visitor.visit_variadic(self, arg)


class Map(Attr):
    """Map all elements in a tuple to new values"""

    def __init__(self, tup: AttrLike, func: Callable[[Symbol], AttrLike]):
        super().__init__()
        self.tup = tup
        self.sym = Symbol()
        self.body = to_attr(func(self.sym))
        self._update_free_sym()

    @property
    def sub_expr(self) -> List['Attr']:
        return [self.tup, self.body]

    @property
    def bounded_sym(self) -> List['Symbol']:
        return [self.sym]

    def accept(self, visitor: 'AttrVisitor', arg: 'ArgType'):
        return visitor.visit_map(self, arg)


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
        self.op = op
        self.index = Symbol()
        self.elem = to_attr(func(self.index))
        self.len = to_attr(length)
        self.init = to_attr(reduce_init[op] if init is None else init)
        self._update_free_sym()

    @property
    def sub_expr(self) -> List['Attr']:
        return [self.elem, self.len, self.init]

    @property
    def bounded_sym(self) -> List['Symbol']:
        return [self.index]

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
        self.op = op
        self.tup = to_attr(tup)
        self.init = to_attr(reduce_init[op] if init is None else init)
        self._update_free_sym()

    @property
    def sub_expr(self) -> List['Attr']:
        return [self.tup, self.init]

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
        self.visit(ran.start, arg)
        self.visit(ran.stop, arg)
        self.visit(ran.step, arg)

    def visit_tuple(self, tup_attr: Tuple, arg: ArgType):
        for f in tup_attr.fields:
            self.visit(f, arg)

    def visit_tuple_len(self, tuple_len: TupleLen, arg: ArgType):
        self.visit(tuple_len.tup, arg)

    def visit_getitem(self, getitem: GetItem, arg: ArgType):
        self.visit(getitem.tup, arg)
        self.visit(getitem.index, arg)

    def visit_slice(self, slc: Slice, arg: ArgType):
        self.visit(slc.start, arg)
        self.visit(slc.stop, arg)
        self.visit(slc.step, arg)

    def visit_getslice(self, getslice: GetSlice, arg: ArgType):
        self.visit(getslice.tup, arg)
        self.visit(getslice.slc, arg)

    def visit_reverse(self, rev: Reverse, arg: ArgType):
        self.visit(rev.tup, arg)

    def visit_unary(self, unary: Unary, arg: ArgType):
        self.visit(unary.attr, arg)

    def visit_binary(self, binary: Binary, arg: ArgType):
        self.visit(binary.lhs, arg)
        self.visit(binary.rhs, arg)

    def visit_cond(self, cond: Cond, arg: ArgType):
        self.visit(cond.pred, arg)
        self.visit(cond.then_br, arg)
        self.visit(cond.else_br, arg)

    def visit_match(self, match: Match, arg: ArgType):
        for c in match.clauses:
            self.visit(c, arg)

    def visit_symbol(self, sym: Symbol, arg: ArgType):
        pass

    def visit_variadic(self, var: Variadic, arg: ArgType):
        pass

    def visit_map(self, m: Map, arg: ArgType):
        pass

    def visit_reduce_indexed(self, red: ReduceIndexed, arg: ArgType):
        pass

    def visit_reduce_tuple(self, red: ReduceTuple, arg: ArgType):
        self.visit(red.tup, arg)
        self.visit(red.init, arg)
