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


AttrLike = Union[Attr, AttrValueType, None]


class Any(Attr):
    """
    Matches any attribute value.
    """
    pass


class Const(Attr):
    """
    A compile-time constant attribute value.
    """

    def __init__(self, value: AttrPrimType):
        super().__init__()
        self.value = value


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


def to_attr(val: AttrLike) -> Attr:
    """
    Create an attribute expression with given value.

    :param val: All types of values that are or can be converted to an attribute expression.
    :return: Attribute expression created from given value.
    """
    if val is None:
        return Any()
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
    EQ = '='
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


class Symbol(Attr):
    """
    A language symbol which can be mapped to attribute value.
    """

    def __init__(self):
        super().__init__()
        self.free_sym = {self}


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


reduce_ops = {
    BinaryOp.ADD, BinaryOp.MUL, BinaryOp.AND, BinaryOp.OR, BinaryOp.MAX, BinaryOp.MIN,
}


class ReduceIndexed(Attr):
    """
    Reduce indexed attribute values with a certain length.
    """

    def __init__(self, op: BinaryOp, init: AttrLike, func: Callable[[Symbol], AttrLike],
                 length: AttrLike):
        """
        Constructor.

        :param op: Binary operator used for reduction.
        :param init: Initial value of reduction.
        :param func: How index is mapped to each element.
        :param length: Length of reduction.
        """
        if op not in reduce_ops:
            raise ValueError(
                'Operator \'{}\' cannot be used for reduction.'.format(op.value)
            )

        super().__init__()
        self.op = op
        self.init = to_attr(init)
        self.index = Symbol()
        self.elem = to_attr(func(self.index))
        self.len = to_attr(length)
        self._update_free_sym()

    @property
    def sub_expr(self) -> List['Attr']:
        return [self.init, self.elem, self.len]

    @property
    def bounded_sym(self) -> List['Symbol']:
        return [self.index]


class ReduceTuple(Attr):
    """
    Reduce values in a tuple.
    """

    def __init__(self, op: BinaryOp, tup: AttrLike, init: AttrLike):
        if op not in reduce_ops:
            raise ValueError(
                'Operator \'{}\' cannot be used for reduction.'.format(op.value)
            )

        super().__init__()
        self.op = op
        self.tup = to_attr(tup)
        self.init = to_attr(init)
        self._update_free_sym()

    @property
    def sub_expr(self) -> List['Attr']:
        return [self.tup, self.init]


ArgType = TypeVar('ArgType')
RetType = TypeVar('RetType')


class AttrVisitor(Generic[ArgType, RetType]):
    def visit(self, attr: Attr, arg: ArgType) -> RetType:
        if isinstance(attr, Any):
            return self.visit_any(attr, arg)
        elif isinstance(attr, Const):
            return self.visit_const(attr, arg)
        elif isinstance(attr, GetAttr):
            return self.visit_getattr(attr, arg)
        elif isinstance(attr, Range):
            return self.visit_range(attr, arg)
        elif isinstance(attr, Tuple):
            return self.visit_tuple(attr, arg)
        elif isinstance(attr, TupleLen):
            return self.visit_tuple_len(attr, arg)
        elif isinstance(attr, GetItem):
            return self.visit_getitem(attr, arg)
        elif isinstance(attr, Slice):
            return self.visit_slice(attr, arg)
        elif isinstance(attr, GetSlice):
            return self.visit_getslice(attr, arg)
        elif isinstance(attr, Binary):
            return self.visit_binary(attr, arg)
        elif isinstance(attr, Cond):
            return self.visit_cond(attr, arg)
        elif isinstance(attr, Symbol):
            return self.visit_symbol(attr, arg)
        elif isinstance(attr, Variadic):
            return self.visit_variadic(attr, arg)
        elif isinstance(attr, Map):
            return self.visit_map(attr, arg)
        elif isinstance(attr, ReduceIndexed):
            return self.visit_reduce_indexed(attr, arg)
        elif isinstance(attr, ReduceTuple):
            return self.visit_reduce_tuple(attr, arg)
        else:
            raise RuntimeError('Unknown attribute type.')

    def visit_any(self, a: Any, arg: ArgType):
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

    def visit_unary(self, unary: Unary, arg: ArgType):
        self.visit(unary.attr, arg)

    def visit_binary(self, binary: Binary, arg: ArgType):
        self.visit(binary.lhs, arg)
        self.visit(binary.rhs, arg)

    def visit_cond(self, cond: Cond, arg: ArgType):
        self.visit(cond.pred, arg)
        self.visit(cond.then_br, arg)
        self.visit(cond.else_br, arg)

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
