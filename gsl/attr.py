import typing as ty
from enum import Enum
from typing import Union, Dict, Type, Callable, Generic, TypeVar, Optional

AttrPrimType = Union[bool, int, float, str]
AttrValueType = Union[AttrPrimType, tuple, list]


class Attr:
    """
    AST for attribute expression.
    """
    value_class = (bool, int, float, str)

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
        self.value = value


class GetAttr(Attr):
    """
    Access attribute from a graph node.
    """

    def __init__(self, pat, name: str):
        from .pat import Pattern
        self.pat: Pattern = pat
        self.name = name


class Range(Attr):
    """
    Produces a tuple with elements in given range.
    """

    def __init__(self, stop: AttrLike, start: AttrLike = None, step: AttrLike = None):
        self.stop = to_attr(stop)
        self.start = to_attr(start)
        self.step = to_attr(step)


class Tuple(Attr):
    """
    Create a list attribute expression.
    """

    def __init__(self, *fields):
        self.fields = [to_attr(e) for e in fields]


class GetItem(Attr):
    """
    Get one item from a tuple attribute with given index.
    """

    def __init__(self, tup: Attr, index: AttrLike):
        self.tup = tup
        self.index = to_attr(index)


class Slice(Attr):
    """
    Create a slice attribute.
    """

    def __init__(self, start: AttrLike = None, stop: AttrLike = None, step: AttrLike = None):
        self.start = to_attr(start)
        self.stop = to_attr(stop)
        self.step = to_attr(step)


class GetSlice(Attr):
    """
    Get slice from a tuple.
    """

    def __init__(self, tup: Attr, slc: Slice):
        self.tup = tup
        self.slc = slc


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
        self.op = uop
        self.attr = to_attr(attr)

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
        self.op = bop
        self.lhs = to_attr(lhs)
        self.rhs = to_attr(rhs)

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
        self.pred = pred
        self.then_br = to_attr(then_br)
        self.else_br = to_attr(else_br)


class Symbol(Attr):
    """
    A language symbol which can be mapped to attribute value.
    """
    pass


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

    def __init__(self, field: AttrLike, index: Optional[Symbol] = None,
                 length: Optional[AttrLike] = None):
        """
        Constructor.

        :param field: Pattern of tuple fields.
        :param index: Symbol mapping to index of tuple field.
        :param length: Attribute expression specifying the length of tuple. In source pattern, it
            will be checked if provided. In target pattern, it is required.
        """
        self.field = to_attr(field)
        self.index = index
        self.len = to_attr(length)


class Reduce(Attr):
    """
    Reduce attribute values in a range.
    """

    reduce_ops = {
        BinaryOp.ADD, BinaryOp.MUL, BinaryOp.AND, BinaryOp.OR, BinaryOp.MAX, BinaryOp.MIN,
    }

    def __init__(self, op: BinaryOp, init: AttrLike, elem: AttrLike, index: Symbol,
                 length: AttrLike):
        """
        Constructor.

        :param op: Binary operator used for reduction.
        :param init: Initial value of reduction.
        :param elem: Pattern of reduced elements.
        :param index:  Symbol mapping to iteration of reduction.
        :param length: Length of reduction.
        """
        if op not in self.reduce_ops:
            raise ValueError(
                'Operator \'{}\' cannot be used for reduction.'.format(op.value)
            )

        self.op = op
        self.init = to_attr(init)
        self.elem = to_attr(elem)
        self.index = index
        self.len = to_attr(length)


class ReduceTuple(Attr):
    """
    Reduce values in a tuple
    """

    def __init__(self, op: BinaryOp, tup: AttrLike, init: AttrLike):
        if op not in Reduce.reduce_ops:
            raise ValueError(
                'Operator \'{}\' cannot be used for reduction.'.format(op.value)
            )

        self.op = op
        self.tup = to_attr(tup)
        self.init = to_attr(init)


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
        elif isinstance(attr, Reduce):
            return self.visit_reduce(attr, arg)
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

    def visit_reduce(self, red: Reduce, arg: ArgType):
        pass

    def visit_reduce_tuple(self, red: ReduceTuple, arg: ArgType):
        self.visit(red.tup, arg)
        self.visit(red.init, arg)
