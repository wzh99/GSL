from enum import Enum
from typing import Union, Dict, Type, Callable, Generic, TypeVar, Optional

import typing as ty

AttrPrimType = Union[bool, int, float, str]
AttrValueType = Union[AttrPrimType, tuple, list]


class Attr:
    """
    AST for attribute expression.
    """
    value_class = (bool, int, float, str)

    def __getitem__(self, index):
        return GetItem(self, to_attr(index))

    def __add__(self, other):
        return Binary(BinaryOp.ADD, self, to_attr(other))

    def __radd__(self, other):
        return Binary(BinaryOp.ADD, to_attr(other), self)

    def __sub__(self, other):
        return Binary(BinaryOp.SUB, self, to_attr(other))

    def __rsub__(self, other):
        return Binary(BinaryOp.SUB, to_attr(other), self)

    def __mul__(self, other):
        return Binary(BinaryOp.MUL, self, to_attr(other))

    def __rmul__(self, other):
        return Binary(BinaryOp.MUL, to_attr(other), self)

    def __floordiv__(self, other):
        return Binary(BinaryOp.FLOOR_DIV, self, to_attr(other))

    def __rfloordiv__(self, other):
        return Binary(BinaryOp.FLOOR_DIV, to_attr(other), self)

    def max(self, other):
        return Binary(BinaryOp.MAX, self, to_attr(other))

    def min(self, other):
        return Binary(BinaryOp.MIN, self, to_attr(other))


AttrConvertible = Union[Attr, AttrValueType, None]


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


class Tuple(Attr):
    """
    Create a list attribute expression.
    """

    def __init__(self, *fields):
        self.fields = [to_attr(e) for e in fields]


class GetItem(Attr):
    """
    Get item from a tuple attribute with given index.
    """

    def __init__(self, seq: Attr, index: Attr):
        self.seq = seq
        self.index = index


def to_attr(val: AttrConvertible) -> Attr:
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


class BinaryOp(Enum):
    ADD = '+'
    SUB = '-'
    MUL = '*'
    FLOOR_DIV = '//'
    MAX = 'max'
    MIN = 'min'


class Binary(Attr):
    """
    Binary expression of attributes..
    """

    def __init__(self, op_name: BinaryOp, lhs: Attr, rhs: Attr):
        self.op = op_name
        self.lhs = lhs
        self.rhs = rhs

    eval_func: Dict[BinaryOp, Dict[ty.Tuple[Type, Type], Callable[[ty.Any, ty.Any], ty.Any]]] = {
        BinaryOp.ADD: {
            (int, int): int.__add__,
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
        BinaryOp.MAX: {
            (int, int): max,
        },
        BinaryOp.MIN: {
            (int, int): min,
        },
    }


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

    def __init__(self, attr: AttrConvertible, index: Optional[Symbol] = None,
                 length: Optional[AttrConvertible] = None):
        """
        Constructor.

        :param attr: Pattern of tuple fields.
        :param index: Symbol mapping to index of tuple field.
        :param length: Attribute expression specifying the length of tuple. In source pattern, it
            will be checked if provided. In target pattern, it is required.
        """
        self.attr = to_attr(attr)
        self.index = index
        self.len = to_attr(length)


class Sum(Attr):
    """
    Summation of attribute values in a given range
    """

    def __init__(self, elem: AttrConvertible, index: Symbol, length: AttrConvertible):
        """
        Constructor.

        :param elem: Pattern of summed elements.
        :param index:  Symbol mapping to iteration of summation.
        :param length: Attribute expression specifying the length of summation.
        """
        self.elem = to_attr(elem)
        self.index = index
        self.len = to_attr(length)


ArgType = TypeVar('ArgType')


class AttrVisitor(Generic[ArgType]):
    def visit(self, attr: Attr, arg: ArgType) -> ty.Any:
        if isinstance(attr, Any):
            return self.visit_any(attr, arg)
        elif isinstance(attr, Const):
            return self.visit_const(attr, arg)
        elif isinstance(attr, GetAttr):
            return self.visit_getattr(attr, arg)
        elif isinstance(attr, Tuple):
            return self.visit_tuple(attr, arg)
        elif isinstance(attr, GetItem):
            return self.visit_getitem(attr, arg)
        elif isinstance(attr, Binary):
            return self.visit_binary(attr, arg)
        elif isinstance(attr, Symbol):
            return self.visit_symbol(attr, arg)
        elif isinstance(attr, Variadic):
            return self.visit_variadic(attr, arg)
        elif isinstance(attr, Sum):
            return self.visit_sum(attr, arg)
        else:
            raise RuntimeError('Unknown attribute type.')

    def visit_any(self, a: Any, arg: ArgType) -> ty.Any:
        pass

    def visit_const(self, const: Const, arg: ArgType) -> ty.Any:
        pass

    def visit_getattr(self, get_attr: GetAttr, arg: ArgType) -> ty.Any:
        pass

    def visit_tuple(self, tup_attr: Tuple, arg: ArgType) -> ty.Any:
        for f in tup_attr.fields:
            self.visit(f, arg)

    def visit_getitem(self, getitem: GetItem, arg: ArgType) -> ty.Any:
        self.visit(getitem.seq, arg)
        self.visit(getitem.index, arg)

    def visit_binary(self, binary: Binary, arg: ArgType) -> ty.Any:
        self.visit(binary.lhs, arg)
        self.visit(binary.rhs, arg)

    def visit_symbol(self, sym: Symbol, arg: ArgType) -> ty.Any:
        pass

    def visit_variadic(self, var: Variadic, arg: ArgType) -> ty.Any:
        pass

    def visit_sum(self, s: Sum, arg: ArgType) -> ty.Any:
        pass
