from enum import Enum
from typing import Union, Any, Dict, Tuple, Type, Callable, Generic, TypeVar, Optional

AttrPrimType = Union[bool, int, float, str]
AttrValueType = Union[AttrPrimType, tuple, list]


class Attr:
    """
    AST for attribute expression.
    """
    value_class = (bool, int, float, str)

    def __getitem__(self, index):
        return GetItemAttr(self, to_attr(index))

    def __add__(self, other):
        return BinaryAttr(BinaryOp.ADD, self, to_attr(other))

    def __radd__(self, other):
        return BinaryAttr(BinaryOp.ADD, to_attr(other), self)

    def __sub__(self, other):
        return BinaryAttr(BinaryOp.SUB, self, to_attr(other))

    def __rsub__(self, other):
        return BinaryAttr(BinaryOp.SUB, to_attr(other), self)

    def __mul__(self, other):
        return BinaryAttr(BinaryOp.MUL, self, to_attr(other))

    def __rmul__(self, other):
        return BinaryAttr(BinaryOp.MUL, to_attr(other), self)

    def __floordiv__(self, other):
        return BinaryAttr(BinaryOp.FLOOR_DIV, self, to_attr(other))

    def __rfloordiv__(self, other):
        return BinaryAttr(BinaryOp.FLOOR_DIV, to_attr(other), self)

    def max(self, other):
        return BinaryAttr(BinaryOp.MAX, self, to_attr(other))

    def min(self, other):
        return BinaryAttr(BinaryOp.MIN, self, to_attr(other))


AttrConvertible = Union[Attr, AttrValueType, None]


class AnyAttr(Attr):
    """
    Matches any attribute value.
    """
    pass


class ConstAttr(Attr):
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


class TupleAttr(Attr):
    """
    Create a list attribute expression.
    """

    def __init__(self, *fields):
        self.fields = [to_attr(e) for e in fields]


class GetItemAttr(Attr):
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
        return AnyAttr()
    elif isinstance(val, Attr):
        return val
    elif isinstance(val, Attr.value_class):
        return ConstAttr(val)
    elif isinstance(val, (tuple, list)):
        return TupleAttr(*val)
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


class BinaryAttr(Attr):
    """
    Binary expression of attributes..
    """

    def __init__(self, op_name: BinaryOp, lhs: Attr, rhs: Attr):
        self.op = op_name
        self.lhs = lhs
        self.rhs = rhs

    eval_func: Dict[BinaryOp, Dict[Tuple[Type, Type], Callable[[Any, Any], Any]]] = {
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

    def __add__(self, pair: Tuple[Symbol, AttrValueType]):
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


class VariadicAttr(Attr):
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


class SumAttr(Attr):
    """
    Summation of attribute values in a given range
    """

    def __init__(self, attr: Attr, index: Symbol, length: AttrConvertible):
        """
        Constructor.

        :param attr: Pattern of summed elements.
        :param index:  Symbol mapping to iteration of summation.
        :param length: Attribute expression specifying the length of summation.
        """
        self.attr = attr
        self.index = index
        self.len = None if length is None else to_attr(length)


ArgType = TypeVar('ArgType')


class AttrVisitor(Generic[ArgType]):
    def visit(self, attr: Attr, arg: ArgType) -> Any:
        if isinstance(attr, AnyAttr):
            return self.visit_any(attr, arg)
        elif isinstance(attr, ConstAttr):
            return self.visit_const(attr, arg)
        elif isinstance(attr, GetAttr):
            return self.visit_getattr(attr, arg)
        elif isinstance(attr, TupleAttr):
            return self.visit_tuple(attr, arg)
        elif isinstance(attr, GetItemAttr):
            return self.visit_getitem(attr, arg)
        elif isinstance(attr, BinaryAttr):
            return self.visit_binary(attr, arg)
        elif isinstance(attr, Symbol):
            return self.visit_symbol(attr, arg)
        elif isinstance(attr, VariadicAttr):
            return self.visit_variadic(attr, arg)
        elif isinstance(attr, SumAttr):
            return self.visit_sum(attr, arg)
        else:
            raise RuntimeError('Unknown attribute type.')

    def visit_any(self, a: AnyAttr, arg: ArgType) -> Any:
        pass

    def visit_const(self, const: ConstAttr, arg: ArgType) -> Any:
        pass

    def visit_getattr(self, get_attr: GetAttr, arg: ArgType) -> Any:
        pass

    def visit_tuple(self, tup_attr: TupleAttr, arg: ArgType) -> Any:
        for f in tup_attr.fields:
            self.visit(f, arg)

    def visit_getitem(self, getitem: GetItemAttr, arg: ArgType) -> Any:
        self.visit(getitem.seq, arg)
        self.visit(getitem.index, arg)

    def visit_binary(self, binary: BinaryAttr, arg: ArgType) -> Any:
        self.visit(binary.lhs, arg)
        self.visit(binary.rhs, arg)

    def visit_symbol(self, sym: Symbol, arg: ArgType) -> Any:
        pass

    def visit_variadic(self, var: VariadicAttr, arg: ArgType) -> Any:
        pass

    def visit_sum(self, s: SumAttr, arg: ArgType) -> Any:
        pass
