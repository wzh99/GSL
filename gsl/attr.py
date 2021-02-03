from enum import Enum
from typing import Union, Any, Dict, Tuple, Type, Callable

AttrValueType = Union[bool, int, float, str]


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


AttrConvertible = Union[Attr, AttrValueType, tuple, list, None]


class AnyAttr(Attr):
    """
    Matches any attribute value.
    """
    pass


class ConstAttr(Attr):
    """
    A compile-time constant attribute value.
    """

    def __init__(self, value: AttrValueType):
        self.value = value


class GetNodeAttr(Attr):
    """
    Access attribute from a graph node.
    """

    def __init__(self, node, name: str):
        self.node = node
        self.name = name


class TupleAttr(Attr):
    """
    Create a list attribute expression.
    """

    def __init__(self, *fields):
        self.fields = tuple([to_attr(e) for e in fields])


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


class AttrVisitor:
    def visit(self, attr: Attr) -> Any:
        if isinstance(attr, AnyAttr):
            return self.visit_any(attr)
        elif isinstance(attr, ConstAttr):
            return self.visit_const(attr)
        elif isinstance(attr, GetNodeAttr):
            return self.visit_get_node(attr)
        elif isinstance(attr, TupleAttr):
            return self.visit_tuple(attr)
        elif isinstance(attr, GetItemAttr):
            return self.visit_getitem(attr)
        elif isinstance(attr, BinaryAttr):
            return self.visit_binary(attr)
        else:
            raise RuntimeError('Unknown attribute type.')

    def visit_any(self, a: AnyAttr) -> Any:
        pass

    def visit_const(self, const: ConstAttr) -> Any:
        pass

    def visit_get_node(self, get_node: GetNodeAttr) -> Any:
        pass

    def visit_tuple(self, tup_attr: TupleAttr) -> Any:
        for f in tup_attr.fields:
            self.visit(f)

    def visit_getitem(self, getitem: GetItemAttr) -> Any:
        self.visit(getitem.seq)
        self.visit(getitem.index)

    def visit_binary(self, binary: BinaryAttr) -> Any:
        pass
