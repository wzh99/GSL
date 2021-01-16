from enum import Enum
from typing import Union, List, Any

AttrValueType = Union[bool, int, float, str]


class AttrExpr:
    """
    AST for attribute expression.
    """
    value_class = (bool, int, float, str)

    def __getitem__(self, index: int):
        return GetItemAttr(self, index)

    def __add__(self, other):
        return BinaryExpr(BinaryOp.ADD, self, to_attr(other))

    def __radd__(self, other):
        return BinaryExpr(BinaryOp.ADD, to_attr(other), self)

    def __sub__(self, other):
        return BinaryExpr(BinaryOp.SUB, self, to_attr(other))

    def __rsub__(self, other):
        return BinaryExpr(BinaryOp.SUB, to_attr(other), self)

    def __mul__(self, other):
        return BinaryExpr(BinaryOp.MUL, self, to_attr(other))

    def __rmul__(self, other):
        return BinaryExpr(BinaryOp.MUL, to_attr(other), self)

    def __floordiv__(self, other):
        return BinaryExpr(BinaryOp.FLOOR_DIV, self, to_attr(other))

    def __rfloordiv__(self, other):
        return BinaryExpr(BinaryOp.FLOOR_DIV, to_attr(other), self)

    def max(self, other):
        return BinaryExpr(BinaryOp.MAX, self, to_attr(other))

    def min(self, other):
        return BinaryExpr(BinaryOp.MIN, self, to_attr(other))


class AnyAttr(AttrExpr):
    """
    Matches any attribute value.
    """
    pass


class ConstAttr(AttrExpr):
    """
    A compile-time constant attribute value.
    """

    def __init__(self, value: AttrValueType):
        self.value = value


class GetAttr(AttrExpr):
    """
    Access attribute from a graph node.
    """

    def __init__(self, node, name: str):
        self.node = node
        self.name = name


class ListAttr(AttrExpr):
    """
    Create a list attribute expression.
    """

    def __init__(self, fields: List[Union[AttrExpr, AttrValueType]]):
        self.fields = [to_attr(e) for e in fields]


class TupleAttr(AttrExpr):
    """
    Create a list attribute expression.
    """

    def __init__(self, *fields):
        self.fields = tuple([to_attr(e) for e in fields])


class GetItemAttr(AttrExpr):
    """
    Get item from a list or tuple attribute with given index
    """

    def __init__(self, seq: AttrExpr, index: int):
        self.seq = seq
        self.index = index


AttrConvertible = Union[AttrExpr, AttrValueType, tuple, list, None]


def to_attr(val: Union[AttrExpr, AttrValueType, tuple, list, None]) -> AttrExpr:
    """
    Create an attribute expression with given value.

    :param val: All types of values that are or can be converted to an attribute expression.
    :return: Attribute expression created from given value.
    """
    if val is None:
        return AnyAttr()
    elif isinstance(val, AttrExpr):
        return val
    elif isinstance(val, AttrExpr.value_class):
        return ConstAttr(val)
    elif isinstance(val, list):
        return ListAttr(val)
    elif isinstance(val, tuple):
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


class BinaryExpr(AttrExpr):
    """
    Binary expression of attribute values.
    """

    def __init__(self, op_name: BinaryOp, lhs: AttrExpr, rhs: AttrExpr):
        self.op = op_name
        self.lhs = lhs
        self.rhs = rhs

    eval_func = {
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
    def visit(self, attr: AttrExpr) -> Any:
        if isinstance(attr, AnyAttr):
            return self.visit_any(attr)
        elif isinstance(attr, ConstAttr):
            return self.visit_const(attr)
        elif isinstance(attr, GetAttr):
            return self.visit_get_attr(attr)
        elif isinstance(attr, ListAttr):
            return self.visit_list(attr)
        elif isinstance(attr, TupleAttr):
            return self.visit_tuple(attr)
        elif isinstance(attr, GetItemAttr):
            return self.visit_getitem(attr)
        elif isinstance(attr, BinaryExpr):
            return self.visit_binary(attr)
        else:
            raise RuntimeError('Unknown attribute type.')

    def visit_any(self, a: AnyAttr) -> Any:
        pass

    def visit_const(self, const: ConstAttr) -> Any:
        pass

    def visit_get_attr(self, get_attr: GetAttr) -> Any:
        pass

    def visit_list(self, list_attr: ListAttr) -> Any:
        for f in list_attr.fields:
            self.visit(f)

    def visit_tuple(self, tup_attr: TupleAttr) -> Any:
        for f in tup_attr.fields:
            self.visit(f)

    def visit_getitem(self, getitem: GetItemAttr) -> Any:
        self.visit(getitem.seq)

    def visit_binary(self, binary: BinaryExpr) -> Any:
        pass
