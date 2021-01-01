from enum import Enum
from typing import Union
import op

AttribValueType = Union[bool, int, tuple, list, str]
attrib_value_class = (bool, int, tuple, list, str)


class AttrExpr:
    """
    AST for attribute expression.
    """

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
        return BinaryExpr(BinaryOp.DIV, self, to_attr(other))

    def __rfloordiv__(self, other):
        return BinaryExpr(BinaryOp.DIV, to_attr(other), self)


class ConstAttr(AttrExpr):
    """
    A compile-time constant attribute value.
    """

    def __init__(self, value: AttribValueType):
        self.value = value


def to_attr(val: Union[AttrExpr, AttribValueType]) -> AttrExpr:
    """
    Create an attribute expression with given value.
    :param val: All types of values that are or can be converted to an attribute expression.
    :return: Attribute expression created from given value.
    """
    if isinstance(val, AttrExpr):
        return val
    elif isinstance(val, attrib_value_class):
        return ConstAttr(val)
    else:
        raise ValueError(
            'Cannot convert value of type \'{}\' to attribute.'.format(val.__class__)
        )


class GetAttr(AttrExpr):
    """
    Access attribute from a graph node.
    """

    def __init__(self, node, name: str):
        # Check if the call node has attribute of this name
        func = op.get_func(node.op)
        attr_names = op.get_func_attr_names(func)
        if not attr_names.__contains__(name):
            raise AttributeError(
                'Attribute \'{}\' not found in op \'{}\''.format(name, node.op)
            )

        self.node = node
        self.name = name


class BinaryOp(Enum):
    ADD = '+'
    SUB = '-'
    MUL = '*'
    DIV = '/'


class BinaryExpr(AttrExpr):
    """
    Binary expression of attribute values.
    """

    def __init__(self, op: BinaryOp, lhs: AttrExpr, rhs: AttrExpr):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs


class AttrVisitor:
    def visit(self, attrib: AttrExpr):
        if isinstance(attrib, ConstAttr):
            return self.visit_const(attrib)
        elif isinstance(attrib, GetAttr):
            return self.visit_get_attr(attrib)
        elif isinstance(attrib, BinaryExpr):
            return self.visit_binary(attrib)
        else:
            raise RuntimeError('Unknown attribute type.')

    def visit_const(self, const: ConstAttr):
        pass

    def visit_get_attr(self, get_attr: GetAttr):
        pass

    def visit_binary(self, binary: BinaryExpr):
        pass
