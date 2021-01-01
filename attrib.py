from enum import Enum
from typing import Union

AttribValueType = Union[bool, int, tuple, list, str]
attrib_value_class = (bool, int, tuple, list, str)


class AttribExpr:
    """
    AST for attribute expression.
    """

    def __add__(self, other):
        return BinaryExpr(BinaryOp.ADD, self, to_attrib(other))

    def __radd__(self, other):
        return BinaryExpr(BinaryOp.ADD, to_attrib(other), self)

    def __sub__(self, other):
        return BinaryExpr(BinaryOp.SUB, self, to_attrib(other))

    def __rsub__(self, other):
        return BinaryExpr(BinaryOp.SUB, to_attrib(other), self)

    def __mul__(self, other):
        return BinaryExpr(BinaryOp.MUL, self, to_attrib(other))

    def __rmul__(self, other):
        return BinaryExpr(BinaryOp.MUL, to_attrib(other), self)

    def __floordiv__(self, other):
        return BinaryExpr(BinaryOp.DIV, self, to_attrib(other))

    def __rfloordiv__(self, other):
        return BinaryExpr(BinaryOp.DIV, to_attrib(other), self)


class ConstAttrib(AttribExpr):
    """
    A compile-time constant attribute value.
    """

    def __init__(self, value: AttribValueType):
        self.value = value


def to_attrib(val: Union[AttribExpr, AttribValueType]) -> AttribExpr:
    """
    Create an attribute expression with given value.
    :param val: All types of values that are or can be converted to an attribute expression.
    :return: Attribute expression created from given value.
    """
    if isinstance(val, AttribExpr):
        return val
    elif isinstance(val, attrib_value_class):
        return ConstAttrib(val)
    else:
        raise ValueError(
            'Cannot convert value of type \'{}\' to attribute.'.format(val.__class__)
        )


class GetAttrib(AttribExpr):
    """
    Access attribute from a graph node.
    """

    def __init__(self, node, name: str):
        self.node = node
        self.name = name


class BinaryOp(Enum):
    ADD = '+'
    SUB = '-'
    MUL = '*'
    DIV = '/'


class BinaryExpr(AttribExpr):
    """
    Binary expression of attribute values.
    """

    def __init__(self, op: BinaryOp, lhs: AttribExpr, rhs: AttribExpr):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs


class AttribVisitor:
    def visit(self, attrib: AttribExpr):
        if isinstance(attrib, ConstAttrib):
            return self.visit_const(attrib)
        elif isinstance(attrib, GetAttrib):
            return self.visit_get_attrib(attrib)
        elif isinstance(attrib, BinaryExpr):
            return self.visit_binary(attrib)
        else:
            raise RuntimeError('Unknown attribute type.')

    def visit_const(self, const: ConstAttrib):
        pass

    def visit_get_attrib(self, get_attrib: GetAttrib):
        pass

    def visit_binary(self, binary: BinaryExpr):
        pass
