from enum import Enum
from typing import Union

AttribValueType = Union[bool, int, tuple, list, str]


class AttribExpr:
    """
    AST for attribute expression.
    """

    def __add__(self, other):
        return BinaryExpr(BinaryOp.ADD, self, other)

    def __sub__(self, other):
        return BinaryExpr(BinaryOp.SUB, self, other)


class ConstAttrib(AttribExpr):
    """
    A compile-time constant attribute value.
    """

    def __init__(self, value: AttribValueType):
        self.value = value


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


class BinaryExpr(AttribExpr):
    """
    Binary expression of attribute values.
    """

    def __init__(self, op: BinaryOp, lhs: AttribExpr, rhs: AttribExpr):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs
