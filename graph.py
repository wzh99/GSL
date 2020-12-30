import numpy as np
from typing import Union


class Node:
    def __neg__(self):
        return Call('negative', self)

    def __add__(self, other):
        return Call('add', self, other)

    def __sub__(self, other):
        return Call('subtract', self, other)

    def __mul__(self, other):
        return Call('multiply', self, other)

    def __truediv__(self, other):
        return Call('divide', self, other)


class Wildcard(Node):
    pass


class Var(Node):
    pass


class Constant(Node):
    def __init__(self, value: Union[int, float, list, np.ndarray]):
        if isinstance(value, (int, float, list)):
            value = np.array(value)
        if not isinstance(value, np.ndarray):
            raise RuntimeError('Not a constant')
        self.value = value


class Tuple(Node):
    def __init__(self, *fields):
        self.fields = fields

    def __getitem__(self, index: int):
        return GetItem(self, index)


class GetItem(Node):
    def __init__(self, tup: Tuple, index: int):
        self.tup = tup
        self.index = index


class Call(Node):
    def __init__(self, op: str, *args):
        self.op = op
        self.args = args
