from graph import *
from subst import Substitution
from tvm.relay import dataflow_pattern as dfp


def test_transpose_transpose():
    # Input
    x = Wildcard()

    # Source graph: (A^T)^T
    y1 = Call('transpose', x, axes=[0, 2, 1])
    y1 = Call('transpose', y1, axes=[0, 2, 1])

    # Target graph: A
    y2 = x
    pass


def test_bias_add_add():
    # Input
    x1 = Wildcard()
    x2 = Wildcard()
    b1 = Var()
    b2 = Var()

    # Source graph: (x1 + b1) + (x2 + b2)
    y1 = Call('nn.bias_add', x1, b1, axis=1) + Call('nn.bias_add', x2, b2, axis=1)

    # Target graph: (x1 + x2) + (b1 + b2)
    y2 = Call('nn.bias_add', x1 + x2, b1 + b2, axis=y1.axis)
    # y2 = Call('concatenate', x1)

    # Build substitution
    subst = Substitution(y1, y2)
    pass


if __name__ == '__main__':
    test_bias_add_add()
