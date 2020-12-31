from graph import *
from work import Workload
from subst import Substitution
from tvm import relay


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

    # Source pattern: (x1 + b1) + (x2 + b2)
    y1 = Call('nn.bias_add', x1, b1, axis=1) + Call('nn.bias_add', x2, b2, axis=1)

    # Target pattern: (x1 + x2) + (b1 + b2)
    y2 = Call('nn.bias_add', x1 + x2, b1 + b2, axis=y1.axis)

    # Build substitution
    subst = Substitution(y1, y2)

    # Create source graph
    x1 = relay.var('x1', shape=[4, 3, 32, 32])
    x2 = relay.var('x2', shape=[4, 3, 32, 32])
    b1 = relay.var('b1', shape=[3])
    b2 = relay.var('b2', shape=[3])
    y = relay.nn.bias_add(x1, b1) + relay.nn.bias_add(x2, b2)
    wl = Workload.from_expr(y)
    wl = subst.apply(wl)
    pass


if __name__ == '__main__':
    test_bias_add_add()
