from tvm import relay

from graph import *
from subst import Substitution
from work import Workload


def test_transpose_transpose():
    # Source graph
    x = relay.var('x', shape=[4, 16, 32])
    x = relay.transpose(x, axes=[0, 2, 1])
    x = relay.transpose(x, axes=[0, 2, 1])
    wl = Workload.from_expr(x)
    print(wl.mod)

    # Input
    x = Wildcard()

    # Source graph: (A^T)^T
    y1 = Call('transpose', x, axes=[0, 2, 1])
    y1 = Call('transpose', y1, axes=[0, 2, 1])

    # Target graph: A
    y2 = x

    # Build substitution
    subst = Substitution(y1, y2)

    # Apply substitution
    wl = subst(wl)
    print(wl.mod)


def test_bias_add_add():
    # Source graph
    x1 = relay.var('x1', shape=[4, 3, 32, 32])
    x2 = relay.var('x2', shape=[4, 3, 32, 32])
    b1 = relay.var('b1', shape=[3])
    b2 = relay.var('b2', shape=[3])
    y = relay.nn.bias_add(x1, b1) + relay.nn.bias_add(x2, b2)
    wl = Workload.from_expr(y)
    print(wl.mod)

    # Input
    x1 = Wildcard()
    x2 = Wildcard()
    b1 = Var()
    b2 = Var()

    # Source pattern: (x1 + b1) + (x2 + b2)
    add1 = Call('nn.bias_add', x1, b1, axis=1)
    add2 = Call('nn.bias_add', x2, b2, axis=add1.axis)
    y1 = add1 + add2

    # Target pattern: (x1 + x2) + (b1 + b2)
    y2 = Call('nn.bias_add', x1 + x2, b1 + b2, axis=add1.axis)

    # Build substitution
    subst = Substitution(y1, y2)

    # Apply substitution
    wl = subst(wl)
    print(wl.mod)


def test_conv_bn():
    # Source graph
    x = relay.var('x', shape=(4, 3, 32, 32))
    w = relay.var('w', shape=(16, 3, 3, 3))
    gamma = relay.var('gamma', shape=(16,))
    beta = relay.var('beta', shape=(16,))
    moving_mean = relay.var('moving_mean', shape=(16,))
    moving_var = relay.var('moving_var', shape=(16,))
    y = relay.nn.conv2d(x, w, padding=(1, 1))
    y = relay.nn.batch_norm(y, gamma, beta, moving_mean, moving_var)[0]
    wl = Workload.from_expr(y)
    print(wl.mod)

    # Input
    x = Wildcard()
    w = Var()
    gamma = Var()
    beta = Var()
    moving_mean = Var()
    moving_var = Var()

    # Source pattern
    conv = Call('nn.conv2d', x, w)
    bn = Call('nn.batch_norm', conv, gamma, beta, moving_mean, moving_var)
    y1 = bn[0]

    # Target pattern
    k = gamma / Call('sqrt', moving_var + bn.epsilon)
    out_chan = gamma.shape[0]
    zeros = Call('zeros', shape=(out_chan, out_chan), dtype=w.dtype)
    diag = Call('expand_dims', Call('matrix_set_diag', zeros, k), axis=0)
    conv_w = Call('reshape', w, newshape=(1, w.shape[0], -1))
    matmul = Call('nn.batch_matmul', diag, Call('transpose', conv_w, axes=[0, 2, 1]))
    fused_w = Call('reshape_like', matmul, w)
    new_conv = Call('nn.conv2d', x, fused_w, strides=conv.strides, padding=conv.padding,
                    dilation=conv.dilation, groups=conv.groups)
    bias = beta - moving_mean * k
    y2 = Call('nn.bias_add', new_conv, bias)

    # Build substitution
    subst = Substitution(y1, y2)

    # Apply substitution
    wl = subst(wl)
    print(wl.mod)


if __name__ == '__main__':
    # test_bias_add_add()
    test_conv_bn()
    pass
