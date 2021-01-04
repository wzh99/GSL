import unittest

from tvm import relay

from gsl.graph import *
from gsl.subst import Substitution
from gsl.work import Workload


class GslTest(unittest.TestCase):
    fontname = 'LM Mono 12 Regular'

    def test_trans_trans(self):
        print('Transpose-Transpose')

        # Source graph
        x = relay.var('x', shape=[4, 16, 32])
        x = relay.transpose(x, axes=[0, 2, 1])
        x = relay.transpose(x, axes=[0, 2, 1])
        wl = Workload.from_expr(x, {'x'})
        # print(wl.mod)

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
        # print(wl.mod)
        self.assertTrue(True)

    def test_bias_add_add(self):
        print('BiasAdd-Add')

        # Source graph
        x1 = relay.var('x1', shape=[4, 3, 32, 32])
        x2 = relay.var('x2', shape=[4, 3, 32, 32])
        b1 = relay.var('b1', shape=[3])
        b2 = relay.var('b2', shape=[3])
        y = relay.nn.bias_add(x1, b1) + relay.nn.bias_add(x2, b2)
        wl = Workload.from_expr(y, {'x1', 'x2'})
        # print(wl.mod)

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
        # print(wl.mod)
        self.assertTrue(True)

    def test_conv_bn(self):
        print('Conv-BatchNorm')

        # Source graph
        x = relay.var('x', shape=(2, 2, 4, 4))
        w = relay.var('w', shape=(2, 2, 3, 3))
        gamma = relay.var('gamma', shape=(2,))
        beta = relay.var('beta', shape=(2,))
        moving_mean = relay.var('moving_mean', shape=(2,))
        moving_var = relay.var('moving_var', shape=(2,))
        y = relay.nn.conv2d(x, w, padding=(1, 1))
        y = relay.nn.batch_norm(y, gamma, beta, moving_mean, moving_var)[0]
        wl = Workload.from_expr(y, {'x'}, name='conv_bn')

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
        # y1.visualize('conv_bn_pat', fontname=self.fontname)

        # Target pattern
        k = gamma / Call('sqrt', moving_var + bn.epsilon)
        out_chan = gamma.shape[0]
        zeros = Call('zeros', shape=(out_chan, out_chan), dtype=w.dtype)
        diag = Call('expand_dims', Call('matrix_set_diag', zeros, k), axis=0)
        conv_w = Call('reshape', w, newshape=(1, w.shape[0], -1))
        matmul = Call('nn.batch_matmul', diag, Call('transpose', conv_w, axes=[0, 2, 1]))
        fused_w = Call('reshape', matmul, newshape=w.shape)
        new_conv = Call('nn.conv2d', x, fused_w, strides=conv.strides, padding=conv.padding,
                        dilation=conv.dilation, groups=conv.groups)
        bias = beta - moving_mean * k
        y2 = Call('nn.bias_add', new_conv, bias)
        # y2.visualize('conv_bias_add_pat', fontname=self.fontname)

        # Build substitution
        subst = Substitution(y1, y2)

        # Apply substitution
        wl = subst(wl, new_name='conv_bias_add')
        # wl.visualize(fontname=self.fontname)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
