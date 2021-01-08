import unittest

from tvm import relay

import rule
from gsl.work import Workload


class RuleTest(unittest.TestCase):
    def test_trans_trans(self):
        print('Transpose-Transpose')

        # Source graph
        x = relay.var('x', shape=(4, 16, 32))
        y = relay.transpose(x, axes=(0, 2, 1))
        y = relay.transpose(y, axes=(0, 2, 1))
        wl = Workload.from_expr(y, {'x'})
        print(wl.mod)

        # Apply substitution
        subst = rule.trans_trans()
        wl = subst(wl)
        print(wl.mod)
        self.assertTrue(True)

    def test_bias_add_add(self):
        print('BiasAdd-Add')

        # Source graph
        x1 = relay.var('x1', shape=(4, 3, 32, 32))
        x2 = relay.var('x2', shape=(4, 3, 32, 32))
        b1 = relay.var('b1', shape=(3,))
        b2 = relay.var('b2', shape=(3,))
        y = relay.nn.bias_add(x1, b1) + relay.nn.bias_add(x2, b2)
        wl = Workload.from_expr(y, {'x1', 'x2'})
        print(wl.mod)

        # Apply substitution
        subst = rule.bias_add_add()
        wl = subst(wl)
        print(wl.mod)
        self.assertTrue(True)

    def test_split_concat(self):
        print('Split-Concat')

        # Source graph
        x = relay.var('x', shape=(2, 4, 4, 4))
        split = relay.split(x, 2, axis=1)
        y = relay.concatenate(split, 1)
        wl = Workload.from_expr(y, {'x'})
        print(wl.mod)

        # Apply substitution
        subst = rule.split_concat()
        wl = subst(wl)
        print(wl.mod)
        self.assertTrue(True)

    def test_parallel_conv(self):
        print('Parallel Conv')

        # Source graph
        x = relay.var('x', shape=(2, 2, 4, 4))
        w1 = relay.var('w1', shape=(2, 2, 3, 3))
        w2 = relay.var('w2', shape=(2, 2, 3, 3))
        conv1 = relay.nn.conv2d(x, w1, padding=(1, 1))
        conv2 = relay.nn.conv2d(x, w2, padding=(1, 1))
        y = relay.concatenate([conv1, conv2], 1)
        wl = Workload.from_expr(y, {'x'})
        print(wl.mod)

        # Apply substitution
        subst = rule.parallel_conv()
        wl = subst(wl)
        print(wl.mod)
        self.assertTrue(True)

    def test_conv_batch_norm(self):
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
        print(wl.mod)

        # Apply substitution
        subst = rule.conv_batch_norm()
        wl = subst(wl, new_name='conv_bias_add')
        print(wl.mod)
        self.assertTrue(True)


class ModelTest(unittest.TestCase):
    def test_resnet(self):
        print('ResNet')

        # Create model
        import model
        net = model.resnet.get_model(3)
        wl = Workload.from_keras(net, {'input_1': model.batch_shape_nchw})
        # wl.visualize()

        # Apply substitution
        subst = rule.conv_batch_norm()
        wl = subst(wl, fast_mode=True, new_name=wl.name + '_subst')
        wl.visualize()
        self.assertTrue(True)

    def test_nasnet(self):
        print('NASNet')

        # Create model
        import model
        net = model.nasnet.get_model(1)
        wl = Workload.from_keras(net, {'input_1': model.batch_shape_nchw})
        # wl.visualize()

        # Apply substitution
        subst = rule.conv_batch_norm()
        wl = subst(wl, fast_mode=True, new_name=wl.name + '_subst')
        wl.visualize()
        self.assertTrue(True)


if __name__ == '__main__':
    suite = unittest.TestSuite(tests=[
        # RuleTest('test_trans_trans'),
        # RuleTest('test_bias_add_add'),
        # RuleTest('test_split_concat'),
        # RuleTest('test_parallel_conv'),
        # RuleTest('test_conv_batch_norm'),
        # ModelTest('test_resnet'),
        ModelTest('test_nasnet'),
    ])
    unittest.TextTestRunner().run(suite)
