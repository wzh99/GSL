import unittest

from tvm import relay

import rule
from gsl import Workload, Substitution, Wildcard
from gsl.pat import Call


class MatchTest(unittest.TestCase):
    def test_pyramid(self):
        print('Match Pyramid')

        # Pattern
        x = Wildcard()
        l1_op1 = Call('exp', x)
        l1_op2 = Call('abs', x)
        l2_op1 = Call('sqrt', l1_op1)
        l2_op2 = Call('add', l1_op1, l1_op2)
        l2_op3 = Call('negative', l1_op2)
        subst = Substitution([l2_op1, l2_op2, l2_op3], [x, x, x])

        # Source graph
        x = relay.var('x', shape=(2, 2, 4, 4))
        l1_op1 = relay.exp(x)
        l1_op2 = relay.abs(x)
        l2_op1 = relay.sqrt(l1_op1)
        l2_op2 = l1_op1 + l1_op2
        l2_op3 = -l1_op2
        y = relay.concatenate([l2_op1, l2_op2, l2_op3], axis=1)
        wl = Workload.from_expr(y, {'x'})
        print(wl.mod)

        # Apply substitution
        wl = subst(wl)
        print(wl.mod)

        self.assertTrue(True)


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

    def test_split_concat_variadic(self):
        print('Split-Concat Variadic')

        # Source graph
        x = relay.var('x', shape=(2, 6, 4, 4))
        split = relay.split(x, 3, axis=1)
        y = relay.concatenate(split, 1)
        wl = Workload.from_expr(y, {'x'})
        print(wl.mod)

        # Apply substitution
        subst = rule.split_concat_variadic()
        wl = subst(wl)
        print(wl.mod)
        self.assertTrue(True)

    def test_bias_add_add(self):
        print('BiasAdd-Add')

        # Source graph
        x1 = relay.var('x1', shape=(2, 3, 4, 4))
        x2 = relay.var('x2', shape=(2, 3, 4, 4))
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

    def test_diamond_conv_add(self):
        print('Diamond Conv-Add')

        # Source graph
        x = relay.var('x', shape=(2, 2, 4, 4))
        w1 = relay.var('w1', shape=(4, 2, 1, 1))
        w2 = relay.var('w2', shape=(4, 2, 1, 1))
        conv1 = relay.nn.conv2d(x, w1)
        conv2 = relay.nn.conv2d(x, w2)
        y = conv1 + conv2
        wl = Workload.from_expr(y, {'x'})
        print(wl.mod)

        # Apply substitution
        subst = rule.diamond_conv_add()
        wl = subst(wl)
        print(wl.mod)
        self.assertTrue(True)

    def test_two_conv_add(self):
        print('Two Conv-Add')

        # Source graph
        x1 = relay.var('x1', shape=(2, 2, 4, 4))
        w1 = relay.var('w1', shape=(4, 2, 1, 1))
        conv1 = relay.nn.conv2d(x1, w1)
        x2 = relay.var('x2', shape=(2, 2, 4, 4))
        w2 = relay.var('w2', shape=(4, 2, 1, 1))
        conv2 = relay.nn.conv2d(x2, w2)
        y = conv1 + conv2
        wl = Workload.from_expr(y, {'x1', 'x2'})
        print(wl.mod)

        # Apply substitution
        subst = rule.two_conv_add()
        wl = subst(wl)
        print(wl.mod)
        self.assertTrue(True)

    def test_conv_shortcut_add(self):
        print('Conv-Shortcut-Add')

        # Source graph
        x = relay.var('x', shape=(2, 2, 4, 4))
        w = relay.var('w', shape=(2, 2, 3, 3))
        conv = relay.nn.conv2d(x, w, padding=(1, 1, 1, 1))
        y = conv + x
        wl = Workload.from_expr(y, {'x'})
        print(wl.mod)

        # Apply substitution
        subst = rule.conv_shortcut_add()
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
        conv = relay.nn.conv2d(x, w, padding=(1, 1, 1, 1))
        y = relay.nn.batch_norm(conv, gamma, beta, moving_mean, moving_var)[0]
        wl = Workload.from_expr(y, {'x'}, name='conv_bn')
        print(wl.mod)

        # Apply substitutions
        for subst in [
            rule.lower_batch_norm(),
            rule.conv_mul(),
        ]:
            wl = subst(wl, fast_mode=True)
        print(wl.mod)
        self.assertTrue(True)

    def test_lower_layer_norm(self):
        print('Lower Layer Normalization')

        # Source graph
        x = relay.var('x', shape=(2, 2, 4))
        gamma = relay.var('gamma', shape=(4,))
        beta = relay.var('beta', shape=(4,))
        y = relay.nn.layer_norm(x, gamma, beta)
        wl = Workload.from_expr(y, {'x'})
        print(wl.mod)

        # Target graph
        subst = rule.lower_layer_norm()
        wl = subst(wl)
        print(wl.mod)
        self.assertTrue(True)

    def test_merge_element_wise(self):
        print('Merge ReLU')

        # Source graph
        x = relay.var('x', shape=(2, 2, 4, 4))
        relu1 = relay.nn.relu(x)
        w1 = relay.var('w1', shape=(2, 2, 3, 3))
        conv1 = relay.nn.conv2d(relu1, w1, padding=(1, 1, 1, 1))
        relu2 = relay.nn.relu(x)
        w2 = relay.var('w2', shape=(2, 2, 3, 3))
        conv2 = relay.nn.conv2d(relu2, w2, padding=(1, 1, 1, 1))
        y = conv1 + conv2
        wl = Workload.from_expr(y, {'x'})
        print(wl.mod)

        # Apply substitution
        subst = rule.merge_element_wise()
        wl = subst(wl)
        print(wl.mod)
        self.assertTrue(True)

    def test_merge_element_wise_variadic(self):
        print('Variadic Merge ReLU')

        # Source graph
        x = relay.var('x', shape=(2, 2, 4, 4))
        relu1 = relay.nn.relu(x)
        w1 = relay.var('w1', shape=(2, 2, 3, 3))
        conv1 = relay.nn.conv2d(relu1, w1, padding=(1, 1, 1, 1))
        relu2 = relay.nn.relu(x)
        w2 = relay.var('w2', shape=(2, 2, 3, 3))
        conv2 = relay.nn.conv2d(relu2, w2, padding=(1, 1, 1, 1))
        relu3 = relay.nn.relu(x)
        w3 = relay.var('w3', shape=(2, 2, 3, 3))
        conv3 = relay.nn.conv2d(relu3, w3, padding=(1, 1, 1, 1))
        y = relay.concatenate([conv1, conv2, conv3], 1)
        wl = Workload.from_expr(y, {'x'})
        print(wl.mod)

        # Apply substitution
        subst = rule.merge_element_wise_variadic()
        wl = subst(wl)
        print(wl.mod)
        self.assertTrue(True)

    def test_parallel_conv(self):
        print('Parallel Conv')

        # Source graph
        x = relay.var('x', shape=(2, 2, 4, 4))
        w1 = relay.var('w1', shape=(2, 2, 3, 3))
        w2 = relay.var('w2', shape=(2, 2, 3, 3))
        conv1 = relay.nn.conv2d(x, w1, padding=(1, 1, 1, 1))
        conv2 = relay.nn.conv2d(x, w2, padding=(1, 1, 1, 1))
        y = conv1 + conv2
        wl = Workload.from_expr(y, {'x'})
        print(wl.mod)

        # Apply substitution
        subst = rule.parallel_conv()
        wl = subst(wl)
        print(wl.mod)
        self.assertTrue(True)

    def test_parallel_conv_variadic(self):
        print('Variadic Parallel Conv')

        # Source graph
        x = relay.var('x', shape=(2, 2, 4, 4))
        w1 = relay.var('w1', shape=(2, 2, 3, 3))
        w2 = relay.var('w2', shape=(2, 2, 3, 3))
        w3 = relay.var('w3', shape=(4, 2, 3, 3))
        conv1 = relay.nn.conv2d(x, w1, padding=(1, 1, 1, 1))
        conv2 = relay.nn.conv2d(x, w2, padding=(1, 1, 1, 1))
        conv3 = relay.nn.conv2d(x, w3, padding=(1, 1, 1, 1))
        y = relay.concatenate([conv1, conv2, conv3], 1)
        wl = Workload.from_expr(y, {'x'})
        print(wl.mod)

        # Apply substitution
        subst = rule.parallel_conv_variadic()
        wl = subst(wl)
        print(wl.mod)
        self.assertTrue(True)

    def test_parallel_conv_expand_kernels(self):
        print('Parallel Conv (Expand Kernels)')

        # Source graph
        x = relay.var('x', shape=(2, 2, 4, 4))
        w1 = relay.var('w1', shape=(2, 2, 1, 1))
        w2 = relay.var('w2', shape=(2, 2, 3, 3))
        conv1 = relay.nn.conv2d(x, w1, padding=(0, 0, 0, 0))
        conv2 = relay.nn.conv2d(x, w2, padding=(1, 1, 1, 1))
        y = conv1 + conv2
        wl = Workload.from_expr(y, {'x'})
        print(wl.mod)

        # Apply substitution
        subst = rule.parallel_conv_expand_kernels()
        wl = subst(wl)
        print(wl.mod)
        self.assertTrue(True)

    def test_parallel_group_conv_variadic(self):
        print('Variadic Group Conv')

        # Source graph
        x = relay.var('x', shape=(2, 4, 4, 4))
        w1 = relay.var('w1', shape=(4, 2, 3, 3))
        w2 = relay.var('w2', shape=(4, 2, 3, 3))
        w3 = relay.var('w3', shape=(4, 2, 3, 3))
        conv1 = relay.nn.conv2d(x, w1, padding=(1, 1, 1, 1), groups=2)
        conv2 = relay.nn.conv2d(x, w2, padding=(1, 1, 1, 1), groups=2)
        conv3 = relay.nn.conv2d(x, w3, padding=(1, 1, 1, 1), groups=2)
        y = relay.concatenate([conv1, conv2, conv3], axis=1)
        wl = Workload.from_expr(y, {'x'})
        print(wl.mod)

        # Apply substitution
        subst = rule.parallel_group_conv_variadic()
        wl = subst(wl)
        print(wl.mod)
        self.assertTrue(True)

    def test_parallel_dense(self):
        print('Parallel Dense')

        # Source graph
        x = relay.var('x', shape=(2, 4))
        w1 = relay.var('w1', shape=(4, 4))
        w2 = relay.var('w2', shape=(4, 4))
        d1 = relay.nn.dense(x, w1)
        d2 = relay.nn.dense(x, w2)
        y = relay.concatenate([d1, d2], axis=-1)
        wl = Workload.from_expr(y, {'x'})

        # Apply substitution
        subst = rule.parallel_dense()
        wl = subst(wl)
        print(wl.mod)
        self.assertTrue(True)

    def test_parallel_dense_variadic(self):
        print('Variadic Parallel Dense')

        # Source graph
        x = relay.var('x', shape=(2, 4))
        w1 = relay.var('w1', shape=(4, 4))
        w2 = relay.var('w2', shape=(4, 4))
        w3 = relay.var('w3', shape=(4, 4))
        d1 = relay.nn.dense(x, w1)
        d2 = relay.nn.dense(x, w2)
        d3 = relay.nn.dense(x, w3)
        y = relay.concatenate([d1, d2, d3], axis=-1)
        wl = Workload.from_expr(y, {'x'})
        print(wl.mod)

        # Apply substitution
        subst = rule.parallel_dense_variadic()
        wl = subst(wl)
        print(wl.mod)
        self.assertTrue(True)

    def test_nasnet_block(self):
        print('Sequential Subst. on a Simplified NASNet Block')

        # Source graph
        x = relay.var('x', shape=(2, 32, 32, 32))
        relu1 = relay.nn.relu(x)
        dep_w1 = relay.var('dep_w1', shape=(32, 1, 3, 3))
        dep_conv1 = relay.nn.conv2d(relu1, dep_w1, padding=(1, 1, 1, 1), groups=32)
        pt_w1 = relay.var('pt_w1', shape=(32, 32, 1, 1))
        pt_conv1 = relay.nn.conv2d(dep_conv1, pt_w1)
        g1 = relay.var('g1', shape=(32,))
        b1 = relay.var('b1', shape=(32,))
        m1 = relay.var('m1', shape=(32,))
        v1 = relay.var('v1', shape=(32,))
        bn1 = relay.nn.batch_norm(pt_conv1, g1, b1, m1, v1)[0]
        relu2 = relay.nn.relu(x)
        w2 = relay.var('dep_w2', shape=(32, 1, 5, 5))
        dep_conv2 = relay.nn.conv2d(relu2, w2, padding=(2, 2, 2, 2), groups=32)
        pt_w2 = relay.var('pt_w2', shape=(32, 32, 1, 1))
        pt_conv2 = relay.nn.conv2d(dep_conv2, pt_w2)
        g2 = relay.var('g2', shape=(32,))
        b2 = relay.var('b2', shape=(32,))
        m2 = relay.var('m2', shape=(32,))
        v2 = relay.var('v2', shape=(32,))
        bn2 = relay.nn.batch_norm(pt_conv2, g2, b2, m2, v2)[0]
        y = bn1 + bn2
        wl = Workload.from_expr(y, {'x'})
        print(wl.mod)

        # Apply substitutions
        for subst in [
            rule.lower_batch_norm(),
            rule.conv_mul(),
            rule.bias_add_add(),
            rule.two_conv_add(),
            rule.merge_element_wise(),
        ]:
            wl = subst(wl, fast_mode=True)
        print(wl.mod)
        self.assertTrue(True)


class ModelTest(unittest.TestCase):
    def test_resnet(self):
        print('ResNet')

        # Create model
        from model.resnet import get_model, batch_shape_nchw
        net = get_model(3)
        wl = Workload.from_keras(net, {'input_1': batch_shape_nchw})

        # Apply substitution
        for subst in [
            rule.lower_batch_norm(),
            rule.conv_mul()
        ]:
            wl = subst(wl)
        wl.visualize()
        self.assertTrue(True)

    def test_nasnet(self):
        print('NASNet')

        # Create model
        from model.nasnet import get_model, batch_shape_nchw
        net = get_model(1)
        wl = Workload.from_keras(net, {'input_1': batch_shape_nchw})
        # import numpy as np
        # x_in = np.random.rand(*batch_shape_nchw)
        # wl.build(target='metal')
        # y1 = wl(input_1=x_in)

        # Apply substitution
        for subst in [
            rule.merge_element_wise_variadic(),
            rule.lower_batch_norm(),
            rule.conv_mul(),
            rule.bias_add_add(),
            rule.two_conv_add(),
        ]:
            wl = subst(wl, fast_mode=True)
        wl.visualize()
        # wl.build(target='metal')
        # y2 = wl(input_1=x_in)
        self.assertTrue(True)

    def test_transformer(self):
        print('Transformer')

        # Create model
        from model.transformer import get_workload
        wl = get_workload(1, 64, 4, 128)

        # Apply substitution
        for subst in [
            rule.parallel_dense_variadic(),
            rule.lower_layer_norm(),
        ]:
            wl = subst(wl)
        wl.visualize()
        self.assertTrue(True)


if __name__ == '__main__':
    suite = unittest.TestSuite(tests=[
        # MatchTest('test_pyramid'),
        # RuleTest('test_trans_trans'),
        # RuleTest('test_split_concat'),
        # RuleTest('test_split_concat_variadic'),
        # RuleTest('test_bias_add_add'),
        # RuleTest('test_diamond_conv_add'),
        # RuleTest('test_two_conv_add'),
        # RuleTest('test_conv_shortcut_add'),
        # RuleTest('test_conv_batch_norm'),
        # RuleTest('test_lower_layer_norm'),
        # RuleTest('test_merge_element_wise'),
        # RuleTest('test_merge_element_wise_variadic'),
        # RuleTest('test_parallel_conv'),
        # RuleTest('test_parallel_conv_expand_kernels'),
        # RuleTest('test_parallel_conv_variadic'),
        # RuleTest('test_parallel_group_conv_variadic')
        # RuleTest('test_parallel_dense'),
        # RuleTest('test_parallel_dense_variadic'),
        # RuleTest('test_nasnet_block'),
        # ModelTest('test_resnet'),
        # ModelTest('test_nasnet'),
        # ModelTest('test_transformer')
    ])
    unittest.TextTestRunner().run(suite)
