from typing import Optional, List

from tvm import transform

import rule
from gsl import attr, pat, op, spec, Workload, Subst
from gsl.util import Timer


class AlgCmp:
    def create_workload(self) -> Workload:
        raise NotImplementedError()

    def get_pass(self) -> Optional[transform.Pass]:
        pass

    def gsl_rules(self) -> List[Subst]:
        raise NotImplementedError()

    def run(self):
        # Create workload
        wl = self.create_workload()

        # Apply pass
        timer = Timer()
        f_pass = self.get_pass()
        if f_pass is not None:
            timer.begin()
            f_pass(wl.mod)
            print(f'Built-in pass: {timer.end()} s')

        # Apply GSL rules
        Subst.profile = True
        for idx, subst in enumerate(self.gsl_rules()):
            print(f'Rule {idx + 1}')
            for _ in range(15):
                subst(wl, fold_params=False)


class InceptionV3(AlgCmp):
    def create_workload(self) -> Workload:
        from tvm.relay.testing import inception_v3
        net, params = inception_v3.get_workload()
        wl = Workload(net, params)
        for subst in [
            rule.lower_batch_norm(),
            rule.conv_mul()
        ]:
            wl = subst(wl)
        return wl

    def gsl_rules(self) -> List[Subst]:
        conv_attrs = ['strides', 'padding', 'dilation', 'groups']

        def parallel_two_conv(num_ops: int):
            # Input
            x = pat.Wildcard()
            w1 = pat.Variable()
            w2 = pat.Variable(shape=(None, None, w1.shape[2], w1.shape[3]))

            # Source pattern
            conv1 = op.Conv2D(x, w1, groups=1)
            conv2 = op.Conv2D(x, w2, **pat.same_attr(conv1, conv_attrs))
            src = [conv1, conv2]
            biases = [pat.Variable(), pat.Variable()]
            if num_ops >= 2:
                src = [op.BiasAdd(y, b, axis=1) for y, b in zip(src, biases)]
            if num_ops >= 3:
                src = [op.ReLU(y) for y in src]

            # Target pattern
            w = op.Concatenate((w1, w2), axis=0)
            y = op.Conv2D(x, w, **pat.same_attr(conv1, conv_attrs))
            if num_ops >= 2:
                y = op.BiasAdd(y, op.Concatenate(biases, axis=0))
            if num_ops >= 3:
                y = op.ReLU(y)
            split = op.Split(y, indices_or_sections=(w1.shape[0],), axis=1)
            tgt = [split[0], split[1]]

            # Build substitution
            return Subst(src, tgt)

        def parallel_three_conv(num_ops: int):
            # Input
            x = pat.Wildcard()
            w1 = pat.Variable()
            w2 = pat.Variable(shape=(None, None, w1.shape[2], w1.shape[3]))
            w3 = pat.Variable(shape=(None, None, w1.shape[2], w1.shape[3]))

            # Source pattern
            conv1 = op.Conv2D(x, w1, groups=1)
            conv2 = op.Conv2D(x, w2, **pat.same_attr(conv1, conv_attrs))
            conv3 = op.Conv2D(x, w3, **pat.same_attr(conv1, conv_attrs))
            src = [conv1, conv2, conv3]
            biases = [pat.Variable() for _ in range(3)]
            if num_ops >= 2:
                src = [op.BiasAdd(y, b, axis=1) for y, b in zip(src, biases)]
            if num_ops >= 3:
                src = [op.ReLU(y) for y in src]

            # Target pattern
            w = op.Concatenate((w1, w2, w3), axis=0)
            y = op.Conv2D(x, w, **pat.same_attr(conv1, conv_attrs))
            if num_ops >= 2:
                y = op.BiasAdd(y, op.Concatenate(biases, axis=0))
            if num_ops >= 3:
                y = op.ReLU(y)
            split = op.Split(y, indices_or_sections=(w1.shape[0], w1.shape[0] + w2.shape[0]),
                             axis=1)
            tgt = [split[0], split[1], split[2]]

            # Build substitution
            return Subst(src, tgt)

        def parallel_conv_variadic(num_ops: int):
            # Input
            x = pat.Wildcard()
            w1 = pat.Variable()
            w = pat.Variable(shape=(None, None, w1.shape[2], w1.shape[3]))
            b1 = pat.Variable()
            b = pat.Variable()

            # Source pattern
            conv1 = op.Conv2D(x, w1, groups=1)
            conv = op.Conv2D(x, w, **pat.same_attr(conv1, conv_attrs))
            templates = [conv, w]
            first = [conv1, w1]
            y, y1 = conv, conv1
            if num_ops >= 2:
                y1 = bias_add1 = op.BiasAdd(y1, b1, axis=1)
                y = bias_add = op.BiasAdd(y, b, axis=1)
                templates += [bias_add, b]
                first += [bias_add1, b1]
            if num_ops >= 3:
                relu1 = op.ReLU(y1)
                y = relu = op.ReLU(y)
                templates += [relu]
                first += [relu1]
            src = pat.Variadic(y, templates=templates, first=first, min_len=2)

            # Target pattern
            i = attr.Symbol()
            w_inst = src(w, i)
            concat = op.Concatenate(
                pat.Variadic(w_inst, templates=[w_inst], index=i, length=src.length), axis=0)
            y = op.Conv2D(x, concat, **pat.same_attr(conv1, conv_attrs))
            if num_ops >= 2:
                i = attr.Symbol()
                b_inst = src(b, i)
                bias_add = op.Concatenate(pat.Variadic(b_inst, templates=[b_inst], index=i,
                                                       length=src.length), axis=0)
                y = op.BiasAdd(y, bias_add, axis=1)
            if num_ops >= 3:
                y = op.ReLU(y)
            split = op.Split(y, axis=1, indices_or_sections=attr.Variadic(
                lambda j: attr.ReduceIndexed(attr.BinaryOp.ADD, lambda k: src(w, k).shape[0], j + 1),
                length=src.length - 1))
            i = attr.Symbol()
            item = split[i]
            tgt = pat.Variadic(item, templates=[item], index=i)

            # Build substitution
            return Subst(src, tgt)

        return [
            parallel_two_conv(3),
            parallel_three_conv(3),
            parallel_conv_variadic(3),
        ]


class Transformer(AlgCmp):
    def create_workload(self) -> Workload:
        from model import transformer
        return transformer.get_workload(6, 64, 4, 128)

    def gsl_rules(self) -> List[Subst]:
        def parallel_three_dense():
            # Input
            x = pat.Wildcard()
            w1 = pat.Variable()
            w2 = pat.Variable()
            w3 = pat.Variable()
            weights = [w1, w2, w3]

            # Source pattern
            src = [op.Dense(x, w) for w in weights]

            # Target pattern
            dense = op.Dense(x, op.Concatenate(weights, axis=0))
            split = op.Split(dense, indices_or_sections=(w1.shape[0], w1.shape[0] + w2.shape[0]),
                             axis=-1)
            tgt = [split[0], split[1], split[2]]

            # Build substitution
            return Subst(src, tgt)

        return [
            rule.parallel_dense(),
            parallel_three_dense(),
            rule.parallel_dense_variadic(),
        ]


class NASNet(AlgCmp):
    def create_workload(self) -> Workload:
        from model import nasnet
        model = nasnet.get_model(6)
        return Workload.from_keras(model, {'input_1': nasnet.batch_shape_nchw})

    def gsl_rules(self) -> List[Subst]:
        def merge_three_element_wise():
            # Input
            x = pat.Wildcard()

            # Source pattern
            ew_op = pat.OpWithTrait(spec.OpTrait.ELEMENT_WISE)
            ew1 = pat.Call(ew_op, x)
            ew2 = pat.Call(ew_op, x)
            ew3 = pat.Call(ew_op, x)

            # Target pattern
            ew = pat.Call(ew_op, x)

            # Build substitution
            return Subst([ew1, ew2, ew3], [ew, ew, ew])

        return [
            rule.merge_element_wise(),
            merge_three_element_wise(),
            rule.merge_element_wise_variadic(),
        ]


if __name__ == '__main__':
    for cls in [
        # InceptionV3,
        # Transformer,
        # NASNet,
    ]:
        cls().run()
