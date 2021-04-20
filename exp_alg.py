from typing import Optional, List

from tvm import transform, relay

from gsl import attr, op, pat, Workload, Subst
from gsl.util import Timer


class AlgCmp:
    def create_workload(self) -> Workload:
        raise NotImplementedError()

    def pre_passes(self) -> List[transform.Pass]:
        return []

    def get_pass(self) -> Optional[transform.Pass]:
        pass

    def define_gsl(self) -> Subst:
        raise NotImplementedError()

    def run(self):
        # Create workload
        wl = self.create_workload()

        # Perform preliminary passes to avoid assertion error
        for p in self.pre_passes():
            wl.mod = p(wl.mod)

        # Apply pass
        timer = Timer()
        f_pass = self.get_pass()
        if f_pass is not None:
            timer.begin()
            f_pass(wl.mod)
            print(f'Built-in pass: {timer.end()} s')

        # Apply GSL rule
        subst = self.define_gsl()
        Subst.profile = True
        subst(wl, fold_params=False)


class InceptionV3(AlgCmp):
    def create_workload(self) -> Workload:
        from tvm.relay.testing import inception_v3
        net, params = inception_v3.get_workload()
        return Workload(net, params)

    def pre_passes(self) -> List[transform.Pass]:
        return [relay.transform.SimplifyInference()]

    def get_pass(self) -> Optional[transform.Pass]:
        return relay.transform.CombineParallelConv2D(min_num_branches=2)

    def define_gsl(self) -> Subst:
        x = pat.Wildcard()
        w1 = pat.Variable()
        conv1 = op.Conv2D(x, w1, groups=1)
        w_shape = attr.Cond(conv1.kernel_layout == 'OIHW', (None, None, w1.shape[2], w1.shape[3]),
                            (w1.shape[0], w1.shape[1], None, None))
        w = pat.Variable(shape=w_shape)
        attrs = ['strides', 'padding', 'dilation', 'groups', 'data_layout', 'kernel_layout',
                 'out_dtype', 'out_layout']
        conv = op.Conv2D(x, w, **pat.same_attr(conv1, attrs))
        src = pat.Variadic(conv, templates=[conv, w], first=[conv1, w1], min_len=2)

        i = attr.Symbol()
        wi = src(w, i)
        concat = op.Concatenate(pat.Variadic(wi, templates=[wi], index=i, length=src.length),
                                axis=attr.Cond(conv1.kernel_layout == 'OIHW', 0, 3))
        conv = op.Conv2D(x, concat, **pat.same_attr(conv1, attrs))
        out_layout = attr.Cond(conv1.out_layout == '', conv1.data_layout, conv1.out_layout)
        split = op.Split(conv, indices_or_sections=attr.Variadic(
            lambda j: attr.ReduceIndexed(attr.BinaryOp.ADD, lambda k: src(w, k).shape[0], j + 1),
            length=src.length - 1), axis=attr.Cond(out_layout == "NCHW", 1, 3))
        i = attr.Symbol()
        item = split[i]
        tgt = pat.Variadic(item, templates=[item], index=i)

        return Subst(src, tgt)


class Transformer(AlgCmp):
    def create_workload(self) -> Workload:
        from model import transformer
        return transformer.get_workload(1, 64, 4, 128)

    def pre_passes(self) -> List[transform.Pass]:
        return [relay.transform.SimplifyInference()]

    def get_pass(self) -> Optional[transform.Pass]:
        return relay.transform.CombineParallelDense(min_num_branches=2, to_batch=False)

    def define_gsl(self) -> Subst:
        x = pat.Wildcard()
        w1 = pat.Variable()
        dense1 = op.Dense(x, w1)
        w = pat.Variable()
        dense = op.Dense(x, w, out_dtype=dense1.out_dtype)
        src = pat.Variadic(dense, templates=[dense, w], first=[dense1, w1], min_len=2)

        i = attr.Symbol()
        wi = src(w, i)
        dense = op.Dense(x, op.Concatenate(
            pat.Variadic(wi, templates=[wi], index=i, length=src.length), axis=0))
        split = op.Split(dense, axis=-1, indices_or_sections=attr.Variadic(
            lambda j: attr.ReduceIndexed(attr.BinaryOp.ADD, lambda k: src(w, k).shape[0], j + 1),
            length=src.length - 1))
        i = attr.Symbol()
        item = split[i]
        tgt = pat.Variadic(item, templates=[item], index=i)

        return Subst(src, tgt)


if __name__ == '__main__':
    for cls in [
        InceptionV3,
        # Transformer,
    ]:
        cls().run()
