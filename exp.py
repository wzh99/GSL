from functools import reduce
from typing import Dict, Optional

from tvm import relay, transform, ir
from tvm.relay import dataflow_pattern as dfp

from gsl import pat, op, attr, Subst, Workload


class SubstTest(dfp.DFPatternCallback):
    def __init__(self, viz_orig=False, viz_pass=False, viz_dfp=False, viz_gsl=False):
        super().__init__()
        self.viz_orig = viz_orig
        self.viz_pass = viz_pass
        self.viz_dfp = viz_dfp
        self.viz_gsl = viz_gsl

    def create_expr(self) -> relay.Expr:
        pass

    def get_pass(self) -> transform.Pass:
        pass

    def define_gsl(self) -> Optional[Subst]:
        pass

    def rewrite_dfp(self, node_map: Dict[dfp.DFPattern, relay.Expr]) -> relay.Expr:
        pass

    def callback(self, pre: relay.Expr, post: relay.Expr, node_map: ir.Map) -> relay.Expr:
        return self.rewrite_dfp(dict([(p, l[0]) for p, l in node_map.items()]))

    def run(self):
        # Create workload
        body = self.create_expr()
        wl = Workload.from_expr(body, set())
        print('Original module:')
        print(wl.mod)
        if self.viz_orig:
            wl.visualize()

        # Apply built-in pass
        pass_mod = self.get_pass()(wl.mod)
        print('After built-in pass:')
        print(pass_mod)
        if self.viz_pass:
            Workload(pass_mod, {}).visualize()

        # Use Relay DFP for substitution
        if self.pattern is not None:
            dfp_body = self.rewrite(wl.mod['main'].body)
            dfp_mod = ir.IRModule(functions={
                'main': relay.Function(relay.analysis.free_vars(dfp_body),
                                       dfp_body)
            })
            dfp_mod = relay.transform.InferType()(dfp_mod)
            print('After DFP rewrite:')
            print(dfp_mod)
            if self.viz_dfp:
                Workload(dfp_mod, {}).visualize()

        # Use GSL for substitution
        rule = self.define_gsl()
        if rule is not None:
            gsl_wl = rule(wl, fast_mode=True, fold_params=False)
            print('After GSL substitution:')
            print(gsl_wl.mod)
            if self.viz_gsl:
                gsl_wl.visualize()


def _pos_axis(axis: int, ndim: int):
    return axis if axis >= 0 else ndim + axis


def _pos_axis_attr(axis: attr.Attr, ndim: attr.Attr):
    return attr.Cond(axis >= 0, axis, ndim + axis)


def _shape(expr: relay.Expr):
    return expr.checked_type.concrete_shape


def _ndim(expr: relay.Expr) -> int:
    return len(_shape(expr))


def _num_new_axis(axis: int, ndim: int):
    return ndim - 1 - axis


def _num_new_axis_attr(axis: attr.Attr, ndim: attr.Attr):
    return ndim - 1 - axis


class SimplifyReshape(SubstTest):
    def create_expr(self) -> relay.Expr:
        x = relay.var('x', shape=(2, 4, 4))
        x = relay.reshape(x, newshape=(2, 16))
        x = relay.reshape(x, newshape=(2, 1, 16))
        return relay.reshape(x, newshape=(2, 2, 8))

    def get_pass(self) -> transform.Pass:
        return relay.transform.SimplifyExpr()

    def define_gsl(self) -> Optional[Subst]:
        x = pat.Wildcard()
        src = op.Reshape(op.Reshape(x))
        tgt = op.Reshape(x, newshape=src.newshape)
        return Subst(src, tgt)


class SimplifyTranspose(SubstTest):
    def create_expr(self) -> relay.Expr:
        x = relay.var('x', shape=(2, 4, 6, 8))
        x = relay.transpose(x)
        x = relay.transpose(x, axes=(0, 3, 1, 2))
        return relay.transpose(x, axes=(3, 1, 0, 2))

    def get_pass(self) -> transform.Pass:
        return relay.transform.SimplifyExpr()

    def define_gsl(self) -> Optional[Subst]:
        x = pat.Wildcard()
        trans1 = op.Transpose(x)
        trans2 = op.Transpose(trans1)

        axes1 = self._get_axes(trans1.axes, x.ndim)
        axes2 = self._get_axes(trans2.axes, x.ndim)
        axes = attr.Map(axes2, lambda a: axes1[a])
        trans = pat.Cond(axes == attr.Range(x.ndim), x, op.Transpose(x, axes))

        return Subst(trans2, trans)

    @staticmethod
    def _get_axes(axes: attr.Attr, ndim: attr.Attr):
        return attr.Cond(axes == attr.NoneAttr(), attr.Reverse(attr.Range(ndim)),
                         attr.Map(axes, lambda a: _pos_axis_attr(a, ndim)))


class LowerBatchNorm(SubstTest):
    def __init__(self):
        super().__init__()

        self.x = dfp.wildcard()
        self.gamma = dfp.is_var()
        self.beta = dfp.is_var()
        self.mean = dfp.is_var()
        self.var = dfp.is_var()
        self.bn = dfp.is_op('nn.batch_norm')(self.x, self.gamma, self.beta, self.mean, self.var)
        self.pattern = dfp.is_tuple_get_item(self.bn, index=0)

    def create_expr(self) -> relay.Expr:
        x = relay.var('x', shape=(2, 4, 4, 4))
        gamma = relay.var('gamma', shape=(4,))
        beta = relay.var('beta', shape=(4,))
        mean = relay.var('mean', shape=(4,))
        var = relay.var('var', shape=(4,))
        bn = relay.nn.batch_norm(x, gamma, beta, mean, var, axis=1)
        return bn[0]

    def get_pass(self) -> transform.Pass:
        return relay.transform.SimplifyInference()

    def rewrite_dfp(self, node_map: Dict[dfp.DFPattern, relay.Expr]) -> relay.Expr:
        bn = node_map[self.bn]
        std = relay.sqrt(node_map[self.var] + relay.const(bn.attrs['epsilon']))
        k: relay.Expr = node_map[self.gamma] / std if bn.attrs['scale'] else 1.0 / std
        bias = node_map[self.beta] - node_map[self.mean] * k if bn.attrs['center'] else \
            -node_map[self.mean] * k
        ndim = _ndim(node_map[self.x])
        axis = _pos_axis(bn.attrs['axis'], ndim)
        k_expand = relay.expand_dims(k, 1, ndim - 1 - axis)
        return relay.nn.bias_add(node_map[self.x] * k_expand, bias)

    def define_gsl(self) -> Optional[Subst]:
        x = pat.Wildcard()
        gamma = pat.Variable()
        beta = pat.Variable()
        mean = pat.Variable()
        var = pat.Variable()

        bn = op.BatchNorm(x, gamma, beta, mean, var)
        y1 = bn[0]

        std = op.Sqrt(var + bn.epsilon)
        k = pat.Cond(bn.scale, gamma / std, 1.0 / std)
        bias = pat.Cond(bn.center, beta - mean * k, -mean * k)
        n_new = _num_new_axis_attr(bn.axis, x.ndim)
        k = pat.Cond(n_new > 0, op.ExpandDims(k, axis=1, num_newaxis=n_new), k)
        y2 = op.BiasAdd(x * k, bias, axis=bn.axis)

        # Build substitution
        return Subst(y1, y2)


class LowerLayerNorm(SubstTest):
    def __init__(self):
        super().__init__()

        self.x = dfp.wildcard()
        self.gamma = dfp.is_var()
        self.beta = dfp.is_var()
        self.ln = dfp.is_op('nn.layer_norm')(self.x, self.gamma, self.beta)
        self.pattern = self.ln

    def create_expr(self) -> relay.Expr:
        x = relay.var('x', shape=(2, 4, 4))
        gamma = relay.var('gamma', shape=(4,))
        beta = relay.var('beta', shape=(4,))
        return relay.nn.layer_norm(x, gamma, beta, axis=-1)

    def get_pass(self) -> transform.Pass:
        return relay.transform.SimplifyInference()

    def rewrite_dfp(self, node_map: Dict[dfp.DFPattern, relay.Expr]) -> relay.Expr:
        x = node_map[self.x]
        ln = node_map[self.ln]
        axis = ln.attrs['axis']
        mean = relay.mean(x, axis=(axis,), keepdims=True)
        demean = x - mean
        # To fairly compare with GSL, we avoid using relay.variance API.
        var = relay.sum(demean * demean, axis=(axis,), keepdims=True) / relay.cast(
            relay.const(x.checked_type.concrete_shape[axis]), dtype=x.checked_type.dtype)
        norm = demean / relay.sqrt(var + relay.const(ln.attrs['epsilon']))
        n_new = _ndim(x) - 1 - _pos_axis(axis, _ndim(x))
        gamma = node_map[self.gamma]
        beta = node_map[self.beta]
        if n_new > 0:
            gamma = relay.expand_dims(gamma, 1, num_newaxis=n_new)
            beta = relay.expand_dims(beta, 1, num_newaxis=n_new)
        scale = norm * gamma if ln.attrs['scale'] else norm
        center = scale + beta if ln.attrs['center'] else scale
        return center

    def define_gsl(self) -> Optional[Subst]:
        x = pat.Wildcard()
        gamma = pat.Variable()
        beta = pat.Variable()

        ln = op.LayerNorm(x, gamma, beta)

        mean = op.Mean(x, axis=(ln.axis,), keepdims=True)
        demean = x - mean
        # We avoid using `variance` op because its API is not consistent with op definition.
        # The same case for LowerGroupNorm and LowerInstanceNorm
        var = op.Sum(demean * demean, axis=(ln.axis,), keepdims=True) / \
              op.Cast(x.shape[ln.axis], dtype=x.dtype)
        norm = demean / op.Sqrt(var + ln.epsilon)
        axis = _pos_axis_attr(ln.axis, x.ndim)
        n_new = _num_new_axis_attr(axis, x.ndim)
        gamma = pat.Cond(n_new > 0, op.ExpandDims(gamma, axis=1, num_newaxis=n_new), gamma)
        beta = pat.Cond(n_new > 0, op.ExpandDims(beta, axis=1, num_newaxis=n_new), beta)
        scale = pat.Cond(ln.scale, norm * gamma, norm)
        center = pat.Cond(ln.center, scale + beta, scale)

        return Subst(ln, center)


class LowerGroupNorm(SubstTest):
    def __init__(self):
        super().__init__()

        self.x = dfp.wildcard()
        self.gamma = dfp.is_var()
        self.beta = dfp.is_var()
        self.gn = dfp.is_op('nn.group_norm')(self.x, self.gamma, self.beta)
        self.pattern = self.gn

    def create_expr(self) -> relay.Expr:
        x = relay.var('x', shape=(2, 4, 4, 4))
        gamma = relay.var('gamma', shape=(4,))
        beta = relay.var('beta', shape=(4,))
        return relay.nn.group_norm(x, gamma, beta, 2)

    def get_pass(self) -> transform.Pass:
        return relay.transform.SimplifyInference()

    def rewrite_dfp(self, node_map: Dict[dfp.DFPattern, relay.Expr]) -> relay.Expr:
        x = node_map[self.x]
        gn = node_map[self.gn]
        axis = gn.attrs['axis']
        n_grp = gn.attrs['num_groups']
        x_shape = _shape(x)
        new_shape = x_shape[:axis] + (n_grp, x_shape[axis] // n_grp) + x_shape[axis + 1:]
        reduced_axes = tuple(range(axis + 1, _ndim(x) + 1))
        reshaped = relay.reshape(x, new_shape)
        mean = relay.mean(reshaped, axis=reduced_axes, keepdims=True)
        demean = reshaped - mean
        n_val = reduce(int.__mul__, new_shape[axis + 1:], 1)
        var = relay.sum(demean * demean, axis=reduced_axes, keepdims=True) / \
              relay.cast(relay.const(n_val), dtype=x.checked_type.dtype)
        norm = demean / relay.sqrt(var + relay.const(gn.attrs['epsilon']))
        norm = relay.reshape(norm, x_shape)
        n_new = _num_new_axis(axis, _ndim(x))
        gamma = node_map[self.gamma]
        beta = node_map[self.beta]
        if n_new > 0:
            gamma = relay.expand_dims(gamma, 1, num_newaxis=n_new)
            beta = relay.expand_dims(beta, 1, num_newaxis=n_new)
        scale = norm * gamma if gn.attrs['scale'] else norm
        center = scale + beta if gn.attrs['center'] else scale
        return center

    def define_gsl(self) -> Optional[Subst]:
        x = pat.Wildcard()
        gamma = pat.Variable()
        beta = pat.Variable()

        gn = op.GroupNorm(x, gamma, beta)

        axis = _pos_axis_attr(gn.axis, x.ndim)
        n_grp = gn.num_groups
        new_shape = x.shape[attr.Slice(stop=axis)] + (n_grp, x.shape[axis] // n_grp) + \
                    x.shape[attr.Slice(start=axis + 1)]
        reduced_axes = attr.Range(start=axis + 1, stop=x.ndim + 1)
        reshaped = op.Reshape(x, new_shape)
        mean = op.Mean(reshaped, axis=reduced_axes, keepdims=True)
        demean = reshaped - mean
        n_val = attr.ReduceTuple(attr.BinaryOp.MUL, new_shape[attr.Slice(start=axis + 1)])
        var = op.Sum(demean * demean, axis=reduced_axes, keepdims=True) / \
              op.Cast(n_val, dtype=x.dtype)
        norm = demean / op.Sqrt(var + gn.epsilon)
        norm = op.Reshape(norm, newshape=x.shape)
        n_new = _num_new_axis_attr(axis, x.ndim)
        gamma = pat.Cond(n_new > 0, op.ExpandDims(gamma, axis=1, num_newaxis=n_new), gamma)
        beta = pat.Cond(n_new > 0, op.ExpandDims(beta, axis=1, num_newaxis=n_new), beta)
        scale = pat.Cond(gn.scale, norm * gamma, norm)
        center = pat.Cond(gn.center, scale + beta, scale)

        return Subst(gn, center)


class LowerInstanceNorm(SubstTest):
    def __init__(self):
        super().__init__()

        self.x = dfp.wildcard()
        self.gamma = dfp.is_var()
        self.beta = dfp.is_var()
        self.norm = dfp.is_op('nn.instance_norm')(self.x, self.gamma, self.beta)
        self.pattern = self.norm

    def create_expr(self) -> relay.Expr:
        x = relay.var('x', shape=(2, 4, 4, 4))
        gamma = relay.var('gamma', shape=(4,))
        beta = relay.var('beta', shape=(4,))
        return relay.nn.instance_norm(x, gamma, beta)

    def get_pass(self) -> transform.Pass:
        return relay.transform.SimplifyInference()

    def rewrite_dfp(self, node_map: Dict[dfp.DFPattern, relay.Expr]) -> relay.Expr:
        x = node_map[self.x]
        ndim = _ndim(x)
        inst_norm = node_map[self.norm]
        axis = inst_norm.attrs['axis']
        reduced_axes = tuple(range(1, axis)) + tuple(range(axis + 1, ndim))
        mean = relay.mean(x, axis=reduced_axes, keepdims=True)
        demean = x - mean
        x_shape = _shape(x)
        n_val = reduce(int.__mul__, map(lambda a: x_shape[a], reduced_axes))
        var = relay.sum(demean * demean, axis=reduced_axes, keepdims=True) / \
              relay.cast(relay.const(n_val), dtype=x.checked_type.dtype)
        norm = demean / relay.sqrt(var + relay.const(inst_norm.attrs['epsilon']))
        n_new = _num_new_axis(axis, ndim)
        gamma = node_map[self.gamma]
        beta = node_map[self.beta]
        if n_new > 0:
            gamma = relay.expand_dims(gamma, 1, num_newaxis=n_new)
            beta = relay.expand_dims(beta, 1, num_newaxis=n_new)
        scale = norm * gamma if inst_norm.attrs['scale'] else norm
        center = scale + beta if inst_norm.attrs['center'] else scale
        return center

    def define_gsl(self) -> Optional[Subst]:
        x = pat.Wildcard()
        gamma = pat.Variable()
        beta = pat.Variable()

        inst_norm = op.InstanceNorm(x, gamma, beta)

        axis = _pos_axis_attr(inst_norm.axis, x.ndim)
        reduced_axes = attr.Range(start=1, stop=axis) + attr.Range(start=axis + 1, stop=x.ndim)
        mean = op.Mean(x, axis=reduced_axes, keepdims=True)
        demean = x - mean
        n_val = attr.ReduceTuple(attr.BinaryOp.MUL, attr.Map(reduced_axes, lambda a: x.shape[a]), 1)
        var = op.Sum(demean * demean, axis=reduced_axes, keepdims=True) / \
              op.Cast(n_val, dtype=x.dtype)
        norm = demean / op.Sqrt(var + inst_norm.epsilon)
        n_new = _num_new_axis_attr(axis, x.ndim)
        gamma = pat.Cond(n_new > 0, op.ExpandDims(gamma, axis=1, num_newaxis=n_new), gamma)
        beta = pat.Cond(n_new > 0, op.ExpandDims(beta, axis=1, num_newaxis=n_new), beta)
        scale = pat.Cond(inst_norm.scale, norm * gamma, norm)
        center = pat.Cond(inst_norm.center, scale + beta, scale)

        return Subst(inst_norm, center)


class LowerL2Norm(SubstTest):
    def __init__(self):
        super().__init__()

        self.x = dfp.wildcard()
        self.l2 = dfp.is_op('nn.l2_normalize')(self.x)
        self.pattern = self.l2

    def create_expr(self) -> relay.Expr:
        x = relay.var('x', shape=(2, 4, 4, 4))
        return relay.nn.l2_normalize(x, 1e-5)

    def get_pass(self) -> transform.Pass:
        return relay.transform.SimplifyInference()

    def rewrite_dfp(self, node_map: Dict[dfp.DFPattern, relay.Expr]) -> relay.Expr:
        x = node_map[self.x]
        l2 = node_map[self.l2]
        return x / relay.sqrt(relay.maximum(relay.sum(x * x, l2.attrs['axis'], keepdims=True),
                                            relay.const(l2.attrs['eps'])))

    def define_gsl(self) -> Optional[Subst]:
        x = pat.Wildcard()
        l2 = op.L2Normalize(x)
        result = x / op.Sqrt(op.Maximum(op.Sum(x * x, l2.axis, keepdims=True), l2.eps))
        return Subst(l2, result)


class CombineParallelConv2D(SubstTest):
    def create_expr(self) -> relay.Expr:
        x = relay.var('x', shape=(2, 2, 4, 4))
        w1 = relay.var('w1', shape=(2, 2, 3, 3))
        w2 = relay.var('w2', shape=(2, 2, 3, 3))
        w3 = relay.var('w3', shape=(4, 2, 3, 3))
        out = [relay.nn.conv2d(x, w, padding=(1, 1, 1, 1)) for w in [w1, w2, w3]]
        return relay.concatenate(out, 1)

    def get_pass(self) -> transform.Pass:
        return relay.transform.CombineParallelConv2D(min_num_branches=2)

    def define_gsl(self) -> Optional[Subst]:
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


class CombineParallelDense(SubstTest):
    def create_expr(self) -> relay.Expr:
        x = relay.var('x', shape=(2, 2))
        w1 = relay.var('w1', shape=(2, 2))
        w2 = relay.var('w2', shape=(2, 2))
        w3 = relay.var('w3', shape=(4, 2))
        out = [relay.nn.dense(x, w) for w in [w1, w2, w3]]
        return relay.concatenate(out, axis=1)

    def get_pass(self) -> transform.Pass:
        return relay.transform.CombineParallelDense(min_num_branches=2, to_batch=False)

    def define_gsl(self) -> Optional[Subst]:
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


class CombineParallelBatchMatmul(SubstTest):
    def create_expr(self) -> relay.Expr:
        x = relay.var('x', shape=(2, 4, 4))
        y1 = relay.var('y1', shape=(2, 2, 4))
        y2 = relay.var('y2', shape=(2, 2, 4))
        y3 = relay.var('y3', shape=(2, 4, 4))
        out = [relay.nn.batch_matmul(x, y) for y in [y1, y2, y3]]
        return relay.concatenate(out, 2)

    def get_pass(self) -> transform.Pass:
        return relay.transform.CombineParallelBatchMatmul(2)

    def define_gsl(self) -> Optional[Subst]:
        x = pat.Wildcard()
        y1 = pat.Variable()
        matmul1 = op.BatchMatmul(x, y1)
        y = pat.Variable(shape=(y1.shape[0], None, None), dtype=y1.dtype)
        matmul = op.BatchMatmul(x, y)
        src = pat.Variadic(matmul, templates=[matmul, y], first=[matmul1, y1], min_len=2)

        i = attr.Symbol()
        yi = src(y, i)
        matmul = op.BatchMatmul(x, op.Concatenate(
            pat.Variadic(yi, templates=[yi], index=i, length=src.length), axis=1))
        split = op.Split(matmul, axis=2, indices_or_sections=attr.Variadic(
            lambda j: attr.ReduceIndexed(attr.BinaryOp.ADD, lambda k: src(y, k).shape[1], j + 1),
            length=src.length - 1))
        i = attr.Symbol()
        item = split[i]
        tgt = pat.Variadic(item, templates=[item], index=i)

        return Subst(src, tgt)


if __name__ == '__main__':
    for cls in [
        # SimplifyReshape,
        # SimplifyTranspose,
        # LowerBatchNorm,
        # LowerLayerNorm,
        # LowerGroupNorm,
        # LowerInstanceNorm,
        # LowerL2Norm,
        # CombineParallelConv2D,
        # CombineParallelDense,
        # CombineParallelBatchMatmul,
    ]:
        cls().run()
