from gsl import *


def trans_trans():
    # Input
    x = pat.Wildcard()

    # Source pattern: (A^T)^T
    y1 = op.Transpose(op.Transpose(x, axes=[0, 2, 1]), axes=[0, 2, 1])

    # Target pattern: A
    y2 = x

    # Build substitution
    return Subst(y1, y2)


def split_concat():
    # Input
    x = pat.Wildcard()

    # Source pattern: concat(split(x, axis=a), axis=a)
    split = op.Split(x, indices_or_sections=2)
    y1 = op.Concatenate((split[0], split[1]), axis=split.axis)

    # Target pattern: x
    y2 = x

    # Build substitution
    return Subst(y1, y2)


def split_concat_variadic():
    # Input
    x = pat.Wildcard()

    # Source pattern: concat(split(x, axis=a), axis=a)
    split = op.Split(x)
    i = attr.Symbol()
    item = split[i]
    y1 = op.Concatenate(pat.Variadic(item, templates=[item], index=i), axis=split.axis)

    # Target pattern: x
    y2 = x

    # Build substitution
    return Subst(y1, y2)


def bias_add_add():
    # Input
    x1 = pat.Wildcard()
    x2 = pat.Wildcard()
    b1 = pat.Var()
    b2 = pat.Var()

    # Source pattern: (x1 + b1) + (x2 + b2)
    add1 = op.BiasAdd(x1, b1, axis=1)
    add2 = op.BiasAdd(x2, b2, axis=add1.axis)
    y1 = add1 + add2

    # Target pattern: (x1 + x2) + (b1 + b2)
    y2 = op.BiasAdd(x1 + x2, b1 + b2, axis=add1.axis)

    # Build substitution
    return Subst(y1, y2)


def diamond_conv_add():
    # Input
    x = pat.Wildcard()
    w1 = pat.Var()
    w2 = pat.Var(shape=w1.shape)

    # Source pattern: conv2d(x, w1) + conv2d(x, w2)
    conv1 = op.Conv2D(x, w1)
    conv2 = op.Conv2D(x, w2, strides=conv1.strides, padding=conv1.padding,
                      dilation=conv1.dilation, groups=conv1.groups)
    y1 = conv1 + conv2

    # Target pattern: conv2d(x, w1 + w2)
    y2 = op.Conv2D(x, w1 + w2, strides=conv1.strides, padding=conv1.padding,
                   dilation=conv1.dilation, groups=conv1.groups)

    # Build substitution
    return Subst(y1, y2)


def two_conv_add():
    # Input
    x1 = pat.Wildcard()
    x2 = pat.Wildcard()
    w1 = pat.Var()
    w2 = pat.Var(shape=w1.shape)

    # Source pattern: conv2d(x1, w1) + conv2d(x2, w2)
    conv1 = op.Conv2D(x1, w1)
    conv2 = op.Conv2D(x2, w2, **pat.same_attr(conv1, ['strides', 'padding', 'dilation', 'groups']))
    y1 = conv1 + conv2

    # Target pattern: conv2d(concat(x1, x2), concat(w1, w2))
    x_concat = op.Concatenate((x1, x2), axis=1)
    w_concat = op.Concatenate((w1, w2), axis=1)
    y2 = op.Conv2D(x_concat, w_concat,
                   **pat.same_attr(conv1, ['strides', 'padding', 'dilation', 'groups']))

    # Build substitution
    return Subst(y1, y2)


def conv_shortcut_add():
    # Input
    x = pat.Wildcard()
    w = pat.Var()

    # Source pattern: conv2d(x, w) + x
    conv = op.Conv2D(x, w)
    y1 = conv + x

    # Target pattern: conv2d(x, w + I)
    zeros = op.Zeros(shape=(w.shape[0],) * 2, dtype=w.dtype)
    diag = op.Ones(shape=(w.shape[0],), dtype=w.dtype)
    eye = op.ExpandDims(op.MatrixSetDiag(zeros, diag), axis=2, num_newaxis=2)
    pad_w, pad_h = (w.shape[2] - 1) // 2, (w.shape[3] - 1) // 2
    new_wt = w + op.Pad(eye, pad_width=((0, 0), (0, 0), (pad_w,) * 2, (pad_h,) * 2))
    y2 = op.Conv2D(x, new_wt, **pat.same_attr(conv, ['strides', 'padding', 'dilation', 'groups']))

    # Build substitution
    return Subst(y1, y2)


def lower_batch_norm():
    # Input
    x = pat.Wildcard()
    gamma = pat.Var()
    beta = pat.Var()
    moving_mean = pat.Var()
    moving_var = pat.Var()

    # Source pattern: batch_norm(x, gamma, beta, mean, var)
    bn = op.BatchNorm(x, gamma, beta, moving_mean, moving_var, axis=1, scale=True, center=True)
    y1 = bn[0]

    # Target pattern: k = gamma / sqrt(var + epsilon), x * k + beta - mean * k
    k = gamma / op.Sqrt(moving_var + bn.epsilon)
    bias = beta - moving_mean * k
    y2 = op.BiasAdd(x * op.ExpandDims(k, axis=1, num_newaxis=2), bias)

    # Build substitution
    return Subst(y1, y2)


def lower_layer_norm():
    # Input
    x = pat.Wildcard()
    gamma = pat.Var()
    beta = pat.Var()

    # Source pattern layer_norm(x, gamma, beta)
    y1 = op.LayerNorm(x, gamma, beta, axis=-1, scale=True, center=True)

    # Target pattern:
    # (data - mean(data)) / sqrt(var(data) + epsilon)) * gamma + beta
    mean = op.Mean(x, axis=(-1,), keepdims=True)
    demean = x - mean
    var = op.Sum(demean * demean, axis=(-1,),
                 keepdims=True) / op.Cast(gamma.shape[0], dtype=gamma.dtype)
    norm = demean / op.Sqrt(var + y1.epsilon)
    y2 = norm * gamma + beta

    # Build substitution
    return Subst(y1, y2)


def conv_mul():
    # Input
    x = pat.Wildcard()
    w = pat.Var()
    k = pat.Var(shape=(None, 1, 1))

    # Source pattern: conv2d(x, w) * k
    conv = op.Conv2D(x, w)
    y1 = conv * k

    # Target pattern: conv2d(x, w * k)
    fused_w = w * op.ExpandDims(k, axis=1, num_newaxis=1)
    y2 = op.Conv2D(x, fused_w,
                   **pat.same_attr(conv, ['strides', 'padding', 'dilation', 'groups']))

    # Build substitution
    return Subst(y1, y2)


def merge_element_wise():
    # Input
    x = pat.Wildcard()

    # Source pattern
    ew_op = pat.OpWithFlag(spec.OpFlag.ELEMENT_WISE)
    ew1 = pat.Call(ew_op, x)
    ew2 = pat.Call(ew_op, x)

    # Target pattern
    ew = pat.Call(ew_op, x)

    # Build substitution
    return Subst([ew1, ew2], [ew, ew])


def merge_element_wise_variadic():
    # Input
    x = pat.Wildcard()

    # Source pattern
    ew_op = pat.OpWithFlag(spec.OpFlag.ELEMENT_WISE)
    call = pat.Call(ew_op, x)
    src = pat.Variadic(call, templates=[call])

    # Target pattern
    ew = pat.Call(ew_op, x)
    tgt = pat.Variadic(ew)

    # Build substitution
    return Subst(src, tgt)


def dispatch_tuple():
    # Source pattern
    x = pat.Wildcard()
    tup = pat.Variadic(x, templates=[x])
    get_item = pat.GetItem(tup)
    src = pat.Variadic(get_item, templates=[get_item])

    # Target pattern
    i = attr.Symbol()
    item = tup(src(i, get_item).index, x)
    tgt = pat.Variadic(item, templates=[item], index=i)

    return Subst(src, tgt)


def parallel_conv():
    # Input
    x = pat.Wildcard()
    w1 = pat.Var()
    w2 = pat.Var(shape=w1.shape)

    # Source pattern
    conv1 = op.Conv2D(x, w1, groups=1)
    conv2 = op.Conv2D(x, w2, **pat.same_attr(conv1, ['strides', 'padding', 'dilation', 'groups']))

    # Target pattern
    w = op.Concatenate((w1, w2), axis=0)
    conv = op.Conv2D(x, w, **pat.same_attr(conv1, ['strides', 'padding', 'dilation', 'groups']))
    split = op.Split(conv, indices_or_sections=2, axis=1)

    # Build substitution
    return Subst([conv1, conv2], [split[0], split[1]])


def parallel_conv_variadic():
    # Input
    x = pat.Wildcard()
    w1 = pat.Var()
    w = pat.Var(shape=(None, None, w1.shape[2], w1.shape[3]))

    # Source pattern
    conv1 = op.Conv2D(x, w1, groups=1)
    conv = op.Conv2D(x, w, **pat.same_attr(conv1, ['strides', 'padding', 'dilation', 'groups']))
    src = pat.Variadic(conv, templates=[conv, w], first=[conv1, w1], min_len=2)

    # Target pattern
    i = attr.Symbol()
    w_inst = src(i, w)
    concat = op.Concatenate(pat.Variadic(w_inst, templates=[w_inst], index=i, length=src.length),
                            axis=0)
    conv = op.Conv2D(x, concat,
                     **pat.same_attr(conv1, ['strides', 'padding', 'dilation', 'groups']))

    i = attr.Symbol()
    j = attr.Symbol()
    split = op.Split(conv, axis=1, indices_or_sections=attr.Variadic(
        attr.Sum(src(j, w).shape[0], j, i + 1), index=i, length=src.length - 1))
    i = attr.Symbol()
    item = split[i]
    tgt = pat.Variadic(item, templates=[item], index=i)

    # Build substitution
    return Subst(src, tgt)


def parallel_conv_expand_kernels():
    # Input
    x = pat.Wildcard()
    w1 = pat.Var()
    w2 = pat.Var(shape=(w1.shape[0], None, None, None))

    # Source pattern
    def same_padding(h: attr.Attr, w: attr.Attr):
        pad_h = (h - 1) // 2
        pad_w = (w - 1) // 2
        return pad_h, pad_w, pad_h, pad_w

    conv1_pad = same_padding(w1.shape[2], w1.shape[3])
    conv1 = op.Conv2D(x, w1, padding=conv1_pad, strides=(1, 1), dilation=(1, 1), groups=1)
    conv2_pad = same_padding(w2.shape[2], w2.shape[3])
    conv2 = op.Conv2D(x, w2, padding=conv2_pad,
                      **pat.same_attr(conv1, ['strides', 'dilation', 'groups']))

    # Target pattern
    max_h, max_w = w1.shape[2].max(w2.shape[2]), w1.shape[3].max(w2.shape[3])
    w1_pad_h = (max_h - w1.shape[2]) // 2
    w1_pad_w = (max_w - w1.shape[3]) // 2
    w1_pad = op.Pad(w1, pad_width=((0, 0), (0, 0), (w1_pad_h,) * 2, (w1_pad_w,) * 2))
    w2_pad_h = (max_h - w2.shape[2]) // 2
    w2_pad_w = (max_w - w2.shape[3]) // 2
    w2_pad = op.Pad(w2, pad_width=((0, 0), (0, 0), (w2_pad_h,) * 2, (w2_pad_w,) * 2))
    concat = op.Concatenate((w1_pad, w2_pad), axis=0)
    new_conv_pad = same_padding(max_h, max_w)
    new_conv = op.Conv2D(x, concat, padding=new_conv_pad,
                         **pat.same_attr(conv1, ['strides', 'dilation', 'groups']))
    split = op.Split(new_conv, indices_or_sections=2, axis=1)

    # Build substitution
    return Subst([conv1, conv2], [split[0], split[1]])


def parallel_group_conv_variadic():
    # Input
    x = pat.Wildcard()
    w1 = pat.Var()
    w = pat.Var(shape=(w1.shape[0], None, w1.shape[2], w1.shape[3]))

    # Source pattern
    conv1 = op.Conv2D(x, w1)
    conv = op.Conv2D(x, w, **pat.same_attr(conv1, ['strides', 'padding', 'dilation', 'groups']))
    src = pat.Variadic(conv, templates=[conv, w], first=[conv1, w1], min_len=2)

    # Target pattern
    i = attr.Symbol()
    w_inst = src(i, w)
    w_concat = op.Concatenate(pat.Variadic(w_inst, templates=[w_inst], index=i, length=src.length),
                              axis=0)
    w_expand = op.ExpandDims(w_concat, axis=1, num_newaxis=2)
    num_conv = src.length
    groups = conv1.groups
    chan_per_group = w1.shape[0] // groups
    w_reshape = op.Reshape(w_expand, newshape=(num_conv, groups, chan_per_group, -2))
    w_trans = op.Transpose(w_reshape, axes=[1, 0, 2, 3, 4, 5])
    w_comb = op.Reshape(w_trans, newshape=(-1, w1.shape[1], w1.shape[2], w1.shape[3]))

    conv = op.Conv2D(x, w_comb,
                     **pat.same_attr(conv1, ['strides', 'padding', 'dilation', 'groups']))
    conv_expand = op.ExpandDims(conv, axis=2, num_newaxis=2)
    conv_reshape = op.Reshape(conv_expand, newshape=(0, groups, num_conv, chan_per_group, -2))
    conv_trans = op.Transpose(conv_reshape, axes=[0, 2, 1, 3, 4, 5])
    conv_comb = op.Squeeze(op.Reshape(conv_trans, newshape=(0, num_conv * w1.shape[0], 1, 1, -2)),
                           axis=[2, 3])

    split = op.Split(conv_comb, indices_or_sections=num_conv, axis=1)
    i = attr.Symbol()
    item = split[i]
    tgt = pat.Variadic(item, [item], index=i)

    # Build substitution
    return Subst(src, tgt)


def parallel_dense():
    # Input
    x = pat.Wildcard()
    w1 = pat.Var()
    w2 = pat.Var()

    # Source pattern
    dense1 = op.Dense(x, w1)
    dense2 = op.Dense(x, w2)

    # Target pattern
    dense = op.Dense(x, op.Concatenate((w1, w2), axis=0))
    split = op.Split(dense, indices_or_sections=(w1.shape[0],), axis=-1)

    # Build substitution
    return Subst([dense1, dense2], [split[0], split[1]])


def parallel_dense_variadic():
    # Input
    x = pat.Wildcard()
    w1 = pat.Var()
    w = pat.Var(shape=(None, w1.shape[1]))

    # Source pattern
    dense = op.Dense(x, w)
    src = pat.Variadic(dense, templates=[dense, w], first=[None, w1], min_len=2)

    # Target pattern
    i = attr.Symbol()
    w_inst = src(i, w)
    dense = op.Dense(x, op.Concatenate(pat.Variadic(w_inst, templates=[w_inst], index=i,
                                                    length=src.length),
                                       axis=0))
    i = attr.Symbol()
    j = attr.Symbol()
    split = op.Split(dense, axis=-1, indices_or_sections=attr.Variadic(
        attr.Sum(src(j, w).shape[0], j, i + 1), index=i, length=src.length - 1))
    i = attr.Symbol()
    item = split[i]
    tgt = pat.Variadic(item, templates=[item], index=i)

    # Build substitution
    return Subst(src, tgt)
