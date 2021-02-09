from gsl import *


def trans_trans():
    # Input
    x = Wildcard()

    # Source pattern: (A^T)^T
    y1 = Transpose(Transpose(x, axes=(0, 2, 1)), axes=(0, 2, 1))

    # Target pattern: A
    y2 = x

    # Build substitution
    return Substitution(y1, y2)


def split_concat():
    # Input
    x = Wildcard()

    # Source pattern: concat(split(x, axis=a), axis=a)
    split = Split(x, indices_or_sections=2)
    y1 = Concatenate((split[0], split[1]), axis=split.axis)

    # Target pattern: x
    y2 = x

    # Build substitution
    return Substitution(y1, y2)


def split_concat_variadic():
    # Inputs
    x = Wildcard()

    # Source pattern: concat(split(x, axis=a), axis=a)
    split = Split(x)
    i = Symbol()
    item = split[i]
    y1 = Concatenate(Variadic(item, templates=[item], index=i), axis=split.axis)

    # Target pattern: x
    y2 = x

    # Build substitution
    return Substitution(y1, y2)


def bias_add_add():
    # Input
    x1 = Wildcard()
    x2 = Wildcard()
    b1 = Var()
    b2 = Var()

    # Source pattern: (x1 + b1) + (x2 + b2)
    add1 = BiasAdd(x1, b1, axis=1)
    add2 = BiasAdd(x2, b2, axis=add1.axis)
    y1 = add1 + add2

    # Target pattern: (x1 + x2) + (b1 + b2)
    y2 = BiasAdd(x1 + x2, b1 + b2, axis=add1.axis)

    # Build substitution
    return Substitution(y1, y2)


def diamond_conv_add():
    # Input
    x = Wildcard()
    w1 = Var()
    w2 = Var(shape=w1.shape)

    # Source pattern: conv2d(x, w1) + conv2d(x, w2)
    conv1 = Conv2D(x, w1)
    conv2 = Conv2D(x, w2, strides=conv1.strides, padding=conv1.padding,
                   dilation=conv1.dilation, groups=conv1.groups)
    y1 = conv1 + conv2

    # Target pattern: conv2d(x, w1 + w2)
    y2 = Conv2D(x, w1 + w2, strides=conv1.strides, padding=conv1.padding,
                dilation=conv1.dilation, groups=conv1.groups)

    # Build substitution
    return Substitution(y1, y2)


def two_conv_add():
    # Input
    x1 = Wildcard()
    x2 = Wildcard()
    w1 = Var()
    w2 = Var(shape=w1.shape)

    # Source pattern: conv2d(x1, w1) + conv2d(x2, w2)
    conv1 = Conv2D(x1, w1)
    conv2 = Conv2D(x2, w2, **same_attr(conv1, ['strides', 'padding', 'dilation', 'groups']))
    y1 = conv1 + conv2

    # Target pattern: conv2d(concat(x1, x2), concat(w1, w2))
    x_concat = Concatenate((x1, x2), axis=1)
    w_concat = Concatenate((w1, w2), axis=1)
    y2 = Conv2D(x_concat, w_concat,
                **same_attr(conv1, ['strides', 'padding', 'dilation', 'groups']))

    # Build substitution
    return Substitution(y1, y2)


def conv_shortcut_add():
    # Input
    x = Wildcard()
    w = Var()

    # Source pattern: conv2d(x, w) + x
    conv = Conv2D(x, w)
    y1 = conv + x

    # Target pattern: conv2d(x, w + I)
    zeros = Zeros(shape=(w.shape[0],) * 2, dtype=w.dtype)
    diag = Ones(shape=(w.shape[0],), dtype=w.dtype)
    eye = ExpandDims(MatrixSetDiag(zeros, diag), axis=2, num_newaxis=2)
    pad_w, pad_h = (w.shape[2] - 1) // 2, (w.shape[3] - 1) // 2
    new_wt = w + Pad(eye, pad_width=((0, 0), (0, 0), (pad_w,) * 2, (pad_h,) * 2))
    y2 = Conv2D(x, new_wt, **same_attr(conv, ['strides', 'padding', 'dilation', 'groups']))

    # Build substitution
    return Substitution(y1, y2)


def simplify_batch_norm():
    # Input
    x = Wildcard()
    gamma = Var()
    beta = Var()
    moving_mean = Var()
    moving_var = Var()

    # Source pattern: batch_norm(x, gamma, beta, mean, var)
    bn = BatchNorm(x, gamma, beta, moving_mean, moving_var)
    y1 = bn[0]

    # Target pattern: k = gamma / sqrt(var + epsilon), x * k + beta - mean * k
    k = gamma / Sqrt(moving_var + bn.epsilon)
    bias = beta - moving_mean * k
    y2 = BiasAdd(x * ExpandDims(k, axis=1, num_newaxis=2), bias)

    # Build substitution
    return Substitution(y1, y2)


def conv_mul():
    # Input
    x = Wildcard()
    w = Var()
    k = Var(shape=(None, 1, 1))

    # Source pattern: conv2d(x, w) * k
    conv = Conv2D(x, w, groups=1)
    y1 = conv * k

    # Target pattern: conv2d(x, matmul(diag(k), reshape(w)))
    zeros = Zeros(shape=(k.shape[0],) * 2, dtype=k.dtype)
    diag = ExpandDims(MatrixSetDiag(zeros, Reshape(k, newshape=(-1,))), axis=0)
    conv_mat = Reshape(w, newshape=(1, w.shape[0], -1))
    matmul = BatchMatmul(diag, Transpose(conv_mat, axes=(0, 2, 1)))
    fused_w = Reshape(matmul, newshape=w.shape)
    y2 = Conv2D(x, fused_w, groups=1, **same_attr(conv, ['strides', 'padding', 'dilation']))

    # Build substitution
    return Substitution(y1, y2)


def conv_batch_norm():
    # Input
    x = Wildcard()
    w = Var()
    gamma = Var()
    beta = Var()
    moving_mean = Var()
    moving_var = Var()

    # Source pattern
    conv = Conv2D(x, w, groups=1)
    bn = BatchNorm(conv, gamma, beta, moving_mean, moving_var)
    y1 = bn[0]

    # Target pattern
    k = gamma / Sqrt(moving_var + bn.epsilon)
    out_chan = gamma.shape[0]
    zeros = Zeros(shape=(out_chan, out_chan), dtype=w.dtype)
    diag = ExpandDims(MatrixSetDiag(zeros, k), axis=0)
    conv_w = Reshape(w, newshape=(1, w.shape[0], -1))
    matmul = BatchMatmul(diag, Transpose(conv_w, axes=[0, 2, 1]))
    fused_w = Reshape(matmul, newshape=w.shape)
    new_conv = Conv2D(x, fused_w,
                      **same_attr(conv, ['strides', 'padding', 'dilation', 'groups']))
    bias = beta - moving_mean * k
    y2 = BiasAdd(new_conv, bias)

    # Build substitution
    return Substitution(y1, y2)


def merge_element_wise():
    # Input
    x = Wildcard()

    # Source pattern
    ew_op = OpWithFlag(OpFlag.ELEMENT_WISE)
    ew1 = Call(ew_op, x)
    ew2 = Call(ew_op, x)

    # Target pattern
    ew = Call(ew_op, x)

    # Build substitution
    return Substitution([ew1, ew2], [ew, ew])


def merge_element_wise_variadic():
    # Input
    x = Wildcard()

    # Source pattern
    ew_op = OpWithFlag(OpFlag.ELEMENT_WISE)
    call = Call(ew_op, x)
    src = Variadic(call, templates=[call])

    # Target pattern
    ew = Call(ew_op, x)
    tgt = Variadic(ew)

    # Build substitution
    return Substitution(src, tgt)


def parallel_conv():
    # Input
    x = Wildcard()
    w1 = Var()
    w2 = Var(shape=w1.shape)

    # Source pattern
    conv1 = Conv2D(x, w1, groups=1)
    conv2 = Conv2D(x, w2, **same_attr(conv1, ['strides', 'padding', 'dilation', 'groups']))

    # Target pattern
    w = Concatenate((w1, w2), axis=0)
    conv = Conv2D(x, w, **same_attr(conv1, ['strides', 'padding', 'dilation', 'groups']))
    split = Split(conv, indices_or_sections=2, axis=1)

    # Build substitution
    return Substitution([conv1, conv2], [split[0], split[1]])


def parallel_conv_variadic():
    # Input
    x = Wildcard()
    w1 = Var()
    w = Var(shape=(None, None, w1.shape[2], w1.shape[3]))

    # Source pattern
    conv1 = Conv2D(x, w1, groups=1)
    conv = Conv2D(x, w, **same_attr(conv1, ['strides', 'padding', 'dilation', 'groups']))
    src = Variadic(conv, templates=[conv, w], first=[conv1, w1], min_len=2)

    # Target pattern
    i = Symbol()
    get_inst = src(i, w)
    concat = Concatenate(Variadic(get_inst, templates=[get_inst], index=i, length=src.length),
                         axis=0)
    conv = Conv2D(x, concat, **same_attr(conv1, ['strides', 'padding', 'dilation', 'groups']))

    i = Symbol()
    j = Symbol()
    split = Split(conv, axis=1,
                  indices_or_sections=VariadicAttr(Sum(src(j, w).shape[0], j, i + 1),
                                                   index=i, length=src.length - 1))
    i = Symbol()
    item = split[i]
    tgt = Variadic(item, templates=[item], index=i)

    # Build substitution
    return Substitution(src, tgt)


def parallel_conv_expand_kernels():
    # Input
    x = Wildcard()
    w1 = Var()
    w2 = Var(shape=(w1.shape[0], None, None, None))

    # Source pattern
    def same_padding(h: Attr, w: Attr):
        pad_h = (h - 1) // 2
        pad_w = (w - 1) // 2
        return pad_h, pad_w, pad_h, pad_w

    conv1_pad = same_padding(w1.shape[2], w1.shape[3])
    conv1 = Conv2D(x, w1, padding=conv1_pad, strides=(1, 1), dilation=(1, 1), groups=1)
    conv2_pad = same_padding(w2.shape[2], w2.shape[3])
    conv2 = Conv2D(x, w2, padding=conv2_pad,
                   **same_attr(conv1, ['strides', 'dilation', 'groups']))

    # Target pattern
    max_h, max_w = w1.shape[2].max(w2.shape[2]), w1.shape[3].max(w2.shape[3])
    w1_pad_h = (max_h - w1.shape[2]) // 2
    w1_pad_w = (max_w - w1.shape[3]) // 2
    w1_pad = Pad(w1, pad_width=((0, 0), (0, 0), (w1_pad_h,) * 2, (w1_pad_w,) * 2))
    w2_pad_h = (max_h - w2.shape[2]) // 2
    w2_pad_w = (max_w - w2.shape[3]) // 2
    w2_pad = Pad(w2, pad_width=((0, 0), (0, 0), (w2_pad_h,) * 2, (w2_pad_w,) * 2))
    concat = Concatenate((w1_pad, w2_pad), axis=0)
    new_conv_pad = same_padding(max_h, max_w)
    new_conv = Conv2D(x, concat, padding=new_conv_pad,
                      **same_attr(conv1, ['strides', 'dilation', 'groups']))
    split = Split(new_conv, indices_or_sections=2, axis=1)

    # Build substitution
    return Substitution([conv1, conv2], [split[0], split[1]])


def parallel_dense():
    # Input
    x = Wildcard()
    w1 = Var()
    w2 = Var()

    # Source pattern
    dense1 = Dense(x, w1)
    dense2 = Dense(x, w2)

    # Target pattern
    dense = Dense(x, Concatenate((w1, w2), axis=0))
    split = Split(dense, indices_or_sections=(w1.shape[0],), axis=-1)

    # Build substitution
    return Substitution([dense1, dense2], [split[0], split[1]])


def parallel_dense_variadic():
    # Input
    x = Wildcard()
    w1 = Var()
    w = Var(shape=(None, w1.shape[1]))

    # Source pattern
    dense = Dense(x, w)
    src = Variadic(dense, templates=[dense, w], first=[None, w1], min_len=2)

    # Target pattern
    i = Symbol()
    get_inst = src(i, w)
    dense = Dense(x, Concatenate(Variadic(get_inst, templates=[get_inst], index=i,
                                          length=src.length),
                                 axis=0))
    i = Symbol()
    j = Symbol()
    split = Split(dense, axis=-1,
                  indices_or_sections=VariadicAttr(Sum(src(j, w).shape[0], j, i + 1),
                                                   index=i, length=src.length - 1))
    i = Symbol()
    item = split[i]
    tgt = Variadic(item, templates=[item], index=i)

    # Build substitution
    return Substitution(src, tgt)
