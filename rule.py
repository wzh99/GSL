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
    conv2 = Conv2D(x2, w2, strides=conv1.strides, padding=conv1.padding,
                   dilation=conv1.dilation, groups=conv1.groups)
    y1 = conv1 + conv2

    # Target pattern: conv2d(concat(x1, x2), concat(w1, w2))
    x_concat = Concatenate((x1, x2), axis=1)
    w_concat = Concatenate((w1, w2), axis=1)
    y2 = Conv2D(x_concat, w_concat, strides=conv1.strides, padding=conv1.padding,
                dilation=conv1.dilation, groups=conv1.groups)

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
    conv = Conv2D(x, w)
    bn = BatchNorm(conv, gamma, beta, moving_mean, moving_var)
    y1 = bn[0]
    # y1.visualize('conv_bn_pat')

    # Target pattern
    k = gamma / Sqrt(moving_var + bn.epsilon)
    out_chan = gamma.shape[0]
    zeros = Zeros(shape=(out_chan, out_chan), dtype=w.dtype)
    diag = ExpandDims(MatrixSetDiag(zeros, k), axis=0)
    conv_w = Reshape(w, newshape=(1, w.shape[0], -1))
    matmul = BatchMatmul(diag, Transpose(conv_w, axes=[0, 2, 1]))
    fused_w = Reshape(matmul, newshape=w.shape)
    new_conv = Conv2D(x, fused_w, strides=conv.strides, padding=conv.padding,
                      dilation=conv.dilation, groups=conv.groups)
    bias = beta - moving_mean * k
    y2 = BiasAdd(new_conv, bias)

    # Build substitution
    return Substitution(y1, y2)


def merge_relu():
    # Input
    x = Wildcard()

    # Source pattern
    relu_1 = ReLU(x)
    relu_2 = ReLU(x)

    # Target pattern
    relu = ReLU(x)

    # Build substitution
    return Substitution([relu_1, relu_2], [relu, relu])


def parallel_conv():
    # Input
    x = Wildcard()
    w1 = Var()
    w2 = Var(shape=w1.shape)

    # Source pattern
    conv1 = Conv2D(x, w1)
    conv2 = Conv2D(x, w2, strides=conv1.strides, padding=conv1.padding,
                   dilation=conv1.dilation, groups=conv1.groups)

    # Target pattern
    w = Concatenate((w1, w2), axis=0)
    conv = Conv2D(x, w, strides=conv1.strides, padding=conv1.padding,
                  dilation=conv1.dilation, groups=conv1.groups)
    split = Split(conv, indices_or_sections=2, axis=1)

    # Build substitution
    return Substitution([conv1, conv2], [split[0], split[1]])


def parallel_conv_expand_kernels():
    # Input
    x = Wildcard()
    w1 = Var()
    w2 = Var(shape=(w1.shape[0], None, None, None))

    # Source pattern
    def same_padding(h: AttrExpr, w: AttrExpr):
        pad_h = (h - 1) // 2
        pad_w = (w - 1) // 2
        return pad_h, pad_w, pad_h, pad_w

    conv1_pad = same_padding(w1.shape[2], w1.shape[3])
    conv1 = Conv2D(x, w1, padding=conv1_pad, strides=(1, 1), dilation=(1, 1))
    conv2_pad = same_padding(w2.shape[2], w2.shape[3])
    conv2 = Conv2D(x, w2, padding=conv2_pad, strides=(1, 1), dilation=(1, 1),
                   groups=conv1.groups)

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
    new_conv = Conv2D(x, concat, padding=new_conv_pad, strides=(1, 1),
                      dilation=(1, 1), groups=conv1.groups)
    split = Split(new_conv, indices_or_sections=2, axis=1)

    # Build substitution
    return Substitution([conv1, conv2], [split[0], split[1]])
