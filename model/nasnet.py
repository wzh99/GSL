from typing import Optional

from tensorflow import keras, Tensor
from tensorflow.keras import layers, regularizers

from .common import bn_eps, batch_size

input_shape_hwc = (32, 32, 3)
input_shape_chw = (3, 32, 32)
batch_shape_nhwc = (batch_size,) + input_shape_hwc
batch_shape_nchw = (batch_size,) + input_shape_chw

# NASNet-A 6 @ 768
num_penult_filters = 768
num_stem_filters = 96
num_reduction = 2

l2_reg = regularizers.l2(1e-4)


# noinspection PyTypeChecker
def get_model(num_stacked: int, load_weights: bool = False) -> keras.Model:
    # Stem from input
    input_tensor = layers.Input(batch_input_shape=batch_shape_nhwc)
    cur_filters = num_penult_filters // ((2 ** num_reduction) * 6)
    prev: Optional[Tensor] = None
    cur = _stem(input_tensor)

    # Build cells
    for red_idx in range(num_reduction + 1):
        # Reduction cell
        if red_idx > 0:
            cur_filters *= 2
            nxt = _reduction(prev, cur, cur_filters, 'red_%d' % red_idx)
            prev, cur = cur, nxt

        # Normal cell
        for normal_idx in range(num_stacked):
            nxt = _normal(prev, cur, cur_filters,
                          'norm_%d_%d' % (red_idx, normal_idx + 1))
            prev, cur = cur, nxt

    # Final output
    x = layers.ReLU(name='final_relu')(cur)
    x = layers.GlobalAvgPool2D(name='global_avg')(x)
    x = layers.Dense(
        10, activation='softmax', kernel_regularizer=l2_reg, name='dense'
    )(x)

    # Build model
    model = keras.Model(inputs=input_tensor, outputs=x,
                        name='nasnet-a_%d_%d' % (num_stacked, num_penult_filters))
    if load_weights:
        weights_path = 'weights/%s.h5' % model.name
        model.load_weights(weights_path, by_name=True)

    return model


# noinspection PyTypeChecker
def _stem(x: Tensor) -> Tensor:
    x = layers.Conv2D(
        num_stem_filters, 3, padding='same', use_bias=False,
        kernel_initializer='he_normal', kernel_regularizer=l2_reg,
        name='stem_conv'
    )(x)
    x = layers.BatchNormalization(
        epsilon=bn_eps, gamma_regularizer=l2_reg, name='stem_bn'
    )(x)
    return x


# noinspection PyTypeChecker
def _reduction(prev: Optional[Tensor], cur: Tensor, num_filters: int, name: str) \
        -> Tensor:
    prev = _fit(prev, cur, num_filters, name + '_fit')
    cur = _squeeze(cur, num_filters, name + '_squeeze')
    blk_1 = layers.add([
        _sep(cur, num_filters, 5, name + '_blk1_sep1', strides=2),
        _sep(prev, num_filters, 7, name + '_blk1_sep2', strides=2)
    ], name=name + '_add1')
    blk_2 = layers.add([
        layers.MaxPool2D(
            pool_size=3, strides=2, padding='same', name=name + '_blk2_max'
        )(cur),
        _sep(prev, num_filters, 7, name + '_blk2_sep', strides=2)
    ], name=name + '_add2')
    blk_3 = layers.add([
        layers.AveragePooling2D(
            pool_size=3, strides=2, padding='same', name=name + '_blk3_avg'
        )(cur),
        _sep(prev, num_filters, 5, name + '_blk3_sep', strides=2)
    ], name=name + '_add3')
    blk_4 = layers.add([
        _sep(blk_1, num_filters, 3, name + '_blk5_sep'),
        layers.MaxPool2D(
            pool_size=3, strides=2, padding='same', name=name + '_blk4_max'
        )(cur)
    ], name=name + '_add4')
    blk_5 = layers.add([
        layers.AveragePooling2D(
            pool_size=3, strides=1, padding='same', name=name + '_blk5_avg'
        )(blk_1),
        blk_2
    ], name=name + '_add5')
    x = layers.concatenate([blk_2, blk_3, blk_4, blk_5], name=name + '_concat')
    return x


# noinspection PyTypeChecker
def _normal(prev: Optional[Tensor], cur: Tensor, num_filters: int, name: str) -> Tensor:
    cur = _squeeze(cur, num_filters, name + '_squeeze')
    prev = _fit(prev, cur, num_filters, name + '_fit')
    blk_1 = layers.add([
        _sep(cur, num_filters, 3, name + '_blk1_sep'),
        cur
    ], name=name + '_add1')
    blk_2 = layers.add([
        _sep(cur, num_filters, 5, name + '_sep1'),
        _sep(prev, num_filters, 3, name + '_sep2')
    ], name=name + '_add2')
    blk_3 = layers.add([
        layers.AveragePooling2D(
            pool_size=3, strides=1, padding='same', name=name + '_blk3_avg'
        )(cur),
        prev
    ], name=name + '_add3')
    blk_4 = layers.add([
        layers.AveragePooling2D(
            pool_size=3, strides=1, padding='same', name=name + '_blk4_avg1'
        )(prev),
        layers.AveragePooling2D(
            pool_size=3, strides=1, padding='same', name=name + '_blk4_avg2'
        )(prev)
    ], name=name + '_add4')
    blk_5 = layers.add([
        _sep(prev, num_filters, 5, name=name + '_blk5_sep1'),
        _sep(prev, num_filters, 3, name=name + '_blk5_sep2')
    ], name=name + '_add5')
    x = layers.concatenate([prev, blk_1, blk_2, blk_3, blk_4, blk_5],
                           name=name + "_concat")
    return x


# noinspection PyTypeChecker
def _sep(x: Tensor, num_filters: int, kernel_size: int, name: str,
         strides: int = 1) -> Tensor:
    x = layers.ReLU(name=name + '_relu1')(x)
    x = layers.SeparableConv2D(
        num_filters, kernel_size, strides=strides, padding='same', use_bias=False,
        kernel_initializer='he_normal', kernel_regularizer=l2_reg,
        name=name + '_conv1'
    )(x)
    x = layers.BatchNormalization(
        epsilon=bn_eps, gamma_regularizer=l2_reg, name=name + '_bn1'
    )(x)
    x = layers.ReLU(name=name + '_relu2')(x)
    x = layers.SeparableConv2D(
        num_filters, kernel_size, padding='same', use_bias=False,
        kernel_initializer='he_normal', kernel_regularizer=l2_reg,
        name=name + '_conv2'
    )(x)
    x = layers.BatchNormalization(
        epsilon=bn_eps, gamma_regularizer=l2_reg, name=name + '_bn2'
    )(x)
    return x


# noinspection PyTypeChecker
def _fit(src: Optional[Tensor], tgt: Tensor, num_filters: int, name: str) -> Tensor:
    if src is None:
        # Directly return target because there is nothing to fit
        return tgt
    if src.shape[2] == tgt.shape[2]:
        # Feature map shapes match, squeeze channels
        return _squeeze(src, num_filters, name + '_squeeze')
    # Shape does not match, down-sample source feature map
    x = layers.ReLU(name=name + '_relu')(src)
    p1 = layers.AveragePooling2D(pool_size=1, strides=2, name=name + '_pool1')(x)
    p1 = layers.Conv2D(
        num_filters // 2, 1, use_bias=False, kernel_initializer='he_normal',
        kernel_regularizer=l2_reg, name=name + '_conv1'
    )(p1)
    p2 = layers.ZeroPadding2D(padding=((0, 1), (0, 1)), name=name + '_pad')(x)
    p2 = layers.Cropping2D(cropping=((1, 0), (1, 0)), name=name + '_crop')(p2)
    p2 = layers.AveragePooling2D(pool_size=1, strides=2, name=name + '_pool2')(p2)
    p2 = layers.Conv2D(
        num_filters // 2, 1, use_bias=False, kernel_initializer='he_normal',
        kernel_regularizer=l2_reg, name=name + '_conv2'
    )(p2)
    x = layers.concatenate([p1, p2], name=name + '_concat')
    x = layers.BatchNormalization(
        epsilon=bn_eps, gamma_regularizer=l2_reg, name=name + '_bn'
    )(x)
    return x


# noinspection PyTypeChecker
def _squeeze(x: Tensor, num_filters: int, name: str) -> Tensor:
    x = layers.ReLU(name=name + "_relu")(x)
    x = layers.Conv2D(
        num_filters, 1, use_bias=False, kernel_initializer='he_normal',
        kernel_regularizer=l2_reg, name=name + '_conv'
    )(x)
    x = layers.BatchNormalization(
        epsilon=bn_eps, gamma_regularizer=l2_reg, name=name + '_bn'
    )(x)
    return x
