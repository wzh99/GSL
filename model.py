from tensorflow import keras
from tensorflow.keras import layers, regularizers

batch_shape_nhwc = (4, 32, 32, 3)
batch_shape_nchw = (4, 3, 32, 32)
bn_eps = 1e-5

l2_reg = regularizers.l2(1e-4)


def resnet(num_stacked: int, load_weights: bool = False) -> keras.Model:
    input_data = layers.Input(batch_input_shape=batch_shape_nhwc)
    x = layers.Conv2D(16, 3, padding='same', use_bias=False,
                      name='input_conv', kernel_regularizer=l2_reg)(input_data)
    for i in range(num_stacked):
        x = _res_block(x, 16, 'feat16_block%d' % (i + 1))
    x = _res_block(x, 32, 'feat32_block1', strides=2)
    for i in range(num_stacked - 1):
        x = _res_block(x, 32, 'feat32_block%d' % (i + 2))
    x = _res_block(x, 64, 'feat64_block1', strides=2)
    for i in range(num_stacked - 1):
        x = _res_block(x, 64, 'feat64_block%d' % (i + 2))
    x = layers.GlobalAvgPool2D(name='global_avg_pool')(x)
    x = layers.Dense(10, activation='softmax', name='dense',
                     kernel_regularizer=l2_reg)(x)
    model = keras.Model(inputs=input_data, outputs=x,
                        name='resnet%d' % (6 * num_stacked + 2))
    if load_weights:
        weights_path = 'weights/%s.h5' % model.name
        model.load_weights(weights_path, by_name=True)
    return model


# noinspection PyTypeChecker
def _res_block(x, filters: int, name: str, strides: int = 1):
    if strides == 1:
        shortcut = x
    else:
        shortcut = layers.Conv2D(filters, 1, strides=strides, use_bias=False,
                                 kernel_regularizer=l2_reg,
                                 name=name + '_proj')(x)
    x = layers.Conv2D(filters, 3, strides=strides, padding='same', use_bias=False,
                      kernel_regularizer=l2_reg, name=name + '_conv1')(x)
    x = layers.BatchNormalization(epsilon=bn_eps, gamma_regularizer=l2_reg,
                                  name=name + '_bn1')(x)
    x = layers.ReLU(name=name + '_relu1')(x)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False,
                      kernel_regularizer=l2_reg, name=name + '_conv2')(x)
    x = layers.BatchNormalization(epsilon=bn_eps, gamma_regularizer=l2_reg,
                                  name=name + '_bn2')(x)
    x = layers.Add(name=name + '_add')([x, shortcut])
    x = layers.ReLU(name=name + '_relu2')(x)
    return x
