from typing import Optional, Union, Tuple

from .pat import PatternConvertible, Attr, Call, Tup, Variadic, filter_attrs


class Abs(Call):
    def __init__(self, data: PatternConvertible):
        super().__init__('call', data)


class Exp(Call):
    def __init__(self, data: PatternConvertible):
        super().__init__('exp', data)


class Sqrt(Call):
    def __init__(self, data: PatternConvertible):
        super().__init__('sqrt', data)


class Zeros(Call):
    def __init__(self, shape: Union[tuple, Attr, None] = None,
                 dtype: Union[str, Attr, None] = None):
        super().__init__('zeros', **filter_attrs({
            'shape': shape, 'dtype': dtype,
        }))


class Ones(Call):
    def __init__(self, shape: Union[tuple, Attr, None] = None,
                 dtype: Union[str, Attr, None] = None):
        super().__init__('ones', **filter_attrs({
            'shape': shape, 'dtype': dtype,
        }))


class Concatenate(Call):
    def __init__(self, data: Union[Tuple[PatternConvertible, ...], Variadic],
                 axis: Optional[int] = None):
        if isinstance(data, Variadic):
            arg = data
        elif isinstance(data, tuple):
            arg = Tup(*data)
        else:
            raise TypeError('Invalid type \'{}\' for input of \'concatenate\'.')
        super().__init__('concatenate', arg, **filter_attrs({
            'axis': axis,
        }))


class Split(Call):
    def __init__(self, data: PatternConvertible,
                 indices_or_sections: Union[int, tuple, Attr, None] = None,
                 axis: Union[int, Attr, None] = None):
        super().__init__('split', data, **filter_attrs({
            'indices_or_sections': indices_or_sections, 'axis': axis,
        }))


class Reshape(Call):
    def __init__(self, data: PatternConvertible,
                 newshape: Union[tuple, Attr, None] = None):
        super().__init__('reshape', data, **filter_attrs({
            'newshape': newshape,
        }))


class Transpose(Call):
    def __init__(self, data: PatternConvertible,
                 axes: Union[list, Attr, None] = None):
        super().__init__('transpose', data, **filter_attrs({
            'axes': axes,
        }))


class ExpandDims(Call):
    def __init__(self, data: PatternConvertible,
                 axis: Union[int, Attr, None] = None,
                 num_newaxis: Union[int, Attr, None] = None):
        super().__init__('expand_dims', data, **filter_attrs({
            'axis': axis, 'num_newaxis': num_newaxis,
        }))


class Squeeze(Call):
    def __init__(self, data: PatternConvertible,
                 axis: Union[tuple, list, Attr, None] = None):
        super().__init__('squeeze', data, **filter_attrs({
            'axis': axis
        }))


class Cast(Call):
    def __init__(self, data: PatternConvertible,
                 dtype: Union[str, Attr, None] = None):
        super().__init__('cast', data, **filter_attrs({
            'dtype': dtype
        }))


class Sum(Call):
    def __init__(self, data: PatternConvertible,
                 axis: Union[int, tuple, Attr, None] = None,
                 keepdims: Union[bool, Attr, None] = None,
                 exclude: Union[bool, Attr, None] = None):
        super().__init__('sum', data, **filter_attrs({
            'axis': axis, 'keepdims': keepdims, 'exclude': exclude,
        }))


class Mean(Call):
    def __init__(self, data: PatternConvertible,
                 axis: Union[int, tuple, Attr, None] = None,
                 keepdims: Union[bool, Attr, None] = None,
                 exclude: Union[bool, Attr, None] = None):
        super().__init__('mean', data, **filter_attrs({
            'axis': axis, 'keepdims': keepdims, 'exclude': exclude,
        }))


class Variance(Call):
    def __init__(self, data: PatternConvertible,
                 axis: Union[int, tuple, Attr, None] = None,
                 keepdims: Union[bool, Attr, None] = False,
                 exclude: Union[bool, Attr, None] = False,
                 unbiased: Union[bool, Attr, None] = False):
        super().__init__('variance', data, **filter_attrs({
            'axis': axis, 'keepdims': keepdims, 'exclude': exclude, 'unbiased': unbiased,
        }))


class MatrixSetDiag(Call):
    def __init__(self, data: PatternConvertible,
                 diagonal: PatternConvertible,
                 k: Union[int, tuple, Attr, None] = None,
                 align: Union[str, Attr, None] = None):
        super().__init__('matrix_set_diag', data, diagonal, **filter_attrs({
            'k': k, 'align': align,
        }))


class Dense(Call):
    def __init__(self, data: PatternConvertible, weight: PatternConvertible):
        super().__init__('nn.dense', data, weight)


class Conv2D(Call):
    def __init__(self, data: PatternConvertible,
                 weight: PatternConvertible,
                 strides: Union[tuple, Attr, None] = None,
                 padding: Union[tuple, Attr, None] = None,
                 dilation: Union[tuple, Attr, None] = None,
                 groups: Union[int, Attr, None] = None):
        super().__init__('nn.conv2d', data, weight, **filter_attrs({
            'strides': strides, 'padding': padding, 'dilation': dilation, 'groups': groups,
        }))


class BatchNorm(Call):
    def __init__(self, data: PatternConvertible,
                 gamma: PatternConvertible,
                 beta: PatternConvertible,
                 moving_mean: PatternConvertible,
                 moving_var: PatternConvertible,
                 axis: Union[int, Attr, None] = None,
                 epsilon: Union[float, Attr, None] = None,
                 center: Union[bool, Attr, None] = None,
                 scale: Union[bool, Attr, None] = None):
        super().__init__('nn.batch_norm', data, gamma, beta, moving_mean, moving_var,
                         **filter_attrs({
                             'axis': axis, 'epsilon': epsilon, 'center': center,
                             'scale': scale,
                         }))


class LayerNorm(Call):
    def __init__(self, data: PatternConvertible,
                 gamma: PatternConvertible,
                 beta: PatternConvertible,
                 axis: Union[int, Attr, None] = None,
                 epsilon: Union[float, Attr, None] = None,
                 center: Union[bool, Attr, None] = None,
                 scale: Union[bool, Attr, None] = None):
        super().__init__('nn.layer_norm', data, gamma, beta,
                         **filter_attrs({
                             'axis': axis, 'epsilon': epsilon, 'center': center,
                             'scale': scale,
                         }))


class BiasAdd(Call):
    def __init__(self, data: PatternConvertible,
                 bias: PatternConvertible,
                 axis: Union[int, Attr, None] = None):
        super().__init__('nn.bias_add', data, bias, **filter_attrs({
            'axis': axis
        }))


class ReLU(Call):
    def __init__(self, data: PatternConvertible):
        super().__init__('nn.relu', data)


class Pad(Call):
    def __init__(self, data: PatternConvertible,
                 pad_width: Union[tuple, Attr, None] = None,
                 pad_value: Union[float, Attr, None] = None,
                 pad_mode: Union[str, Attr, None] = None):
        super().__init__('nn.pad', data, **filter_attrs({
            'pad_width': pad_width, 'pad_value': pad_value, 'pad_mode': pad_mode
        }))


class BatchMatmul(Call):
    def __init__(self, x: PatternConvertible, y: PatternConvertible):
        super().__init__('nn.batch_matmul', x, y)
