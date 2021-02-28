from typing import Optional, Union, Tuple

from . import attr, pat


class Abs(pat.Call):
    def __init__(self, data: pat.PatternConvertible):
        super().__init__('abs', data)


class Exp(pat.Call):
    def __init__(self, data: pat.PatternConvertible):
        super().__init__('exp', data)


class Sqrt(pat.Call):
    def __init__(self, data: pat.PatternConvertible):
        super().__init__('sqrt', data)


class Zeros(pat.Call):
    def __init__(self, shape: Union[tuple, attr.Attr, None] = None,
                 dtype: Union[str, attr.Attr, None] = None):
        super().__init__('zeros', **pat.filter_attrs({
            'shape': shape, 'dtype': dtype,
        }))


class Ones(pat.Call):
    def __init__(self, shape: Union[tuple, attr.Attr, None] = None,
                 dtype: Union[str, attr.Attr, None] = None):
        super().__init__('ones', **pat.filter_attrs({
            'shape': shape, 'dtype': dtype,
        }))


class Concatenate(pat.Call):
    def __init__(self, data: Union[Tuple[pat.PatternConvertible, ...], pat.Variadic],
                 axis: Optional[int] = None):
        if isinstance(data, pat.Variadic):
            arg = data
        elif isinstance(data, tuple):
            arg = pat.Tuple(*data)
        else:
            raise TypeError('Invalid type \'{}\' for input of \'concatenate\'.')
        super().__init__('concatenate', arg, **pat.filter_attrs({
            'axis': axis,
        }))


class Split(pat.Call):
    def __init__(self, data: pat.PatternConvertible,
                 indices_or_sections: Union[int, tuple, attr.Attr, None] = None,
                 axis: Union[int, attr.Attr, None] = None):
        super().__init__('split', data, **pat.filter_attrs({
            'indices_or_sections': indices_or_sections, 'axis': axis,
        }))


class Reshape(pat.Call):
    def __init__(self, data: pat.PatternConvertible,
                 newshape: Union[tuple, attr.Attr, None] = None):
        super().__init__('reshape', data, **pat.filter_attrs({
            'newshape': newshape,
        }))


class Transpose(pat.Call):
    def __init__(self, data: pat.PatternConvertible,
                 axes: Union[list, attr.Attr, None] = None):
        super().__init__('transpose', data, **pat.filter_attrs({
            'axes': axes,
        }))


class ExpandDims(pat.Call):
    def __init__(self, data: pat.PatternConvertible,
                 axis: Union[int, attr.Attr, None] = None,
                 num_newaxis: Union[int, attr.Attr, None] = None):
        super().__init__('expand_dims', data, **pat.filter_attrs({
            'axis': axis, 'num_newaxis': num_newaxis,
        }))


class Squeeze(pat.Call):
    def __init__(self, data: pat.PatternConvertible,
                 axis: Union[tuple, list, attr.Attr, None] = None):
        super().__init__('squeeze', data, **pat.filter_attrs({
            'axis': axis
        }))


class Cast(pat.Call):
    def __init__(self, data: pat.PatternConvertible,
                 dtype: Union[str, attr.Attr, None] = None):
        super().__init__('cast', data, **pat.filter_attrs({
            'dtype': dtype
        }))


class Sum(pat.Call):
    def __init__(self, data: pat.PatternConvertible,
                 axis: Union[int, tuple, attr.Attr, None] = None,
                 keepdims: Union[bool, attr.Attr, None] = None,
                 exclude: Union[bool, attr.Attr, None] = None):
        super().__init__('sum', data, **pat.filter_attrs({
            'axis': axis, 'keepdims': keepdims, 'exclude': exclude,
        }))


class Mean(pat.Call):
    def __init__(self, data: pat.PatternConvertible,
                 axis: Union[int, tuple, attr.Attr, None] = None,
                 keepdims: Union[bool, attr.Attr, None] = None,
                 exclude: Union[bool, attr.Attr, None] = None):
        super().__init__('mean', data, **pat.filter_attrs({
            'axis': axis, 'keepdims': keepdims, 'exclude': exclude,
        }))


class Variance(pat.Call):
    def __init__(self, data: pat.PatternConvertible,
                 axis: Union[int, tuple, attr.Attr, None] = None,
                 keepdims: Union[bool, attr.Attr, None] = False,
                 exclude: Union[bool, attr.Attr, None] = False,
                 unbiased: Union[bool, attr.Attr, None] = False):
        super().__init__('variance', data, **pat.filter_attrs({
            'axis': axis, 'keepdims': keepdims, 'exclude': exclude, 'unbiased': unbiased,
        }))


class MatrixSetDiag(pat.Call):
    def __init__(self, data: pat.PatternConvertible,
                 diagonal: pat.PatternConvertible,
                 k: Union[int, tuple, attr.Attr, None] = None,
                 align: Union[str, attr.Attr, None] = None):
        super().__init__('matrix_set_diag', data, diagonal, **pat.filter_attrs({
            'k': k, 'align': align,
        }))


class Dense(pat.Call):
    def __init__(self, data: pat.PatternConvertible, weight: pat.PatternConvertible):
        super().__init__('nn.dense', data, weight)


class Conv2D(pat.Call):
    def __init__(self, data: pat.PatternConvertible,
                 weight: pat.PatternConvertible,
                 strides: Union[tuple, attr.Attr, None] = None,
                 padding: Union[tuple, attr.Attr, None] = None,
                 dilation: Union[tuple, attr.Attr, None] = None,
                 groups: Union[int, attr.Attr, None] = None):
        super().__init__('nn.conv2d', data, weight, **pat.filter_attrs({
            'strides': strides, 'padding': padding, 'dilation': dilation, 'groups': groups,
        }))


class BatchNorm(pat.Call):
    def __init__(self, data: pat.PatternConvertible,
                 gamma: pat.PatternConvertible,
                 beta: pat.PatternConvertible,
                 moving_mean: pat.PatternConvertible,
                 moving_var: pat.PatternConvertible,
                 axis: Union[int, attr.Attr, None] = None,
                 epsilon: Union[float, attr.Attr, None] = None,
                 center: Union[bool, attr.Attr, None] = None,
                 scale: Union[bool, attr.Attr, None] = None):
        super().__init__('nn.batch_norm', data, gamma, beta, moving_mean, moving_var,
                         **pat.filter_attrs({
                             'axis': axis, 'epsilon': epsilon, 'center': center,
                             'scale': scale,
                         }))


class LayerNorm(pat.Call):
    def __init__(self, data: pat.PatternConvertible,
                 gamma: pat.PatternConvertible,
                 beta: pat.PatternConvertible,
                 axis: Union[int, attr.Attr, None] = None,
                 epsilon: Union[float, attr.Attr, None] = None,
                 center: Union[bool, attr.Attr, None] = None,
                 scale: Union[bool, attr.Attr, None] = None):
        super().__init__('nn.layer_norm', data, gamma, beta,
                         **pat.filter_attrs({
                             'axis': axis, 'epsilon': epsilon, 'center': center,
                             'scale': scale,
                         }))


class BiasAdd(pat.Call):
    def __init__(self, data: pat.PatternConvertible,
                 bias: pat.PatternConvertible,
                 axis: Union[int, attr.Attr, None] = None):
        super().__init__('nn.bias_add', data, bias, **pat.filter_attrs({
            'axis': axis
        }))


class ReLU(pat.Call):
    def __init__(self, data: pat.PatternConvertible):
        super().__init__('nn.relu', data)


class Pad(pat.Call):
    def __init__(self, data: pat.PatternConvertible,
                 pad_width: Union[tuple, attr.Attr, None] = None,
                 pad_value: Union[float, attr.Attr, None] = None,
                 pad_mode: Union[str, attr.Attr, None] = None):
        super().__init__('nn.pad', data, **pat.filter_attrs({
            'pad_width': pad_width, 'pad_value': pad_value, 'pad_mode': pad_mode
        }))


class BatchMatmul(pat.Call):
    def __init__(self, x: pat.PatternConvertible, y: pat.PatternConvertible):
        super().__init__('nn.batch_matmul', x, y)
