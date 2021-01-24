from typing import Optional

from .pat import *

AttrOpt = Optional[Attr]


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
    def __init__(self, shape: Union[tuple, AttrOpt] = None,
                 dtype: Union[str, AttrOpt] = None):
        super().__init__('zeros', **_filter_attrs({
            'shape': shape, 'dtype': dtype,
        }))


class Ones(Call):
    def __init__(self, shape: Union[tuple, AttrOpt] = None,
                 dtype: Union[str, AttrOpt] = None):
        super().__init__('ones', **_filter_attrs({
            'shape': shape, 'dtype': dtype,
        }))


class Concatenate(Call):
    def __init__(self, data: Tuple[PatternConvertible, ...],
                 axis: Optional[int] = None):
        super().__init__('concatenate', Tup(*data), **_filter_attrs({
            'axis': axis,
        }))


class Split(Call):
    def __init__(self, data: PatternConvertible,
                 indices_or_sections: Union[int, tuple, AttrOpt] = None,
                 axis: Union[int, AttrOpt] = None):
        super().__init__('split', data, **_filter_attrs({
            'indices_or_sections': indices_or_sections, 'axis': axis,
        }))


class Reshape(Call):
    def __init__(self, data: PatternConvertible,
                 newshape: Union[tuple, AttrOpt] = None):
        super().__init__('reshape', data, **_filter_attrs({
            'newshape': newshape,
        }))


class Transpose(Call):
    def __init__(self, data: PatternConvertible,
                 axes: Union[tuple, list, AttrOpt] = None):
        super().__init__('transpose', data, **_filter_attrs({
            'axes': axes,
        }))


class ExpandDims(Call):
    def __init__(self, data: PatternConvertible,
                 axis: Union[int, AttrOpt] = None,
                 num_newaxis: Union[int, AttrOpt] = None):
        super().__init__('expand_dims', data, **_filter_attrs({
            'axis': axis, 'num_newaxis': num_newaxis,
        }))


class MatrixSetDiag(Call):
    def __init__(self, data: PatternConvertible,
                 diagonal: PatternConvertible,
                 k: Union[int, tuple, AttrOpt] = None,
                 align: Union[str, AttrOpt] = None):
        super().__init__('matrix_set_diag', data, diagonal, **_filter_attrs({
            'k': k, 'align': align,
        }))


class Conv2D(Call):
    def __init__(self, data: PatternConvertible,
                 weight: PatternConvertible,
                 strides: Union[tuple, AttrOpt] = None,
                 padding: Union[tuple, AttrOpt] = None,
                 dilation: Union[tuple, AttrOpt] = None,
                 groups: Union[int, AttrOpt] = None):
        super().__init__('nn.conv2d', data, weight, **_filter_attrs({
            'strides': strides, 'padding': padding, 'dilation': dilation, 'groups': groups,
        }))


class BatchNorm(Call):
    def __init__(self, data: PatternConvertible,
                 gamma: PatternConvertible,
                 beta: PatternConvertible,
                 moving_mean: PatternConvertible,
                 moving_var: PatternConvertible,
                 axis: Union[int, AttrOpt] = None,
                 epsilon: Union[float, AttrOpt] = None,
                 center: Union[bool, AttrOpt] = None,
                 scale: Union[bool, AttrOpt] = None):
        super().__init__('nn.batch_norm', data, gamma, beta, moving_mean, moving_var,
                         **_filter_attrs({
                             'axis': axis, 'epsilon': epsilon, 'center': center,
                             'scale': scale,
                         }))


class BiasAdd(Call):
    def __init__(self, data: PatternConvertible,
                 bias: PatternConvertible,
                 axis: Union[int, AttrOpt] = None):
        super().__init__('nn.bias_add', data, bias, **_filter_attrs({
            'axis': axis
        }))


class ReLU(Call):
    def __init__(self, data: PatternConvertible):
        super().__init__('nn.relu', data)


class Pad(Call):
    def __init__(self, data: PatternConvertible,
                 pad_width: Union[tuple, AttrOpt] = None,
                 pad_value: Union[float, AttrOpt] = None,
                 pad_mode: Union[str, AttrOpt] = None):
        super().__init__('nn.pad', data, **_filter_attrs({
            'pad_width': pad_width, 'pad_value': pad_value, 'pad_mode': pad_mode
        }))


class BatchMatmul(Call):
    def __init__(self, x: PatternConvertible, y: PatternConvertible):
        super().__init__('nn.batch_matmul', x, y)


def _filter_attrs(attrs: Dict[str, Any]) -> Dict[str, Any]:
    filtered = {}
    for k, v in attrs.items():
        if v is not None:
            filtered[k] = v
    return filtered
