from typing import Union, Tuple, List

from . import pat
from .attr import Attr
from .pat import Call, PatternLike


class Abs(Call):
    def __init__(self, data: PatternLike):
        super().__init__('abs', data)


class Exp(Call):
    def __init__(self, data: PatternLike):
        super().__init__('exp', data)


class Sqrt(Call):
    def __init__(self, data: PatternLike):
        super().__init__('sqrt', data)


class Maximum(Call):
    def __init__(self, lhs: PatternLike, rhs: PatternLike):
        super().__init__('maximum', lhs, rhs)


class Zeros(Call):
    def __init__(self, shape: Union[tuple, Attr, None] = None,
                 dtype: Union[str, Attr, None] = None):
        super().__init__('zeros', **pat.filter_attrs({
            'shape': shape, 'dtype': dtype,
        }))


class ZerosLike(Call):
    def __init__(self, data: PatternLike):
        super().__init__('zeros_like', data)


class Ones(Call):
    def __init__(self, shape: Union[tuple, Attr, None] = None,
                 dtype: Union[str, Attr, None] = None):
        super().__init__('ones', **pat.filter_attrs({
            'shape': shape, 'dtype': dtype,
        }))


class OnesLike(Call):
    def __init__(self, data: PatternLike):
        super().__init__('ones_like', data)


class Full(Call):
    def __init__(self, fill_value: PatternLike,
                 shape: Union[tuple, Attr, None] = None,
                 dtype: Union[str, Attr, None] = None):
        super().__init__('full', fill_value, **pat.filter_attrs({
            'shape': shape, 'dtype': dtype,
        }))


class FullLike(Call):
    def __init__(self, data: PatternLike, fill_value: PatternLike):
        super().__init__('full_like', data, fill_value)


class BroadcastTo(Call):
    def __init__(self, data: PatternLike,
                 shape: Union[tuple, Attr, None] = None):
        super().__init__('broadcast_to', data, **pat.filter_attrs({
            'shape': shape
        }))


class Concatenate(Call):
    def __init__(self, data: Union[Tuple[PatternLike, ...], List[PatternLike], pat.Variadic],
                 axis: Union[int, Attr, None] = None):
        if isinstance(data, pat.Variadic):
            arg = data
        elif isinstance(data, (tuple, list)):
            arg = pat.Tuple(*data)
        else:
            raise TypeError('Invalid type \'{}\' for input of \'concatenate\'.')
        super().__init__('concatenate', arg, **pat.filter_attrs({
            'axis': axis,
        }))


class Split(Call):
    def __init__(self, data: PatternLike,
                 indices_or_sections: Union[int, tuple, Attr, None] = None,
                 axis: Union[int, Attr, None] = None):
        super().__init__('split', data, **pat.filter_attrs({
            'indices_or_sections': indices_or_sections, 'axis': axis,
        }))


class Reshape(Call):
    def __init__(self, data: PatternLike,
                 newshape: Union[tuple, Attr, None] = None):
        super().__init__('reshape', data, **pat.filter_attrs({
            'newshape': newshape,
        }))


class ReshapeLike(Call):
    def __init__(self, data: PatternLike, shape_like: PatternLike,
                 lhs_begin: Union[int, Attr, None] = None,
                 lhs_end: Union[int, Attr, None] = None,
                 rhs_begin: Union[int, Attr, None] = None,
                 rhs_end: Union[int, Attr, None] = None):
        super().__init__('reshape_like', data, shape_like, **pat.filter_attrs({
            'lhs_begin': lhs_begin, 'lhs_end': lhs_end, 'rhs_begin': rhs_begin,
            'rhs_end': rhs_end,
        }))


class Transpose(Call):
    def __init__(self, data: PatternLike,
                 axes: Union[list, Attr, None] = None):
        super().__init__('transpose', data, **pat.filter_attrs({
            'axes': axes,
        }))


class LayoutTransform(Call):
    def __init__(self, data: PatternLike,
                 src_layout: Union[str, Attr, None] = None,
                 dst_layout: Union[str, Attr, None] = None):
        super().__init__('layout_transform', data, **pat.filter_attrs({
            'src_layout': src_layout, 'dst_layout': dst_layout,
        }))


class ExpandDims(Call):
    def __init__(self, data: PatternLike,
                 axis: Union[int, Attr, None] = None,
                 num_newaxis: Union[int, Attr, None] = None):
        super().__init__('expand_dims', data, **pat.filter_attrs({
            'axis': axis, 'num_newaxis': num_newaxis,
        }))


class Squeeze(Call):
    def __init__(self, data: PatternLike,
                 axis: Union[tuple, list, Attr, None] = None):
        super().__init__('squeeze', data, **pat.filter_attrs({
            'axis': axis
        }))


class Cast(Call):
    def __init__(self, data: PatternLike,
                 dtype: Union[str, Attr, None] = None):
        super().__init__('cast', data, **pat.filter_attrs({
            'dtype': dtype
        }))


class Sum(Call):
    def __init__(self, data: PatternLike,
                 axis: Union[int, tuple, Attr, None] = None,
                 keepdims: Union[bool, Attr, None] = None,
                 exclude: Union[bool, Attr, None] = None):
        super().__init__('sum', data, **pat.filter_attrs({
            'axis': axis, 'keepdims': keepdims, 'exclude': exclude,
        }))


class Mean(Call):
    def __init__(self, data: PatternLike,
                 axis: Union[int, tuple, Attr, None] = None,
                 keepdims: Union[bool, Attr, None] = None,
                 exclude: Union[bool, Attr, None] = None):
        super().__init__('mean', data, **pat.filter_attrs({
            'axis': axis, 'keepdims': keepdims, 'exclude': exclude,
        }))


class MatrixSetDiag(Call):
    def __init__(self, data: PatternLike,
                 diagonal: PatternLike,
                 k: Union[int, tuple, Attr, None] = None,
                 align: Union[str, Attr, None] = None):
        super().__init__('matrix_set_diag', data, diagonal, **pat.filter_attrs({
            'k': k, 'align': align,
        }))


class Dense(Call):
    def __init__(self, data: PatternLike,
                 weight: PatternLike,
                 units: Union[int, Attr, None] = None,
                 out_dtype: Union[str, Attr, None] = None):
        super().__init__('nn.dense', data, weight, **pat.filter_attrs({
            'units': units, 'out_dtype': out_dtype,
        }))


class Conv2D(Call):
    def __init__(self, data: PatternLike,
                 weight: PatternLike,
                 strides: Union[tuple, Attr, None] = None,
                 padding: Union[tuple, Attr, None] = None,
                 dilation: Union[tuple, Attr, None] = None,
                 groups: Union[int, Attr, None] = None,
                 channels: Union[int, Attr, None] = None,
                 kernel_size: Union[tuple, Attr, None] = None,
                 data_layout: Union[str, Attr, None] = None,
                 kernel_layout: Union[str, Attr, None] = None,
                 out_layout: Union[str, Attr, None] = None,
                 out_dtype: Union[str, Attr, None] = None):
        super().__init__('nn.conv2d', data, weight, **pat.filter_attrs({
            'strides': strides, 'padding': padding, 'dilation': dilation, 'groups': groups,
            'channels': channels, 'kernel_size': kernel_size, 'data_layout': data_layout,
            'kernel_layout': kernel_layout, 'out_layout': out_layout, 'out_dtype': out_dtype,
        }))


class BatchNorm(Call):
    def __init__(self, data: PatternLike,
                 gamma: PatternLike,
                 beta: PatternLike,
                 moving_mean: PatternLike,
                 moving_var: PatternLike,
                 axis: Union[int, Attr, None] = None,
                 epsilon: Union[float, Attr, None] = None,
                 center: Union[bool, Attr, None] = None,
                 scale: Union[bool, Attr, None] = None):
        super().__init__('nn.batch_norm', data, gamma, beta, moving_mean, moving_var,
                         **pat.filter_attrs({
                             'axis': axis, 'epsilon': epsilon, 'center': center,
                             'scale': scale,
                         }))


class LayerNorm(Call):
    def __init__(self, data: PatternLike,
                 gamma: PatternLike,
                 beta: PatternLike,
                 axis: Union[int, Attr, None] = None,
                 epsilon: Union[float, Attr, None] = None,
                 center: Union[bool, Attr, None] = None,
                 scale: Union[bool, Attr, None] = None):
        super().__init__('nn.layer_norm', data, gamma, beta,
                         **pat.filter_attrs({
                             'axis': axis, 'epsilon': epsilon, 'center': center,
                             'scale': scale,
                         }))


class GroupNorm(Call):
    def __init__(self, data: PatternLike,
                 gamma: PatternLike,
                 beta: PatternLike,
                 num_groups: Union[int, Attr, None] = None,
                 axis: Union[int, Attr, None] = None,
                 epsilon: Union[float, Attr, None] = None,
                 center: Union[bool, Attr, None] = None,
                 scale: Union[bool, Attr, None] = None):
        super().__init__('nn.group_norm', data, gamma, beta,
                         **pat.filter_attrs({
                             'num_groups': num_groups, 'axis': axis, 'epsilon': epsilon,
                             'center': center, 'scale': scale,
                         }))


class InstanceNorm(Call):
    def __init__(self, data: PatternLike,
                 gamma: PatternLike,
                 beta: PatternLike,
                 axis: Union[int, Attr, None] = None,
                 epsilon: Union[float, Attr, None] = None,
                 center: Union[bool, Attr, None] = None,
                 scale: Union[bool, Attr, None] = None):
        super().__init__('nn.instance_norm', data, gamma, beta,
                         **pat.filter_attrs({
                             'axis': axis, 'epsilon': epsilon, 'center': center,
                             'scale': scale,
                         }))


class L2Normalize(Call):
    def __init__(self, data: PatternLike,
                 eps: Union[float, Attr, None] = None,
                 axis: Union[tuple, Attr, None] = None):
        super().__init__('nn.l2_normalize', data, **pat.filter_attrs({
            'eps': eps, 'axis': axis,
        }))


class BiasAdd(Call):
    def __init__(self, data: PatternLike,
                 bias: PatternLike,
                 axis: Union[int, Attr, None] = None):
        super().__init__('nn.bias_add', data, bias, **pat.filter_attrs({
            'axis': axis
        }))


class ReLU(Call):
    def __init__(self, data: PatternLike):
        super().__init__('nn.relu', data)


class Pad(Call):
    def __init__(self, data: PatternLike,
                 pad_width: Union[tuple, Attr, None] = None,
                 pad_value: Union[float, Attr, None] = None,
                 pad_mode: Union[str, Attr, None] = None):
        super().__init__('nn.pad', data, **pat.filter_attrs({
            'pad_width': pad_width, 'pad_value': pad_value, 'pad_mode': pad_mode
        }))


class BatchMatmul(Call):
    def __init__(self, x: PatternLike, y: PatternLike):
        super().__init__('nn.batch_matmul', x, y)
