from inspect import signature
from types import FunctionType
from typing import List

from tvm.relay import *

# Specify number of input tensors of Relay API.
num_inputs = {
    # Algebraic operators
    negative: 1,
    add: 2,
    subtract: 2,
    multiply: 2,
    divide: 2,
    abs: 1,
    exp: 1,
    sqrt: 1,

    # Tensor generation
    zeros: 0,

    # Tensor transformations
    concatenate: 1,
    split: 1,
    reshape: 1,
    transpose: 1,
    expand_dims: 1,
    matrix_set_diag: 2,

    # Neural network operators
    nn.conv2d: 2,
    nn.batch_norm: 5,
    nn.bias_add: 2,
    nn.max_pool2d: 1,
    nn.relu: 1,
    nn.pad: 1,
    nn.dense: 2,
    nn.softmax: 1,
    nn.batch_matmul: 2,
}


def get_func(name: str) -> FunctionType:
    return eval(name)


def get_attr_names(func: FunctionType) -> List[str]:
    if not num_inputs.__contains__(func):
        raise ValueError(
            'Specification of function \'{}\' is not found.'.format(func.__name__)
        )
    num_input = num_inputs[func]
    return list(signature(func).parameters.keys())[num_input:]
