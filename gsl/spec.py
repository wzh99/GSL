from enum import IntFlag, auto
from inspect import signature
from types import FunctionType
from typing import List

from tvm.relay import *


class OpProperty(IntFlag):
    ELEMENT_WISE = auto()
    BROADCASTING = auto()


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
    ones: 0,

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
    nn.relu: 1,
    nn.pad: 1,
    nn.batch_matmul: 2,
}


def get_func(name: str) -> FunctionType:
    return eval(name)


def get_attr_names(func: FunctionType) -> List[str]:
    if func not in num_inputs:
        raise ValueError(
            'Specification of Relay API \'{}\' is not found.'.format(func.__name__)
        )
    num_input = num_inputs[func]
    return list(signature(func).parameters.keys())[num_input:]
