from enum import IntFlag, auto
from inspect import signature
from types import FunctionType
from typing import List, Dict

from tvm.ir import Op
from tvm.relay import *


class OpFlag(IntFlag):
    ELEMENT_WISE = auto()
    BROADCASTING = auto()


flag = 'flag'

# Specify certain properties of ops.
op_spec: Dict[FunctionType, Dict[str, Any]] = {
    # Algebraic operators
    negative: {
        flag: OpFlag.ELEMENT_WISE,
    },
    add: {
        flag: OpFlag.ELEMENT_WISE | OpFlag.BROADCASTING,
    },
    subtract: {
        flag: OpFlag.ELEMENT_WISE | OpFlag.BROADCASTING,
    },
    multiply: {
        flag: OpFlag.ELEMENT_WISE | OpFlag.BROADCASTING,
    },
    divide: {
        flag: OpFlag.ELEMENT_WISE | OpFlag.BROADCASTING,
    },
    abs: {
        flag: OpFlag.ELEMENT_WISE,
    },
    exp: {
        flag: OpFlag.ELEMENT_WISE,
    },
    sqrt: {
        flag: OpFlag.ELEMENT_WISE,
    },

    # Tensor generation

    # Tensor transformations

    # Neural network operators
    nn.relu: {
        flag: OpFlag.ELEMENT_WISE
    },
}


def get_func(name: str) -> FunctionType:
    return eval(name)


def get_num_inputs(name: str) -> int:
    return Op.get(name).num_inputs


def get_op_attr_names(name: str) -> List[str]:
    func = get_func(name)
    num_inputs = get_num_inputs(name)
    return list(signature(func).parameters.keys())[num_inputs:]


def match_flag(name: str, required_flag: OpFlag) -> bool:
    func = get_func(name)
    if func not in op_spec:
        return False
    return required_flag in op_spec[func][flag]


def _init_spec():
    for d in op_spec.values():
        if flag not in d:
            d[flag] = OpFlag(0)


_init_spec()
