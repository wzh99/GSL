from enum import IntFlag, auto
from inspect import signature
from types import FunctionType
from typing import List, Dict

from tvm.ir import Op
from tvm.relay import *


class OpTrait(IntFlag):
    ELEMENT_WISE = auto()
    BROADCASTING = auto()


trait = 'trait'

# Specify certain properties of ops.
op_spec: Dict[FunctionType, Dict[str, Any]] = {
    # Algebraic operators
    negative: {
        trait: OpTrait.ELEMENT_WISE,
    },
    add: {
        trait: OpTrait.ELEMENT_WISE | OpTrait.BROADCASTING,
    },
    subtract: {
        trait: OpTrait.ELEMENT_WISE | OpTrait.BROADCASTING,
    },
    multiply: {
        trait: OpTrait.ELEMENT_WISE | OpTrait.BROADCASTING,
    },
    divide: {
        trait: OpTrait.ELEMENT_WISE | OpTrait.BROADCASTING,
    },
    abs: {
        trait: OpTrait.ELEMENT_WISE,
    },
    exp: {
        trait: OpTrait.ELEMENT_WISE,
    },
    sqrt: {
        trait: OpTrait.ELEMENT_WISE,
    },

    # Tensor generation

    # Tensor transformations

    # Neural network operators
    nn.relu: {
        trait: OpTrait.ELEMENT_WISE
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


def match_trait(name: str, required: OpTrait) -> bool:
    func = get_func(name)
    if func not in op_spec:
        return False
    return required in op_spec[func][trait]


def _init_spec():
    for d in op_spec.values():
        if trait not in d:
            d[trait] = OpTrait(0)


_init_spec()
