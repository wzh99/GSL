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
op_trait: Dict[FunctionType, OpTrait] = {
    # Algebraic operators
    negative: OpTrait.ELEMENT_WISE,
    add: OpTrait.ELEMENT_WISE | OpTrait.BROADCASTING,
    subtract: OpTrait.ELEMENT_WISE | OpTrait.BROADCASTING,
    multiply: OpTrait.ELEMENT_WISE | OpTrait.BROADCASTING,
    divide: OpTrait.ELEMENT_WISE | OpTrait.BROADCASTING,
    abs: OpTrait.ELEMENT_WISE,
    exp: OpTrait.ELEMENT_WISE,
    sqrt: OpTrait.ELEMENT_WISE,

    # Tensor generation

    # Tensor transformations

    # Neural network operators
    nn.relu: OpTrait.ELEMENT_WISE
}


def get_api(name: str) -> FunctionType:
    return eval(name)


def get_num_inputs(name: str) -> int:
    return Op.get(name).num_inputs


def get_op_attr_names(name: str) -> List[str]:
    func = get_api(name)
    num_inputs = get_num_inputs(name)
    return list(signature(func).parameters.keys())[num_inputs:]


def match_trait(name: str, required: OpTrait) -> bool:
    func = get_api(name)
    if func not in op_trait:
        return False
    return required in op_trait[func]
