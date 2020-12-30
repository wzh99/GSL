from graph import *
from types import FunctionType
from inspect import signature, Parameter
import op


class Substitution:
    def __init__(self, src: Node, tgt: Node):
        # Check source and target
        _SrcChecker().visit(src)
        _TgtChecker().visit(tgt)

        # Store source and target
        self.src = src
        self.tgt = tgt
        pass


class _SrcChecker(NodeVisitor):
    def visit_call(self, call: Call) -> Any:
        # Visit arguments
        super().visit_call(call)

        # Check number of inputs
        func = op.name_to_func(call.op)
        num_input = op.num_inputs[func]
        _check_num_input(num_input, call)

        # Check whether specified attributes really exist
        _validate_attrib(func, call)


def _check_num_input(num_input: int, call: Call):
    if num_input != len(call.args):
        raise RuntimeError('Expect {} input tensor(s), got {}'.format(
            num_input, len(call.args)
        ))


def _validate_attrib(func: FunctionType, call: Call):
    num_input = op.num_inputs[func]
    func_attrib = list(signature(func).parameters.keys())[num_input:]
    for a in call.attrib.keys():
        if not func_attrib.__contains__(a):
            raise RuntimeError('Invalid attribute: {}'.format(a))


class _TgtChecker(NodeVisitor):
    def visit_call(self, call: Call) -> Any:
        # Visit arguments
        super().visit_call(call)

        # Check number of inputs
        func = op.name_to_func(call.op)
        num_input = op.num_inputs[func]
        _check_num_input(num_input, call)

        # Check whether specified attributes really exist
        _validate_attrib(func, call)

        # Check whether all non-default attributes are provided
        required = set()
        for name, param in list(signature(func).parameters.items())[num_input:]:
            if param.default == Parameter.empty:
                required.add(name)
        required.difference_update(call.attrib.keys())
        if len(required) != 0:
            raise RuntimeError('Attributes {} are not provided for \'{}\''.format(
                tuple(required), call.op
            ))
