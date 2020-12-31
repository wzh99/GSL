from inspect import signature, Parameter
from types import FunctionType
from typing import Set
from tvm.relay import dataflow_pattern as dfp

import op
from graph import *


class Substitution:
    """
    Represents a graph substitution rule.
    """

    def __init__(self, src: Node, tgt: Node):
        """
        Constructor.
        :param src: Source graph pattern.
        :param tgt: Target graph pattern.
        """
        # Check source and target
        src_checker = _SourceChecker()
        src_checker.visit(src)
        _TargetChecker(src_checker.wildcard_vars).visit(tgt)

        # Store source and target
        self.src = src
        self.tgt = tgt

        # Create source pattern for matching in Relay data-flow callback
        self.src_pat = _PatternCreator().visit(src)
        pass


class _SourceChecker(NodeVisitor):
    wildcard_vars: Set[Union[Wildcard, Var]] = set()

    def visit_wildcard(self, wildcard: Wildcard) -> Any:
        self.wildcard_vars.add(wildcard)

    def visit_var(self, var: Var) -> Any:
        self.wildcard_vars.add(var)

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
        raise ValueError('Expect {} input tensor(s), got {}.'.format(
            num_input, len(call.args)
        ))


def _validate_attrib(func: FunctionType, call: Call):
    num_input = op.num_inputs[func]
    func_attrib = list(signature(func).parameters.keys())[num_input:]
    for arg in call.attrib.keys():
        if not func_attrib.__contains__(arg):
            raise AttributeError('Invalid attribute: {}.'.format(arg))


class _TargetChecker(NodeVisitor):
    def __init__(self, wildcard_vars: Set[Union[Wildcard, Var]]):
        super().__init__()
        self.wildcard_vars = wildcard_vars

    def visit_wildcard(self, wildcard: Wildcard) -> Any:
        if not self.wildcard_vars.__contains__(wildcard):
            raise ValueError(
                'Target graph contains wildcard node not defined in source graph.'
            )

    def visit_var(self, var: Var) -> Any:
        if not self.wildcard_vars.__contains__(var):
            raise ValueError(
                'Target graph contains variable node not defined in source graph.'
            )

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
            raise AttributeError('Attributes {} are not provided for \'{}\'.'.format(
                tuple(required), call.op
            ))


class _PatternCreator(NodeVisitor):
    def visit_wildcard(self, wildcard: Wildcard) -> dfp.DFPattern:
        return dfp.wildcard()

    def visit_var(self, var: Var) -> dfp.DFPattern:
        return dfp.is_var()

    def visit_const(self, const: Const) -> dfp.DFPattern:
        return dfp.is_constant()

    def visit_call(self, call: Call) -> dfp.DFPattern:
        super().visit_call(call)
        args = [self.visited[node] for node in call.args]
        return dfp.is_op(call.op)(*args)

    # noinspection PyTypeChecker
    def visit_tuple(self, tp: Tuple) -> dfp.DFPattern:
        super().visit_tuple(tp)
        fields = [self.visited[node] for node in tp.fields]
        return dfp.is_tuple(fields)

    def visit_getitem(self, getitem: GetItem) -> dfp.DFPattern:
        super().visit_getitem(getitem)
        return dfp.is_tuple_get_item(self.visited[getitem.tup], index=getitem.index)
