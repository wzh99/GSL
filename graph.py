from typing import Dict, Any

import numpy as np
import op

from attrib import *


class Node:
    """
    Base class for all graph pattern nodes. This class cannot be instantiated.
    """

    def __getitem__(self, index: int):
        return GetItem(self, index)

    def __neg__(self):
        return Call('negative', self)

    def __add__(self, other):
        return Call('add', self, to_node(other))

    def __radd__(self, other):
        return Call('add', to_node(other), self)

    def __sub__(self, other):
        return Call('subtract', self, to_node(other))

    def __rsub__(self, other):
        return Call('subtract', to_node(other), self)

    def __mul__(self, other):
        return Call('multiply', self, to_node(other))

    def __rmul__(self, other):
        return Call('multiply', to_node(other), self)

    def __truediv__(self, other):
        return Call('divide', self, to_node(other))

    def __rtruediv__(self, other):
        return Call('divide', to_node(other), self)


class Wildcard(Node):
    """
    A wildcard node matches all nodes in graph. Target graph cannot contain wildcard nodes not
    defined in source graph.
    """
    pass


class Var(Node):
    """
    A variable node matches input tensor of the model. Target graph cannot contain variable nodes
    not defined in source graph.
    """
    # Ad-hoc attributes
    attrs = ['shape', 'dtype']

    def __getattr__(self, name: str):
        if not Var.attrs.__contains__(name):
            raise AttributeError('Attribute \'{}\' not found in variable node.'.format(name))
        return GetAttr(self, name)


ConstValueType = Union[int, float, list, np.ndarray]


class Const(Node):
    """
    A constant nodes stores constants in graph.
    """
    value_class = (int, float, list, np.ndarray, AttrExpr)

    def __init__(self, value: Union[ConstValueType, AttrExpr, None]):
        """
        Constructor
        :param value: In source graph, if the value is provided, it only matches nodes with the
        same value. Otherwise, any constant node will match. If the node also appears in target
        graph, the value will be copied to new graph.
        New constant nodes can be created in target graph. In target graph, the constant nodes can
        also be specified by an attribute expression with respect to nodes in source graph. \
        """
        if value is None:
            self.value = None
        elif isinstance(value, (AttrExpr, np.ndarray)):
            self.value = value
        elif isinstance(value, (int, float, list)):
            self.value = np.array(value)
        else:
            raise TypeError(
                'Cannot create constant node from value of type {}.'.format(value.__class__)
            )


def to_node(val: Union[Node, ConstValueType]) -> Node:
    """
    Create a graph pattern node with given value
    :param val: All types of values that are or can be converted to an graph pattern node.
    :return: Graph pattern node created from given value.
    """
    if isinstance(val, Node):
        return val
    elif isinstance(val, Const.value_class):
        return Const(val)
    else:
        raise TypeError('Cannot convert to graph pattern node.')


class Call(Node):
    def __init__(self, op_name: str, *args, **raw_attr):
        self.op = op_name
        self.args = args

        # Check number of inputs
        func = op.get_func(op_name)
        num_input = op.num_inputs[func]
        if num_input != len(args):
            raise ValueError(
                'Expect {} input tensor(s), got {}.'.format(num_input, len(args))
            )

        # Convert raw attribute values to attribute nodes if necessary
        self.attrs = dict([(name, to_attr(val)) for name, val in raw_attr.items()])

        # Check if attributes really exists in op
        attr_names = op.get_func_attr_names(func)
        for name, val in self.attrs.items():
            if not attr_names.__contains__(name):
                raise AttributeError(
                    'Attribute \'{}\' not found in op \'{}\'.'.format(name, op_name)
                )

    def __getattr__(self, name: str):
        func = op.get_func(self.op)
        attr_names = op.get_func_attr_names(func)
        if not attr_names.__contains__(name):
            raise AttributeError(
                'Attribute \'{}\' not found in op \'{}\'.'.format(name, self.op)
            )
        return GetAttr(self, name)


class Tuple(Node):
    def __init__(self, *raw_fields):
        self.fields = tuple([to_node(f) for f in raw_fields])


class GetItem(Node):
    def __init__(self, tup: Node, index: int):
        self.tup = tup
        self.index = index


class NodeVisitor:
    def __init__(self):
        self.visited: Dict[Node, Any] = dict()

    def visit(self, node: Node):
        if self.visited.__contains__(node):
            return self.visited[node]
        if isinstance(node, Wildcard):
            ret = self.visit_wildcard(node)
        elif isinstance(node, Var):
            ret = self.visit_var(node)
        elif isinstance(node, Const):
            ret = self.visit_const(node)
        elif isinstance(node, Call):
            ret = self.visit_call(node)
        elif isinstance(node, Tuple):
            ret = self.visit_tuple(node)
        elif isinstance(node, GetItem):
            ret = self.visit_getitem(node)
        else:
            raise RuntimeError('Unknown node type.')
        self.visited[node] = ret
        return ret

    def visit_wildcard(self, wildcard: Wildcard) -> Any:
        pass

    def visit_var(self, var: Var) -> Any:
        pass

    def visit_const(self, const: Const) -> Any:
        pass

    def visit_call(self, call: Call) -> Any:
        for arg in call.args:
            self.visit(arg)

    def visit_tuple(self, tp: Tuple) -> Any:
        for f in tp.fields:
            self.visit(f)

    def visit_getitem(self, getitem: GetItem) -> Any:
        self.visit(getitem.tup)
