from typing import Dict, Any

import numpy as np

from attrib import *


class Node:
    """
    Base class for all graph pattern nodes. This class cannot be instantiated.
    """

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
    pass


ConstValueType = Union[int, float, list, np.ndarray]
const_value_class = (int, float, list, np.ndarray)


class Const(Node):
    """
    A constant nodes stores constants in graph.
    """

    def __init__(self, value: Union[ConstValueType, None]):
        """
        Constructor
        :param value: In source graph, if the value is provided, it only matches nodes with the
        same value. Otherwise, any constant node will match. If the node also appears in target
        graph, the value will be copied to new graph. New constant nodes can also be created in
        target graph.
        """
        if value is None:
            return
        if isinstance(value, (int, float, list)):
            value = np.array(value)
        if not isinstance(value, np.ndarray):
            raise TypeError('Not a constant')
        self.value = value


def to_node(val: Union[Node, ConstValueType]) -> Node:
    """
    Create a graph pattern node with given value
    :param val: All types of values that are or can be converted to an graph pattern node.
    :return: Graph pattern node created from given value.
    """
    if isinstance(val, Node):
        return val
    elif isinstance(val, const_value_class):
        return Const(val)
    else:
        raise TypeError('Cannot convert to graph pattern node.')


class Call(Node):
    def __init__(self, op: str, *args: Node, **raw_attr):
        self.op = op
        self.args = args

        # Convert values to constant attributes if necessary
        attrs = raw_attr.copy()
        for name, val in raw_attr.items():
            attrs[name] = to_attr(val)
        self.attrs = attrs

    def __getattr__(self, name: str):
        return GetAttr(self, name)


class Tuple(Node):
    def __init__(self, *raw_fields):
        self.fields = tuple([to_node(f) for f in raw_fields])

    def __getitem__(self, index: int):
        return GetItem(self, index)


class GetItem(Node):
    def __init__(self, tup: Tuple, index: int):
        if index >= len(tup.fields):
            raise ValueError('Index {} out of bound.'.format(index))
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
