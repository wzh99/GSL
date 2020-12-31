from typing import Union, Dict, Any

import numpy as np

from attrib import GetAttrib, ConstAttrib


class Node:
    """
    Base class for all graph nodes. This class cannot be instantiated.
    """

    def __neg__(self):
        return Call('negative', self)

    def __add__(self, other):
        return Call('add', self, other)

    def __sub__(self, other):
        return Call('subtract', self, other)

    def __mul__(self, other):
        return Call('multiply', self, other)

    def __truediv__(self, other):
        return Call('divide', self, other)


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


class Const(Node):
    """
    A constant nodes stores constants in graph.
    """

    def __init__(self, value: Union[int, float, list, np.ndarray, None]):
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


class Call(Node):
    def __init__(self, op: str, *args: Node, **attrib):
        self.op = op
        self.args = args

        # Convert values to constant attributes if necessary
        proc_attrib = attrib.copy()
        for name, val in attrib.items():
            if isinstance(val, (bool, int, tuple, list, str)):
                proc_attrib[name] = ConstAttrib(val)
        self.attrib = proc_attrib

    def __getattr__(self, name: str):
        return GetAttrib(self, name)


class Tuple(Node):
    def __init__(self, *fields):
        self.fields = fields

    def __getitem__(self, index: int):
        return GetItem(self, index)


class GetItem(Node):
    def __init__(self, tup: Tuple, index: int):
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
            raise RuntimeError('Unhandled node case.')
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
