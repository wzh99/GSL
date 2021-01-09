from typing import Dict, Any

import numpy as np
from graphviz import Digraph
from tvm import relay

from . import op
from .attr import *
from .util import default_font_name


class Node:
    """
    Base class for all graph pattern nodes. This class cannot be instantiated.
    """

    def __init__(self):
        self.succ: List[Node] = []

    @property
    def pred(self):
        return []

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

    def visualize(self, name: str, path: str = 'out', font_name: str = default_font_name, **attrs):
        """
        Visualize this graph pattern node.
        :param name: Name of the file.
        :param path: Directory to store the file.
        :param font_name: Name of the font used to display node texts.
        :param attrs: Other attributes for GraphViz to plot the nodes.
        """
        graph = Digraph(name=name)
        _PatternVisualizer(graph, fontname=font_name, **attrs).visit(self)
        graph.view(directory=path)


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
    avail_attrs = {'shape', 'dtype'}

    def __init__(self, **raw_attrs):
        super().__init__()

        # Check attributes for variable
        self.attrs: Dict[str, AttrExpr] = {}
        for name, attr in raw_attrs.items():
            if not self.avail_attrs.__contains__(name):
                raise AttributeError(
                    'Attribute \'{}\' not found in variable node'.format(name)
                )
            self.attrs[name] = to_attr(attr)

    def __getattr__(self, name: str):
        if not Var.avail_attrs.__contains__(name):
            raise AttributeError('Attribute \'{}\' not found in variable node.'.format(name))
        return GetAttr(self, name)

    @staticmethod
    def get_expr_attr(var: relay.Var, name: str):
        if name == 'shape':
            return var.type_annotation.concrete_shape
        elif name == 'dtype':
            return var.type_annotation.dtype
        else:
            raise RuntimeError('Invalid attribute name.')


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
        super().__init__()
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
    """
    Represents an operator call.
    """

    def __init__(self, op_name: str, *args: Node, **raw_attr):
        super().__init__()
        self.op = op_name
        self.args = list(args)

        # Check number of inputs
        func = op.get_func(op_name)
        num_input = op.num_inputs[func]
        if num_input != len(args):
            raise ValueError(
                'Expect {} input tensor(s), got {}.'.format(num_input, len(args))
            )

        # Set self as output of arguments
        for a in args:
            a.succ.append(self)

        # Convert raw attribute values to attribute nodes if necessary
        self.attrs = dict([(name, to_attr(val)) for name, val in raw_attr.items()])

        # Check if specified attributes really exists in op
        attr_names = op.get_attr_names(func)
        for name, val in self.attrs.items():
            if not attr_names.__contains__(name):
                raise AttributeError(
                    'Attribute \'{}\' not found in op \'{}\'.'.format(name, op_name)
                )

    @property
    def pred(self):
        return self.args

    def __getattr__(self, name: str):
        # Check if attribute name is valid
        func = op.get_func(self.op)
        attr_names = op.get_attr_names(func)
        if not attr_names.__contains__(name):
            raise AttributeError(
                'Attribute \'{}\' not found in op \'{}\'.'.format(name, self.op)
            )

        return GetAttr(self, name)


class Tuple(Node):
    def __init__(self, *raw_fields: Node):
        super().__init__()
        self.fields = [to_node(f) for f in raw_fields]
        for f in raw_fields:
            f.succ.append(self)

    @property
    def pred(self):
        return self.fields


class GetItem(Node):
    def __init__(self, tup: Node, index: int):
        super().__init__()
        self.tup = tup
        self.index = index
        tup.succ.append(self)

    @property
    def pred(self):
        return [self.tup]


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

    def visit_tuple(self, tup: Tuple) -> Any:
        for f in tup.fields:
            self.visit(f)

    def visit_getitem(self, getitem: GetItem) -> Any:
        self.visit(getitem.tup)


class _PatternVisualizer(NodeVisitor):
    def __init__(self, graph: Digraph, **attrs):
        super().__init__()
        self.graph = graph
        self.attrs = attrs
        self.counter = 0

    def visit_wildcard(self, wildcard: Wildcard) -> Any:
        node_id = self._next_id()
        self.graph.node(node_id, label='*', **self.attrs)
        return node_id

    def visit_var(self, var: Var) -> Any:
        node_id = self._next_id()
        self.graph.node(node_id, label='Var', **self.attrs)
        return node_id

    def visit_const(self, const: Const) -> Any:
        node_id = self._next_id()
        self.graph.node(node_id, label='Const', **self.attrs)
        return node_id

    def visit_call(self, call: Call) -> Any:
        node_id = self._next_id()
        self.graph.node(node_id, label=call.op, **self.attrs)
        for a in call.args:
            self.graph.edge(self.visit(a), node_id)
        return node_id

    def visit_tuple(self, tup: Tuple) -> Any:
        node_id = self._next_id()
        self.graph.node(node_id, label='(,)', **self.attrs)
        for f in tup.fields:
            self.graph.edge(self.visit(f), node_id)
        return node_id

    def visit_getitem(self, getitem: GetItem) -> Any:
        node_id = self._next_id()
        self.graph.node(node_id, label='.{}'.format(getitem.index), **self.attrs)
        self.graph.edge(self.visit(getitem.tup), node_id)
        return node_id

    def _next_id(self) -> str:
        cur_id = str(self.counter)
        self.counter += 1
        return cur_id


class AttrEvaluator(AttrVisitor):
    def __init__(self, pat_to_expr: Dict[Node, relay.Expr]):
        self.pat_to_expr = pat_to_expr

    def visit_any(self, a: AnyAttr):
        return None

    def visit_const(self, const: ConstAttr):
        return const.value

    def visit_get_attr(self, get_attr: GetAttr):
        # Get actual expression from map
        node = get_attr.node
        name = get_attr.name
        expr = self.pat_to_expr[node]

        # Access attribute according to type of node
        if isinstance(node, Call):
            return expr.attrs[name]
        elif isinstance(node, Var):
            assert isinstance(expr, relay.Var)
            return Var.get_expr_attr(expr, name)
        else:
            raise RuntimeError('Impossible case.')

    def visit_list(self, list_attr: ListAttr):
        return [self.visit(f) for f in list_attr.fields]

    def visit_tuple(self, tup_attr: TupleAttr):
        return tuple([self.visit(f) for f in tup_attr.fields])

    def visit_getitem(self, getitem: GetItemAttr):
        return self.visit(getitem.seq)[getitem.index]

    def visit_binary(self, binary: BinaryExpr):
        raise NotImplementedError()
