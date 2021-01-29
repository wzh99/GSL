from typing import List

import numpy as np

from . import spec
from .attr import *
from .util import default_font_name


class Pattern:
    """
    Base class for all pattern graph nodes. This class cannot be directly instantiated.
    """

    def __init__(self):
        self.succ: List[Pattern] = []
        self.src_idx = -1
        self.in_tgt = False

    @property
    def pred(self):
        return []

    @property
    def is_output(self) -> bool:
        return len(self.succ) == 0

    @property
    def in_src(self):
        return self.src_idx != -1

    @property
    def is_used(self):
        return self.in_src or self.in_tgt

    # Shared attributes
    shared_attrs = {'shape', 'dtype'}

    def __getattr__(self, name: str):
        if name not in self.shared_attrs:
            raise AttributeError('Attribute \'{}\' not found in pattern node.'.format(name))
        return GetNodeAttr(self, name)

    def __getitem__(self, index: int):
        return GetItem(self, index)

    def __neg__(self):
        return Call('negative', self)

    def __add__(self, other):
        return Call('add', self, to_pat(other))

    def __radd__(self, other):
        return Call('add', to_pat(other), self)

    def __sub__(self, other):
        return Call('subtract', self, to_pat(other))

    def __rsub__(self, other):
        return Call('subtract', to_pat(other), self)

    def __mul__(self, other):
        return Call('multiply', self, to_pat(other))

    def __rmul__(self, other):
        return Call('multiply', to_pat(other), self)

    def __truediv__(self, other):
        return Call('divide', self, to_pat(other))

    def __rtruediv__(self, other):
        return Call('divide', to_pat(other), self)

    def visualize(self, name: str, path: str = 'out', font_name: str = default_font_name, **attrs):
        """
        Visualize this graph pattern node.

        :param name: Name of the file.
        :param path: Directory to store the file.
        :param font_name: Name of the font used to display node texts.
        :param attrs: Other attributes for GraphViz to plot the nodes.
        """
        from graphviz import Digraph
        graph = Digraph(name=name)
        _Visualizer(graph, fontname=font_name, **attrs).visit(self)
        graph.view(directory=path)


class Wildcard(Pattern):
    """
    A wildcard node matches all nodes in graph. Target graph cannot contain wildcard nodes not
    defined in source graph.
    """
    pass


class Var(Pattern):
    """
    A variable node matches input tensor of the model. Target graph cannot contain variable nodes
    not defined in source graph.
    """

    lhs_attrs = {'shape', 'dtype'}

    def __init__(self, **raw_attrs):
        super().__init__()

        # Check attributes for variable
        self.attrs: Dict[str, Attr] = {}
        for name, attr in raw_attrs.items():
            if name not in self.lhs_attrs:
                raise AttributeError(
                    'Attribute \'{}\' cannot appear on lhs of a constraint for '
                    'variables.'.format(name)
                )
            self.attrs[name] = to_attr(attr)


ConstValueType = Union[int, float, list, np.ndarray]


class Const(Pattern):
    """
    A constant nodes stores constants in graph.
    """
    value_class = (int, float, list, np.ndarray, Attr)

    def __init__(self, value: Union[ConstValueType, Attr, None]):
        """
        Constructor.

        :param value: In source graph, if the value is provided, it only matches nodes with the
            same value. Otherwise, any constant node will match. If the node also appears in
            target graph, the value will be copied to new graph. New constant nodes can be created
            in target graph. In target graph, the constant nodes can also be specified by an
            attribute expression with respect to nodes in source graph.
        """
        super().__init__()
        if value is None:
            self.value = None
        elif isinstance(value, (Attr, np.ndarray)):
            self.value = value
        elif isinstance(value, (int, float, list)):
            self.value = np.array(value)
        else:
            raise TypeError(
                'Cannot create constant node from value of type {}.'.format(value.__class__)
            )


PatternConvertible = Union[Pattern, ConstValueType]


def to_pat(val: PatternConvertible) -> Pattern:
    """
    Create a graph pattern node with given value.

    :param val: All types of values that are or can be converted to an graph pattern node.
    :return: Graph pattern node created from given value.
    """
    if isinstance(val, Pattern):
        return val
    elif isinstance(val, Const.value_class):
        return Const(val)
    else:
        raise TypeError('Cannot convert to pattern graph node.')


class Op(Pattern):
    def __str__(self):
        pass


class ConcreteOp(Op):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def __str__(self):
        return self.name


class OpWithFlag(Op):
    def __init__(self, flag: spec.OpFlag):
        super().__init__()
        self.flag = flag

    def __str__(self):
        return self.flag.name


class Call(Pattern):
    """
    Represents an operator call.
    """

    def __init__(self, op: Union[Op, str, spec.OpFlag], *args: PatternConvertible, **raw_attr):
        super().__init__()
        self.args = [to_pat(a) for a in args]

        # Convert valid alternatives of Op to node
        if isinstance(op, str):
            op = ConcreteOp(op)
        elif isinstance(op, spec.OpFlag):
            op = OpWithFlag(op)
        self.op = op

        # Check number of inputs
        if isinstance(op, ConcreteOp):
            num_input = spec.get_num_inputs(op.name)
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
        if isinstance(op, ConcreteOp):
            attr_names = spec.get_op_attr_names(op.name)
            for name, val in self.attrs.items():
                if name not in attr_names:
                    raise AttributeError(
                        'Attribute \'{}\' not found in op \'{}\'.'.format(name, op.name)
                    )

    @property
    def pred(self):
        return self.args

    def __getattr__(self, name: str):
        # Check shared attributes first
        if name in self.shared_attrs:
            return super().__getattr__(name)

        # Validate attribute name for concrete op
        if isinstance(self.op, ConcreteOp):
            attr_names = spec.get_op_attr_names(self.op.name)
            if name not in attr_names:
                raise AttributeError(
                    'Attribute \'{}\' not found in op \'{}\'.'.format(name, self.op)
                )

        return GetNodeAttr(self, name)


def same_attr(pat: Pattern, attrs: List[str]) -> Dict[str, Attr]:
    """
    Create a attribute expression  dictionary with form a=p.a where a is attribute name from the
    list, and p is a pattern node. This function is especially useful for specifying attribute
    identity constraints between two nodes.

    :param pat: The pattern node from which attributes are accessed.
    :param attrs: List of attribute names.
    :return: Attribute expression dictionary where eah=xh entry has the form a=p.a.
    """
    return dict([(a, pat.__getattr__(a)) for a in attrs])


class Tup(Pattern):
    def __init__(self, *raw_fields: PatternConvertible):
        super().__init__()
        self.fields = [to_pat(f) for f in raw_fields]
        for f in raw_fields:
            f.succ.append(self)

    @property
    def pred(self):
        return self.fields


class GetItem(Pattern):
    def __init__(self, tup: Pattern, index: int):
        super().__init__()
        self.tup = tup
        self.index = index
        tup.succ.append(self)

    @property
    def pred(self):
        return [self.tup]


class PatternVisitor:
    def __init__(self):
        self.visited: Dict[Pattern, Any] = dict()

    def visit(self, node: Pattern):
        if node in self.visited:
            return self.visited[node]
        if isinstance(node, Wildcard):
            ret = self.visit_wildcard(node)
        elif isinstance(node, Var):
            ret = self.visit_var(node)
        elif isinstance(node, Const):
            ret = self.visit_const(node)
        elif isinstance(node, Call):
            ret = self.visit_call(node)
        elif isinstance(node, Op):
            ret = self.visit_op(node)
        elif isinstance(node, Tup):
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
        self.visit(call.op)
        for arg in call.args:
            self.visit(arg)

    def visit_op(self, op: Op) -> Any:
        pass

    def visit_tuple(self, tup: Tup) -> Any:
        for f in tup.fields:
            self.visit(f)

    def visit_getitem(self, getitem: GetItem) -> Any:
        self.visit(getitem.tup)


class _Visualizer(PatternVisitor):
    def __init__(self, graph, **attrs):
        super().__init__()
        from graphviz import Digraph
        self.graph: Digraph = graph
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
        self.graph.node(node_id, label=str(call.op), **self.attrs)
        for a in call.args:
            self.graph.edge(self.visit(a), node_id)
        return node_id

    def visit_tuple(self, tup: Tup) -> Any:
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
