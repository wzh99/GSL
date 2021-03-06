from typing import List, Dict, Callable, Union, Any, Optional, Generic, TypeVar

import numpy as np

from . import attr, spec
from .util import default_font_name


class Pattern:
    """
    Base class for all pattern graph nodes. This class cannot be directly instantiated.
    """

    NOT_IN_SRC = -1

    def __init__(self):
        self.attrs: Dict[str, attr.Attr] = {}
        self.succ: List[Pattern] = []
        self.src_idx = self.NOT_IN_SRC
        self.in_tgt = False
        self.is_template = False

    @property
    def pred(self) -> List['Pattern']:
        return []

    @property
    def is_output(self) -> bool:
        return len(self.succ) == 0

    @property
    def in_src(self) -> bool:
        return self.src_idx != self.NOT_IN_SRC

    @property
    def src_succ(self) -> List['Pattern']:
        return list(filter(lambda p: p.in_src and not p.is_template, self.succ))

    @property
    def is_used(self) -> bool:
        return self.in_src or self.in_tgt

    def check_all(self, predicate: Callable[['Pattern'], bool]) -> bool:
        if not predicate(self):
            return False
        for p in self.pred:
            if not p.check_all(predicate):
                return False
        return True

    def check_any(self, predicate: Callable[['Pattern'], bool]) -> bool:
        if predicate(self):
            return True
        for p in self.pred:
            if p.check_any(predicate):
                return True
        return False

    @property
    def avail_attrs(self) -> List[str]:
        return []

    def has_attr(self, name: str) -> bool:
        return name in self.avail_attrs

    def __getitem__(self, *item):
        return GetItem(self, item[0])

    def __getattr__(self, name: str) -> attr.GetAttr:
        if not self.has_attr(name):
            raise AttributeError(
                'Attribute {} not found in pattern.'.format(name)
            )
        return attr.GetAttr(self, name)

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

    def __contains__(self, sub: 'Pattern') -> bool:
        return self.check_any(lambda p: p is sub)

    def clear(self):
        for p in self.pred:
            p.clear()

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
        _Visualizer(graph, fontname=font_name, **attrs).visit(self, None)
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

    tensor_attrs = ['shape', 'dtype', 'ndim']

    def __init__(self, shape: Union[tuple, attr.Attr, None] = None,
                 dtype: Union[str, attr.Attr, None] = None,
                 ndim: Union[int, attr.Attr, None] = None):
        super().__init__()

        # Check attributes for variable
        raw_attrs = filter_attrs(dict(zip(
            self.tensor_attrs,
            [shape, dtype, ndim]
        )))
        for n, a in raw_attrs.items():
            if n in self.tensor_attrs:
                self.attrs[n] = attr.to_attr(a)
            else:
                raise AttributeError(
                    'Attribute {} not found in variable node.'.format(n)
                )

    @property
    def avail_attrs(self) -> List[str]:
        return self.tensor_attrs


def filter_attrs(attrs: Dict[str, Any]) -> Dict[str, Any]:
    filtered = {}
    for k, v in attrs.items():
        if v is not None:
            filtered[k] = v
    return filtered


ConstValueType = Union[int, float, list, np.ndarray]


class Const(Pattern):
    """
    A constant nodes stores constants in graph.
    """
    value_class = (int, float, list, np.ndarray, attr.Attr)

    def __init__(self, value: Union[ConstValueType, attr.Attr, None]):
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
        elif isinstance(value, (attr.Attr, np.ndarray)):
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


class OpWithTrait(Op):
    def __init__(self, trait: spec.OpTrait):
        super().__init__()
        self.trait = trait

    def __str__(self):
        return self.trait.name


class Call(Pattern):
    """
    Represents an operator call.
    """

    def __init__(self, op: Union[Op, str, spec.OpTrait], *args: PatternConvertible, **raw_attr):
        super().__init__()
        self.args = [to_pat(a) for a in args]

        # Convert valid alternatives of Op to node
        if isinstance(op, str):
            op = ConcreteOp(op)
        elif isinstance(op, spec.OpTrait):
            op = OpWithTrait(op)
        self.op = op

        # Check number of inputs
        if isinstance(op, ConcreteOp):
            num_input = spec.get_num_inputs(op.name)
            if num_input != len(args):
                raise ValueError(
                    'Expect {} input tensor(s), got {}.'.format(num_input, len(args))
                )

        # Set self as output of arguments
        for a in self.args:
            a.succ.append(self)

        # Convert raw attribute values to attribute nodes if necessary
        self.attrs.update(
            dict([(name, attr.to_attr(val)) for name, val in raw_attr.items()])
        )

        # Check if specified attributes really exists in op
        if isinstance(op, ConcreteOp):
            attr_names = spec.get_op_attr_names(op.name)
            for name in self.attrs.keys():
                if name not in attr_names:
                    raise AttributeError(
                        'Attribute \'{}\' not found in op \'{}\'.'.format(name, op.name)
                    )

    @property
    def pred(self):
        return self.args

    def has_attr(self, name: str) -> bool:
        if isinstance(self.op, ConcreteOp):
            attr_names = spec.get_op_attr_names(self.op.name)
            return name in attr_names
        else:
            return True


def same_attr(pat: Pattern, attrs: List[str]) -> Dict[str, attr.Attr]:
    """
    Create a attribute expression dictionary with form `a=p.a` where `a` is attribute name from
    the list, and `p` is a pattern node. This function is especially useful for specifying
    attribute identity constraints between two nodes.

    :param pat: The pattern node from which attributes are accessed.
    :param attrs: List of attribute names.
    :return: Attribute expression dictionary where each entry has form a=p.a.
    """
    return dict([(a, pat.__getattr__(a)) for a in attrs])


class Tuple(Pattern):
    def __init__(self, *raw_fields: PatternConvertible):
        super().__init__()
        self.fields = [to_pat(f) for f in raw_fields]
        for f in self.fields:
            f.succ.append(self)

    @property
    def pred(self):
        return self.fields


class GetItem(Pattern):
    def __init__(self, tup: Pattern, index: attr.AttrConvertible = None):
        super().__init__()
        self.tup = tup
        self.idx = attr.to_attr(index)
        tup.succ.append(self)

    @property
    def pred(self):
        return [self.tup]

    @property
    def avail_attrs(self) -> List[str]:
        return ['index']


class Variadic(Pattern):
    """
    Matches tuple with arbitrary input, each with the same pattern. It can also be used to specify
    arbitrary number of output nodes, each with the same pattern.
    """

    def __init__(self, pat: Pattern,
                 templates: Optional[List[Pattern]] = None,
                 first: Optional[List[Optional[Pattern]]] = None,
                 index: Optional[attr.Symbol] = None,
                 length: Optional[attr.AttrConvertible] = None,
                 min_len: Optional[int] = None):
        """
        Constructor.

        :param pat: The common pattern of tuple fields.
        :param templates: Sub-patterns serve as templates, that is, for each of the expression
            matched, a unique pattern will be created. Note that if at least one sub-pattern is a
            template, the whole pattern must be a template as well.
        :param first: Pattern used in firstly matched expression. Sometimes, there are constraints
            imposed between template patterns. In this situation, pattern used for matching the
            first expression is different from the rest. The i-th pattern in first list will be
            used for the i-th template at first match.
        :param index: Symbol that indicates the index of matched expression.
        :param length: An attribute expression that indicates how long the pattern will be. In
            source pattern, it is optional. If provided, the length of tuple will be checked.
            In target pattern, it is required to specify the length of constructed tuple.
        :param min_len: An integer specifying the minimum number of fields that could be a match
            of this variadic. This only works for variadic source output pattern. In other cases,
            this value will be ignored.
        """
        super().__init__()

        # Add successor to pattern
        self.pat = pat
        pat.succ.append(self)

        # Check templates
        if templates is not None and len(templates) > 0:  # at least one pattern is a template
            if pat not in templates:
                raise ValueError(
                    'Template pattern must be duplicated if any of its sub-pattern should be '
                    'duplicated.'
                )
            if first is not None:
                if len(templates) != len(first):
                    raise ValueError(
                        'Number of first symbols must match number of duplicated symbols.'
                    )
            else:
                first = [None] * len(templates)
        else:
            templates = []
            first = []

        # Map template to first
        self.templates: List[Pattern] = templates
        self.first: List[Optional[Pattern]] = first
        self.tpl_to_fst = dict(zip(templates, first))
        for t, f in self.tpl_to_fst.items():
            if isinstance(t, Variadic):
                raise TypeError(
                    'Variadic cannot be a template pattern.'
                )
            if not pat.check_any(lambda p: p is t):
                raise ValueError(
                    'Template is not sub-pattern of field pattern.'
                )
            if f is not None and t.__class__ != f.__class__:
                raise TypeError(
                    'Template and first pattern must be of same type.'
                )
            t.is_template = True

        # Initialize index and length
        self.index = index
        self.len: Optional[attr.Attr] = None
        if length is not None:
            self.len = attr.to_attr(length)
        self.min_len = min_len

        # Initialize records during substitution
        self.pat_inst: List[Pattern] = []
        self.tpl_inst: List[Dict[Pattern, Pattern]] = []

    @property
    def pred(self):
        return self.pat_inst

    @property
    def avail_attrs(self) -> List[str]:
        return ['length']

    def __call__(self, tpl: Pattern, index: attr.AttrConvertible):
        if tpl not in self.templates:
            raise ValueError('Pattern is not template of this variadic.')
        return GetInst(self, tpl, attr.to_attr(index))

    def __len__(self) -> int:
        return len(self.pat_inst)

    def get_inst(self, idx: int, t: Pattern):
        if idx >= len(self):
            raise RuntimeError(
                'Index {} out of bound {}.'.format(idx, len(self))
            )
        inst_map = self.tpl_inst[idx]
        if t not in inst_map:
            raise RuntimeError(
                'Template pattern not found in instance mapping.'
            )
        return inst_map[t]

    def clear(self):
        super().clear()
        while len(self.pat_inst) > 0:
            self.rollback()

    def has_first(self, t: Pattern) -> bool:
        return self.tpl_to_fst[t] is not None

    def instantiate(self) -> Pattern:
        # Instantiate templates
        visitor = _PatInst(self)
        inst = visitor.visit(self.pat, None)
        inst_map = visitor.map

        # Maps templates to first instances
        if len(self) == 0:
            for tpl in self.templates:
                if self.has_first(tpl):
                    inst_map[tpl] = self.tpl_to_fst[tpl]

        # Add record
        self.pat_inst.append(inst)
        self.tpl_inst.append(inst_map)

        return inst

    def rollback(self):
        self.pat_inst.pop()
        inst_map = self.tpl_inst.pop()
        for tpl, inst in inst_map.items():
            if inst not in self.first:
                for p in inst.pred:
                    p.succ.remove(inst)


class GetInst(Pattern):
    def __init__(self, var: Variadic, tpl: Pattern, index: attr.Attr):
        super().__init__()
        self.var = var
        self.tpl = tpl
        self.idx = index

    def has_attr(self, name: str) -> bool:
        return True


ArgType = TypeVar('ArgType')


class PatternVisitor(Generic[ArgType]):
    def __init__(self):
        self.visited: Dict[Pattern, Any] = dict()

    def visit(self, pat: Pattern, arg: ArgType) -> Any:
        if pat in self.visited:
            return self.visited[pat]
        if isinstance(pat, Wildcard):
            ret = self.visit_wildcard(pat, arg)
        elif isinstance(pat, Var):
            ret = self.visit_var(pat, arg)
        elif isinstance(pat, Const):
            ret = self.visit_const(pat, arg)
        elif isinstance(pat, Call):
            ret = self.visit_call(pat, arg)
        elif isinstance(pat, Tuple):
            ret = self.visit_tuple(pat, arg)
        elif isinstance(pat, GetItem):
            ret = self.visit_getitem(pat, arg)
        elif isinstance(pat, Variadic):
            ret = self.visit_variadic(pat, arg)
        elif isinstance(pat, GetInst):
            ret = self.visit_get_instance(pat, arg)
        else:
            raise RuntimeError('Unknown node type.')
        self.visited[pat] = ret
        return ret

    def visit_wildcard(self, wildcard: Wildcard, arg: ArgType) -> Any:
        pass

    def visit_var(self, var: Var, arg: ArgType) -> Any:
        pass

    def visit_const(self, const: Const, arg: ArgType) -> Any:
        pass

    def visit_call(self, call: Call, arg: ArgType) -> Any:
        for a in call.args:
            self.visit(a, arg)

    def visit_tuple(self, tup: Tuple, arg: ArgType) -> Any:
        for f in tup.fields:
            self.visit(f, arg)

    def visit_getitem(self, getitem: GetItem, arg: ArgType) -> Any:
        self.visit(getitem.tup, arg)

    def visit_variadic(self, var: Variadic, arg: ArgType) -> Any:
        self.visit(var.pat, arg)

    def visit_get_instance(self, get_inst: GetInst, arg: ArgType) -> Any:
        self.visit(get_inst.var, arg)

    # Visitor will not dispatch this method
    def visit_op(self, op: Op, arg: ArgType) -> Any:
        pass


class _PatInst(PatternVisitor[None]):
    def __init__(self, var: Variadic):
        super().__init__()
        self.var = var
        self.index = len(var)
        self.map: Dict[Pattern, Pattern] = {}

    def visit(self, pat: Pattern, arg: None) -> Pattern:
        if pat in self.var.templates:  # current pattern is a template
            if self.index == 0 and self.var.has_first(pat):  # this template has first instance
                return self.var.tpl_to_fst[pat]
            else:
                # Instantiate template and copy attributes to instance
                inst: Pattern = super().visit(pat, arg)
                inst.is_template = False
                inst.src_idx = pat.src_idx
                inst.in_tgt = pat.in_tgt
                self.map[pat] = inst  # map template to created instance
                return inst
        else:
            return pat  # not a template, keep it

    def visit_wildcard(self, wildcard: Wildcard, arg: None) -> Pattern:
        return Wildcard()

    def visit_var(self, var: Var, arg: None) -> Pattern:
        return Var(**var.attrs)

    def visit_const(self, const: Const, arg: None) -> Pattern:
        return Const(const.value)

    def visit_call(self, call: Call, arg: None) -> Pattern:
        args = self._visit_pred(call, arg)
        return Call(call.op, *args, **call.attrs)

    def visit_tuple(self, tup: Tuple, arg: None) -> Pattern:
        fields = self._visit_pred(tup, arg)
        return Tuple(*fields)

    def visit_getitem(self, getitem: GetItem, arg: None) -> Pattern:
        return GetItem(self.visit(getitem.tup, arg), getitem.idx)

    def visit_variadic(self, var: Variadic, arg: None) -> Pattern:
        raise RuntimeError('Unreachable.')

    def visit_get_instance(self, get_inst: GetInst, arg: None) -> Pattern:
        return GetInst(get_inst.var, get_inst.tpl, get_inst.idx)

    def _visit_pred(self, pat: Pattern, arg: None) -> List[Pattern]:
        return [self.visit(p, arg) for p in pat.pred]


class _Visualizer(PatternVisitor[None]):
    def __init__(self, graph, **attrs):
        super().__init__()
        from graphviz import Digraph
        self.graph: Digraph = graph
        self.attrs = attrs
        self.counter = 0

    def visit_wildcard(self, wildcard: Wildcard, arg: None) -> Any:
        node_id = self._next_id()
        self.graph.node(node_id, label='*', **self.attrs)
        return node_id

    def visit_var(self, var: Var, arg: None) -> Any:
        node_id = self._next_id()
        self.graph.node(node_id, label='Var', **self.attrs)
        return node_id

    def visit_const(self, const: Const, arg: None) -> Any:
        node_id = self._next_id()
        self.graph.node(node_id, label='Const', **self.attrs)
        return node_id

    def visit_call(self, call: Call, arg: None) -> Any:
        node_id = self._next_id()
        self.graph.node(node_id, label=str(call.op), **self.attrs)
        for a in call.args:
            self.graph.edge(self.visit(a, arg), node_id)
        return node_id

    def visit_tuple(self, tup: Tuple, arg: None) -> Any:
        node_id = self._next_id()
        self.graph.node(node_id, label='(,)', **self.attrs)
        for f in tup.fields:
            self.graph.edge(self.visit(f, arg), node_id)
        return node_id

    def visit_getitem(self, getitem: GetItem, arg: None) -> Any:
        node_id = self._next_id()
        self.graph.node(node_id, label='.{}'.format(getitem.idx), **self.attrs)
        self.graph.edge(self.visit(getitem.tup, arg), node_id)
        return node_id

    def visit_variadic(self, var: Variadic, arg: ArgType) -> Any:
        node_id = self._next_id()
        self.graph.node(node_id, label='(...)', **self.attrs)
        self.graph.edge(self.visit(var.pat, arg), node_id)
        return node_id

    def visit_get_instance(self, get_inst: GetInst, arg: ArgType) -> Any:
        node_id = self._next_id()
        self.graph.node(node_id, label='[i, x]', **self.attrs)
        self.graph.edge(self.visit(get_inst.var, arg), node_id)
        return node_id

    def _next_id(self) -> str:
        cur_id = str(self.counter)
        self.counter += 1
        return cur_id
