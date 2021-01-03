from typing import Dict, Union, List, Tuple

import numpy as np
from graphviz import Digraph
from tvm import ir, runtime, relay

from . import dtype


class Workload:
    """
    A workload contains computation graph of a model and its parameters.
    """

    def __init__(self, mod: ir.IRModule, params: Dict[str, Union[runtime.NDArray, np.ndarray]],
                 name: str = ''):
        """
        Constructor.
        :param mod: Relay IR module defining computation graph of the model.
        :param params: Mapping from parameter names to values. Internally, the values are stored
        in `np.ndarray`s. NDArray values will be converted to that type.
        """
        self.mod = relay.transform.InferType()(mod)
        self.params = dict([(key, self._cvt_param(val)) for key, val in params.items()])
        self.name = name

    @staticmethod
    def _cvt_param(x: Union[runtime.NDArray, np.ndarray]) -> np.ndarray:
        if isinstance(x, runtime.NDArray):
            return np.array(x.asnumpy(), dtype=dtype)
        else:
            return np.array(x, dtype=dtype)

    @staticmethod
    def from_expr(expr: relay.Expr):
        """
        Create a workload from a Relay expression. All free variables become parameters of the
        function. Model parameters will be randomly generated.
        :param expr: Body expression of function
        :return: Workload object created from this expression.
        """
        # Create module
        free_vars: List[relay.Var] = relay.analysis.free_vars(expr)
        main = relay.Function(free_vars, expr)
        mod = ir.IRModule(functions={'main': main})

        # Generate random parameters
        params: Dict[str, np.ndarray] = dict()
        for v in free_vars:
            shape: Tuple[int] = v.type_annotation.concrete_shape
            params[v.name_hint] = np.random.rand(*shape)

        return Workload(mod, params)

    def visualize(self, path: str = ''):
        """
        Visualize computation graph of this workload.
        :param path: Path to save graph visualization.
        """
        graph = Digraph(name=self.name)
        _GraphVizVisitor(graph).visit_function(self.mod['main'])
        graph.view(directory=path)


class _GraphVizVisitor(relay.ExprVisitor):
    def __init__(self, graph: Digraph):
        super().__init__()
        self.graph = graph
        self.node_id: Dict[relay.Expr, str] = {}
        self.counter = 0

    def visit(self, expr):
        if self.node_id.__contains__(expr):
            return
        super().visit(expr)

    def visit_var(self, var: relay.Var):
        expr_id = self._register_node(var)
        self.graph.node(expr_id, label=var.name_hint)

    def visit_constant(self, const: relay.Constant):
        expr_id = self._register_node(const)
        self.graph.node(expr_id, label='const')

    def visit_call(self, call: relay.Call):
        expr_id = self._register_node(call)
        self.graph.node(expr_id, label=call.op.name)
        for arg in call.args:
            self.visit(arg)
            self.graph.edge(self.node_id[arg], expr_id)

    def visit_tuple(self, tup: relay.Tuple):
        expr_id = self._register_node(tup)
        self.graph.node(expr_id, label='(,)')
        for field in tup.fields:
            self.visit(field)
            self.graph.edge(self.node_id[field], expr_id)

    def visit_tuple_getitem(self, getitem: relay.TupleGetItem):
        expr_id = self._register_node(getitem)
        self.graph.node(expr_id, label='.%d' % getitem.index)
        self.visit(getitem.tuple_value)
        self.graph.edge(self.node_id[getitem.tuple_value], expr_id)

    def _register_node(self, expr: relay.Expr) -> str:
        cur_id = str(self.counter)
        self.node_id[expr] = cur_id
        self.counter += 1
        return cur_id
