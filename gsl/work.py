from typing import Dict, Union, List, Tuple, Optional, Set

import numpy as np
from graphviz import Digraph
import tvm
from tvm import ir, runtime, relay, transform

from . import _default_dtype


class Workload:
    """
    A workload contains computation graph of a model and its parameters.
    """

    def __init__(self, mod: ir.IRModule, params: Dict[str, Union[runtime.NDArray, np.ndarray]],
                 dtype: str = _default_dtype, name: str = ''):
        """
        Constructor.
        :param mod: Relay IR module defining computation graph of the model.
        :param params: Mapping from parameter names to values. Internally, the values are stored
        in `np.ndarray`s. NDArray values will be converted to that type.
        """
        self.mod = transform.Sequential(passes=[
            _AlterDType(dtype),
            relay.transform.InferType()
        ])(mod)
        self.params = dict([(key, self._cvt_param(val)) for key, val in params.items()])
        self.name = name
        self.executor: Optional[relay.build_module.GraphExecutor] = None
        self.func = None

    @staticmethod
    def _cvt_param(x: Union[runtime.NDArray, np.ndarray]) -> np.ndarray:
        if isinstance(x, runtime.NDArray):
            return np.array(x.asnumpy(), dtype=_default_dtype)
        else:
            return np.array(x, dtype=_default_dtype)

    @staticmethod
    def from_expr(expr: relay.Expr, input_names: Set[str], dtype: str = _default_dtype):
        """
        Create a workload from a Relay expression. All free variables become parameters of the
        function. Model parameters will be randomly generated.
        :param expr: Body expression of function
        :param input_names: Set of names of input tensors
        :param dtype: Data type of input tensors.
        :return: Workload object created from this expression.
        """
        # Create module
        free_vars: List[relay.Var] = relay.analysis.free_vars(expr)
        main = relay.Function(free_vars, expr)
        mod = ir.IRModule(functions={'main': main})

        # Generate random parameters
        params: Dict[str, np.ndarray] = dict()
        for v in free_vars:
            if input_names.__contains__(v.name_hint):  # skip input tensors
                continue
            shape: Tuple[int] = v.type_annotation.concrete_shape
            params[v.name_hint] = np.random.rand(*shape)

        return Workload(mod, params, dtype=dtype)

    def build(self, target: str = 'llvm', **config):
        """
        Build workload to run on a certain target platform.
        :param target: The corresponding target.
        :param config: Configurations of building workload.
        """
        with transform.PassContext(opt_level=0, config=config):
            self.executor = relay.create_executor(kind='graph', mod=self.mod,
                                                  ctx=tvm.context(target), target=target)
            self.func = self.executor.evaluate()

    def __call__(self, **inputs) -> np.ndarray:
        """
        Execute workload with given inputs
        :param inputs: Input tensors of workload.
        :return: Computation result in numpy array.
        """
        return self.func(**inputs, **self.params).asnumpy()

    def visualize(self, path: str = ''):
        """
        Visualize computation graph of this workload.
        :param path: Path to save graph visualization.
        """
        graph = Digraph(name=self.name)
        _GraphVizVisitor(graph).visit_function(self.mod['main'])
        graph.view(directory=path)


@relay.transform.function_pass(opt_level=0)
class _AlterDType:
    def __init__(self, tgt_ty: str):
        self.var_mut = _VarDTypeMutator(tgt_ty)

    def transform_function(self, func: relay.Function, _mod: ir.IRModule,
                           _ctx: transform.PassContext) -> relay.Function:
        return self.var_mut.visit(func)

    def __call__(self, mod: ir.IRModule) -> ir.IRModule: ...


class _VarDTypeMutator(relay.ExprMutator):
    def __init__(self, tgt_ty: str):
        super().__init__()
        self.ty_mut = _TensorDTypeMutator(tgt_ty)

    def visit_function(self, fn: relay.Function):
        new_func = super().visit_function(fn)
        return relay.Function(new_func.params, new_func.body)

    def visit_var(self, var: relay.Var):
        new_ty = self.ty_mut.visit(var.type_annotation)
        return relay.Var(name_hint=var.name_hint, type_annotation=new_ty)


class _TensorDTypeMutator(relay.TypeMutator):
    def __init__(self, tgt_ty: str):
        super().__init__()
        self.tgt_ty = tgt_ty

    def visit_tensor_type(self, tt: relay.TensorType):
        return relay.TensorType(tt.concrete_shape, dtype=self.tgt_ty)


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
