from typing import Dict, Union, List, Tuple, Optional, Set, Any

import numpy as np
import tvm
from graphviz import Digraph
from tvm import ir, runtime, relay, transform
from .util import default_font_name

default_dtype = 'float32'


class Workload:
    """
    A workload contains computation graph of a model and its parameters.
    """

    def __init__(self, mod: ir.IRModule, params: Dict[str, Union[runtime.NDArray, np.ndarray]],
                 dtype: str = default_dtype, name: str = ''):
        """
        Constructor.

        :param mod: Relay IR module defining computation graph of the model.
        :param params: Mapping from parameter names to values. Internally, the values are stored
            in `np.ndarray`s. NDArray values will be converted to that type.
        :param dtype: Data type of new workload.
        :param name: Name of the new workload.
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
            return np.array(x.asnumpy(), dtype=default_dtype)
        else:
            return np.array(x, dtype=default_dtype)

    @staticmethod
    def from_expr(expr: relay.Expr, input_names: Set[str], dtype: str = default_dtype,
                  name: str = ''):
        """
        Create a workload from a Relay expression. All free variables become parameters of the
        function. Model parameters will be randomly generated.

        :param expr: Body expression of function
        :param input_names: Set of names of input tensors
        :param dtype: Data type of input tensors.
        :param name: Name of the model.
        :return: Workload object created from this expression.
        """
        # Create module
        free_vars: List[relay.Var] = relay.analysis.free_vars(expr)
        main = relay.Function(free_vars, expr)
        mod = ir.IRModule(functions={'main': main})

        # Generate random parameters
        params: Dict[str, np.ndarray] = dict()
        for v in free_vars:
            if v.name_hint in input_names:  # skip input tensors
                continue
            shape: Tuple[int] = v.type_annotation.concrete_shape
            params[v.name_hint] = np.random.rand(*shape)

        return Workload(mod, params, dtype=dtype, name=name)

    @staticmethod
    def from_keras(model, shape: Dict[str, Tuple[int, ...]], dtype: str = default_dtype):
        """
        Build workload from a Keras model.

        :param model: The Keras model to be converted.
        :param shape: Specification of input shapes.
        :param dtype: Data type of target workload.
        :return: The built workload object.
        """
        mod, params = relay.frontend.from_keras(model, shape=shape)
        return Workload(mod, params, dtype=dtype, name=model.name)

    def build(self, target: str = 'llvm', config: Dict[str, Any] = None):
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
        Execute workload with given inputs.
        Note that the workload can only be called after it is built.

        :param inputs: Input tensors of workload.
        :return: Computation result in numpy array.
        """
        if self.func is None:
            raise RuntimeError('The workload has not been built yet.')
        return self.func(**inputs, **self.params).asnumpy()

    def visualize(self, path: str = 'out', font_name: str = default_font_name, **attrs):
        """
        Visualize computation graph of this workload.

        :param path: Path to save graph visualization.
        :param font_name: Name of the font used to display node texts.
        :param attrs: Other attributes for GraphViz to plot the nodes.
        """
        graph = Digraph(name=self.name)
        viz = _ExprVisualizer(graph, fontname=font_name, **attrs)
        viz.visit_function(self.mod['main'])
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


class _ExprVisualizer(relay.ExprVisitor):
    def __init__(self, graph: Digraph, **attrs):
        super().__init__()
        self.graph = graph
        self.attrs = attrs
        self.counter = 0

    def visit(self, expr):
        if expr in self.memo_map:
            return self.memo_map[expr]
        return super().visit(expr)

    def visit_var(self, var: relay.Var):
        expr_id = self._next_id()
        self.graph.node(expr_id, label='%' + var.name_hint, **self.attrs)
        return expr_id

    def visit_constant(self, const: relay.Constant):
        expr_id = self._next_id()
        self.graph.node(expr_id, label='const', **self.attrs)
        return expr_id

    def visit_call(self, call: relay.Call):
        expr_id = self._next_id()
        self.graph.node(expr_id, label=call.op.name, **self.attrs)
        for a in call.args:
            self.graph.edge(self.visit(a), expr_id)
        return expr_id

    def visit_tuple(self, tup: relay.Tuple):
        expr_id = self._next_id()
        self.graph.node(expr_id, label='(,)', **self.attrs)
        for f in tup.fields:
            self.graph.edge(self.visit(f), expr_id)
        return expr_id

    def visit_tuple_getitem(self, getitem: relay.TupleGetItem):
        expr_id = self._next_id()
        self.graph.node(expr_id, label='.{}'.format(getitem.index), **self.attrs)
        self.graph.edge(self.visit(getitem.tuple_value), expr_id)
        return expr_id

    def _next_id(self) -> str:
        cur_id = str(self.counter)
        self.counter += 1
        return cur_id
