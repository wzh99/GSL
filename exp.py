from typing import Tuple, Set, Dict, Optional

from tvm import relay, transform, ir
from tvm.relay import dataflow_pattern as dfp

from gsl import *


class SubstTest(dfp.DFPatternCallback):
    def __init__(self, viz_orig=False, viz_pass=False, viz_dfp=False, viz_gsl=False):
        super().__init__()
        self.pattern = self.define_dfp()
        self.viz_orig = viz_orig
        self.viz_pass = viz_pass
        self.viz_dfp = viz_dfp
        self.viz_gsl = viz_gsl

    def create_expr(self) -> Tuple[relay.Expr, Set[str]]:
        pass

    def get_pass(self) -> transform.Pass:
        pass

    def define_gsl(self) -> Optional[Subst]:
        pass

    def define_dfp(self) -> Optional[dfp.DFPattern]:
        pass

    def rewrite_dfp(self, node_map: Dict[dfp.DFPattern, relay.Expr]) -> relay.Expr:
        pass

    def callback(self, pre: relay.Expr, post: relay.Expr, node_map: ir.Map) -> relay.Expr:
        return self.rewrite_dfp(dict([(p, l[0]) for p, l in node_map.items()]))

    def run(self):
        # Create workload
        body, input_names = self.create_expr()
        wl = Workload.from_expr(body, input_names)
        print('Original module:')
        print(wl.mod)
        if self.viz_orig:
            wl.visualize()

        # Apply built-in pass
        pass_mod = self.get_pass()(wl.mod)
        print('After built-in pass:')
        print(pass_mod)
        if self.viz_pass:
            Workload(pass_mod, {}).visualize()

        # Use Relay DFP for substitution
        if self.pattern is not None:
            dfp_body = self.rewrite(body)
            dfp_mod = ir.IRModule(functions={
                'main': relay.Function(relay.analysis.free_vars(dfp_body),
                                       dfp_body)
            })
            print('After DFP rewrite:')
            print(dfp_mod)
            if self.viz_dfp:
                Workload(dfp_mod, {}).visualize()

        # Use GSL for substitution
        rule = self.define_gsl()
        if rule is not None:
            gsl_wl = rule(wl, fast_mode=True, fold_params=False)
            print('After GSL substitution:')
            print(gsl_wl.mod)
            if self.viz_gsl:
                gsl_wl.visualize()


class SimplifyBatchNorm(SubstTest):

    def create_expr(self) -> Tuple[relay.Expr, Set[str]]:
        x = relay.var('x', shape=(2, 2, 4, 4))
        gamma = relay.var('gamma', shape=(2,))
        beta = relay.var('beta', shape=(2,))
        mean = relay.var('mean', shape=(2,))
        var = relay.var('var', shape=(2,))
        bn = relay.nn.batch_norm(x, gamma, beta, mean, var)
        return bn[0], {'x'}

    def get_pass(self) -> transform.Pass:
        return relay.transform.SimplifyInference()

    def define_gsl(self) -> Optional[Subst]:
        # Input
        x = pat.Wildcard()
        gamma = pat.Variable()
        beta = pat.Variable()
        mean = pat.Variable()
        var = pat.Variable()

        # Source pattern: batch_norm(x, gamma, beta, mean, var)
        bn = op.BatchNorm(x, gamma, beta, mean, var)
        y1 = bn[0]

        # Target pattern: k = gamma / sqrt(var + epsilon), x * k + beta - mean * k
        std = op.Sqrt(var + bn.epsilon)
        k = pat.Cond(bn.scale, gamma / std, 1.0 / std)
        bias = pat.Cond(bn.center, beta - mean * k, -mean * k)
        y2 = op.BiasAdd(x * op.ExpandDims(k, axis=1, num_newaxis=x.ndim - 1 - bn.axis), bias)

        # Build substitution
        return Subst(y1, y2)


if __name__ == '__main__':
    for case in [
        SimplifyBatchNorm(),
    ]:
        case.run()
