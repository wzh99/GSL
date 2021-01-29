# GSL

GSL (pronounced Gie-sel) is a declarative domain-specific language, embedded in Python, for defining graph substitution rules in deep learning compilers. 

## Introduction

Deep learning compilers perform graph substitutions on computation graphs of models. Common graph substitution passes follow a match-capture-rewrite process, which may lead to repetition of boilerplate code. GSL helps users focuses on essence of graph substitution - structure of source and target patterns, as well as constraints imposed on them, thus reducing repetitive work. 

Current implementation of GSL is based on [TVM Relay IR](https://tvm.apache.org/docs/dev/relay_intro.html) and can be used as an alternative to its [pattern language](https://tvm.apache.org/docs/langref/relay_pattern.html#pattern-language-design). It can be ported to other graph-level IRs because of their similarity. 

## Feature

* **Declarative**. The language makes users focus on essence of graph substitutions. They don't need to care about details of substitution algorithm. 
* **Simple**. The language makes full use of Python features to make it simple and concise.  Substitution rules can be defined and applied to the graph at any time, with only a few lines of code. 
* **Expressive**. The language support patterns with any number of output nodes. Complex constraints on operator attributes could be specified using attribute expressions. 

## Dependency

* tvm>=0.7
* numpy
* graphviz (if graph visualization feature is needed)

## Language

Here we use a simple case, which has only one output node, to demonstrate this language. Consider the diamond-shaped convolution and addition pattern. In the source graph, an input is parallelly convolved with two kernels with identical shape, and the result is added together. This is equivalent to convolving the input with element-wise addition of two kernels. In simplest form, rule may look like this: 

```python
conv2d(x, w1) + conv2d(x, w2) = conv2d(x, w1 + w2)
```

Here we use GSL to define this rule, which looks like this: 

```python
from gsl import *

# Input
x = Wildcard()
w1 = Var()
w2 = Var(shape=w1.shape)

# Source pattern: conv2d(x, w1) + conv2d(x, w2)
conv1 = Conv2D(x, w1)
conv2 = Conv2D(x, w2, strides=conv1.strides, padding=conv1.padding, 
               dilation=conv1.dilation, groups=conv1.groups)
y1 = conv1 + conv2

# Target pattern: conv2d(x, w1 + w2)
y2 = Conv2D(x, w1 + w2, 
            **same_attr(conv1, ['strides', 'padding', 'dilation', 'groups']))

# Build substitution
subst = Substitution(y1, y2)
```

At the very beginning, we import all the necessary classes and functions from package `gsl`. 

First, we define all the input nodes, including wildcards (which matches any expression) and variables. The second kernel must have identical shape with the first, so we provide a keyword argument as an attribute constraint. The `shape` keyword on the left hand side is an attribute of variable `w2`. The right hand side is an attribute expression which gets attribute `shape` of `w1`.

Second, we define source pattern graph. For the first convolution,  it has input nodes `x` and `w1`, so we just need to define `conv1 = Conv2D(x, w1)`. For the second convolution, it has input nodes `x` and `w2`, and should have identical strides, padding, dilation and groups with `conv1`, so we define `conv2 = Conv2D(x, w2, strides=conv1.strides, padding=conv1.padding, dilation=conv1.dilation, groups=conv1.groups)`. The output is addition of the results of two convolutions, so we define `y1 = conv1 + conv2`. 

Then the target pattern, which is a single convolution. The kernel is addition of `w1` and `w2`. The target convolution must also have identical strides, padding, dilation and groups with either of the convolutions in source pattern. Instead of writing the four equations which share the same form `p=a.p`, we provide `same_attr` shorthand to specify that the four attributes of `y2` are identical with the ones of `conv1`. 

Finally, we build substitution with source and target patterns. The code is very straightforward. The `Substitution` class will perform semantics checking on the pattern. After checking, it can be applied to deep learning workloads. 

## Executor

The pattern carries essential information for graph substitution. The executor then used this information to actually perform substitution on real computation graphs. The executor uses a bidirectional matching algorithm to handle patterns with multiple nodes, and falls back to a simpler algorithm for single node. Here we just shows how the rule above applies to a Relay workload. 

We define a simple test case with Relay API, then create a `Workload` object. TVM provides module (which describes a computation graph) and parameter dictionary separately. `Workload` just combines these two things. Since we don't have trained parameters for this workload, we let static method `from_expr` create random parameters for us.  Set `{'x'}` indicates variable `x` is input of the model and should not be included in parameter dictionary. 

```python
from tvm import relay

x = relay.var('x', shape=(2, 2, 4, 4))
w1 = relay.var('w1', shape=(4, 2, 1, 1))
w2 = relay.var('w2', shape=(4, 2, 1, 1))
conv1 = relay.nn.conv2d(x, w1)
conv2 = relay.nn.conv2d(x, w2)
y = conv1 + conv2
wl = Workload.from_expr(y, {'x'})
print(wl.mod)
```

```
def @main(%x: Tensor[(2, 2, 4, 4), float32], %w1: Tensor[(4, 2, 1, 1), float32], %w2: Tensor[(4, 2, 1, 1), float32]) -> Tensor[(2, 4, 4, 4), float32] {
  %0 = nn.conv2d(%x, %w1, padding=[0, 0, 0, 0]) /* ty=Tensor[(2, 4, 4, 4), float32] */;
  %1 = nn.conv2d(%x, %w2, padding=[0, 0, 0, 0]) /* ty=Tensor[(2, 4, 4, 4), float32] */;
  add(%0, %1) /* ty=Tensor[(2, 4, 4, 4), float32] */
}
```

Code for application of substitution is quite simple, just one line. Then we can print the module after substitution.

```python
new_wl = subst(wl)
print(new_wl.mod)
```

```
def @main(%x: Tensor[(2, 2, 4, 4), float32], %v_param_1: Tensor[(4, 2, 1, 1), float32]) -> Tensor[(2, 4, 4, 4), float32] {
  nn.conv2d(%x, %v_param_1, padding=[0, 0, 0, 0]) /* ty=Tensor[(2, 4, 4, 4), float32] */
}
```

We can see that the two convolutions are fused to one, a new kernel is created and the original two are removed. We can run the workload on same input to test its correctness.

```python
import numpy as np

x = np.random.rand(2, 2, 4, 4)
wl.build()
y1 = wl(x=x)
new_wl.build()
y2 = new_wl(x=x)
print(np.max(np.abs(y2 - y1)))
```

```
2.3841858e-07
```

The floating point computational difference is very small, this means the substitution is correct. 

