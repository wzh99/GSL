# GSL

GSL (pronounced Gie-sel, acronym for Graph Substitution Language) is a declarative domain-specific language, embedded in Python, for defining graph substitution rules in deep learning compilers. 

## Introduction

Deep learning compilers perform graph substitutions on computation graphs of models. Common graph substitution passes follow a match-capture-rewrite process, which may lead to repetition of boilerplate code. GSL helps users focuses on essence of graph substitution - structure of source and target patterns, as well as constraints imposed on them, thus reducing repetitive work. 

Current implementation of GSL is based on [TVM Relay IR](https://tvm.apache.org/docs/dev/relay_intro.html) and can be used as an alternative to its [pattern language](https://tvm.apache.org/docs/langref/relay_pattern.html#pattern-language-design). It can be ported to other graph-level IRs because of their similarity. 

## Features

* **Declarative**. The language makes users focus on essence of graph substitutions. They don't need to care about details of substitution algorithm. 
* **Simple**. The language makes full use of Python features to make it simple and concise.  Substitution rules can be defined and applied to the graph with only a few lines of code. 
* **Expressive**. The language support patterns with multiple, and even variadic output nodes. Complex constraints on operator attributes could be specified using attribute expressions. 

## Dependencies

* tvm>=0.7
* numpy
* graphviz (if graph visualization feature is needed)

## Usage

Here we use a simple case to demonstrate this language. Consider the diamond-shaped convolution and addition pattern, which can be fused to a single convolution. In simplest form, rule may look like this: 

```python
conv2d(x, w1) + conv2d(x, w2) = conv2d(x, w1 + w2)
```

Write code in GSL to define this rule: 

```python
from gsl import *

# Input
x = pat.Wildcard()
w1 = pat.Var()
w2 = pat.Var(shape=w1.shape)

# Source pattern: conv2d(x, w1) + conv2d(x, w2)
conv1 = op.Conv2D(x, w1)
conv2 = op.Conv2D(x, w2, strides=conv1.strides, padding=conv1.padding,
                  dilation=conv1.dilation, groups=conv1.groups)
y1 = conv1 + conv2

# Target pattern: conv2d(x, w1 + w2)
y2 = op.Conv2D(x, w1 + w2, strides=conv1.strides, padding=conv1.padding,
               dilation=conv1.dilation, groups=conv1.groups)

# Build substitution
subst = Subst(y1, y2)
```

Then we define a simple test case with Relay API, and create a `Workload` object. 

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

Apply the substitution to the workload. 

```python
new_wl = subst(wl)
print(new_wl.mod)
```

```
def @main(%x: Tensor[(2, 2, 4, 4), float32], %v_param_1: Tensor[(4, 2, 1, 1), float32]) -> Tensor[(2, 4, 4, 4), float32] {
  nn.conv2d(%x, %v_param_1, padding=[0, 0, 0, 0]) /* ty=Tensor[(2, 4, 4, 4), float32] */
}
```

We can see that the two convolutions are fused to one, that a new kernel is created and that the original two are removed. We can run the workload on same input to test its correctness.

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

The floating point computational difference is very small, which means the substitution is correct.

## Examples

GSL supports source and target patterns with multiple output nodes. The following substitution fuses two parallel convolutions, with same number of output channels, into one. 

```python
# Input
x = pat.Wildcard()
w1 = pat.Var()
w2 = pat.Var(shape=w1.shape)

# Source pattern
conv1 = op.Conv2D(x, w1, groups=1)
conv2 = op.Conv2D(x, w2, **pat.same_attr(conv1, ['strides', 'padding', 'dilation', 'groups']))

# Target pattern
w = op.Concatenate((w1, w2), axis=0)
conv = op.Conv2D(x, w, **pat.same_attr(conv1, ['strides', 'padding', 'dilation', 'groups']))
split = op.Split(conv, indices_or_sections=2, axis=1)

# Build substitution
return Subst([conv1, conv2], [split[0], split[1]])
```

GSL can also support variadic pattern, including variadic tuple fields and variadic output nodes. The following substitution removes a split and a following concatenate operation along the same axis. 

```python
# Input
x = pat.Wildcard()

# Source pattern: concat(split(x, axis=a), axis=a)
split = op.Split(x)
i = attr.Symbol()
item = split[i]
y1 = op.Concatenate(pat.Variadic(item, templates=[item], index=i), axis=split.axis)

# Target pattern: x
y2 = x

# Build substitution
subst = Subst(y1, y2)
```

The following is a variadic version of fusing parallel convolutions. This substitution also allows number of output channels to be different. 

```python
# Input
x = pat.Wildcard()
w1 = pat.Var()
w = pat.Var(shape=(None, None, w1.shape[2], w1.shape[3]))

# Source pattern
conv1 = op.Conv2D(x, w1, groups=1)
conv = op.Conv2D(x, w, **pat.same_attr(conv1, ['strides', 'padding', 'dilation', 'groups']))
src = pat.Variadic(conv, templates=[conv, w], first=[conv1, w1], min_len=2)

# Target pattern
i = attr.Symbol()
w_inst = src(i, w)
concat = op.Concatenate(pat.Variadic(w_inst, templates=[w_inst], index=i, length=src.length),
                        axis=0)
conv = op.Conv2D(x, concat,
                 **pat.same_attr(conv1, ['strides', 'padding', 'dilation', 'groups']))

i = attr.Symbol()
j = attr.Symbol()
split = op.Split(conv, axis=1, indices_or_sections=attr.Variadic(
    attr.Sum(src(j, w).shape[0], j, i + 1), index=i, length=src.length - 1))
i = attr.Symbol()
item = split[i]
tgt = pat.Variadic(item, templates=[item], index=i)

# Build substitution
subst = Subst(src, tgt)
```

For more examples of GSL, see [rule.py](rule.py). 
