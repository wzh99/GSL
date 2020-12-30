from tvm.relay import *

# Specify number of input tensors of Relay API.
num_input = {
    # Basic math functions
    negative: 1,
    abs: 1,
    exp: 1,
    sqrt: 1,
    add: 2,
    subtract: 2,
    multiply: 2,
    divide: 2,

    # Tensor transformations
    concatenate: 1,
    split: 1,
    reshape: 1,
    transpose: 1,

    # Neural network operators
    nn.conv2d: 2,
    nn.batch_norm: 5,
    nn.max_pool2d: 1,
    nn.relu: 1,
    nn.dense: 2,
    nn.batch_matmul: 2,
}
