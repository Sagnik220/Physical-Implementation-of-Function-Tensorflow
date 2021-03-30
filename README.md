# Zfit-CERN-SFT
 Physical Shape Function Implementation using Tensorflow

## Implementation of physical shape function
## Description
zfit is a highly scalable and customizable model manipulation and fitting library. It uses TensorFlow as its computational backend and is optimised for simple and direct manipulation of probability density functions. Purely built in Python, the usage is targeted towards the High Energy Physics analysis ecosystem.

While zfit in the core is focused to provide the basic building blocks, it misses compared to other libraries functions that are especially useful in physics. Since it is built on top of TensorFlow, which has a performance penalty when using normal Python, these functions have to be implemented in zfit.

This project aims at implementing a more difficult lineshape, namely the faddeeva or voigt profile. This can either be done by creating a C++ kernel for CPU as well as GPU or by implementing it using the TensorFlow library.

## Requirements
Python, Numpy, (maybe C++ and CUDA)

## Examples:

## Usage
```python
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import numba

#Problem 1
# This is an example to give you an idea
@tf.function(autograph=False)
def log_abs(x):
    """EXAMPLE IMPLEMENTATION: Return the log of the absolute of x element-wise"""
    return tf.math.log(tf.math.abs(x))
In [4]:
 log_abs(5.0)
 
 Out[4]:
<tf.Tensor: shape=(), dtype=float32, numpy=1.609438>
 
#Problem 2
@tf.function(autograph=False)
def sum_cos_sin(x, coeff_cos, coeff_sin):
    """Return the sum of the cos and sin of x element-wise, cos and sin scaled by coeff_cos and coeff_sin respectively."""
    return coeff_sin * tf.math.sin(x) + coeff_cos * tf.math.cos(x)
In [6]:
sum_cos_sin(5.0,1.0,2.0)

Out[6]:
<tf.Tensor: shape=(), dtype=float32, numpy=-1.6341864>
