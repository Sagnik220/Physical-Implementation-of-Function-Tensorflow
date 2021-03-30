# Zfit-CERN-SFT
 Physical Shape Function Implementation using Tensorflow

## Implementation of physical shape function
## Description
zfit is a highly scalable and customizable model manipulation and fitting library. It uses TensorFlow as its computational backend and is optimised for simple and direct manipulation of probability density functions. Purely built in Python, the usage is targeted towards the High Energy Physics analysis ecosystem.

While zfit in the core is focused to provide the basic building blocks, it misses compared to other libraries functions that are especially useful in physics. Since it is built on top of TensorFlow, which has a performance penalty when using normal Python, these functions have to be implemented in zfit.

This project aims at implementing a more difficult lineshape, namely the faddeeva or voigt profile. This can either be done by creating a C++ kernel for CPU as well as GPU or by implementing it using the TensorFlow library.

## Requirements
Python, Numpy, (maybe C++ and CUDA)
