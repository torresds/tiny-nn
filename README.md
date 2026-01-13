# tiny-nn

`tiny-nn` is a small neural network implementation developed from scratch in C++ as a hobby project.

Formally, an **artificial neural network** can be defined as a parametric function  

$$
f_\theta : \mathbb{R}^n \rightarrow \mathbb{R}^m
$$

constructed as a composition of affine transformations and non-linear activation functions, where the parameters $(\theta\)$ are learned by minimizing a loss function through gradient-based optimization.

The goal of this project is to study and implement these concepts explicitly at a low level, matrix operations, backpropagation, optimization, and numerical stability, without relying on existing machine learning frameworks or numerical libraries.

## Design principles

- Explicit tensor operations and memory layout  
- Clear separation between mathematical core and neural network components  
- No external ML or numerical libraries  
- Minimal abstractions to keep learning and debugging transparent  

The CPU implementation is treated as a **reference baseline**, prioritizing correctness and clarity before more advanced execution models.

## Implemented features

- 2D tensor with contiguous row-major memory representation
- Core linear algebra operations:
  - Matrix multiplication (MatMul) with cache-friendly access via transpose
  - Transpose
  - Elementwise operations (add, sub, mul) and scalar multiply
  - Row-wise bias addition and row-sum reduction
- Fully connected (Dense) layer:
  - Forward pass: \(Y = XW + b\)
  - Backward pass with explicit gradient accumulation
- Activation functions:
  - ReLU
  - Sigmoid
- Loss functions:
  - Binary Cross-Entropy with logits (numerically stable formulation)
  - Mean Squared Error (MSE) for regression
- Stochastic Gradient Descent (SGD)
- Extensive correctness validation:
  - Unit tests for tensor ops and layers
  - Numerical gradient checking (finite differences)
  - Stability tests for extreme logits
  - Move semantics and gradient accumulation tests

## Examples

### XOR classification (non-linear separability)

The XOR problem is a classical benchmark in neural network literature because it is **not linearly separable**. A single linear model cannot solve XOR; at least one hidden layer with a non-linear activation is required.

This makes XOR a minimal test for:
- correct implementation of non-linear activations,
- correct gradient propagation through hidden layers,
- stable optimization behavior.

```bash
./build/xor_train
````

The model converges to high accuracy, indicating correct forward/backward computation and optimization.

### Linear regression (Dense + MSE)

A linear regression example is provided to validate the simplest case of supervised learning:

$$
y = w_1 x_1 + w_2 x_2 + b
$$


Synthetic data is generated with known parameters and noise. Training should recover the original weights, serving as a sanity check for:

* affine layers,
* MSE loss,
* gradient descent behavior.

```bash
./build/linear_regression
```

## Benchmarks

Basic benchmarks are included to track performance and guide optimization work:

* `bench_matmul`: micro-benchmark for raw matrix multiplication across sizes
* `bench_mlp`: forward/backward-heavy benchmark where MatMul dominates execution time

```bash
./build/bench_matmul
./build/bench_mlp
```

These benchmarks are intended for **relative comparisons across versions**, not absolute performance claims.

## Tests

```bash
./build/tiny_nn_tests
# or:
ctest --test-dir build
```

The test suite emphasizes correctness and numerical stability over performance.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Notes:

* Non-MSVC builds enable aggressive optimizations (`-O3 -march=native -ffast-math`)
* OpenMP is used when available (currently for MatMul parallelization)

## Scope and limitations

This project is **not** intended to compete with established machine learning libraries.

Instead, it serves as:

* a learning and experimentation tool,
* a correctness reference for low-level NN components,
* a foundation for studying optimization strategies and execution models.

## Planned features / future work

* Softmax + CrossEntropy
* Sequential container for model composition
* Additional activation functions (Tanh)
* Mini-batch and data-loading utilities
* More detailed benchmarking and profiling
* Further CPU optimizations (tiling, cache blocking, packing)
* Optional SIMD kernels
* CUDA-accelerated backend