# tiny-nn

<div align="center">

![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)
![CMake](https://img.shields.io/badge/CMake-3.20+-green.svg)
![License](https://img.shields.io/badge/license-MIT-orange.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

*A minimalist neural network implementation from scratch in C++*

</div>

---

Formally, an **artificial neural network** can be defined as a parametric function:

$$
f_\theta : \mathbb{R}^n \rightarrow \mathbb{R}^m
$$

constructed as a composition of affine transformations and non-linear activation functions, where the parameters $(\theta)$ are learned by minimizing a loss function through gradient-based optimization.

**Goal**: Study and implement these concepts explicitly at a low level—matrix operations, backpropagation, optimization, and numerical stability, without relying on existing machine learning frameworks or numerical libraries.

---

## Design Principles

- **Explicit tensor operations** and memory layout  
- **Clear separation** between mathematical core and neural network components  
- **No external ML or numerical libraries**  
- **Minimal abstractions** to keep learning and debugging transparent  

The CPU implementation is treated as a **reference baseline**, prioritizing correctness and clarity before more advanced execution models.

---

## Implemented Features

### Core Math
- 2D tensor with contiguous row-major memory representation
- Matrix multiplication (MatMul) with cache-friendly access patterns
- Transpose and elementwise operations (add, sub, mul, div)
- Broadcasted bias addition and row-sum reductions

### Neural Network Components
- **Layers**: Fully connected (`Dense`) with explicit gradient accumulation
- **Containers**: `Sequential` for modular model composition
- **Activations**: `ReLU`, `Sigmoid`

### Optimization & Loss Functions
- **Optimizers**: `SGD`, `Adam` (with momentum and bias correction)
- **Losses**: 
  - Binary Cross-Entropy with logits
  - Softmax Cross-Entropy with logits
  - Mean Squared Error (MSE)

### Data Engineering
- `Dataset` and `DataLoader` abstractions for batching and shuffling

### Validation & Testing
- Unit tests for tensor operations and layers
- Numerical gradient checking (finite differences)
- Stability tests for extreme logits

---

## Examples

The provided examples serve as validation benchmarks for different components of the library.

### Multi-class Classification (Gaussian Blobs)
Evaluates the orchestration of high-level abstractions: `Sequential`, `Adam`, and `DataLoader`. Validates the numerical stability of **Softmax Cross-Entropy** and the convergence efficiency of adaptive momentum methods.

```bash
./build/classify_blobs
```

### XOR Classification (Non-Linear Separability)
A mandatory benchmark for any neural network library. Confirms correct implementation of **backpropagation through hidden layers** and the crucial role of non-linear activation functions.

```bash
./build/xor_train
```

### Linear Regression
Baseline sanity check for the most primitive supervised learning components. Verifies that **Dense layer** transformations and **MSE loss** are mathematically correct.

```bash
./build/linear_regression
```

---

## Benchmarks

Performance tracking via micro-benchmarks:

| Benchmark      | Description                                      |
| -------------- | ------------------------------------------------ |
| `bench_matmul` | Raw matrix multiplication across varying sizes   |
| `bench_mlp`    | Forward/backward pass latency (MatMul-dominated) |

```bash
./build/bench_matmul
./build/bench_mlp
```

---

## Tests

```bash
ctest --test-dir build
# or:
./build/tiny_nn_tests
```

---

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

**Notes:**
- Non-MSVC builds enable aggressive optimizations (`-O3 -march=native -ffast-math`)
- OpenMP is used when available for MatMul parallelization

---

## Project Roadmap

### Completed
- [x] **Math**: Contiguous 2D Tensor, MatMul, Transpose
- [x] **Modules**: `Dense` layer, `Sequential` container
- [x] **Optimization**: `SGD` and `Adam` (with momentum)
- [x] **Losses**: MSE, Binary Cross-Entropy, Softmax Cross-Entropy
- [x] **Data Engineering**: `Dataset` and `DataLoader` abstractions

### Planned
- [ ] **Initialization**: He and Xavier/Glorot weight initialization schemes
- [ ] **Activations**: `Tanh` and `LeakyReLU`
- [ ] **Performance**: CPU optimizations (tiling, cache-blocking)
- [ ] **Linear Algebra**: Optional SIMD kernels (AVX/NEON)
- [ ] **Hardware Acceleration**: Preliminary CUDA backend support

---

<div align="center">

**Made with ❤️ and C++**

</div>