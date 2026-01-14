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

### Model Persistence
- **Binary checkpoint format** for efficient save/load of model parameters
- **Named parameters API** (`Module::named_parameters()`) for parameter enumeration
- **Convenience methods** (`Sequential::save()` and `Sequential::load()`) for checkpoint management
- **Shape validation** and integrity checks during deserialization

### Validation & Testing
- Unit tests for tensor operations and layers
- Numerical gradient checking (finite differences)
- Stability tests for extreme logits
- Checkpoint save/load validation

---

## Examples

The provided examples serve as validation benchmarks for different components of the library.

### XOR Classification (Non-Linear Separability)
A mandatory benchmark for any neural network library. Confirms correct implementation of **backpropagation through hidden layers** and the crucial role of non-linear activation functions.

```bash
./build/xor_train
```

<details>
<summary>Example Output</summary>

```
epoch    1 | loss 0.749525 | acc 0.750
epoch  250 | loss 0.190809 | acc 1.000
epoch  500 | loss 0.061450 | acc 1.000
epoch 1000 | loss 0.018638 | acc 1.000
epoch 2000 | loss 0.006475 | acc 1.000
epoch 5000 | loss 0.001857 | acc 1.000

Final logits:
  x=(0.000,0.000) -> logit=-6.263 target=0.000
  x=(0.000,1.000) -> logit=6.773 target=1.000
  x=(1.000,0.000) -> logit=5.918 target=1.000
  x=(1.000,1.000) -> logit=-6.381 target=0.000
```
</details>

### Multi-class Classification (Gaussian Blobs)
Evaluates the orchestration of high-level abstractions: `Sequential`, `Adam`, and `DataLoader`. Validates the numerical stability of **Softmax Cross-Entropy** and the convergence efficiency of adaptive momentum methods.

```bash
./build/classify_blobs
```

### Checkpoint Save/Load Demo
Demonstrates the **binary checkpoint format** for model persistence. Trains a model, saves parameters to disk, loads them into a fresh model instance, and verifies prediction consistency.

```bash
./build/save_load_demo
```

<details>
<summary>Example Output</summary>

```
--- Save/Load Demo ---
[Step 1] Training Initial Model...
[Step 2] Saving model to 'demo_model.tnn'...
[Step 3] Creating fresh model (random weights)...
[Step 4] Loading checkpoint...
[Step 5] Verifying predictions...
Difference between original and loaded predictions: 0
SUCCESS: Predictions match!
```
</details>

### Train-Save-Load-Resume Workflow
End-to-end workflow demonstrating **training resumption** from checkpoints. Trains a model, saves state, loads into a new instance, and continues training—validating that learned representations persist correctly.

```bash
./build/train_resume_blobs
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

<details>
<summary>Test Results</summary>

```
Running tiny-nn tests...
[PASS] Sanity check
[PASS] Error macro check
[PASS] Tensor creation
[PASS] Matmul simple
[PASS] Transpose
[PASS] Add
[PASS] Dense grad check
[PASS] BCE stability
[PASS] BCE normal
[PASS] MSE simple
[PASS] Softmax sanity
[PASS] Softmax stability
[PASS] Softmax grad check
[PASS] Move semantics
[PASS] Move assignment
[PASS] Grad accumulation
[PASS] Adam simple
[PASS] Make blobs
[PASS] DataLoader batching
[PASS] Save/Load checkpoint
--------------------------------------------------
All 20 tests passed!
```
</details>

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
- [x] **Model Persistence**: Binary checkpoint save/load with `named_parameters()` API

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