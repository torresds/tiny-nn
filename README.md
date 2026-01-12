# tiny-nn

`tiny-nn` is a small neural network implementation developed from scratch in C++ as a hobby project.

The main goal of this project is to study and implement the fundamental building blocks of neural networks at a low level, including matrix operations, backpropagation, gradient-based optimization, and numerical stability, without relying on existing machine learning frameworks.

## Design principles

- Explicit tensor operations and memory layout  
- Clear separation between mathematical core and neural network components  
- No external ML or numerical libraries  
- Minimal abstractions to keep the learning process transparent  

This project treats the CPU implementation as a reference baseline before introducing more advanced execution models.

## Implemented features

- 2D tensor with contiguous memory representation
- Core matrix operations (matrix multiplication, transpose, elementwise ops)
- Fully connected (Dense) layers
- ReLU and Sigmoid activation functions
- Binary Cross-Entropy loss with logits (numerically stable formulation)
- Stochastic Gradient Descent (SGD)
- End-to-end training loop with forward and backward passes

## Example

A minimal XOR classification example is provided to validate the implementation of non-linear learning and backpropagation.

```bash
./build/xor_train
````

The model converges to perfect accuracy, indicating correct gradient propagation and stable optimization behavior.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Scope and limitations

This project is not intended to compete with established machine learning libraries.

Instead, it serves as:

* a learning tool,
* a correctness reference, and
* a foundation for experimentation with optimization and execution models.

Performance optimizations are deliberately secondary at this stage.

## Planned features / future work

* Sequential container to simplify model construction
* Mean Squared Error (MSE) loss for regression tasks
* Additional activation functions (Tanh)
* Mini-batch training support
* Numerical gradient checking for validation
* Basic benchmarking utilities
* CPU optimizations (tiling, cache-friendly matmul)
* CUDA-accelerated backend