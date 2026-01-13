#include "core/math.h"
#include <cmath>
#include <cassert>
#include "core/error.h"

namespace tf {

Tensor matmul(const Tensor& A, const Tensor& B) {
  CHECK(A.cols == B.rows, "matmul mismatch: " << A.shape_str() << " * " << B.shape_str());
  Tensor C(A.rows, B.cols, 0.0f);

  // to optimize the matmul we can transpose B and access it linearly in the inner loop (dot product)
  Tensor Bt = transpose(B);
  // for now we use OpenMP to parallelize the outer loop
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < A.rows; ++i) {
    const size_t a_row_offset = (size_t)i * (size_t)A.cols;
    const float* A_ptr = A.data.data() + a_row_offset;
    
    for (int j = 0; j < B.cols; ++j) {
      const size_t b_row_offset = (size_t)j * (size_t)Bt.cols; // Bt dims: (B.cols x B.rows) -> (cols x A.cols)
      const float* B_ptr = Bt.data.data() + b_row_offset;
      
      float sum = 0.0f;
      for (int k = 0; k < A.cols; ++k) {
        sum += A_ptr[k] * B_ptr[k];
      }
      
      C(i, j) = sum;
    }
  }
  return C;
}

Tensor transpose(const Tensor& A) {
  Tensor T(A.cols, A.rows, 0.0f);
  for (int i = 0; i < A.rows; ++i)
    for (int j = 0; j < A.cols; ++j)
      T(j, i) = A(i, j);
  return T;
}

Tensor add(const Tensor& A, const Tensor& B) {
  CHECK(A.rows == B.rows && A.cols == B.cols, "add mismatch: " << A.shape_str() << " + " << B.shape_str());
  Tensor C(A.rows, A.cols);
  for (size_t i = 0; i < A.size(); ++i) C.data[i] = A.data[i] + B.data[i];
  return C;
}

Tensor sub(const Tensor& A, const Tensor& B) {
  CHECK(A.rows == B.rows && A.cols == B.cols, "sub mismatch: " << A.shape_str() << " - " << B.shape_str());
  Tensor C(A.rows, A.cols);
  for (size_t i = 0; i < A.size(); ++i) C.data[i] = A.data[i] - B.data[i];
  return C;
}

Tensor mul(const Tensor& A, const Tensor& B) {
  CHECK(A.rows == B.rows && A.cols == B.cols, "element-wise mul mismatch: " << A.shape_str() << " * " << B.shape_str());
  Tensor C(A.rows, A.cols);
  for (size_t i = 0; i < A.size(); ++i) C.data[i] = A.data[i] * B.data[i];
  return C;
}

Tensor mul_scalar(const Tensor& A, float s) {
  Tensor C(A.rows, A.cols);
  for (size_t i = 0; i < A.size(); ++i) C.data[i] = A.data[i] * s;
  return C;
}

Tensor add_bias_rowwise(const Tensor& X, const Tensor& b) {
  CHECK(b.rows == 1 && b.cols == X.cols, "add_bias_rowwise mismatch: X=" << X.shape_str() << ", b=" << b.shape_str());
  Tensor Y(X.rows, X.cols);
  for (int i = 0; i < X.rows; ++i) {
    const size_t row = (size_t)i * (size_t)X.cols;
    for (int j = 0; j < X.cols; ++j) {
      Y.data[row + (size_t)j] = X.data[row + (size_t)j] + b(0, j);
    }
  }
  return Y;
}

Tensor sum_rows(const Tensor& X) {
  Tensor s(1, X.cols, 0.0f);
  for (int i = 0; i < X.rows; ++i) {
    const size_t row = (size_t)i * (size_t)X.cols;
    for (int j = 0; j < X.cols; ++j) {
      s(0, j) += X.data[row + (size_t)j];
    }
  }
  return s;
}

Tensor relu(const Tensor& X) {
  Tensor Y(X.rows, X.cols);
  for (size_t i = 0; i < X.size(); ++i) Y.data[i] = (X.data[i] > 0.0f) ? X.data[i] : 0.0f;
  return Y;
}

Tensor relu_backward(const Tensor& X, const Tensor& dY) {
  CHECK(X.rows == dY.rows && X.cols == dY.cols, "relu_backward mismatch: X=" << X.shape_str() << ", dY=" << dY.shape_str());
  Tensor dX(X.rows, X.cols);
  for (size_t i = 0; i < X.size(); ++i) dX.data[i] = (X.data[i] > 0.0f) ? dY.data[i] : 0.0f;
  return dX;
}

static inline float sigmoid_scalar(float x) {
  if (x >= 0.0f) {
    float z = std::exp(-x);
    return 1.0f / (1.0f + z);
  } else {
    float z = std::exp(x);
    return z / (1.0f + z);
  }
}

Tensor sigmoid(const Tensor& X) {
  Tensor Y(X.rows, X.cols);
  for (size_t i = 0; i < X.size(); ++i) Y.data[i] = sigmoid_scalar(X.data[i]);
  return Y;
}

Tensor sigmoid_backward_from_output(const Tensor& sigmoid_out, const Tensor& dY) {
  CHECK(sigmoid_out.rows == dY.rows && sigmoid_out.cols == dY.cols, 
        "sigmoid_backward mismatch: out=" << sigmoid_out.shape_str() << ", dY=" << dY.shape_str());
  Tensor dX(sigmoid_out.rows, sigmoid_out.cols);
  for (size_t i = 0; i < sigmoid_out.size(); ++i) {
    const float s = sigmoid_out.data[i];
    dX.data[i] = dY.data[i] * s * (1.0f - s);
  }
  return dX;
}

float mean(const Tensor& X) {
  float acc = 0.0f;
  for (auto v : X.data) acc += v;
  return acc / (float)X.size();
}

}