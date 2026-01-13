#include "utils/test_utils.h"
#include "core/tensor.h"
#include "core/math.h"
#include "core/error.h"

using namespace tf;

void test_tensor_creation() {
    Tensor t(2, 3, 1.5f);
    ASSERT_EQ(t.rows, 2);
    ASSERT_EQ(t.cols, 3);
    ASSERT_EQ(t.size(), 6);
    
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            ASSERT_EQ(t(i, j), 1.5f);
        }
    }

    Tensor z = Tensor::zeros(2, 2);
    ASSERT_EQ(z(0, 0), 0.0f);
    
    Tensor o = Tensor::ones(2, 2);
    ASSERT_EQ(o(0, 0), 1.0f);
}

void test_matmul_simple() {
    
    Tensor A(2, 3);
    
    
    A(0, 0) = 1; A(0, 1) = 2; A(0, 2) = 3;
    A(1, 0) = 4; A(1, 1) = 5; A(1, 2) = 6;

    Tensor B(3, 2);
    
    
    
    B(0, 0) = 7; B(0, 1) = 8;
    B(1, 0) = 9; B(1, 1) = 1;
    B(2, 0) = 2; B(2, 1) = 3;

    Tensor C = matmul(A, B);
    
    ASSERT_EQ(C.rows, 2);
    ASSERT_EQ(C.cols, 2);

    
    
    
    

    ASSERT_EQ(C(0, 0), 31.0f);
    ASSERT_EQ(C(0, 1), 19.0f);
    ASSERT_EQ(C(1, 0), 85.0f);
    ASSERT_EQ(C(1, 1), 55.0f);
}

void test_transpose() {
    Tensor A(2, 3);
    A(0, 0) = 1; A(0, 1) = 2; A(0, 2) = 3;
    A(1, 0) = 4; A(1, 1) = 5; A(1, 2) = 6;

    Tensor T = transpose(A);
    ASSERT_EQ(T.rows, 3);
    ASSERT_EQ(T.cols, 2);
    
    ASSERT_EQ(T(0, 0), 1);
    ASSERT_EQ(T(1, 0), 2);
    ASSERT_EQ(T(2, 0), 3);
    ASSERT_EQ(T(0, 1), 4);
    ASSERT_EQ(T(1, 1), 5);
    ASSERT_EQ(T(2, 1), 6);
}

void test_add() {
    Tensor A = Tensor::ones(2, 2);
    Tensor B = Tensor::ones(2, 2);
    Tensor C = add(A, B);
    ASSERT_EQ(C(0, 0), 2.0f);
}
