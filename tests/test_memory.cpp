#include "utils/test_utils.h"
#include "core/tensor.h"
#include <iostream>

using namespace tf;

void test_move_semantics() {
    // create big tensor
    Tensor a(1000, 1000, 1.0f);
    const float* original_ptr = a.data.data();
    
    // move construct
    Tensor b = std::move(a);
    
    // ptr should be same (no malloc)
    ASSERT_TRUE(b.data.data() == original_ptr);
    ASSERT_EQ(b.rows, 1000);
    
    // old one should be empty/zeroed
    ASSERT_EQ(a.rows, 0);
    ASSERT_EQ(a.cols, 0);
    ASSERT_EQ(a.data.size(), 0);
}

void test_move_assignment() {
    Tensor a(500, 500, 2.0f);
    const float* ptr_a = a.data.data();
    
    Tensor b(10, 10, 0.0f); // small garbage
    
    // move assign
    b = std::move(a);
    
    // b got a's guts
    ASSERT_TRUE(b.data.data() == ptr_a);
    ASSERT_EQ(b(0,0), 2.0f);
    
    // a is dead
    ASSERT_EQ(a.rows, 0);
}
