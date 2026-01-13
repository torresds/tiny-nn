#include "utils/test_utils.h"
#include "core/tensor.h"
#include <iostream>

using namespace tf;

void test_move_semantics() {
    
    Tensor a(1000, 1000, 1.0f);
    const float* original_ptr = a.data.data();
    
    
    Tensor b = std::move(a);
    
    
    ASSERT_TRUE(b.data.data() == original_ptr);
    ASSERT_EQ(b.rows, 1000);
    
    
    ASSERT_EQ(a.rows, 0);
    ASSERT_EQ(a.cols, 0);
    ASSERT_EQ(a.data.size(), 0);
}

void test_move_assignment() {
    Tensor a(500, 500, 2.0f);
    const float* ptr_a = a.data.data();
    
    Tensor b(10, 10, 0.0f); 
    
    
    b = std::move(a);
    
    
    ASSERT_TRUE(b.data.data() == ptr_a);
    ASSERT_EQ(b(0,0), 2.0f);
    
    
    ASSERT_EQ(a.rows, 0);
}
