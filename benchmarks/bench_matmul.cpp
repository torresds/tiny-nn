#include "utils/timer.h"
#include "core/tensor.h"
#include "core/math.h"
#include <vector>
#include <iostream>

using namespace tf;

void bench_matmul(int size) {
    Tensor A(size, size, 1.0f);
    Tensor B(size, size, 1.0f);
    
    std::string name = "MatMul " + std::to_string(size) + "x" + std::to_string(size);
    {
        bench::Timer t(name);
        Tensor C = matmul(A, B);
        if(C(0,0) == -123123) std::cout << "impossible";
    }
}

int main() {
    std::cout << "--- running matmul benchmarks ---" << std::endl;
    
    bench_matmul(64);
    
    bench_matmul(128);
    bench_matmul(256);
    bench_matmul(512);
    
    bench_matmul(1024); 
    
    return 0;
}
