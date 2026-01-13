cmake_minimum_required(VERSION 3.20)
project(tiny-nn LANGUAGES CXX)

add_library(tiny-nn
  src/core/tensor.cpp
  src/core/rng.cpp
  src/core/math.cpp
  src/nn/dense.cpp
  src/nn/activations.cpp
  src/nn/losses.cpp
  src/optim/sgd.cpp
  src/optim/adam.cpp
  src/data/dataloader.cpp
  src/data/toy_datasets.cpp
  src/nn/sequential.cpp
)

add_library(tiny-nn::tiny-nn ALIAS tiny-nn)

target_compile_features(tiny-nn PUBLIC cxx_std_17)

target_include_directories(tiny-nn PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

option(TINY_NN_ENABLE_WARNINGS "Enable warnings" ON)

if (TINY_NN_ENABLE_WARNINGS)
  if (MSVC)
    target_compile_options(tiny-nn PRIVATE /W4 /permissive-)
  else()
    target_compile_options(tiny-nn PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endif()

# Optimizations
if(NOT MSVC)
  target_compile_options(tiny-nn PRIVATE -O3 -march=native -ffast-math)
endif()

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(tiny-nn PUBLIC OpenMP::OpenMP_CXX)
endif()

# Examples
add_executable(xor_train examples/xor_train.cpp)
target_link_libraries(xor_train PRIVATE tiny-nn::tiny-nn)
add_executable(linear_regression examples/linear_regression.cpp)
target_link_libraries(linear_regression PRIVATE tiny-nn::tiny-nn)

add_executable(classify_blobs examples/classify_blobs.cpp)
target_link_libraries(classify_blobs PRIVATE tiny-nn::tiny-nn)

# Tests
enable_testing()
add_executable(tiny_nn_tests 
    tests/run_tests.cpp
    tests/test_core.cpp
    tests/test_grad.cpp
    tests/test_losses.cpp
    tests/test_memory.cpp
    tests/test_accum.cpp
    tests/test_optim.cpp
    tests/test_data.cpp
)
target_include_directories(tiny_nn_tests PRIVATE tests)
target_link_libraries(tiny_nn_tests PRIVATE tiny-nn::tiny-nn)

add_test(NAME AllTests COMMAND tiny_nn_tests)

# Benchmarks
add_executable(bench_matmul benchmarks/bench_matmul.cpp)
target_include_directories(bench_matmul PRIVATE benchmarks)
target_link_libraries(bench_matmul PRIVATE tiny-nn::tiny-nn)

add_executable(bench_mlp benchmarks/bench_mlp.cpp)
target_include_directories(bench_mlp PRIVATE benchmarks)
target_link_libraries(bench_mlp PRIVATE tiny-nn::tiny-nn)