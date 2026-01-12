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

add_executable(xor_train examples/xor_train.cpp)
target_link_libraries(xor_train PRIVATE tiny-nn::tiny-nn)