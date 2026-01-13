#pragma once
#include <chrono>
#include <iostream>
#include <string>

namespace tf {
namespace bench {

class Timer {
public:
    Timer(const std::string& name) : name_(name) {
        start_ = std::chrono::high_resolution_clock::now();
    }

    ~Timer() {
        stop();
    }

    void stop() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_).count();
        std::cout << "[BENCH] " << name_ << ": " << duration << " ms" << std::endl;
    }

private:
    std::string name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

}
}
