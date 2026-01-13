#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <functional>

namespace tf {
namespace test {

inline int tests_run = 0;
inline int tests_passed = 0;

#define ANSI_RED "\033[31m"
#define ANSI_GREEN "\033[32m"
#define ANSI_RESET "\033[0m"

inline void run_test(const std::string& name, std::function<void()> func) {
    tests_run++;
    try {
        func();
        std::cout << ANSI_GREEN << "[PASS] " << ANSI_RESET << name << std::endl;
        tests_passed++;
    } catch (const std::exception& e) {
        std::cout << ANSI_RED << "[FAIL] " << ANSI_RESET << name << " - " << e.what() << std::endl;
    } catch (...) {
        std::cout << ANSI_RED << "[FAIL] " << ANSI_RESET << name << " - Unknown error" << std::endl;
    }
}

inline void print_summary() {
    std::cout << "--------------------------------------------------" << std::endl;
    if (tests_passed == tests_run) {
        std::cout << ANSI_GREEN << "All " << tests_run << " tests passed!" << ANSI_RESET << std::endl;
    } else {
        std::cout << ANSI_RED << tests_passed << "/" << tests_run << " tests passed." << ANSI_RESET << std::endl;
    }
}

#define ASSERT_TRUE(cond) do { \
    if (!(cond)) throw std::runtime_error("Assertion failed: " #cond); \
} while(0)

#define ASSERT_EQ(a, b) do { \
    if ((a) != (b)) throw std::runtime_error("Assertion failed: " #a " == " #b); \
} while(0)

#define ASSERT_NEAR(a, b, tol) do { \
    if (std::abs((a) - (b)) > (tol)) throw std::runtime_error("Assertion failed: |" #a " - " #b "| <= " #tol); \
} while(0)

}
}
