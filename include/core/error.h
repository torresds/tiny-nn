#pragma once
#include <stdexcept>
#include <string>
#include <sstream>
#include <iostream>

namespace tf {

#define THROW_ERROR(msg) do { \
    std::ostringstream __oss; \
    __oss << "Error at " << __FILE__ << ":" << __LINE__ << ": " << msg; \
    throw std::runtime_error(__oss.str()); \
} while(0)

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        THROW_ERROR(msg); \
    } \
} while(0)

}
