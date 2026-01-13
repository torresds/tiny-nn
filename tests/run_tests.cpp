#include "utils/test_utils.h"
#include "core/error.h"

void test_sanity() {
    ASSERT_TRUE(true);
    ASSERT_EQ(1 + 1, 2);
}

void test_error_macro() {
    try {
        CHECK(false, "This should fail");
        throw std::runtime_error("CHECK failed to throw");
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        if (msg.find("This should fail") == std::string::npos) {
             throw std::runtime_error("Incorrect error message caught");
        }
    }
}

void test_tensor_creation();
void test_matmul_simple();
void test_transpose();
void test_add();
void test_dense_grad_check();

void test_bce_stability();
void test_bce_normal();
void test_mse_simple();

// Mem tests
void test_move_semantics();
void test_move_assignment();

// Accum tests
void test_grad_accumulation();

int main() {
    std::cout << "Running tiny-nn tests..." << std::endl;
    
    tf::test::run_test("Sanity check", test_sanity);
    tf::test::run_test("Error macro check", test_error_macro);

    tf::test::run_test("Tensor creation", test_tensor_creation);
    tf::test::run_test("Matmul simple", test_matmul_simple);
    tf::test::run_test("Transpose", test_transpose);
    tf::test::run_test("Add", test_add);
    
    tf::test::run_test("Dense grad check", test_dense_grad_check);
    
    tf::test::run_test("BCE stability", test_bce_stability);
    tf::test::run_test("BCE normal", test_bce_normal);
    tf::test::run_test("MSE simple", test_mse_simple);

    tf::test::run_test("Move semantics", test_move_semantics);
    tf::test::run_test("Move assignment", test_move_assignment);
    
    tf::test::run_test("Grad accumulation", test_grad_accumulation);

    tf::test::print_summary();
    return (tf::test::tests_passed == tf::test::tests_run) ? 0 : 1;
}
