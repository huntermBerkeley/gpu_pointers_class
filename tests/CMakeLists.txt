cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

message("Parsing tests")
#mainline tests
ConfigureExecutablePTR(pointer_test "${CMAKE_CURRENT_SOURCE_DIR}/src/pointer_test.cu" "${PTR_TESTS_BINARY_DIR}")

ConfigureExecutablePTR(double_ptr_test "${CMAKE_CURRENT_SOURCE_DIR}/src/double_ptr_test.cu" "${PTR_TESTS_BINARY_DIR}")