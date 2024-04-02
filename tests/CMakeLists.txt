cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

message("Parsing tests")
#mainline tests
ConfigureExecutablePTR(pointer_test "${CMAKE_CURRENT_SOURCE_DIR}/src/pointer_test.cu" "${PTR_TESTS_BINARY_DIR}")

# target_link_libraries(cache_test PRIVATE bght)
# target_link_libraries(cache_test PRIVATE warpcore)
# target_link_libraries(cache_test PRIVATE hashing_project)
# target_link_libraries(cache_test PRIVATE ssl crypto)

