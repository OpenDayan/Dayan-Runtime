cmake_minimum_required(VERSION 3.10)
project(npu-driver LANGUAGES C CXX)
set(SOURCE src/NPUDriver.c src/NPURuntime.c src/NPUMemoryManagement.cpp)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(USE_FPGA "Enable FPGA mode" OFF)

if(USE_FPGA)
    add_compile_definitions(USE_FPGA)
    message(STATUS "Using FPGA mode")
endif()

option(DEV_PREFIX "Device prefix for NPU device files" "")
if(DEV_PREFIX)
    add_compile_definitions(DEV_PREFIX="${DEV_PREFIX}")
    message(STATUS "Using device prefix: ${DEV_PREFIX}")
endif()

if(USE_FPGA AND DEV_PREFIX)
    message(FATAL_ERROR "Cannot set DEV_PREFIX when USE_FPGA is enabled")
endif()

add_executable(npu-test_driver ${SOURCE} test/test_driver.cpp)
add_executable(npu-test_driver_memset ${SOURCE} test/test_driver_memset.cpp)
add_executable(npu-test_runtime ${SOURCE} test/test_runtime.cpp)
add_executable(npu-test_matmul_e4m3q ${SOURCE} test/test_runtime_matmul_e4m3q.cpp)
add_executable(npu-test_matmul ${SOURCE} test/test_runtime_matmul_fp32_fp32_fp32.cpp)
add_executable(npu-test_matmul_fp16_fp16_fp32 ${SOURCE} test/test_runtime_matmul_fp16_fp16_fp32.cpp)
add_executable(npu-test_matmul_bf16_bf16_fp32 ${SOURCE} test/test_runtime_matmul_bf16_bf16_fp32.cpp)
add_executable(npu-test_matmul_e4m3_e4m3_fp32 ${SOURCE} test/test_runtime_matmul_e4m3_e4m3_fp32.cpp)
add_executable(npu-test_elementwise ${SOURCE} test/test_runtime_elementwise.cpp)
add_executable(npu-test_reduce ${SOURCE} test/test_runtime_reduce.cpp)
add_executable(npu-test_convert ${SOURCE} test/test_runtime_convert.cpp)
add_executable(npu-test_lut ${SOURCE} test/test_runtime_lut.cpp)

include_directories(include)
include_directories(../npu-kernels/testdata_gen)
include_directories(../npu-kernels/include)

include(CTest)
if(BUILD_TESTING)
    find_package(Threads REQUIRED)
    add_executable(npu-test-abort-handshake
        src/NPUDriver.c
        test/test_abort_handshake.c)
    target_link_libraries(npu-test-abort-handshake PRIVATE Threads::Threads)
    add_test(NAME npu-abort-handshake COMMAND npu-test-abort-handshake)

    add_executable(npu-test-abort-timeout
        src/NPUDriver.c
        test/test_abort_timeout.c)
    target_compile_definitions(npu-test-abort-timeout PRIVATE
        NPU_ABORT_ACK_TIMEOUT_MS=10)
    add_test(NAME npu-abort-timeout COMMAND npu-test-abort-timeout)
    set_tests_properties(npu-abort-timeout PROPERTIES TIMEOUT 2)

    add_executable(npu-test-command-quiescence
        src/NPUDriver.c
        test/test_command_quiescence.c)
    target_link_libraries(npu-test-command-quiescence PRIVATE Threads::Threads)
    add_test(NAME npu-command-quiescence COMMAND npu-test-command-quiescence)

    add_executable(npu-test-start-abort-handoff
        src/NPUDriver.c
        test/test_start_abort_handoff.c)
    target_link_libraries(npu-test-start-abort-handoff PRIVATE Threads::Threads)
    add_test(NAME npu-start-abort-handoff COMMAND npu-test-start-abort-handoff)

    add_executable(npu-test-start-address-validation
        src/NPUDriver.c
        test/test_start_address_validation.c)
    target_compile_definitions(npu-test-start-address-validation PRIVATE
        NPU_ABORT_ACK_TIMEOUT_MS=10)
    add_test(NAME npu-start-address-validation COMMAND npu-test-start-address-validation)
    set_tests_properties(npu-start-address-validation PROPERTIES TIMEOUT 2)
endif()

set(CMAKE_EXE_LINKER_FLAGS "-static")

find_library(DL_LIB dl)
if(DL_LIB)
    target_link_libraries(npu-test_driver PRIVATE ${DL_LIB})
    target_link_libraries(npu-test_driver_memset PRIVATE ${DL_LIB})
    target_link_libraries(npu-test_runtime PRIVATE ${DL_LIB})
    target_link_libraries(npu-test_matmul_e4m3q PRIVATE ${DL_LIB})
    target_link_libraries(npu-test_matmul PRIVATE ${DL_LIB})
    target_link_libraries(npu-test_matmul_fp16_fp16_fp32 PRIVATE ${DL_LIB})
    target_link_libraries(npu-test_matmul_bf16_bf16_fp32 PRIVATE ${DL_LIB})
    target_link_libraries(npu-test_matmul_e4m3_e4m3_fp32 PRIVATE ${DL_LIB})
    target_link_libraries(npu-test_elementwise PRIVATE ${DL_LIB})
    target_link_libraries(npu-test_reduce PRIVATE ${DL_LIB})
    target_link_libraries(npu-test_convert PRIVATE ${DL_LIB})
    target_link_libraries(npu-test_lut PRIVATE ${DL_LIB})
endif()
