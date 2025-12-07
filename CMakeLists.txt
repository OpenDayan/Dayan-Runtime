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
add_executable(npu-test_runtime ${SOURCE} test/test_runtime.cpp ../npu-kernels/general_entry_fpga.packed.c)
add_executable(npu-test_matmul_e4m3q ${SOURCE} test/test_runtime_matmul_e4m3q.cpp ../npu-kernels/general_entry_fpga.packed.c)
add_executable(npu-test_matmul ${SOURCE} test/test_runtime_matmul_fp32_fp32_fp32.cpp ../npu-kernels/general_entry_fpga.packed.c)
add_executable(npu-test_matmul_fp16_fp16_fp32 ${SOURCE} test/test_runtime_matmul_fp16_fp16_fp32.cpp ../npu-kernels/general_entry_fpga.packed.c)
add_executable(npu-test_matmul_bf16_bf16_fp32 ${SOURCE} test/test_runtime_matmul_bf16_bf16_fp32.cpp ../npu-kernels/general_entry_fpga.packed.c)
add_executable(npu-test_matmul_e4m3_e4m3_fp32 ${SOURCE} test/test_runtime_matmul_e4m3_e4m3_fp32.cpp ../npu-kernels/general_entry_fpga.packed.c)
add_executable(npu-test_elementwise ${SOURCE} test/test_runtime_elementwise.cpp ../npu-kernels/general_entry_fpga.packed.c)
add_executable(npu-test_reduce ${SOURCE} test/test_runtime_reduce.cpp ../npu-kernels/general_entry_fpga.packed.c)
add_executable(npu-test_convert ${SOURCE} test/test_runtime_convert.cpp ../npu-kernels/general_entry_fpga.packed.c)
add_executable(npu-test_lut ${SOURCE} test/test_runtime_lut.cpp ../npu-kernels/general_entry_fpga.packed.c)

include_directories(include)
include_directories(../npu-kernels/testdata_gen)
include_directories(../npu-kernels/include)

set(CMAKE_EXE_LINKER_FLAGS "-static")
