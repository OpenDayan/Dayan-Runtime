cmake_minimum_required(VERSION 3.10)
project(npu-driver LANGUAGES C CXX)
set(SOURCE src/NPUDriver.c src/NPURuntime.c)

add_executable(npu-test_driver ${SOURCE} test/test_driver.cpp)
add_executable(npu-test_runtime ${SOURCE} test/test_runtime.cpp test/test_runtime_kernel.cpp)

include_directories(include)

set(CMAKE_EXE_LINKER_FLAGS "-static")
