cmake_minimum_required(VERSION 3.10)
project(npu-runtime-test LANGUAGES C CXX)

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../NPU-Driver/src/*.c src/*.c test/*.cpp)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE include ${CMAKE_CURRENT_SOURCE_DIR}/../include)


# target_link_libraries(${PROJECT_NAME} PRIVATE some_library)

set(CMAKE_EXE_LINKER_FLAGS "-static")
