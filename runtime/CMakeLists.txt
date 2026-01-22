cmake_minimum_required(VERSION 3.15)
project(runtime_gm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra -Wpedantic)

# Ascend path
if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
else()
    message(FATAL_ERROR "ASCEND_HOME_PATH is not set, CANN runtime is required")
endif()

# Compile kernels
add_subdirectory(aicpu)
add_subdirectory(aicore)

# Python bindings (optional - requires Python and pybind11)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(python)
endif()

# Compile host graph executor
add_executable(graph_executor graphbuilder.cpp host/devicerunner.cpp host/memoryallocator.cpp graph/graph.cpp)

target_include_directories(graph_executor PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/graph
)

# Link CANN runtime (must be available)
target_include_directories(graph_executor PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/pkg_inc
    ${ASCEND_HOME_PATH}/pkg_inc/runtime
    ${ASCEND_HOME_PATH}/pkg_inc/profiling
)
target_link_directories(graph_executor PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64
)

target_link_libraries(graph_executor PRIVATE runtime)

message(STATUS "ASCEND_HOME_PATH = ${ASCEND_HOME_PATH}")
