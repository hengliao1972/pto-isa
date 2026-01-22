cmake_minimum_required(VERSION 3.15)

# This CMakeLists.txt builds the Python bindings for PTO Runtime

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Fetch pybind11 if not available
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.11.1
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(pybind11)

# Create Python module
pybind11_add_module(pto_runtime bindings.cpp
    ../host/devicerunner.cpp
    ../host/memoryallocator.cpp
    ../graph/graph.cpp
)

# Include directories
target_include_directories(pto_runtime PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/pkg_inc
    ${ASCEND_HOME_PATH}/pkg_inc/runtime
    ${ASCEND_HOME_PATH}/pkg_inc/profiling
)

# Link directories
target_link_directories(pto_runtime PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64
)

# Link libraries
target_link_libraries(pto_runtime PRIVATE runtime)

# Set output directory to current directory so Python can import easily
set_target_properties(pto_runtime PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

message(STATUS "Python bindings will be built at: ${CMAKE_CURRENT_BINARY_DIR}")
