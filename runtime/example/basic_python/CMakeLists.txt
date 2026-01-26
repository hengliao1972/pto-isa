cmake_minimum_required(VERSION 3.15)

# This CMakeLists.txt sets up the Python basic example
# It copies the necessary files to the build directory

# Copy the Python example script
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/graphbuilder.py
    ${CMAKE_CURRENT_BINARY_DIR}/graphbuilder.py
    COPYONLY
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/gemm16_ptoas.py
    ${CMAKE_CURRENT_BINARY_DIR}/gemm16_ptoas.py
    COPYONLY
)

# Copy the run helper script
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/run.sh
    ${CMAKE_CURRENT_BINARY_DIR}/run.sh
    COPYONLY
)

# Copy kernel source files to the build directory
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/kernels)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/kernel_add.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/kernels/kernel_add.cpp
    COPYONLY
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/kernel_add_scalar.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/kernels/kernel_add_scalar.cpp
    COPYONLY
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels/kernel_mul.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/kernels/kernel_mul.cpp
    COPYONLY
)

# Create a custom target to ensure files are copied
add_custom_target(python_example ALL
    DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/graphbuilder.py
        ${CMAKE_CURRENT_BINARY_DIR}/gemm16_ptoas.py
        ${CMAKE_CURRENT_BINARY_DIR}/run.sh
        ${CMAKE_CURRENT_BINARY_DIR}/kernels/kernel_add.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/kernels/kernel_add_scalar.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/kernels/kernel_mul.cpp
    COMMENT "Setting up Python basic example"
)

message(STATUS "Python basic example will be available at: ${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "To run: cd ${CMAKE_CURRENT_BINARY_DIR} && ./run.sh <device_id>")
message(STATUS "Or manually: cd ${CMAKE_CURRENT_BINARY_DIR} && PYTHONPATH=../../python python3 graphbuilder.py <device_id>")
