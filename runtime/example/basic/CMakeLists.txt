cmake_minimum_required(VERSION 3.16.3)

# Get the Bisheng compiler and linker paths from ASCEND_HOME_PATH (supports multiple CANN layouts).
find_program(
    BISHENG_COMPILER
    NAMES ccec bisheng
    HINTS
        "${ASCEND_HOME_PATH}/bin"
        "${ASCEND_HOME_PATH}/compiler/ccec_compiler/bin"
    REQUIRED
)
find_program(
    BISHENG_LD
    NAMES ld.lld
    HINTS
        "${ASCEND_HOME_PATH}/bin"
        "${ASCEND_HOME_PATH}/compiler/ccec_compiler/bin"
    REQUIRED
)

# Get PTO_ISA_ROOT from parent (set in runtime/aicore/CMakeLists.txt)
# For now, fetch it here too
include(FetchContent)

FetchContent_Declare(
    pto-isa
    GIT_REPOSITORY https://gitcode.com/cann/pto-isa.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
)

FetchContent_GetProperties(pto-isa)
if(NOT pto-isa_POPULATED)
    FetchContent_Populate(pto-isa)
endif()

set(PTO_ISA_ROOT ${pto-isa_SOURCE_DIR})

# Source files for the example kernels
set(KERNEL_ADD ${CMAKE_CURRENT_SOURCE_DIR}/kernels/kernel_add.cpp)
set(KERNEL_ADD_SCALAR ${CMAKE_CURRENT_SOURCE_DIR}/kernels/kernel_add_scalar.cpp)
set(KERNEL_MUL ${CMAKE_CURRENT_SOURCE_DIR}/kernels/kernel_mul.cpp)

# Object files (AIC only - cube architecture)
set(KERNEL_OBJ_ADD ${CMAKE_CURRENT_BINARY_DIR}/kernel_add_aic.o)
set(KERNEL_OBJ_ADD_SCALAR ${CMAKE_CURRENT_BINARY_DIR}/kernel_add_scalar_aic.o)
set(KERNEL_OBJ_MUL ${CMAKE_CURRENT_BINARY_DIR}/kernel_mul_aic.o)

# Compiler flags for AICore kernels
set(AI_CORE_FLAGS
    "-c -O3 -g -x cce -Wall -std=c++17 \
    --cce-aicore-only \
    -mllvm -cce-aicore-stack-size=0x8000 \
    -mllvm -cce-aicore-function-stack-size=0x8000 \
    -mllvm -cce-aicore-record-overflow=false \
    -mllvm -cce-aicore-addr-transform \
    -mllvm -cce-aicore-dcci-insert-for-scalar=false \
    -DMEMORY_BASE \
    -I${PTO_ISA_ROOT}/include \
    -I${PTO_ISA_ROOT}/include/pto \
    -I${CMAKE_CURRENT_SOURCE_DIR}/../.."
)
separate_arguments(AI_CORE_FLAGS)

# Compile the example kernels
add_custom_target(example_kernels ALL
    DEPENDS ${KERNEL_ADD} ${KERNEL_ADD_SCALAR} ${KERNEL_MUL}

    # Compile kernel_add (AIC only)
    COMMAND ${BISHENG_COMPILER} ${AI_CORE_FLAGS} -D__AIC__ --cce-aicore-arch=dav-c220-cube
            -o ${KERNEL_OBJ_ADD} ${KERNEL_ADD}

    # Compile kernel_add_scalar (AIC only)
    COMMAND ${BISHENG_COMPILER} ${AI_CORE_FLAGS} -D__AIC__ --cce-aicore-arch=dav-c220-cube
            -o ${KERNEL_OBJ_ADD_SCALAR} ${KERNEL_ADD_SCALAR}

    # Compile kernel_mul (AIC only)
    COMMAND ${BISHENG_COMPILER} ${AI_CORE_FLAGS} -D__AIC__ --cce-aicore-arch=dav-c220-cube
            -o ${KERNEL_OBJ_MUL} ${KERNEL_MUL}

    BYPRODUCTS ${KERNEL_OBJ_ADD} ${KERNEL_OBJ_ADD_SCALAR} ${KERNEL_OBJ_MUL}
    COMMENT "Compiling example kernels for basic example"
)

# Build the graphbuilder executable
add_executable(graphbuilder
    graphbuilder.cpp
    ${CMAKE_SOURCE_DIR}/host/devicerunner.cpp
    ${CMAKE_SOURCE_DIR}/host/memoryallocator.cpp
    ${CMAKE_SOURCE_DIR}/host/binary_loader.cpp
    ${CMAKE_SOURCE_DIR}/host/kernel_compiler.cpp
    ${CMAKE_SOURCE_DIR}/graph/graph.cpp
)

target_include_directories(graphbuilder PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/graph
)

# Link CANN runtime
target_include_directories(graphbuilder PRIVATE
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/include/experiment/msprof
    ${ASCEND_HOME_PATH}/pkg_inc
    ${ASCEND_HOME_PATH}/pkg_inc/runtime
    ${ASCEND_HOME_PATH}/pkg_inc/profiling
)
target_link_directories(graphbuilder PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64
)

target_link_libraries(graphbuilder PRIVATE runtime)

# Make sure kernels are compiled before the executable
add_dependencies(graphbuilder example_kernels)

message(STATUS "Basic example will be built at: ${CMAKE_CURRENT_BINARY_DIR}/graphbuilder")
