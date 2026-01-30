# PTO Runtime2 Makefile
#
# Builds the PTO Runtime2 library with buffer management
# Based on: docs/runtime_buffer_manager_methods.md

CC = gcc
AR = ar
CFLAGS = -Wall -Wextra -O3 -fPIC -std=c11
DEBUG_FLAGS = -g -DDEBUG -O0
DEBUG_BALANCE_FLAGS = -g -DPTO2_DEBUG_LOAD_BALANCE -O2
INCLUDES = -I. -I./runtime_a2a3_sim/core_model
LDFLAGS = -lpthread

# Source files
SRCS = \
	pto_shared_memory.c \
	pto_ring_buffer.c \
	pto_tensormap.c \
	pto_logical_tensor.c \
	pto_interval_tree.c \
	pto_scheduler.c \
	pto_orchestrator.c \
	pto_runtime2.c \
	pto_runtime2_sim.c \
	pto_worker.c \
	pto_runtime2_threaded.c

OBJS = $(SRCS:.c=.o)

# Headers
HEADERS = \
	pto_runtime2_types.h \
	pto_shared_memory.h \
	pto_ring_buffer.h \
	pto_tensormap.h \
	pto_logical_tensor.h \
	pto_interval_tree.h \
	pto_scheduler.h \
	pto_orchestrator.h \
	pto_runtime2.h \
	pto_runtime2_sim.h \
	pto_worker.h \
	pto_runtime2_threaded.h

# Output library
LIB_NAME = libpto_runtime2
STATIC_LIB = $(LIB_NAME).a
SHARED_LIB = $(LIB_NAME).so

# Core model library (optional)
CORE_MODEL_DIR = runtime_a2a3_sim/core_model
CORE_MODEL_LIB = $(CORE_MODEL_DIR)/liba2a3_core.a

# Test programs
TEST_DIR = tests
TEST_SRCS = $(wildcard $(TEST_DIR)/*.c)
TEST_BINS = $(TEST_SRCS:.c=)

.PHONY: all clean test debug lib shared core_model install help

all: lib

# Build static library
lib: $(STATIC_LIB)

$(STATIC_LIB): $(OBJS)
	$(AR) rcs $@ $^

# Build shared library
shared: $(SHARED_LIB)

$(SHARED_LIB): $(OBJS)
	$(CC) -shared -o $@ $^

# Compile source files
%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

# Debug build
debug: CFLAGS += $(DEBUG_FLAGS)
debug: clean lib

# Debug load balance (prints worker states when getting tasks)
debug_balance: CFLAGS += $(DEBUG_BALANCE_FLAGS)
debug_balance: clean lib

# Build with core model integration
core_model: CFLAGS += -DA2A3_CORE_SIM_AVAILABLE
core_model: $(CORE_MODEL_LIB)
core_model: lib

$(CORE_MODEL_LIB):
	$(MAKE) -C $(CORE_MODEL_DIR) lib

# Test targets
test: lib
	@mkdir -p $(TEST_DIR)
	@echo "Running tests..."
	@if [ -f $(TEST_DIR)/test_runtime2.c ]; then \
		$(CC) $(CFLAGS) $(INCLUDES) -o $(TEST_DIR)/test_runtime2 \
			$(TEST_DIR)/test_runtime2.c -L. -lpto_runtime2 $(LDFLAGS) && \
		./$(TEST_DIR)/test_runtime2; \
	else \
		echo "No tests found. Creating example test..."; \
	fi

# Create test directory and example test
$(TEST_DIR)/test_runtime2.c:
	@mkdir -p $(TEST_DIR)
	@echo '#include "pto_runtime2.h"' > $@
	@echo '#include <stdio.h>' >> $@
	@echo '' >> $@
	@echo 'int main(void) {' >> $@
	@echo '    printf("PTO Runtime2 Test\\n");' >> $@
	@echo '    PTO2Runtime* rt = pto2_runtime_create(PTO2_MODE_SIMULATE);' >> $@
	@echo '    if (!rt) {' >> $@
	@echo '        printf("FAILED: Could not create runtime\\n");' >> $@
	@echo '        return 1;' >> $@
	@echo '    }' >> $@
	@echo '    printf("Runtime created successfully\\n");' >> $@
	@echo '    pto2_runtime_print_stats(rt);' >> $@
	@echo '    pto2_runtime_destroy(rt);' >> $@
	@echo '    printf("PASSED\\n");' >> $@
	@echo '    return 0;' >> $@
	@echo '}' >> $@

# BGEMM test (multi-threaded)
bgemm: lib
	@echo "Building BGEMM test..."
	$(CC) $(CFLAGS) $(INCLUDES) -o $(TEST_DIR)/test_bgemm_runtime2 \
		$(TEST_DIR)/test_bgemm_runtime2.c -L. -lpto_runtime2 $(LDFLAGS)
	@echo "Running BGEMM test (multi-threaded)..."
	@cd $(TEST_DIR) && ./test_bgemm_runtime2 4 4 4 4

# BGEMM large test (multi-threaded)
bgemm-large: lib
	@echo "Building BGEMM test..."
	$(CC) $(CFLAGS) $(INCLUDES) -o $(TEST_DIR)/test_bgemm_runtime2 \
		$(TEST_DIR)/test_bgemm_runtime2.c -L. -lpto_runtime2 $(LDFLAGS)
	@echo "Running large BGEMM test (multi-threaded)..."
	@cd $(TEST_DIR) && ./test_bgemm_runtime2 16 8 8 8

# BGEMM flow control test (small task window to trigger flow control)
# Total tasks = 8*8*8*4*2 = 4096, window = 128, needs ~32 flow control cycles
bgemm-flow: lib
	@echo "Building BGEMM test..."
	$(CC) $(CFLAGS) $(INCLUDES) -o $(TEST_DIR)/test_bgemm_runtime2 \
		$(TEST_DIR)/test_bgemm_runtime2.c -L. -lpto_runtime2 $(LDFLAGS)
	@echo "Running BGEMM flow control test (small window)..."
	@cd $(TEST_DIR) && ./test_bgemm_runtime2 8 8 8 4 128

# Clean
clean:
	rm -f $(OBJS) $(STATIC_LIB) $(SHARED_LIB)
	rm -f $(TEST_DIR)/test_runtime2 $(TEST_DIR)/test_bgemm_runtime2
	rm -f $(TEST_DIR)/*.json

clean-all: clean
	$(MAKE) -C $(CORE_MODEL_DIR) clean

# Installation
PREFIX ?= /usr/local

install: lib
	install -d $(PREFIX)/lib
	install -d $(PREFIX)/include/pto_runtime2
	install -m 644 $(STATIC_LIB) $(PREFIX)/lib/
	install -m 644 $(HEADERS) $(PREFIX)/include/pto_runtime2/

.PHONY: install

# BGEMM alias
bgemm-mt: bgemm

# Help
help:
	@echo "PTO Runtime2 Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all        - Build static library (default)"
	@echo "  lib        - Build static library"
	@echo "  shared     - Build shared library"
	@echo "  debug      - Build with debug symbols"
	@echo "  core_model - Build with A2A3 core model integration"
	@echo "  test       - Build and run tests"
	@echo "  bgemm      - Build and run BGEMM test (multi-threaded)"
	@echo "  bgemm-large- Build and run large BGEMM test (multi-threaded)"
	@echo "  clean      - Remove build artifacts"
	@echo "  clean-all  - Remove all artifacts including core model"
	@echo "  install    - Install to PREFIX (default: /usr/local)"
	@echo "  help       - Show this help"
	@echo ""
	@echo "Variables:"
	@echo "  CC         - C compiler (default: gcc)"
	@echo "  CFLAGS     - Additional compiler flags"
	@echo "  PREFIX     - Installation prefix (default: /usr/local)"
	@echo ""
	@echo "BGEMM Test Usage:"
	@echo "  ./tests/test_bgemm_runtime2 [batch] [m] [n] [k]"
