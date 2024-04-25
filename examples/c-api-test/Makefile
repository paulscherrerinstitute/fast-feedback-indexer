FFBIDX_PKG = fast_indexer

ifneq ($(MAKECMDGOALS), help)

ifndef CUDART_PKG
  $(error please define CUDART_PKG as the pkg-config name to use for finding the cuda runtime)
endif

CUDART_OK := $(shell pkg-config --exists $(CUDART_PKG) && echo yes)
ifneq ($(CUDART_OK), yes)
  $(error $(CUDART_PKG) package not found with pkg-config)
endif

FFBIDX_OK := $(shell pkg-config --exists $(FFBIDX_PKG) && echo yes)
ifneq ($(FFBIDX_OK), yes)
  $(error $(FFBIDX_PKG) package not found with pkg-config)
endif

COMPILER_FLAGS := $(CFLAGS) $(shell pkg-config --cflags $(CUDART_PKG)) $(shell pkg-config --cflags $(FFBIDX_PKG)) # $(shell pkg-config --cflags $(EIGEN_PKG))
LIBRARY_FLAGS := $(LDFLAGS) $(shell pkg-config --libs $(CUDART_PKG)) $(shell pkg-config --libs $(FFBIDX_PKG))

TARGET := c-api-test

.PHONY: clean

$(TARGET): $(TARGET).c
	$(CC) $(COMPILER_FLAGS) $< -o $@ $(LIBRARY_FLAGS)

clean:
	rm $(TARGET)

endif

.PHONY: help

help:
	@echo "Makefile using pkg-config"
	@echo "requirements:"
	@echo "  - CUDA runtime"
	@echo "  - Fast feedback indexer library"
	@echo "targets:"
	@echo "  - $(TARGET): produces the executable (default)"
	@echo "  - clean: remove the executable"
	@echo "env vars:"
	@echo "  - CC"
	@echo "  - CFLAGS"
	@echo "  - LDFLAGS"
	@echo "  - CUDART_PKG    pkg-config package name for cuda runtime (mandatory)"
