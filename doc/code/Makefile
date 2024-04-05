FFBIDX_PKG = fast_indexer
EIGEN_PKG = eigen3

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

EIGEN_OK := $(shell pkg-config --exists $(EIGEN_PKG) && echo yes)
ifneq ($(EIGEN_OK), yes)
$(error $(EIGEN_PKG) package not found with pkg-config)
endif

COMPILER_FLAGS := $(CXXFLAGS) $(shell pkg-config --cflags $(CUDART_PKG)) $(shell pkg-config --cflags $(FFBIDX_PKG)) $(shell pkg-config --cflags $(EIGEN_PKG))
LIBRARY_FLAGS := $(LDFLAGS) $(shell pkg-config --libs $(CUDART_PKG)) $(shell pkg-config --libs $(FFBIDX_PKG))

.PHONY: all clean

all: raw-api convenience-api

raw-api: raw-api.cpp
	$(CXX) $(COMPILER_FLAGS) $< -o $@ $(LIBRARY_FLAGS)

convenience-api: convenience-api.cpp
	$(CXX) $(COMPILER_FLAGS) $< -o $@ $(LIBRARY_FLAGS)

clean:
	rm -f raw-api convenience-api

endif

.PHONY: help

help:
	@echo "Makefile using pkg-config"
	@echo "requirements:"
	@echo "  - CUDA runtime"
	@echo "  - Eigen3 header only library"
	@echo "  - Fast feedback indexer library"
	@echo "targets:"
	@echo "  - all: raw-api and convenience-api executables (default)"
	@echo "  - raw-api: produces the raw-api executable"
	@echo "  - convenience-api: produces the convenience api executable"
	@echo "  - clean: remove the executables"
	@echo "env vars:"
	@echo "  - CXX"
	@echo "  - CXXFLAGS"
	@echo "  - LDFLAGS"
	@echo "  - CUDART_PKG    pkg-config package name for cuda runtime (mandatory)"
