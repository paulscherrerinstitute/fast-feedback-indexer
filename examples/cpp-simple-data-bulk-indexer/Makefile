FFBIDX_PKG = fast_indexer
EIGEN_PKG = eigen3

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

TARGET := simple-data-bulk-indexer

$(TARGET): $(TARGET).cpp
	$(CXX) $(COMPILER_FLAGS) $< -o $@ $(LIBRARY_FLAGS)

clean:
	rm $(TARGET)

# Sorry if this help is not super useful, as the package vars need to be set correctly even for this target
help:
	@echo "Makefile using pkg-config"
	@echo "requirements:"
	@echo "  - CUDA runtime"
	@echo "  - Eigen3 header only library"
	@echo "  - Fast feedback indexer library"
	@echo "taregts:"
	@echo "  - $(TARGET): produces the executable (default)"
	@echo "  - clean: remove the executable"
	@echo "env vars:"
	@echo "  - CXX"
	@echo "  - CXXFLAGS"
	@echo "  - LDFLAGS"
	@echo "  - CUDART_PKG    pkg-config package name for cuda runtime (mandatory)"

.PHONY: clean help
