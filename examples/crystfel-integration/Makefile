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

LIBN := ffbi
LIB := lib$(LIBN).so

COMPILER_FLAGS := $(CXXFLAGS) -fPIC $(shell pkg-config --cflags $(CUDART_PKG)) $(shell pkg-config --cflags $(FFBIDX_PKG)) $(shell pkg-config --cflags $(EIGEN_PKG))
LINKER_FLAGS := $(LDFLAGS) -L. -l$(LIBN) $(shell pkg-config --libs $(FFBIDX_PKG)) $(shell pkg-config --libs $(CUDART_PKG))

all: $(LIB) clean

$(LIB): c-wrapper.o
	$(CXX) -shared $(COMPILER_FLAGS) $< -o $@

c-wrapper.o: c-wrapper.cpp ffbidx/c-wrapper.h
	$(CXX) $(COMPILER_FLAGS) $< -c -o $@

simple-data-test: simple-data-test.c ffbidx/c-wrapper.h $(LIB)
	$(CC) $(CFLAGS) $< $(LINKER_FLAGS) -o $@

test: simple-data-test
	cat ../../data/simple/files/image0_local.txt | LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:. ./simple-data-test

clean:
	rm -f c-wrapper.o simple-data-test

proper: clean
	rm $(LIB)

.PHONY: test clean proper all
