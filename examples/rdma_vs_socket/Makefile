TARGET  := rdma_vs_socket

CPP_OBJS     := Main.o Common.o LatencyTest.o ThroughputTest.o
C_OBJS       :=
CU_OBJS      :=
HEADER_FILES := Common.hpp LatencyTest.hpp ThroughputTest.hpp unikorn_instrumentation.h ValidationKernels.hpp

COMMON_FLAGS :=
ifeq ($(DEBUG),Yes)
    COMMON_FLAGS += -g -O0
else
    COMMON_FLAGS += -O2
endif
COMMON_FLAGS += -I../../inc

LIBS  := -L../../lib -ltakyon
LINKER := g++

# Determine to OS to know if RDMA or CUDA can be supported
OS_TYPE := $(shell uname -s)

# RDMA additions
ifeq ($(RDMA),Yes)
    ifeq ($(OS_TYPE),Linux)
        LIBS  += -lrdmacm -libverbs
    endif
endif

# CUDA additions
CUDA_HOME := /usr/local/cuda
ifeq ($(CUDA),Yes)
    ifeq ($(OS_TYPE),Linux)
        CU_OBJS += ValidationKernels.o
        COMMON_FLAGS += -DENABLE_CUDA -I$(CUDA_HOME)/include
        LIBS  += -LL$(CUDA_HOME)/lib64 -lcudart -lcuda
        LINKER := $(CUDA_HOME)/bin/nvcc
    endif
endif

# Unikorn additions
ifeq ($(UNIKORN),Yes)
    UNIKORN_HOME := ../../../EventAnalyzer
    COMMON_FLAGS += -DENABLE_UNIKORN_RECORDING
    COMMON_FLAGS += -DUNIKORN_RELEASE_BUILD
    COMMON_FLAGS += -I$(UNIKORN_HOME)/inc
    C_OBJS       += unikorn.o unikorn_file_flush.o unikorn_clock_gettime.o
    HEADER_FILES += unikorn.h unikorn_clock.h unikorn_file_flush.h
    vpath %.h $(UNIKORN_HOME)/inc
    vpath %.c $(UNIKORN_HOME)/src
endif

C_FLAGS   := -std=gnu99 -Wall -Werror -Wextra -pthread $(COMMON_FLAGS)
CPP_FLAGS := -std=c++20 -Wall -Werror -Wextra -pthread $(COMMON_FLAGS)
CU_FLAGS  := -std=c++20 $(COMMON_FLAGS) -arch=sm_121

vpath %.h ../../inc

all: $(TARGET)

clean:
	rm -f *.o
	rm -f *.events
	rm -f *~
	rm -f $(TARGET)

$(CPP_OBJS): %.o: %.cpp $(HEADER_FILES)
	g++ $(CPP_FLAGS) -c $< -o $@

$(C_OBJS): %.o: %.c $(HEADER_FILES)
	gcc $(C_FLAGS) -c $< -o $@

$(CU_OBJS): %.o: %.cu $(HEADER_FILES)
	$(CUDA_HOME)/bin/nvcc $(CU_FLAGS) -c $< -o $@

$(TARGET): $(C_OBJS) $(CPP_OBJS) $(CU_OBJS)
	$(LINKER) $(C_OBJS) $(CPP_OBJS) $(CU_OBJS) $(LIBS) -o $@
