TARGET  := rdma_vs_socket

CPP_OBJS     := Main.o Common.o LatencyTest.o ThroughputTest.o
C_OBJS       :=
HEADER_FILES := Common.hpp LatencyTest.hpp ThroughputTest.hpp unikorn_instrumentation.h

COMMON_FLAGS :=
ifeq ($(DEBUG),Yes)
    COMMON_FLAGS += -g -O0
else
    COMMON_FLAGS += -O2
endif
COMMON_FLAGS += -Wall -Werror -Wextra -pthread
COMMON_FLAGS += -I../../inc

LIBS  := -L../../lib -ltakyon -pthread

# Determine to OS to know if RDMA or CUDA can be supported
OS_TYPE := $(shell uname -s)

# RDMA additions
ifeq ($(RDMA),Yes)
    ifeq ($(OS_TYPE),Linux)
        LIBS  += -lrdmacm -libverbs
    endif
endif

# CUDA additions
ifeq ($(CUDA),Yes)
    ifeq ($(OS_TYPE),Linux)
        COMMON_FLAGS += -DENABLE_CUDA -I/usr/local/cuda/include
        LIBS  += -L/usr/local/cuda/lib64 -lcuda -lcudart
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

C_FLAGS   := -std=gnu99 $(COMMON_FLAGS)
CPP_FLAGS := -std=c++20 $(COMMON_FLAGS)

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

$(TARGET): $(C_OBJS) $(CPP_OBJS)
	g++ $(C_OBJS) $(CPP_OBJS) $(LIBS) -o $@
