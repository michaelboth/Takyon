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

# RDMA additions
ifeq ($(RDMA),Yes)
    LIBS  += -lrdmacm -libverbs
endif

# CUDA additions
ifeq ($(CUDA),Yes)
    COMMON_FLAGS += -DENABLE_CUDA -I/usr/local/cuda/include
    LIBS  += -L/usr/local/cuda/lib64 -lcuda -lcudart
endif

# Unikorn additions
UNIKORN_HOME := ../../../EventAnalyzer
COMMON_FLAGS += -I$(UNIKORN_HOME)/inc
ifeq ($(UNIKORN),Yes)
    COMMON_FLAGS += -DENABLE_UNIKORN_RECORDING
    COMMON_FLAGS += -DUNIKORN_RELEASE_BUILD
    C_OBJS       += unikorn.o unikorn_file_flush.o unikorn_clock_gettime.o
    HEADER_FILES += unikorn.h unikorn_clock.h unikorn_file_flush.h
endif

C_FLAGS   := -std=gnu99 $(COMMON_FLAGS)
CPP_FLAGS := -std=c++20 $(COMMON_FLAGS)

vpath %.h ../../inc
vpath %.h $(UNIKORN_HOME)/inc
vpath %.c $(UNIKORN_HOME)/src

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
