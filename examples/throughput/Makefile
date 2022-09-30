TARGET_MT  := throughput_mt
TARGET_MP  := throughput_mp

C_FLAGS := -std=gnu99
ifeq ($(DEBUG),Yes)
    C_FLAGS += -g -O0
else
    C_FLAGS += -O2
endif
C_FLAGS += -Wall -Werror -Wextra -pthread
C_FLAGS += -I../../inc -I../../src/utils

C_FLAGS_MP :=
LDFLAGS := -std=gnu99 -Wall -Werror
C_LIBS  := -L../../lib -ltakyon -pthread
HEADER_FILES := throughput.h

OS_TYPE := $(shell uname -s)
ifeq ($(MMAP),Yes)
    C_FLAGS_MP += -DENABLE_MMAP
    ifeq ($(OS_TYPE),Linux)
        C_LIBS  += -lrt
    endif
endif

ifeq ($(RDMA),Yes)
    ifeq ($(OS_TYPE),Linux)
        C_LIBS  += -lrdmacm -libverbs
    endif
endif

ifeq ($(CUDA),Yes)
    C_FLAGS += -DENABLE_CUDA -I/usr/local/cuda/include
    C_LIBS  += -L/usr/local/cuda/lib64 -lcuda -lcudart
endif

vpath %.h ../../inc
vpath %.h ../../src/utils

all: $(TARGET_MT) $(TARGET_MP)

clean:
	rm -f *.o
	rm -f *~
	rm -f $(TARGET_MT) $(TARGET_MP)

$(TARGET_MT): throughput.c main_inter_thread.c ../../lib/libtakyon.a
	gcc $(C_FLAGS) -c throughput.c -o throughput_mt.o
	gcc $(C_FLAGS) -c main_inter_thread.c -o main_inter_thread.o
	gcc $(LDFLAGS) -o $@ throughput_mt.o main_inter_thread.o $(C_LIBS)

$(TARGET_MP): throughput.c main_inter_processor.c ../../lib/libtakyon.a
	gcc $(C_FLAGS) $(C_FLAGS_MP) -c throughput.c -o throughput_mp.o
	gcc $(C_FLAGS) -c main_inter_processor.c -o main_inter_processor.o
	gcc $(LDFLAGS) -o $@ throughput_mp.o main_inter_processor.o $(C_LIBS)
