TARGET_MT  := hello_mt
TARGET_MP  := hello_mp

C_FLAGS := -std=gnu99
ifeq ($(DEBUG),Yes)
    C_FLAGS += -g -O0
else
    C_FLAGS += -O2
endif
C_FLAGS += -Wall -Werror -Wextra -pthread
C_FLAGS += -I../../inc

C_FLAGS_MT :=
LDFLAGS := -std=gnu99 -Wall -Werror
C_LIBS  := -L../../lib -ltakyon -pthread
HEADER_FILES := hello.h

OS_TYPE := $(shell uname -s)
ifeq ($(MMAP),Yes)
    C_FLAGS_MT += -DENABLE_MMAP -I../../src/utils
    ifeq ($(OS_TYPE),Linux)
        C_LIBS  += -lrt
    endif
endif

ifeq ($(RDMA),Yes)
#+    C_FLAGS +=
#+    C_LIBS  += -libverbs -lrdmacm
endif

ifeq ($(CUDA),Yes)
    C_FLAGS += -DENABLE_CUDA -I/usr/local/cuda/include
    C_LIBS  += -L/usr/local/cuda/lib64 -lcudart
endif

vpath %.h ../../inc
vpath %.h ../../src/utils

all: $(TARGET_MT) $(TARGET_MP)

clean:
	rm -f *.o
	rm -f *~
	rm -f $(TARGET_MT) $(TARGET_MP)

$(TARGET_MT): hello.c main_inter_thread.c ../../lib/libtakyon.a
	gcc $(C_FLAGS) -c hello.c -o hello_mt.o
	gcc $(C_FLAGS) -c main_inter_thread.c -o main_inter_thread.o
	gcc $(LDFLAGS) -o $@ hello_mt.o main_inter_thread.o $(C_LIBS)

$(TARGET_MP): hello.c main_inter_processor.c ../../lib/libtakyon.a
	gcc $(C_FLAGS) $(C_FLAGS_MT) -c hello.c -o hello_mp.o
	gcc $(C_FLAGS) -c main_inter_processor.c -o main_inter_processor.o
	gcc $(LDFLAGS) -o $@ hello_mp.o main_inter_processor.o $(C_LIBS)
