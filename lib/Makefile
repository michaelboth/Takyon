C_FLAGS := -std=gnu99
ifeq ($(DEBUG),Yes)
    C_FLAGS += -g -O0
else
    C_FLAGS += -O2
endif
C_FLAGS += -fPIC
C_FLAGS += -Wall
C_FLAGS += -Werror
C_FLAGS += -Wextra
C_FLAGS += -I../inc -I../src/utils -I../src/interconnects -I../src/core

C_OBJS := takyon.o
C_OBJS += supported_interconnects.o
C_OBJS += utils_arg_parser.o
C_OBJS += utils_endian.o

HEADER_FILES := takyon.h
HEADER_FILES += takyon_private.h
HEADER_FILES += supported_interconnects.h
HEADER_FILES += utils_arg_parser.h
HEADER_FILES += utils_endian.h

LIBRARY := libtakyon.a

NEED_utils_time := No
NEED_utils_ipc := No
NEED_utils_socket := No
NEED_utils_thread_cond_timed_wait := No
NEED_utils_ephemeral_port_manager := No

#---------------------------------------------
# InterThread
#---------------------------------------------
ifeq ($(InterThread),Yes)
    C_FLAGS      += -DENABLE_InterThread
    NEED_utils_time := Yes
    NEED_utils_thread_cond_timed_wait := Yes
    C_OBJS       += takyon_inter_thread_manager.o
    C_OBJS       += interconnect_InterThread.o
    HEADER_FILES += takyon_inter_thread_manager.h
    HEADER_FILES += interconnect_InterThread.h
endif

#---------------------------------------------
# InterProcess
#---------------------------------------------
ifeq ($(InterProcess),Yes)
    C_FLAGS      += -DENABLE_InterProcess
    NEED_utils_time := Yes
    NEED_utils_ipc := Yes
    NEED_utils_thread_cond_timed_wait := Yes
    NEED_utils_socket := Yes
#+    NEED_utils_ephemeral_port_manager := Yes
    C_OBJS       += interconnect_InterProcess.o
    HEADER_FILES += interconnect_InterProcess.h
endif

#---------------------------------------------
# TcpSocket
#---------------------------------------------
ifeq ($(TcpSocket),Yes)
    C_FLAGS      += -DENABLE_TcpSocket
    NEED_utils_time := Yes
    NEED_utils_thread_cond_timed_wait := Yes
    NEED_utils_socket := Yes
    NEED_utils_ephemeral_port_manager := Yes
    C_OBJS       += interconnect_TcpSocket.o
    HEADER_FILES += interconnect_TcpSocket.h
endif

#---------------------------------------------
# UdpSocket
#---------------------------------------------
ifeq ($(UdpSocket),Yes)
    C_FLAGS      += -DENABLE_UdpSocket
    NEED_utils_time := Yes
    NEED_utils_socket := Yes
    C_OBJS       += interconnect_UdpSocket.o
    HEADER_FILES += interconnect_UdpSocket.h
endif

#---------------------------------------------
# Various utilities
#---------------------------------------------
ifeq ($(NEED_utils_time),Yes)
    HEADER_FILES += utils_time.h
    C_OBJS       += utils_time_linux.o
endif

ifeq ($(NEED_utils_ipc),Yes)
    HEADER_FILES += utils_ipc.h
    C_OBJS       += utils_ipc_linux.o
    ifeq ($(CUDA),Yes)
        C_OBJS   += utils_ipc_cuda.o
    endif
endif

ifeq ($(NEED_utils_socket),Yes)
    HEADER_FILES += utils_socket.h
    C_OBJS       += utils_socket_linux.o
endif

ifeq ($(NEED_utils_thread_cond_timed_wait),Yes)
    C_FLAGS      += -pthread
    HEADER_FILES += utils_thread_cond_timed_wait.h
    C_OBJS       += utils_thread_cond_timed_wait.o
endif

ifeq ($(NEED_utils_ephemeral_port_manager),Yes)
    C_FLAGS      += -DENABLE_EPHEMERAL_PORT_MANAGER
    C_OBJS       += utils_socket_ephemeral_port_manager.o
endif

ifeq ($(CUDA),Yes)
    C_FLAGS += -DENABLE_CUDA -I/usr/local/cuda/include
endif

vpath %.c ../src/utils
vpath %.h ../src/utils
vpath %.c ../src/interconnects
vpath %.h ../src/interconnects
vpath %.c ../src/core
vpath %.h ../src/core
vpath %.h ../inc

all: $(LIBRARY)

clean:
	rm -f *.o
	rm -f *~
	rm -f $(LIBRARY)

$(C_OBJS): %.o: %.c $(HEADER_FILES)
	gcc $(C_FLAGS) -c $< -o $@

$(LIBRARY): $(C_OBJS)
	ar rcs $@ $^
