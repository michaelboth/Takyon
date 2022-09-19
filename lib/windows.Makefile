LIBRARY = takyon.lib

# Check if release distribution is enabled
!IF "$(DEBUG)" == "Yes"
OPTIMIZATION_C_FLAGS  = -Zi -MDd # Debug: -MTd or -MDd
!ELSE
OPTIMIZATION_C_FLAGS  = -O2 -MD # Release: -MT means static linking, and -MD means dynamic linking.
!ENDIF

NEED_utils_time = No
NEED_utils_ipc = No
NEED_utils_thread_cond_timed_wait = No
NEED_utils_socket = No
NEED_utils_ephemeral_port_manager = No

#---------------------------------------------
# InterThread
#---------------------------------------------
!IF "$(InterThread)" == "Yes"
InterThread_C_FLAGS = -DENABLE_InterThread
NEED_utils_time = Yes
NEED_utils_thread_cond_timed_wait = Yes
InterThread_C_OBJS = takyon_inter_thread_manager.obj interconnect_InterThread.obj
!ENDIF

#---------------------------------------------
# InterProcess
#---------------------------------------------
!IF "$(InterProcess)" == "Yes"
InterProcess_C_FLAGS = -DENABLE_InterProcess
NEED_utils_time = Yes
NEED_utils_ipc = Yes
NEED_utils_thread_cond_timed_wait = Yes
NEED_utils_socket = Yes
InterProcess_C_OBJS = interconnect_InterProcess.obj
!ENDIF

#---------------------------------------------
# SocketTcp
#---------------------------------------------
!IF "$(SocketTcp)" == "Yes"
SocketTcp_C_FLAGS = -DENABLE_SocketTcp
NEED_utils_time = Yes
NEED_utils_thread_cond_timed_wait = Yes
NEED_utils_socket = Yes
NEED_utils_ephemeral_port_manager = Yes
SocketTcp_C_OBJS = interconnect_SocketTcp.obj
!ENDIF

#---------------------------------------------
# SocketUdp
#---------------------------------------------
!IF "$(SocketUdp)" == "Yes"
SocketUdp_C_FLAGS = -DENABLE_SocketUdp
NEED_utils_time = Yes
NEED_utils_socket = Yes
NEED_utils_thread_cond_timed_wait = Yes
SocketUdp_C_OBJS = interconnect_SocketUdp.obj
!ENDIF

#---------------------------------------------
# Various utilities
#---------------------------------------------
!IF "$(NEED_utils_time)" == "Yes"
utils_time_C_OBJS = utils_time_windows.obj
!ENDIF

!IF "$(NEED_utils_ipc)" == "Yes"
utils_ipc_C_OBJS = utils_ipc_windows.obj
!IF "$(CUDA)" == "Yes"
utils_ipc_C_OBJS = utils_ipc_cuda.obj
!ENDIF
!ENDIF

!IF "$(NEED_utils_socket)" == "Yes"
utils_socket_C_OBJS = utils_socket_windows.obj
!ENDIF

!IF "$(NEED_utils_thread_cond_timed_wait)" == "Yes"
utils_thread_cond_timed_wait_C_FLAGS = -Ic:/pthreads4w/install/include
utils_thread_cond_timed_wait_C_OBJS = utils_thread_cond_timed_wait.obj
!ENDIF

!IF "$(NEED_utils_ephemeral_port_manager)" == "Yes"
utils_ephemeral_port_manager_C_FLAGS = -DENABLE_EPHEMERAL_PORT_MANAGER
utils_ephemeral_port_manager_C_OBJS = utils_socket_ephemeral_port_manager.obj
!ENDIF

!IF "$(CUDA)" == "Yes"
CUDA_HOME = "c:\cuda"
CUDA_C_FLAGS = -DENABLE_CUDA -I"$(CUDA_HOME)\include"
!ENDIF

#---------------------------------------------
# Put it all together
#---------------------------------------------
C_INCS  = -I../inc -I../src/utils -I../src/interconnects -I../src/core

# -std:c11
# -std:c17
C_FLAGS = $(OPTIMIZATION_C_FLAGS) -nologo -WX -W3 -D_CRT_SECURE_NO_WARNINGS $(C_INCS) \
 $(CUDA_C_FLAGS) \
 $(InterThread_C_FLAGS) $(utils_thread_cond_timed_wait_C_FLAGS) \
 $(InterProcess_C_FLAGS) \
 $(SocketTcp_C_FLAGS) $(utils_ephemeral_port_manager_C_FLAGS) \
 $(SocketUdp_C_FLAGS)

C_OBJS  = takyon.obj supported_interconnects.obj utils_arg_parser.obj utils_endian.obj \
 $(InterThread_C_OBJS) $(utils_time_C_OBJS) $(utils_thread_cond_timed_wait_C_OBJS) \
 $(InterProcess_C_OBJS) $(utils_ipc_C_OBJS) $(utils_socket_C_OBJS) \
 $(SocketTcp_C_OBJS) $(utils_ephemeral_port_manager_C_OBJS) \
 $(SocketUdp_C_OBJS)

.SUFFIXES: .c

all: $(LIBRARY)

clean:
	-del *.lib
	-del *.obj
	-del *.pdb
	-del *~

{..\src\core}.c{}.obj::
	cl -c $(C_FLAGS) -Fo $<

{..\src\utils}.c{}.obj::
	cl -c $(C_FLAGS) -Fo $<

{..\src\interconnects}.c{}.obj::
	cl -c $(C_FLAGS) -Fo $<

$(LIBRARY): $(C_OBJS)
	lib -nologo $(C_OBJS) -out:$(LIBRARY)
