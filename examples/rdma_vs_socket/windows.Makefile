# Check if release distribution is enabled
!IF "$(DEBUG)" == "Yes"
OPTIMIZATION_C_FLAGS  = -Zi -MDd # Debug: -MTd or -MDd
!ELSE
OPTIMIZATION_C_FLAGS  = -O2 -MD # Release: -MT means static linking, and -MD means dynamic linking.
!ENDIF

# pthreads
THREAD_C_FLAGS = -Ic:/pthreads4w/install/include
!IF "$(DEBUG)" == "Yes"
THREAD_C_LIBS = c:/pthreads4w/install/lib/libpthreadVC3d.lib -nodefaultlib:LIBCMT.LIB
!ELSE
THREAD_C_LIBS = c:/pthreads4w/install/lib/libpthreadVC3.lib -nodefaultlib:LIBCMT.LIB
!ENDIF

# Takyon Libs
TAKYON_C_LIBS = ../../lib/takyon.lib

# CUDA
!IF "$(CUDA)" == "Yes"
CUDA_HOME = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7
CUDA_C_FLAGS = -DENABLE_CUDA -I"$(CUDA_HOME)\include"
CUDA_C_LIBS = "$(CUDA_HOME)\lib\x64\cudart.lib"
!ENDIF

# Unikorn
UNIKORN_C_FLAGS =
UNIKORN_C_OBJS =
!IF "$(UNIKORN)" == "Yes"
UNIKORN_HOME   = ../../../EventAnalyzer
UNIKORN_C_FLAGS = -DENABLE_UNIKORN_RECORDING -DUNIKORN_RELEASE_BUILD -I$(UNIKORN_HOME)/inc
UNIKORN_C_OBJS = unikorn.obj unikorn_file_flush.obj unikorn_clock_queryperformancecounter.obj
!ENDIF

C_FLAGS    = $(OPTIMIZATION_C_FLAGS) -nologo -WX -W3 -D_CRT_SECURE_NO_WARNINGS -I. -I../../inc $(CUDA_C_FLAGS) $(THREAD_C_FLAGS) $(UNIKORN_C_FLAGS)
CPP_FLAGS  = $(OPTIMIZATION_C_FLAGS) -nologo -WX -W3 -D_CRT_SECURE_NO_WARNINGS -std:c++latest -EHsc -I. -I../../inc $(CUDA_C_FLAGS) $(THREAD_C_FLAGS) $(UNIKORN_C_FLAGS)
LINK_FLAGS = -nologo -incremental:no -manifest:embed -subsystem:console
C_LIBS     = ../../lib/takyon.lib $(CUDA_C_LIBS) $(THREAD_C_LIBS) Ws2_32.lib
CPP_OBJS   = .\Common.obj .\LatencyTest.obj .\Main.obj .\ThroughputTest.obj .\Validation.obj

TARGET = rdma_vs_socket.exe

.SUFFIXES: .c .cpp

all: $(TARGET)

{..\..\..\EventAnalyzer\src}.c{}.obj::
	cl -c $(C_FLAGS) -Fo $<

{.\}.cpp{.\}.obj::
	cl -c $(CPP_FLAGS) -Fo.\ $<

$(TARGET): $(UNIKORN_C_OBJS) $(CPP_OBJS)
	link $(LINK_FLAGS) $(UNIKORN_C_OBJS) $(CPP_OBJS) $(C_LIBS) -out:$(TARGET) 

clean:
	-del $(TARGET)
	-del *.obj
	-del *.pdb
	-del *.events
	-del *~
