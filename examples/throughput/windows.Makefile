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

# MMAP
!IF "$(MMAP)" == "Yes"
MMAP_C_FLAGS = -DENABLE_MMAP
!ENDIF

# CUDA
!IF "$(CUDA)" == "Yes"
CUDA_HOME = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7
CUDA_C_FLAGS = -DENABLE_CUDA -I"$(CUDA_HOME)\include"
CUDA_C_LIBS = "$(CUDA_HOME)\lib\x64\cudart.lib"
!ENDIF

C_FLAGS = $(OPTIMIZATION_C_FLAGS) -nologo -WX -W3 -D_CRT_SECURE_NO_WARNINGS -I. -I../../inc -I../../src/utils $(MMAP_C_FLAGS) $(CUDA_C_FLAGS) $(THREAD_C_FLAGS)
LINK_FLAGS = -nologo -incremental:no -manifest:embed -subsystem:console
C_LIBS = ../../lib/takyon.lib $(CUDA_C_LIBS) $(THREAD_C_LIBS) Ws2_32.lib
TARGET_MT = throughput_mt.exe
TARGET_MP = throughput_mp.exe

.SUFFIXES: .c

all: $(TARGET_MT) $(TARGET_MP)

$(TARGET_MT): throughput.c main_inter_thread.c
	cl -c $(C_FLAGS) -Fothroughput_mt.obj throughput.c
	cl -c $(C_FLAGS) -Fo main_inter_thread.c
	link $(LINK_FLAGS) throughput_mt.obj main_inter_thread.obj $(C_LIBS) -out:$(TARGET_MT)

$(TARGET_MP): throughput.c main_inter_processor.c
	cl -c $(C_FLAGS) -Fothroughput_mp.obj throughput.c
	cl -c $(C_FLAGS) -Fo main_inter_processor.c
	link $(LINK_FLAGS) throughput_mp.obj main_inter_processor.obj $(C_LIBS) -out:$(TARGET_MP)

clean:
	-del $(TARGET_MT)
	-del $(TARGET_MP)
	-del *.obj
	-del *.pdb
	-del *~
