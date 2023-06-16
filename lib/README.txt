This folder and makefiles are just a convenience for creating the Takyon library
instead of directly compiling the Takyon source code into your application.
Make sure to use a C99 or higher compiler.

Linux and Mac
  Required:
    - Linux: gcc and make, Mac: xCode
  Extras:
    - Linux: CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit
  Build:
    > cd takyon/lib
    Specification
      > make [DEBUG=Yes] [InterThread=Yes] [InterProcess=Yes] [SocketTcp=Yes] [SocketUdp=Yes] [RdmaUDMulticast=Yes] [Rdma=Yes] [CUDA=Yes] [DisableExtraErrorChecking=Yes]
    Example
      > make DEBUG=Yes InterThread=Yes InterProcess=Yes SocketTcp=Yes SocketUdp=Yes RdmaUDMulticast=Yes Rdma=Yes CUDA=Yes DisableExtraErrorChecking=Yes

  When linking the library into your app, add the following to the link line:
    -L../../lib -ltakyon -pthread


Windows
  Required:
    - Visual Studio 2019 or greater x64 console
    - pthreads4w is installed
      1. Get the source code from: https://sourceforge.net/projects/pthreads4w/
      2. Unzip, rename to 'pthreads4w' and put in the C:\ folder
      3. Start a Visual Studio x64 native shell
      > cd c:\pthreads4w
      > nmake VC VC-debug VC-static VC-static-debug install DESTROOT=.\install
  Extras:
    - CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit
  Build:
    > cd takyon\lib
    Specification
      > nmake -f windows.Makefile [DEBUG=Yes] [InterThread=Yes] [InterProcess=Yes] [SocketTcp=Yes] [SocketUdp=Yes] [CUDA=Yes] [DisableExtraErrorChecking=Yes]
    Example
      > nmake -f windows.Makefile DEBUG=Yes InterThread=Yes InterProcess=Yes SocketTcp=Yes SocketUdp=Yes CUDA=Yes DisableExtraErrorChecking=Yes

  When linking the library into your app add the following to the link line:
    OPTIMIZED: ../../lib/takyon.lib c:/pthreads4w/install/lib/libpthreadVC3.lib -nodefaultlib:LIBCMT.LIB
    DEBUG:     ../../lib/takyon.lib c:/pthreads4w/install/lib/libpthreadVC3d.lib -nodefaultlib:LIBCMT.LIB
