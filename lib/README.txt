This folder and makefiles are just a convenience for creating the Takyon library
instead of directly compiling the Takyon source code into your application.

Linux and Mac
  Build:
    > cd takyon/lib
    Specification
      > make [DEBUG=Yes] [InterThread=Yes] [InterProcess=Yes] [TcpSocket=Yes] [UdpSocket=Yes] [CUDA=Yes]
    Example
      > make DEBUG=Yes InterThread=Yes InterProcess=Yes TcpSocket=Yes UdpSocket=Yes CUDA=Yes

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
  Build:
    > cd takyon\lib
    Specification
      > nmake -f windows.Makefile [DEBUG=Yes] [InterThread=Yes] [InterProcess=Yes] [TcpSocket=Yes] [UdpSocket=Yes] [CUDA=Yes]
    Example
      > nmake -f windows.Makefile DEBUG=Yes InterThread=Yes InterProcess=Yes TcpSocket=Yes UdpSocket=Yes CUDA=Yes

  When linking the library into your app add the following to the link line:
    ../../lib/takyon.lib -nodefaultlib:MSVCRTD.LIB c:/pthreads4w/install/lib/libpthreadVC3.lib -nodefaultlib:LIBCMT.LIB