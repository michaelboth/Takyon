Shows the basics of Takyon one-sided communication; i.e. one way read or one way write.
A one-sided transfer does not involve the remote side, but both endpoint's need to call
takyonCreate() in order for the endpoint to know the remote addresses.


Files
  - hello.c                 Core hello algorithm that is portable to one-sided capable Takyon Providers
  - hello.h                 Header file to prototype the hello() function
  - main_inter_processor.c  Start the hello() function as a single process
  - main_inter_thread.c     Start two hello() functions in separate threads
  - Makefile                Build for Linux and Mac
  - README.txt              This file
  - windows.Makefile        Build for Windows


Mac and Linux
  First build the Takyon library: see lib/README.txt
  Build app:
    > make [DEBUG=Yes] [MMAP=Yes] [RDMA=Yes] [CUDA=Yes]
  Testing
    Inter-Thread
      > ./hello_mt "InterThreadRC -pathID=1" 10
    Inter-Process
      A> ./hello_mp A "InterProcessRC -pathID=1" 10
      B> ./hello_mp B "InterProcessRC -pathID=1" 10
    RDMA RC (Reliable Connected)
      A> ./hello_mp A "RdmaRC -client -remoteIP=192.168.50.234 -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" 10
      B> ./hello_mp B "RdmaRC -server -localIP=192.168.50.234 -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" 10
  Clean:
    > make clean


Windows
  First build the Takyon library: see lib/README.txt
  Build app:
    > nmake -f windows.Makefile [DEBUG=Yes] [MMAP=Yes] [RDMA=Yes] [CUDA=Yes]
  Testing
    Same as linux but remove the './' before 'hello'
  Clean:
    > nmake clean
