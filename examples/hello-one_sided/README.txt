Shows the basics of Takyon one-sided communication; i.e. one way read or one way write.
A one-sided transfer does not involve the remote side, but both endpoint's need to call
takyonCreate() in order for the endpoint to know the remote addresses.

Mac and Linux
  First build the Takyon library: see lib/README.txt
  Build app:
    > make [DEBUG=Yes] [MMAP=Yes] [RDMA=Yes] [CUDA=Yes]
  Testing
    InterThread
      > ./hello_mt "InterThread -pathID=1" 10
    InterProcess
      A> ./hello_mp A "InterProcess -pathID=1" 10
      B> ./hello_mp B "InterProcess -pathID=1" 10
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
