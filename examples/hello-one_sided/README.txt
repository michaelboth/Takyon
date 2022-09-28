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
    RDMA RC (reliable connected)
      A> ./hello_mp A "RdmaRC -client -remoteIP=192.168.50.234 -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" 10
      B> ./hello_mp B "RdmaRC -server -localIP=192.168.50.234 -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" 10
    RDMA UC (unreliable connected; messages may be quietly dropped)
      A> ./hello_mp A "RdmaUC -client -remoteIP=192.168.50.234 -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" 10
      B> ./hello_mp B "RdmaUC -server -localIP=192.168.50.234 -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" 10
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
