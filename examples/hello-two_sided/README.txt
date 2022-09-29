Shows the basics of Takyon two-sided communication; i.e. a combination of send/recv.
This touches on most of Takyon's features.

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
    SocketTcp (local socket, run A and B on same CPU)
      A> ./hello_mp A "SocketTcp -local -pathID=1" 10
      B> ./hello_mp B "SocketTcp -local -pathID=1" 10
    SocketTcp (different CPUs, user defined port number)
      A> ./hello_mp A "SocketTcp -client -remoteIP=127.0.0.1 -port=23456" 10
      B> ./hello_mp B "SocketTcp -server -localIP=127.0.0.1 -port=23456 -reuse" 10
    SocketTcp (different CPUs, OS implicitly determines port number)
      A> ./hello_mp A "SocketTcp -client -remoteIP=127.0.0.1 -ephemeralID=1" 10
      B> ./hello_mp B "SocketTcp -server -localIP=127.0.0.1 -ephemeralID=1" 10
    SocketUdp (unicast)
      A> ./hello_mp A "SocketUdpSend -unicast -remoteIP=127.0.0.1 -port=23456" 10
      B> ./hello_mp B "SocketUdpRecv -unicast -localIP=127.0.0.1 -port=23456 -reuse" 10
    SocketUdp (multicast)
      A> ./hello_mp A "SocketUdpSend -multicast -localIP=127.0.0.1 -groupIP=233.23.33.56 -port=23456" 10
      B> ./hello_mp B "SocketUdpRecv -multicast -localIP=127.0.0.1 -groupIP=233.23.33.56 -port=23456 -reuse" 10
    RDMA RC (reliable connected)
      A> ./hello_mp A "RdmaRC -client -remoteIP=192.168.50.234 -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" 10
      B> ./hello_mp B "RdmaRC -server -localIP=192.168.50.234 -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" 10
    RDMA UC (unreliable connected; messages may be quietly dropped)
      A> ./hello_mp A "RdmaUC -client -remoteIP=192.168.50.234 -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" 50
      B> ./hello_mp B "RdmaUC -server -localIP=192.168.50.234 -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" 25
    RDMA UD unicast (unreliable; messages may be quietly dropped)
      A> ./hello_mp A "RdmaUDUnicastSend -client -remoteIP=192.168.50.234 -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" 50
      B> ./hello_mp B "RdmaUDUnicastRecv -server -localIP=192.168.50.234 -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" 25
    RDMA UD (multicast)
      A> ./hello_mp A "RdmaUDMulticastSend -localIP=192.168.50.234 -groupIP=233.23.33.56" 50
      B> ./hello_mp B "RdmaUDMulticastRecv -localIP=192.168.50.234 -groupIP=233.23.33.56" 25

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
