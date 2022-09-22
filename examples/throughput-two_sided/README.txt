Show the max sustained throughput of a Takyon two-sided provider; i.e. a combination of send/recv.

Mac and Linux
  First build the Takyon library: see lib/README.txt
  Build app:
    > make [DEBUG=Yes] [MMAP=Yes] [RDMA=Yes] [CUDA=Yes]
  Testing
    InterThread
      > ./throughput_mt "InterThread -pathID=1"
    InterProcess
      A> ./throughput_mp A "InterProcess -pathID=1"
      B> ./throughput_mp B "InterProcess -pathID=1"
    SocketTcp (local socket, run A and B on same CPU)
      A> ./throughput_mp A "SocketTcp -local -pathID=1"
      B> ./throughput_mp B "SocketTcp -local -pathID=1"
    SocketTcp (different CPUs, user defined port number)
      A> ./throughput_mp A "SocketTcp -client -remoteIP=127.0.0.1 -port=23456"
      B> ./throughput_mp B "SocketTcp -server -localIP=127.0.0.1 -port=23456 -reuse"
    SocketTcp (different CPUs, OS implicitly determines port number)
      A> ./throughput_mp A "SocketTcp -client -remoteIP=127.0.0.1 -ephemeralID=1"
      B> ./throughput_mp B "SocketTcp -server -localIP=127.0.0.1 -ephemeralID=1"
    SocketUdp (unicast)
      A> ./throughput_mp A "SocketUdpSend -unicast -remoteIP=127.0.0.1 -port=23456"
      B> ./throughput_mp B "SocketUdpRecv -unicast -localIP=127.0.0.1 -port=23456 -reuse"
    SocketUdp (multicast)
      A> ./throughput_mp A "SocketUdpSend -multicast -localIP=127.0.0.1 -groupIP=233.23.33.56 -port=23456"
      B> ./throughput_mp B "SocketUdpRecv -multicast -localIP=127.0.0.1 -groupIP=233.23.33.56 -port=23456 -reuse"
  Clean:
    > make clean

Windows
  First build the Takyon library: see lib/README.txt
  Build app:
    > nmake -f windows.Makefile [DEBUG=Yes] [MMAP=Yes] [RDMA=Yes] [CUDA=Yes]
  Testing
    Same as linux but remove the './' before 'throughput'
  Clean:
    > nmake clean
