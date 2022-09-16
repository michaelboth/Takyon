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
    TcpSocket (local socket, run A and B on same CPU)
      A> ./hello_mp A "TcpSocket -local -pathID=1" 10
      B> ./hello_mp B "TcpSocket -local -pathID=1" 10
    TcpSocket (different CPUs, user defined port number)
      A> ./hello_mp A "TcpSocket -client -remoteIP=127.0.0.1 -port=23456" 10
      B> ./hello_mp B "TcpSocket -server -localIP=127.0.0.1 -port=23456 -reuse" 10
    TcpSocket (different CPUs, OS implicitly determines port number)
      A> ./hello_mp A "TcpSocket -client -remoteIP=127.0.0.1 -pathID=1" 10
      B> ./hello_mp B "TcpSocket -server -localIP=127.0.0.1 -pathID=1" 10
    UdpSocket (unicast)
      A> ./hello_mp A "UdpSocketSend -unicast -remoteIP=127.0.0.1 -port=23456" 10
      B> ./hello_mp B "UdpSocketRecv -unicast -localIP=127.0.0.1 -port=23456 -reuse" 10
    UdpSocket (multicast)
      A> ./hello_mp A "UdpSocketSend -multicast -localIP=127.0.0.1 -groupIP=233.23.33.56 -port=23456" 10
      B> ./hello_mp B "UdpSocketRecv -multicast -localIP=127.0.0.1 -groupIP=233.23.33.56 -port=23456 -reuse" 10
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
