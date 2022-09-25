Show the max sustained throughput of a Takyon provider; supports both one-sided and two-sided.
  Usage:
    throughput_mt       "<provider>" [-h] [-n=<uint32>] [-b=<uint64>] [-s=<uint32>] [-r=<uint32>] [-e] [-o] [-v]
    throughput_mp <A|B> "<provider>" [-h] [-n=<uint32>] [-b=<uint64>] [-s=<uint32>] [-r=<uint32>] [-e] [-o] [-v]

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
    RDMA UD (multicast)
      A> ./throughput_mp A "RdmaUDMulticastSend -localIP=192.168.50.234 -groupIP=233.23.33.56" -n=1000000 -b=4096
      B> ./throughput_mp B "RdmaUDMulticastRecv -localIP=192.168.50.234 -groupIP=233.23.33.56" -n=1000000 -b=4136       # Need 40 extra bytes for the RDMA GRH
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
