Show the max sustained throughput of a Takyon provider; supports both one-sided and two-sided.
  To see the usage and options, run with -h


Files
  - main_inter_processor.c  Start the throughput() function as a single process
  - main_inter_thread.c     Start two throughput() functions in separate threads
  - Makefile                Build for Linux and Mac
  - README.txt              This file
  - throughput.c            Core throughput algorithm that is portable to all Takyon Providers
  - throughput.h            Header file to prototype the throughput() function
  - windows.Makefile        Build for Windows


Mac and Linux
  First build the Takyon library: see lib/README.txt
  Build app:
    > make [DEBUG=Yes] [MMAP=Yes] [RDMA=Yes] [CUDA=Yes]
  Testing
    Inter-Thread
      > ./throughput_mt "InterThread -pathID=1"
      > ./throughput_mt "InterThread -pathID=1" -o
    Inter-Process
      A> ./throughput_mp A "InterProcess -pathID=1"
      B> ./throughput_mp B "InterProcess -pathID=1"
      A> ./throughput_mp A "InterProcess -pathID=1" -o
      B> ./throughput_mp B "InterProcess -pathID=1" -o
    Local Socket: avoids full IP stack since runs in the same OS instance
      A> ./throughput_mp A "SocketTcp -local -pathID=1"
      B> ./throughput_mp B "SocketTcp -local -pathID=1"
    TCP Socket, user defined port number
      A> ./throughput_mp A "SocketTcp -client -remoteIP=127.0.0.1 -port=23456"
      B> ./throughput_mp B "SocketTcp -server -localIP=127.0.0.1 -port=23456 -reuse"
    TCP Socket, OS implicitly determines port number
      A> ./throughput_mp A "SocketTcp -client -remoteIP=127.0.0.1 -ephemeralID=1"
      B> ./throughput_mp B "SocketTcp -server -localIP=127.0.0.1 -ephemeralID=1"
    UDP Unicast Socket: only one receiver, messages may be quietly dropped
      A> ./throughput_mp A "SocketUdpSend -unicast -remoteIP=127.0.0.1 -port=23456"
      B> ./throughput_mp B "SocketUdpRecv -unicast -localIP=127.0.0.1 -port=23456 -reuse"
    UDP Multicast Socket: one or more receivers, messages may be quietly dropped
      A> ./throughput_mp A "SocketUdpSend -multicast -localIP=127.0.0.1 -groupIP=233.23.33.56 -port=23456"
      B> ./throughput_mp B "SocketUdpRecv -multicast -localIP=127.0.0.1 -groupIP=233.23.33.56 -port=23456 -reuse"
    RDMA RC (Reliable Connected)
      A> ./throughput_mp A "RdmaRC -client -remoteIP=192.168.50.234 -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" -n=1000000 -b=40960 -s=10 -r=100
      B> ./throughput_mp B "RdmaRC -server -localIP=192.168.50.234 -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" -n=1000000 -b=40960 -r=100
      A> ./throughput_mp A "RdmaRC -client -remoteIP=192.168.50.234 -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" -n=1000000 -b=40960 -s=10 -r=100 -o
      B> ./throughput_mp B "RdmaRC -server -localIP=192.168.50.234 -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" -n=1000000 -b=40960 -r=100 -o
    RDMA UC (Unreliable Connected): only one receiver, messages may be quietly dropped
      A> ./throughput_mp A "RdmaUC -client -remoteIP=192.168.50.234 -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" -n=1000000 -b=40960 -s=10 -r=100
      B> ./throughput_mp B "RdmaUC -server -localIP=192.168.50.234 -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" -n=1000000 -b=40960 -r=100
    RDMA Unicast UD (Unreliable Datagram): only one receiver, messages may be quietly dropped
      A> ./throughput_mp A "RdmaUDUnicastSend -client -remoteIP=192.168.50.234 -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" -n=10000000 -b=1024 -s=100
      B> ./throughput_mp B "RdmaUDUnicastRecv -server -localIP=192.168.50.234 -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1 -gidIndex=0" -n=10000000 -b=1064 -r=1000       # Need 40 extra bytes for the RDMA GRH
    RDMA Multicast UD (Unreliable Datagram): one or more receivers, messages may be quietly dropped
      A> ./throughput_mp A "RdmaUDMulticastSend -localIP=192.168.50.234 -groupIP=233.23.33.56" -n=10000000 -b=1024 -s=100
      B> ./throughput_mp B "RdmaUDMulticastRecv -localIP=192.168.50.234 -groupIP=233.23.33.56" -n=10000000 -b=1064 -r=1000       # Need 40 extra bytes for the RDMA GRH
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
