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
    Inter-Thread (reliable)
      > ./throughput_mt "InterThreadRC -pathID=1"
      > ./throughput_mt "InterThreadRC -pathID=1" -write
      > ./throughput_mt "InterThreadRC -pathID=1" -read
    Inter-Thread (unreliable)
      > ./throughput_mt "InterThreadUC -pathID=1"
    Inter-Process (reliable)
      A> ./throughput_mp A "InterProcessRC -pathID=1"
      B> ./throughput_mp B "InterProcessRC -pathID=1"
      A> ./throughput_mp A "InterProcessRC -pathID=1" -write
      B> ./throughput_mp B "InterProcessRC -pathID=1" -write
      A> ./throughput_mp A "InterProcessRC -pathID=1" -read
      B> ./throughput_mp B "InterProcessRC -pathID=1" -read
    Inter-Process (unreliable)
      A> ./throughput_mp A "InterProcessUC -pathID=1"
      B> ./throughput_mp B "InterProcessUC -pathID=1"
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
      A> ./throughput_mp A "RdmaRC -client -remoteIP=192.168.50.234 -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1" -i=1000000 -bytes=40960 -sbufs=10 -dbufs=100
      B> ./throughput_mp B "RdmaRC -server -localIP=192.168.50.234 -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1" -i=1000000 -bytes=40960 -sbufs=10 -dbufs=100
      A> ./throughput_mp A "RdmaRC -client -remoteIP=192.168.50.234 -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1" -i=1000000 -bytes=40960 -sbufs=10 -dbufs=10 -write
      B> ./throughput_mp B "RdmaRC -server -localIP=192.168.50.234 -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1" -i=1000000 -bytes=40960 -sbufs=10 -dbufs=10 -write
      A> ./throughput_mp A "RdmaRC -client -remoteIP=192.168.50.234 -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1" -i=1000000 -bytes=40960 -sbufs=10 -dbufs=10 -read
      B> ./throughput_mp B "RdmaRC -server -localIP=192.168.50.234 -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1" -i=1000000 -bytes=40960 -sbufs=10 -dbufs=10 -read
    RDMA UC (Unreliable Connected): only one receiver, messages may be quietly dropped
      A> ./throughput_mp A "RdmaUC -client -remoteIP=192.168.50.234 -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1" -i=1000000 -bytes=40960 -sbufs=10 -dbufs=100
      B> ./throughput_mp B "RdmaUC -server -localIP=192.168.50.234 -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1" -i=1000000 -bytes=40960 -dbufs=100
    RDMA Unicast UD (Unreliable Datagram): only one receiver, messages may be quietly dropped
      A> ./throughput_mp A "RdmaUDUnicastSend -client -remoteIP=192.168.50.234 -port=23456 -rdmaDevice=mlx5_0 -rdmaPort=1" -i=10000000 -bytes=1024 -sbufs=100
      B> ./throughput_mp B "RdmaUDUnicastRecv -server -localIP=192.168.50.234 -port=23456 -reuse -rdmaDevice=mlx5_0 -rdmaPort=1" -i=10000000 -bytes=1064 -dbufs=1000       # Need 40 extra bytes for the RDMA GRH
    RDMA Multicast UD (Unreliable Datagram): one or more receivers, messages may be quietly dropped
      A> ./throughput_mp A "RdmaUDMulticastSend -localIP=192.168.50.234 -groupIP=233.23.33.56" -i=10000000 -bytes=1024 -sbufs=100
      B> ./throughput_mp B "RdmaUDMulticastRecv -localIP=192.168.50.234 -groupIP=233.23.33.56" -i=10000000 -bytes=1064 -dbufs=1000       # Need 40 extra bytes for the RDMA GRH
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
