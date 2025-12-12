This is a performance test that is designed to test two-sided (send->recv) socket and RDMA communications.

Files
  - Common.cpp                 A simple set of common functionality used by all test modes.
  - Common.hpp
  - LatencyTest.cpp            Self contained latency test. Makes it easy to see how Takyon communication is setup and used.
  - LatencyTest.hpp
  - Main.cpp                   Collects the command line arguments into a single data structure and then runs the appropriate test
  - Makefile                   Build, but only for Linux
  - provider_params.txt        Defines the Takyton provider parameters for each of Socket-TCP, Socket-UDP, RDMA-RC, RDMA-UC, and RDMA-UD
  - unikorn_instrumentation.h  Provides nanosecond timing analysis of the algorithm, which makes it much easier to know how to tune the algorithm


Steps to Run
  First build the Takyon library (see lib/README.txt for more details):
    For a debug build:
      > cd <takyon-folder>/lib
      > make DEBUG=Yes SocketTcp=Yes SocketUdp=Yes [Rdma=Yes] [CUDA=Yes]
    For a release build:
      > cd <takyon-folder>/lib
      > make SocketTcp=Yes SocketUdp=Yes [Rdma=Yes] [CUDA=Yes] DisableExtraErrorChecking=Yes

  Build app:
    > make [DEBUG=Yes] [RDMA=Yes] [CUDA=Yes]

  Running:
    To see the usage and options, run with -h or without any args
    Latency Testing
      Console 1:
        > ./rdma_vs_socket lat provider_params.txt RC A [opther-optional-args such as -poll]
      Console 2:
        > ./rdma_vs_socket lat provider_params.txt RC B [opther-optional-args such as -poll]
    Throughput Testing
      Console 1:
        > ./rdma_vs_socket tp provider_params.txt RC A [opther-optional-args]
      Console 2:
        > ./rdma_vs_socket tp provider_params.txt RC B [opther-optional-args]

  Clean:
    > make clean
