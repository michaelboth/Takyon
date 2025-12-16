This is a performance test that is designed to test two-sided (send->recv) point-to-point socket and RDMA communications.

NOTES:
  - Suported providers:
     - Socket-TCP
     - RDMA-RC (reliable connected)
     - RDMA-UC (unreliable connected, messages could be dropped and cause the app to exit)
  - If RDMA is not available, this app can be built just for sockets.
  - This app supports CUDA for GPU processing. It also supports integrated GPUs and discrete GPUs. Zero copy tranfers to GPU memory are supported if using GPUDirect or on an integrated GPU.


Files
  - Main.cpp                   Collects the command line arguments into a single data structure and then runs the appropriate test
  - LatencyTest.cpp            Self contained latency test. Makes it easy to see how Takyon communication is setup and used.
  - LatencyTest.hpp
  - ThroughputTest.cpp         Self contained throughput test. Makes it easy to see how Takyon communication is setup and used.
  - ThroughputTest.hpp
  - Common.cpp                 A simple set of common functionality used by all test modes.
  - Common.hpp
  - Validation.cpp             Convenience functions used by all test moves to validate the messages being sent/received
  - Validation.hpp
  - ValidationKernels.cu       CUDA kernels to support validation if processing is set to run on the GPU. Kernels are used by Validate.cpp
  - ValidationKernels.hpp
  - unikorn_instrumentation.h  Provides nanosecond timing analysis of the algorithm, which makes it much easier to know how to tune the algorithm
  - Makefile                   Build for Linux and Mac. Mac only supports sockets.
  - windows.Makefile           Windows build, but only supports sockets
  - provider_params.txt        Defines the Takyton provider parameters for each of Socket-TCP, RDMA-RC, and RDMA-UC


Steps to Run (Linux and Mac):
  First build the Takyon library (see lib/README.txt for more details):
    For a debug build:
      > cd <takyon-folder>/lib
      > make DEBUG=Yes SocketTcp=Yes [Rdma=Yes] [CUDA=Yes]
    For a release build:
      > cd <takyon-folder>/lib
      > make SocketTcp=Yes [Rdma=Yes] [CUDA=Yes] DisableExtraErrorChecking=Yes

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


Steps to Run (Windows):
  First build the Takyon library for sockets (see lib/README.txt for more details):
  Build app:
    > nmake -f windows.Makefile [DEBUG=Yes] [CUDA=Yes]
  Testing
    Same as linux but remove the './' before 'throughput'
  Clean:
    > nmake clean
