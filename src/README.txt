Only compile the files that are needed to avoid cluttering your application

core:
  takyon.c             The core of Takyon; i.e. the glue between the application and the Takyon providers
  takyon_private.h     Defines opaque structure for provider pointers, plus some other general helper functionality


providers:
  Defining a new provider
    supported_providers.c      Define your custom provider here
    supported_providers.h      Function prototype to get a provider's capabilities
  The supported Providers:
    provider_InterProcess.*    InterProcessRC, InterProcessUC
    provider_InterThread.*     InterThreadRC, InterThreadUC
    provider_Rdma.*            RdmaRC, RdmaUC, RdmaUDUnicastSend, RdmaUDUnicastRecv
    provider_RdmaUDMulticast.* RdmaUDMulticastSend, RdmaUDMulticastRecv
    provider_SocketTcp.*       SocketTcp
    provider_SocketUdp.*       SocketUdpSend, SocketUdpRecv


utils:
  utils_arg_parser.*             Parse the Takyon Provider text
  utils_endian.*                 Endian check and swapping
  utils_inter_thread_manager.*   Provides ability for two threads to connect with each other to create a point-to-point path
  utils_ipc*                     Inter-process functionality for to create shared memory and exchange shared memory address between endpoints
  utils_rdma_verbs.*             RDMA verbs functionality used by RDMA providers
  utils_socket*                  Socket functionality to coordinate between endpoints; used by InterProcess Socket and RDMA providers
  utils_thread_cond_timed_wait.* A simple wrapper to use a conditional variable with a timeout
  utils_time*                    Sleep and clock time stamp functions
