// Copyright 2022 Michael Both
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "supported_providers.h"
#include <string.h>
#ifdef ENABLE_InterThread
  #include "provider_InterThread.h"
#endif
#ifdef ENABLE_InterProcess
  #include "provider_InterProcess.h"
#endif
#ifdef ENABLE_SocketTcp
  #include "provider_SocketTcp.h"
#endif
#ifdef ENABLE_SocketUdp
  #include "provider_SocketUdp.h"
#endif
#ifdef ENABLE_RdmaUDMulticast
  #include "provider_RdmaUDMulticast.h"
#endif
#ifdef ENABLE_RdmaUC
  #include "provider_RdmaUC.h"
#endif

typedef struct {
  // Name
  char name[TAKYON_MAX_PROVIDER_CHARS];

  // Functions
  bool (*create)(TakyonPath *path, uint32_t post_recv_count, TakyonRecvRequest *recv_requests, double timeout_seconds);
  bool (*destroy)(TakyonPath *path, double timeout_seconds);
  bool (*oneSided)(TakyonPath *path, TakyonOneSidedRequest *request, double timeout_seconds, bool *timed_out_ret);
  bool (*isOneSidedDone)(TakyonPath *path, TakyonOneSidedRequest *request, double timeout_seconds, bool *timed_out_ret);
  bool (*send)(TakyonPath *path, TakyonSendRequest *request, uint32_t piggy_back_message, double timeout_seconds, bool *timed_out_ret);
  bool (*isSent)(TakyonPath *path, TakyonSendRequest *request, double timeout_seconds, bool *timed_out_ret);
  bool (*postRecvs)(TakyonPath *path, uint32_t request_count, TakyonRecvRequest *requests);
  bool (*isRecved)(TakyonPath *path, TakyonRecvRequest *request, double timeout_seconds, bool *timed_out_ret, uint64_t *bytes_received_ret, uint32_t *piggy_back_message_ret);

  // Capabilities
  bool piggy_back_messages_supported;  // True if comm allows sending a 32bit message piggy backed on the primary message
  bool multi_sub_buffers_supported;    // True if more than one sub buffer can be in a single transfer
  bool zero_byte_messages_supported;   // True if can send zero byte messages
} CommInterface;

static CommInterface L_interfaces[] = {
#ifdef ENABLE_InterThread
                                       { .name = "InterThread",
                                         .create = interThreadCreate,
                                         .destroy = interThreadDestroy,
                                         .oneSided = interThreadOneSided,
                                         .isOneSidedDone = NULL,
                                         .send = interThreadSend,
                                         .isSent = NULL,
                                         .postRecvs = interThreadPostRecvs,
                                         .isRecved = interThreadIsRecved,
                                         .piggy_back_messages_supported = true,
                                         .multi_sub_buffers_supported = true,
                                         .zero_byte_messages_supported = true
                                       },
#endif

#ifdef ENABLE_InterProcess
                                       { .name = "InterProcess",
                                         .create = interProcessCreate,
                                         .destroy = interProcessDestroy,
                                         .oneSided = interProcessOneSided,
                                         .isOneSidedDone = NULL,
                                         .send = interProcessSend,
                                         .isSent = NULL,
                                         .postRecvs = interProcessPostRecvs,
                                         .isRecved = interProcessIsRecved,
                                         .piggy_back_messages_supported = true,
                                         .multi_sub_buffers_supported = true,
                                         .zero_byte_messages_supported = true
                                       },
#endif

#ifdef ENABLE_SocketTcp
                                       { .name = "SocketTcp",
                                         .create = tcpSocketCreate,
                                         .destroy = tcpSocketDestroy,
                                         .oneSided = NULL,
                                         .isOneSidedDone = NULL,
                                         .send = tcpSocketSend,
                                         .isSent = NULL,
                                         .postRecvs = NULL,
                                         .isRecved = tcpSocketIsRecved,
                                         .piggy_back_messages_supported = true,
                                         .multi_sub_buffers_supported = true,
                                         .zero_byte_messages_supported = true
                                       },
#endif

#ifdef ENABLE_SocketUdp
                                       { .name = "SocketUdpSend",
                                         .create = udpSocketCreate,
                                         .destroy = udpSocketDestroy,
                                         .oneSided = NULL,
                                         .isOneSidedDone = NULL,
                                         .send = udpSocketSend,
                                         .isSent = NULL,
                                         .postRecvs = NULL,
                                         .isRecved = NULL,
                                         .piggy_back_messages_supported = false,
                                         .multi_sub_buffers_supported = false,
                                         .zero_byte_messages_supported = false
                                       },
                                       { .name = "SocketUdpRecv",
                                         .create = udpSocketCreate,
                                         .destroy = udpSocketDestroy,
                                         .oneSided = NULL,
                                         .isOneSidedDone = NULL,
                                         .send = NULL,
                                         .isSent = NULL,
                                         .postRecvs = NULL,
                                         .isRecved = udpSocketIsRecved,
                                         .piggy_back_messages_supported = false,
                                         .multi_sub_buffers_supported = false,
                                         .zero_byte_messages_supported = false
                                       },
#endif

#ifdef ENABLE_RdmaUDMulticast
                                       { .name = "RdmaUDMulticastSend",
                                         .create = rdmaUDMulticastCreate,
                                         .destroy = rdmaUDMulticastDestroy,
                                         .oneSided = NULL,
                                         .isOneSidedDone = NULL,
                                         .send = rdmaUDMulticastSend,
                                         .isSent = rdmaUDMulticastIsSent,
                                         .postRecvs = NULL,
                                         .isRecved = NULL,
                                         .piggy_back_messages_supported = true,
                                         .multi_sub_buffers_supported = true,
                                         .zero_byte_messages_supported = true
                                       },
                                       { .name = "RdmaUDMulticastRecv",
                                         .create = rdmaUDMulticastCreate,
                                         .destroy = rdmaUDMulticastDestroy,
                                         .oneSided = NULL,
                                         .isOneSidedDone = NULL,
                                         .send = NULL,
                                         .isSent = NULL,
                                         .postRecvs = rdmaUDMulticastPostRecvs,
                                         .isRecved = rdmaUDMulticastIsRecved,
                                         .piggy_back_messages_supported = true,
                                         .multi_sub_buffers_supported = true,
                                         .zero_byte_messages_supported = true
                                       },
#endif

#ifdef ENABLE_RdmaUC
                                       { .name = "RdmaUC",
                                         .create = rdmaUCCreate,
                                         .destroy = rdmaUCDestroy,
                                         .oneSided = rdmaUCOneSided,
                                         .isOneSidedDone = rdmaUCIsOneSidedDone,
                                         .send = rdmaUCSend,
                                         .isSent = rdmaUCIsSent,
                                         .postRecvs = rdmaUCPostRecvs,
                                         .isRecved = rdmaUCIsRecved,
                                         .piggy_back_messages_supported = true,
                                         .multi_sub_buffers_supported = true,
                                         .zero_byte_messages_supported = true
                                       },
#endif
};

bool setProviderFunctionsAndCapabilities(const char *provider_name, TakyonComm *comm, TakyonPathCapabilities *capabilities) {
  uint64_t num_interfaces = (sizeof(L_interfaces) / sizeof(CommInterface));
  for (uint64_t i=0; i<num_interfaces; i++) {
    CommInterface *interface = &L_interfaces[i];
    if (strcmp(interface->name, provider_name) == 0) {
      // Functions
      comm->create         = interface->create;
      comm->destroy        = interface->destroy;
      comm->oneSided       = interface->oneSided;
      comm->isOneSidedDone = interface->isOneSidedDone;
      comm->send           = interface->send;
      comm->isSent         = interface->isSent;
      comm->postRecvs      = interface->postRecvs;
      comm->isRecved       = interface->isRecved;

      // Supported functions and capabilities
      capabilities->OneSided_supported            = interface->create != NULL;
      capabilities->IsOneSidedDone_supported      = interface->destroy != NULL;
      capabilities->OneSided_supported            = interface->oneSided != NULL;
      capabilities->IsOneSidedDone_supported      = interface->isOneSidedDone != NULL;
      capabilities->Send_supported                = interface->send != NULL;
      capabilities->IsSent_supported              = interface->isSent != NULL;
      capabilities->PostRecvs_supported           = interface->postRecvs != NULL;
      capabilities->IsRecved_supported            = interface->isRecved != NULL;
      capabilities->piggy_back_messages_supported = interface->piggy_back_messages_supported;
      capabilities->multi_sub_buffers_supported   = interface->multi_sub_buffers_supported;
      capabilities->zero_byte_messages_supported  = interface->zero_byte_messages_supported;

      return true;
    }
  }
  return false;
}
