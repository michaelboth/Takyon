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

#include "supported_interconnects.h"
#include <string.h>
#ifdef ENABLE_InterThread
  #include "interconnect_InterThread.h"
#endif
#ifdef ENABLE_InterProcess
  #include "interconnect_InterProcess.h"
#endif
#ifdef ENABLE_TcpSocket
  #include "interconnect_TcpSocket.h"
#endif
#ifdef ENABLE_UdpSocket
  #include "interconnect_UdpSocket.h"
#endif

typedef struct {
  // Name
  char name[TAKYON_MAX_INTERCONNECT_CHARS];

  // Functions
  bool (*create)(TakyonPath *path, uint32_t post_recv_count, TakyonRecvRequest *recv_requests, double timeout_seconds);
  bool (*destroy)(TakyonPath *path, double timeout_seconds);
  bool (*oneSided)(TakyonPath *path, TakyonOneSidedRequest *request, double timeout_seconds, bool *timed_out_ret);
  bool (*isOneSidedDone)(TakyonPath *path, TakyonOneSidedRequest *request, double timeout_seconds, bool *timed_out_ret);
  bool (*send)(TakyonPath *path, TakyonSendRequest *request, uint32_t piggy_back_message, double timeout_seconds, bool *timed_out_ret);
  bool (*isSent)(TakyonPath *path, TakyonSendRequest *request, double timeout_seconds, bool *timed_out_ret);
  bool (*postRecvs)(TakyonPath *path, uint32_t request_count, TakyonRecvRequest *requests);
  bool (*isRecved)(TakyonPath *path, TakyonRecvRequest *request, double timeout_seconds, bool *timed_out_ret, uint64_t *bytes_received_ret, uint32_t *piggy_back_message_ret);

  // Features
  bool piggy_back_message_supported;      // True if comm allows sending a 32bit message piggy backed on the primary message
  bool multi_sub_buffers_supported;       // True if more than one sub buffer can be in a single transfer
  bool zero_byte_message_supported;       // True if can send zero byte messages
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
                                         .piggy_back_message_supported = true,
                                         .multi_sub_buffers_supported = true,
                                         .zero_byte_message_supported = true
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
                                         .piggy_back_message_supported = true,
                                         .multi_sub_buffers_supported = true,
                                         .zero_byte_message_supported = true
                                       },
#endif
#ifdef ENABLE_TcpSocket
                                       { .name = "TcpSocket",
                                         .create = tcpSocketCreate,
                                         .destroy = tcpSocketDestroy,
                                         .oneSided = NULL,
                                         .isOneSidedDone = NULL,
                                         .send = tcpSocketSend,
                                         .isSent = NULL,
                                         .postRecvs = NULL,
                                         .isRecved = tcpSocketIsRecved,
                                         .piggy_back_message_supported = true,
                                         .multi_sub_buffers_supported = true,
                                         .zero_byte_message_supported = true
                                       },
#endif
#ifdef ENABLE_UdpSocket
                                       { .name = "UdpSocketSend",
                                         .create = udpSocketCreate,
                                         .destroy = udpSocketDestroy,
                                         .oneSided = NULL,
                                         .isOneSidedDone = NULL,
                                         .send = udpSocketSend,
                                         .isSent = NULL,
                                         .postRecvs = NULL,
                                         .isRecved = NULL,
                                         .piggy_back_message_supported = false,
                                         .multi_sub_buffers_supported = false,
                                         .zero_byte_message_supported = false
                                       },
                                       { .name = "UdpSocketRecv",
                                         .create = udpSocketCreate,
                                         .destroy = udpSocketDestroy,
                                         .oneSided = NULL,
                                         .isOneSidedDone = NULL,
                                         .send = NULL,
                                         .isSent = NULL,
                                         .postRecvs = NULL,
                                         .isRecved = udpSocketIsRecved,
                                         .piggy_back_message_supported = false,
                                         .multi_sub_buffers_supported = false,
                                         .zero_byte_message_supported = false
                                       },
#endif
};

bool setInterconnectFunctionsAndFeatures(const char *interconnect_name, TakyonComm *comm, TakyonPathFeatures *features) {
  uint64_t num_interfaces = (sizeof(L_interfaces) / sizeof(CommInterface));
  for (uint64_t i=0; i<num_interfaces; i++) {
    CommInterface *interface = &L_interfaces[i];
    if (strcmp(interface->name, interconnect_name) == 0) {
      // Functions
      comm->create         = interface->create;
      comm->destroy        = interface->destroy;
      comm->oneSided       = interface->oneSided;
      comm->isOneSidedDone = interface->isOneSidedDone;
      comm->send           = interface->send;
      comm->isSent         = interface->isSent;
      comm->postRecvs      = interface->postRecvs;
      comm->isRecved       = interface->isRecved;

      // Supported functions and features
      features->OneSided_supported           = interface->create != NULL;
      features->IsOneSidedDone_supported     = interface->destroy != NULL;
      features->OneSided_supported           = interface->oneSided != NULL;
      features->IsOneSidedDone_supported     = interface->isOneSidedDone != NULL;
      features->Send_supported               = interface->send != NULL;
      features->IsSent_supported             = interface->isSent != NULL;
      features->PostRecvs_supported          = interface->postRecvs != NULL;
      features->IsRecved_supported           = interface->isRecved != NULL;
      features->piggy_back_message_supported = interface->piggy_back_message_supported;
      features->multi_sub_buffers_supported  = interface->multi_sub_buffers_supported;
      features->zero_byte_message_supported  = interface->zero_byte_message_supported;

      return true;
    }
  }
  return false;
}
