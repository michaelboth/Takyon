//     Copyright 2022 Michael Both
//     Licensed under the Apache License, Version 2.0 (the "License");
//     you may not use this file except in compliance with the License.
//     You may obtain a copy of the License at
//         http://www.apache.org/licenses/LICENSE-2.0
//     Unless required by applicable law or agreed to in writing, software
//     distributed under the License is distributed on an "AS IS" BASIS,
//     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//     See the License for the specific language governing permissions and
//     limitations under the License.

#ifndef _utils_rdma_verbs_h_
#define _utils_rdma_verbs_h_

#include "takyon.h"
#include <rdma/rdma_cma.h>

typedef enum {
  RDMA_PROTOCOL_UD_MULTICAST,
  RDMA_PROTOCOL_UD_UNICAST,
  RDMA_PROTOCOL_UC,
  RDMA_PROTOCOL_RC
} RdmaProtocol;

typedef struct {
  RdmaProtocol protocol;
  bool is_sender;
  struct rdma_event_channel *event_ch;
  struct rdma_cm_id *id;
  struct ibv_qp *qp;
  struct ibv_comp_channel *comp_ch;
  struct ibv_cq *cq;
  struct ibv_ah *multicast_ah;
  struct sockaddr multicast_addr;
  uint32_t multicast_qp_num;
  uint32_t multicast_qkey;
  unsigned int nevents_to_ack;
} RdmaEndpoint;

typedef struct {
  struct ibv_mr *mr;
} RdmaBuffer;

typedef struct {
  struct ibv_sge *sges; // Must be a contiguous array, size == attrs.max_sub_buffers_per_send_request
} RdmaSendRequest;

typedef struct {
  struct ibv_sge *sges; // Must be a contiguous array, size == attrs.max_sub_buffers_per_recv_request
  struct ibv_recv_wr recv_wr;
} RdmaRecvRequest;

#ifdef __cplusplus
extern "C"
{
#endif

extern RdmaEndpoint *rdmaCreateMulticastEndpoint(TakyonPath *path, const char *local_NIC_ip_addr, const char *multicast_group_ip_addr, bool is_sender,
                                          uint32_t max_send_wr, uint32_t max_recv_wr, uint32_t max_send_sges, uint32_t max_recv_sges,
                                          uint32_t recv_request_count, TakyonRecvRequest *recv_requests,
                                          double timeout_seconds, char *error_message, int max_error_message_chars);
extern bool rdmaDestroyEndpoint(TakyonPath *path, RdmaEndpoint *endpoint, double timeout_seconds, char *error_message, int max_error_message_chars);

extern bool rdmaPostRecvs(TakyonPath *path, RdmaEndpoint *endpoint, uint32_t request_count, TakyonRecvRequest *requests, char *error_message, int max_error_message_chars);
extern bool rdmaIsRecved(RdmaEndpoint *endpoint, TakyonRecvRequest *request, double timeout_seconds, bool *timed_out_ret, char *error_message, int max_error_message_chars, uint64_t *bytes_received_ret, uint32_t *piggy_back_message_ret);

extern bool rdmaStartSend(TakyonPath *path, RdmaEndpoint *endpoint, TakyonSendRequest *request, uint32_t piggy_back_message, double timeout_seconds, bool *timed_out_ret, char *error_message, int max_error_message_chars);
extern bool rdmaIsSent(RdmaEndpoint *endpoint, TakyonSendRequest *request, double timeout_seconds, bool *timed_out_ret, char *error_message, int max_error_message_chars);

#ifdef __cplusplus
}
#endif

#endif
