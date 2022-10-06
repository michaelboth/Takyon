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
  uint32_t mtu_bytes;          //          RDMA MTU is less than network MTU. Default is detect min at endpoints (does not detect min in intermediate switches. Valid values are 256, 512, 1024, 2048, and 4096.
  uint8_t gid_index;           //          RDMA's global address index. Default is 0.
  uint8_t service_level;       //          Network quality of service level. Default is 0.
  uint8_t hop_limit;           //          Max routers to travel through. Default is 1.
  uint8_t retransmit_timeout;  // RC Only. Time to verify packet transmission (ACK or NACK). Default is 14. Range is [0 .. 31], for value meanings, see www.rdmamojo.com/2013/01/12/ibv_modify_qp/
  uint8_t retry_cnt;           // RC Only. Retransmit attempts (without a NACK) before erroring. Default is 7, and max is 7.
  uint8_t rnr_retry;           // RC Only. Retransmit attempts (due to NACKs) before erroring. Default is 6. 7 is the max, but means infinite.
  uint8_t min_rnr_timer;       // RC only. Incoming receive not ready. Default is 12. Range is [0 .. 31], for value meanings, see www.rdmamojo.com/2013/01/12/ibv_modify_qp/
} RdmaAppOptions;

typedef struct {
  RdmaProtocol protocol;
  bool is_sender;
  struct ibv_context *context; // Used if not used connection manager
  struct rdma_event_channel *event_ch;
  struct rdma_cm_id *id;
  struct ibv_pd *pd;
  struct ibv_qp *qp;
  struct ibv_comp_channel *send_comp_ch;
  struct ibv_comp_channel *recv_comp_ch;
  struct ibv_cq *send_cq;
  struct ibv_cq *recv_cq;
  unsigned int num_send_events_to_ack;
  unsigned int num_recv_events_to_ack;
  bool connection_broken;
  uint32_t mtu_bytes;
  // Multicast
  struct ibv_ah *multicast_ah;
  struct sockaddr multicast_addr;
  uint32_t multicast_qp_num;
  uint32_t multicast_qkey;
  // Unicast
  struct ibv_ah *unicast_sender_ah;
  uint32_t unicast_remote_qp_num;
  uint32_t unicast_remote_qkey;
} RdmaEndpoint;

typedef struct {
  TakyonPath *path;
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
extern RdmaEndpoint *rdmaCreateEndpoint(TakyonPath *path, bool is_endpointA, int read_pipe_fd, enum ibv_qp_type qp_type, bool is_UD_sender, const char *rdma_device_name, uint32_t rdma_port_id,
					uint32_t max_send_wr, uint32_t max_recv_wr, uint32_t max_send_sges, uint32_t max_recv_sges,
					uint32_t recv_request_count, TakyonRecvRequest *recv_requests,
					RdmaAppOptions app_options, double timeout_seconds, char *error_message, int max_error_message_chars);
extern bool rdmaDestroyEndpoint(TakyonPath *path, RdmaEndpoint *endpoint, char *error_message, int max_error_message_chars);

extern bool rdmaEndpointPostRecvs(TakyonPath *path, RdmaEndpoint *endpoint, uint32_t request_count, TakyonRecvRequest *requests, char *error_message, int max_error_message_chars);
extern bool rdmaEndpointIsRecved(RdmaEndpoint *endpoint, uint64_t expected_transfer_id, int read_pipe_fd, bool use_polling_completion, uint32_t usec_sleep_between_poll_attempts, double timeout_seconds, bool *timed_out_ret, char *error_message, int max_error_message_chars, uint64_t *bytes_received_ret, uint32_t *piggyback_message_ret);

extern bool rdmaEndpointStartSend(TakyonPath *path, RdmaEndpoint *endpoint, enum ibv_wr_opcode transfer_mode, uint64_t transfer_id, uint32_t sub_buffer_count, TakyonSubBuffer *sub_buffers, struct ibv_sge *sge_list, uint64_t remote_addr, uint32_t rkey, uint32_t piggyback_message, bool use_is_sent_notification, char *error_message, int max_error_message_chars);
extern bool rdmaEndpointIsSent(RdmaEndpoint *endpoint, uint64_t expected_transfer_id, enum ibv_wc_opcode expected_opcode, int read_pipe_fd, bool use_polling_completion, uint32_t usec_sleep_between_poll_attempts, double timeout_seconds, bool *timed_out_ret, char *error_message, int max_error_message_chars);

#ifdef __cplusplus
}
#endif

#endif
