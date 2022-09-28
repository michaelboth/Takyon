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

#include "provider_RdmaUDMulticast.h"
#include "takyon_private.h"
#include "utils_rdma_verbs.h"
#include "utils_arg_parser.h"
#include "utils_time.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Supported formats:
//   RDMA UD multicast: No connection between endpoints
//     - Data packets can be dropped, arrive out of order, or duplicated
//     - Max transfer size is RDMA MTU (up to 4KB)
//     - Useful where all bytes may not arrive (unreliable): e.g. live stream video or music
//   ---------------------------------------------------------------------------
//   Multicast: one sender, zero to many receivers
//     "RdmaUDMulticastSend -localIP=<ip_addr>|<hostname> -groupIP=<multicast_ip_addr>"
//     "RdmaUDMulticastRecv -localIP=<ip_addr>|<hostname> -groupIP=<multicast_ip_addr>"
//
//   Argument descriptions:
//     -groupIP=<multicast_ip_addr>: Valid multicast addresses: 224.0.0.0 through 239.255.255.255, but some are reserved

typedef struct {
  RdmaEndpoint *endpoint;
  uint32_t curr_rdma_request_index;
  RdmaSendRequest *rdma_send_request_list;
  RdmaRecvRequest *rdma_recv_request_list;
  RdmaBuffer *rdma_buffer_list;
  struct ibv_sge *sge_list;
} PrivateTakyonPath;

bool rdmaUDMulticastCreate(TakyonPath *path, uint32_t post_recv_count, TakyonRecvRequest *recv_requests, double timeout_seconds) {
  TakyonComm *comm = (TakyonComm *)path->private;
  char error_message[MAX_ERROR_MESSAGE_CHARS];

  // Get the name of the provider
  char provider_name[TAKYON_MAX_PROVIDER_CHARS];
  if (!argGetProvider(path->attrs.provider, provider_name, TAKYON_MAX_PROVIDER_CHARS, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to get provider name: %s\n", error_message);
    return false;
  }

  // Get all posible flags and values
  bool is_a_send = (strcmp(provider_name, "RdmaUDMulticastSend") == 0);
  bool is_a_recv = (strcmp(provider_name, "RdmaUDMulticastRecv") == 0);
  // -localIP=<ip_addr>|<hostname>|Any
  char local_ip_addr[TAKYON_MAX_PROVIDER_CHARS];
  bool local_ip_addr_found = false;
  bool ok = argGetText(path->attrs.provider, "-localIP=", local_ip_addr, TAKYON_MAX_PROVIDER_CHARS, &local_ip_addr_found, error_message, MAX_ERROR_MESSAGE_CHARS);
  if (!ok) {
    TAKYON_RECORD_ERROR(path->error_message, "provider argument -localIP=<ip_addr>|<hostname> is invalid: %s\n", error_message);
    return false;
  }
  if (!local_ip_addr_found) {
    TAKYON_RECORD_ERROR(path->error_message, "RdmaUDMulticast needs the argument: -localIP=<ip_addr>|<hostname>\n");
    return false;
  }
  // -groupIP=<multicast_ip_addr>
  char group_ip_addr[TAKYON_MAX_PROVIDER_CHARS];
  bool group_ip_addr_found = false;
  ok = argGetText(path->attrs.provider, "-groupIP=", group_ip_addr, TAKYON_MAX_PROVIDER_CHARS, &group_ip_addr_found, error_message, MAX_ERROR_MESSAGE_CHARS);
  if (!ok) {
    TAKYON_RECORD_ERROR(path->error_message, "provider argument -groupIP=<multicast_ip_addr> is invalid: %s\n", error_message);
    return false;
  }
  if (!group_ip_addr_found) {
    TAKYON_RECORD_ERROR(path->error_message, "RdmaUDMulticast needs the argument: -groupIP=<multicast_ip_addr>\n");
    return false;
  }
  {
    int tokens[4];
    int ntokens = sscanf(group_ip_addr, "%d.%d.%d.%d", &tokens[0], &tokens[1], &tokens[2], &tokens[3]);
    if (ntokens != 4 || tokens[0] < 224 || tokens[0] > 239 || tokens[1] < 0 || tokens[1] > 255 || tokens[2] < 0 || tokens[2] > 255 || tokens[3] < 0 || tokens[3] > 255) {
      TAKYON_RECORD_ERROR(path->error_message, "-groupIP=<multicast_ip_addr> must be in the range 224.0.0.0 through 239.255.255.255\n");
      return false;
    }
  }

  // Validate arguments
  int num_modes = (is_a_send ? 1 : 0) + (is_a_recv ? 1 : 0);
  if (num_modes != 1) {
    TAKYON_RECORD_ERROR(path->error_message, "RdmaUDMulticast must be one of RdmaUDMulticastSend or RdmaUDMulticastRecv\n");
    return false;
  }

  // Allocate the private data
  PrivateTakyonPath *private_path = calloc(1, sizeof(PrivateTakyonPath));
  if (private_path == NULL) {
    TAKYON_RECORD_ERROR(path->error_message, "Out of memory\n");
    return false;
  }
  comm->data = private_path;

  // Create the RDMA structures that will hold the SGE list and any other needed info
  if (is_a_send) {
    if (path->attrs.verbosity & TAKYON_VERBOSITY_CREATE_DESTROY_MORE) printf("  Sender: Max send requests=%d, sub buffers per request=%d\n", path->attrs.max_pending_send_and_one_sided_requests, path->attrs.max_sub_buffers_per_send_request);
    if (path->attrs.max_pending_send_and_one_sided_requests == 0) {
      TAKYON_RECORD_ERROR(path->error_message, "path->attrs.max_pending_send_and_one_sided_requests must be > 0 for RdmaUDMulticastSend\n");
      goto failed;
    }
    private_path->rdma_send_request_list = (RdmaSendRequest *)calloc(path->attrs.max_pending_send_and_one_sided_requests, sizeof(RdmaSendRequest));
    if (private_path->rdma_send_request_list == NULL) {
      TAKYON_RECORD_ERROR(path->error_message, "Out of memory\n");
      goto failed;
    }
    uint32_t num_sges = path->attrs.max_pending_send_and_one_sided_requests * path->attrs.max_sub_buffers_per_send_request;
    if (num_sges > 0) {
      private_path->sge_list = (struct ibv_sge *)calloc(num_sges, sizeof(struct ibv_sge));
      if (private_path->sge_list == NULL) {
        TAKYON_RECORD_ERROR(path->error_message, "Out of memory\n");
        goto failed;
      }
      for (uint32_t i=0; i<path->attrs.max_pending_send_and_one_sided_requests; i++) {
        private_path->rdma_send_request_list[i].sges = &private_path->sge_list[i*path->attrs.max_sub_buffers_per_send_request];
      }
    }
  } else {
    if (path->attrs.verbosity & TAKYON_VERBOSITY_CREATE_DESTROY_MORE) printf("  Recver: Max recv requests=%d, sub buffers per request=%d\n", path->attrs.max_pending_recv_requests, path->attrs.max_sub_buffers_per_recv_request);
    if (path->attrs.max_pending_recv_requests == 0) {
      TAKYON_RECORD_ERROR(path->error_message, "path->attrs.max_pending_recv_requests must be > 0 for RdmaUDMulticastRecv\n");
      goto failed;
    }
    private_path->rdma_recv_request_list = (RdmaRecvRequest *)calloc(path->attrs.max_pending_recv_requests, sizeof(RdmaRecvRequest));
    if (private_path->rdma_recv_request_list == NULL) {
      TAKYON_RECORD_ERROR(path->error_message, "Out of memory\n");
      goto failed;
    }
    uint32_t num_sges = path->attrs.max_pending_recv_requests * path->attrs.max_sub_buffers_per_recv_request;
    if (num_sges > 0) {
      private_path->sge_list = (struct ibv_sge *)calloc(num_sges, sizeof(struct ibv_sge));
      if (private_path->sge_list == NULL) {
        TAKYON_RECORD_ERROR(path->error_message, "Out of memory\n");
        goto failed;
      }
      for (uint32_t i=0; i<path->attrs.max_pending_recv_requests; i++) {
        private_path->rdma_recv_request_list[i].sges = &private_path->sge_list[i*path->attrs.max_sub_buffers_per_recv_request];
      }
    }
  }

  // Create the info need for RDMA buffer registrations
  private_path->rdma_buffer_list = calloc(path->attrs.buffer_count, sizeof(RdmaBuffer));
  if (private_path->rdma_buffer_list == NULL) {
    TAKYON_RECORD_ERROR(path->error_message, "Out of memory\n");
    goto failed;
  }
  for (uint32_t i=0; i<path->attrs.buffer_count; i++) {
    private_path->rdma_buffer_list[i].path = path;   // Make sure each buffer knows it's for this path: need for verifications later on
    path->attrs.buffers[i].private = &private_path->rdma_buffer_list[i];
  }

  // Prepare for endpoint creation to post recvs
  if (is_a_recv) {
    for (uint32_t i=0; i<post_recv_count; i++) {
      RdmaRecvRequest *rdma_request = &private_path->rdma_recv_request_list[private_path->curr_rdma_request_index];
      private_path->curr_rdma_request_index = (private_path->curr_rdma_request_index + 1) % path->attrs.max_pending_recv_requests;
      recv_requests[i].private = rdma_request;
    }
  }

  // Create the one-sided RDMA endpoint
  RdmaEndpoint *endpoint;
  if (is_a_send) {
    endpoint = rdmaCreateMulticastEndpoint(path, local_ip_addr, group_ip_addr, true, path->attrs.max_pending_send_and_one_sided_requests, 0, path->attrs.max_sub_buffers_per_send_request, 0, 0, NULL, timeout_seconds, error_message, MAX_ERROR_MESSAGE_CHARS);
    if (endpoint == NULL) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to create RDMA multicast sender: %s\n", error_message);
      goto failed;
    }
  } else {
    endpoint = rdmaCreateMulticastEndpoint(path, local_ip_addr, group_ip_addr, false, 0, path->attrs.max_pending_recv_requests, 0, path->attrs.max_sub_buffers_per_recv_request, post_recv_count, recv_requests, timeout_seconds, error_message, MAX_ERROR_MESSAGE_CHARS);
    if (endpoint == NULL) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to create RDMA multicast recver: %s\n", error_message);
      goto failed;
    }
  }
  // Store the results
  private_path->endpoint = endpoint;

  // Ready to start transferring
  return true;

 failed:
  if (private_path->rdma_send_request_list != NULL) free(private_path->rdma_send_request_list);
  if (private_path->rdma_recv_request_list != NULL) free(private_path->rdma_recv_request_list);
  if (private_path->sge_list != NULL) free(private_path->sge_list);
  if (private_path->rdma_buffer_list != NULL) free(private_path->rdma_buffer_list);
  free(private_path);
  return false;
}

bool rdmaUDMulticastDestroy(TakyonPath *path, double timeout_seconds) {
  (void)timeout_seconds; // Quiet compiler checking
  TakyonComm *comm = (TakyonComm *)path->private;
  PrivateTakyonPath *private_path = (PrivateTakyonPath *)comm->data;
  RdmaEndpoint *endpoint = private_path->endpoint;

  // Let RDMA flush and pend transfers
  clockSleepYield(MICROSECONDS_TO_SLEEP_BEFORE_DISCONNECTING);

  // Disconnect
  char error_message[MAX_ERROR_MESSAGE_CHARS];
  if (!rdmaDestroyEndpoint(path, endpoint, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to destroy RDMA endpoint: %s\n", error_message);
    return false;
  }

  // Free memory
  if (private_path->rdma_send_request_list != NULL) free(private_path->rdma_send_request_list);
  if (private_path->rdma_recv_request_list != NULL) free(private_path->rdma_recv_request_list);
  if (private_path->sge_list != NULL) free(private_path->sge_list);
  if (private_path->rdma_buffer_list != NULL) free(private_path->rdma_buffer_list);
  free(private_path);

  return true;
}

bool rdmaUDMulticastSend(TakyonPath *path, TakyonSendRequest *request, uint32_t piggy_back_message, double timeout_seconds, bool *timed_out_ret) {
  (void)timeout_seconds; // Quiet compiler
  TakyonComm *comm = (TakyonComm *)path->private;
  PrivateTakyonPath *private_path = (PrivateTakyonPath *)comm->data;
  RdmaEndpoint *endpoint = private_path->endpoint;
  if (timed_out_ret != NULL) *timed_out_ret = false;
  char error_message[MAX_ERROR_MESSAGE_CHARS];

  // Make sure sending is allowed
#ifdef DEBUG_BUILD
  if (!endpoint->is_sender) {
    TAKYON_RECORD_ERROR(path->error_message, "This RDMA multicast endpoint can only be used for receiving.\n");
    return false;
  }
#endif

  // Validate message attributes
#ifdef DEBUG_BUILD
  for (uint32_t i=0; i<request->sub_buffer_count; i++) {
    TakyonSubBuffer *sub_buffer = &request->sub_buffers[i];
    if (sub_buffer->buffer_index >= path->attrs.buffer_count) {
      TAKYON_RECORD_ERROR(path->error_message, "'sub_buffer->buffer_index == %d out of range\n", sub_buffer->buffer_index);
      return false;
    }
    TakyonBuffer *src_buffer = &path->attrs.buffers[sub_buffer->buffer_index];
    RdmaBuffer *rdma_buffer = (RdmaBuffer *)src_buffer->private;
    if (rdma_buffer->path != path) {
      TAKYON_RECORD_ERROR(path->error_message, "'sub_buffer[%d].buffer_index is not from this Takyon path\n", i);
      return false;
    }
    uint64_t src_bytes = sub_buffer->bytes;
    if (src_bytes > (src_buffer->bytes - sub_buffer->offset)) {
      TAKYON_RECORD_ERROR(path->error_message, "Bytes = %ju, offset = %ju exceeds src buffer (bytes = %ju)\n", src_bytes, sub_buffer->offset, src_buffer->bytes);
      return false;
    }
  }
#endif

  // Get the next unused rdma_request
  RdmaSendRequest *rdma_request = &private_path->rdma_send_request_list[private_path->curr_rdma_request_index];
  private_path->curr_rdma_request_index = (private_path->curr_rdma_request_index + 1) % path->attrs.max_pending_send_and_one_sided_requests;

  // Start the 'send' RDMA transfer
  enum ibv_wr_opcode transfer_mode = IBV_WR_SEND_WITH_IMM;
  uint64_t transfer_id = (uint64_t)request;
  struct ibv_sge *sge_list = rdma_request->sges;
  uint64_t remote_addr = 0;
  uint32_t rkey = 0;
  if (!rdmaEndpointStartSend(path, endpoint, transfer_mode, transfer_id, request->sub_buffer_count, request->sub_buffers, sge_list, remote_addr, rkey, piggy_back_message, request->use_is_sent_notification, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to start the RDMA send: %s\n", error_message);
    return false;
  }

  return true;
}

bool rdmaUDMulticastIsSent(TakyonPath *path, TakyonSendRequest *request, double timeout_seconds, bool *timed_out_ret) {
  TakyonComm *comm = (TakyonComm *)path->private;
  PrivateTakyonPath *private_path = (PrivateTakyonPath *)comm->data;
  RdmaEndpoint *endpoint = private_path->endpoint;
  if (timed_out_ret != NULL) *timed_out_ret = false;
  char error_message[MAX_ERROR_MESSAGE_CHARS];

  // Make sure sending is allowed
#ifdef DEBUG_BUILD
  if (!endpoint->is_sender) {
    TAKYON_RECORD_ERROR(path->error_message, "This RDMA multicast endpoint can only be used for receiving.\n");
    return false;
  }
#endif

  // See if the RDMA message is sent
  uint64_t expected_transfer_id = (uint64_t)request;
  if (!rdmaEndpointIsSent(endpoint, expected_transfer_id, request->use_polling_completion, request->usec_sleep_between_poll_attempts, timeout_seconds, timed_out_ret, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to wait for RDMA send to complete: %s\n", error_message);
    return false;
  }

  return true;
}

bool rdmaUDMulticastPostRecvs(TakyonPath *path, uint32_t request_count, TakyonRecvRequest *requests) {
  TakyonComm *comm = (TakyonComm *)path->private;
  PrivateTakyonPath *private_path = (PrivateTakyonPath *)comm->data;
  RdmaEndpoint *endpoint = private_path->endpoint;
  char error_message[MAX_ERROR_MESSAGE_CHARS];

  // Make sure recving is allowed
#ifdef DEBUG_BUILD
  if (endpoint->is_sender) {
    TAKYON_RECORD_ERROR(path->error_message, "This RDMA multicast endpoint can only be used for sending.\n");
    return false;
  }
#endif

  // Validate message attributes
#ifdef DEBUG_BUILD
  for (uint32_t i=0; i<request_count; i++) {
    TakyonRecvRequest *request = &requests[i];
    for (uint32_t j=0; j<request->sub_buffer_count; j++) {
      TakyonSubBuffer *sub_buffer = &request->sub_buffers[j];
      if (sub_buffer->buffer_index >= path->attrs.buffer_count) {
	TAKYON_RECORD_ERROR(path->error_message, "'sub_buffer->buffer_index == %d out of range\n", sub_buffer->buffer_index);
	return false;
      }
      TakyonBuffer *dest_buffer = &path->attrs.buffers[sub_buffer->buffer_index];
      RdmaBuffer *rdma_buffer = (RdmaBuffer *)dest_buffer->private;
      if (rdma_buffer->path != path) {
        TAKYON_RECORD_ERROR(path->error_message, "'sub_buffer[%d].buffer_index is not from this Takyon path\n", i);
        return false;
      }
      uint64_t dest_bytes = sub_buffer->bytes;
      if (dest_bytes > (dest_buffer->bytes - sub_buffer->offset)) {
	TAKYON_RECORD_ERROR(path->error_message, "Bytes = %ju, offset = %ju exceeds recv buffer (bytes = %ju)\n", dest_bytes, sub_buffer->offset, dest_buffer->bytes);
	return false;
      }
    }
  }
#endif

  // Get the next block of unused rdma requests
  for (uint32_t i=0; i<request_count; i++) {
    RdmaRecvRequest *rdma_request = &private_path->rdma_recv_request_list[private_path->curr_rdma_request_index];
    private_path->curr_rdma_request_index = (private_path->curr_rdma_request_index + 1) % path->attrs.max_pending_recv_requests;
    requests[i].private = rdma_request;
  }

  // Post the recv requests
  if (!rdmaEndpointPostRecvs(path, endpoint, request_count, requests, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to post RDMA recv requests: %s\n", error_message);
    return false;
  }

  return true;
}

bool rdmaUDMulticastIsRecved(TakyonPath *path, TakyonRecvRequest *request, double timeout_seconds, bool *timed_out_ret, uint64_t *bytes_received_ret, uint32_t *piggy_back_message_ret) {
  TakyonComm *comm = (TakyonComm *)path->private;
  PrivateTakyonPath *private_path = (PrivateTakyonPath *)comm->data;
  RdmaEndpoint *endpoint = private_path->endpoint;
  if (timed_out_ret != NULL) *timed_out_ret = false;
  char error_message[MAX_ERROR_MESSAGE_CHARS];

  // Make sure recving is allowed
#ifdef DEBUG_BUILD
  if (endpoint->is_sender) {
    TAKYON_RECORD_ERROR(path->error_message, "This RDMA multicast endpoint can only be used for sending.\n");
    return false;
  }
#endif

  // Wait for the message
  uint64_t expected_transfer_id = (uint64_t)request;
  if (!rdmaEndpointIsRecved(endpoint, expected_transfer_id, request->use_polling_completion, request->usec_sleep_between_poll_attempts, timeout_seconds, timed_out_ret, error_message, MAX_ERROR_MESSAGE_CHARS, bytes_received_ret, piggy_back_message_ret)) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to recv RDMA message: %s\n", error_message);
    return false;
  }

  return true;
}
