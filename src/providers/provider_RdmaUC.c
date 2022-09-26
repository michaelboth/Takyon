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

#include "provider_RdmaUC.h"
#include "takyon_private.h"
#include "utils_rdma_verbs.h"
#include "utils_socket.h"
#include "utils_arg_parser.h"
#include "utils_time.h"
#include "utils_endian.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/*+ also use this for RDMA RC */

// Supported formats:
//   RDMA UC (Unreliable connected)
//     - Max message size is 1 GB
//     - Messages can be dropped
//     - Useful where all bytes may not arrive (unreliable): e.g. live stream video or music
//   ---------------------------------------------------------------------------
//     "RdmaUC -client -remoteIP=<ip_addr>|<hostname> -port=<port_number> -rdmaPort=<local_rdma_port_number>"
//     "RdmaUC -server -localIP=<ip_addr>|<hostname>|Any -port=<port_number> [-reuse] -rdmaPort=<local_rdma_port_number>"
//
//   Argument descriptions:
//     -port=<port_number> = [1024 .. 65535]
//     -rdmaPort=<local_rdma_port_number> = 1 .. max ports on local RDMA NIC

typedef struct {
  uint64_t bytes;
  uint64_t raddr;
  uint32_t rkey;
} RemoteTakyonBuffer;

typedef struct {
  // Socket
  int socket_fd;
  bool thread_started;
  pthread_t disconnect_detection_thread_id;
  bool connection_failed;

  // Rdma
  RdmaEndpoint *endpoint;
  RdmaBuffer *rdma_buffer_list;
  uint32_t curr_rdma_send_request_index;
  uint32_t curr_rdma_recv_request_index;
  RdmaSendRequest *rdma_send_request_list;
  RdmaRecvRequest *rdma_recv_request_list;
  struct ibv_sge *send_sge_list;
  struct ibv_sge *recv_sge_list;

  // Remote buffers
  uint32_t remote_buffer_count;
  RemoteTakyonBuffer *remote_buffers;
} PrivateTakyonPath;

static void *disconnectDetectionThread(void *user_data) {
  PrivateTakyonPath *private_path = (PrivateTakyonPath *)user_data;
  // Wait for either a socket disconnect, or for takyonDestroy() to get called
  uint32_t dummy;
  int64_t timeout_nano_seconds = -1; // Wait forever
  char error_message[MAX_ERROR_MESSAGE_CHARS];
  if (!socketRecv(private_path->socket_fd, &dummy, sizeof(dummy), false, timeout_nano_seconds, NULL, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    private_path->connection_failed = true;
  }
  return NULL;
}

static bool sendRdmaBufferInfo(TakyonPath *path, PrivateTakyonPath *private_path, int64_t timeout_nano_seconds) {
  bool is_big_endian = endianIsBig();
  uint32_t buffer_count = path->attrs.buffer_count;
  char error_message[MAX_ERROR_MESSAGE_CHARS];

  // Send endian
  if (!socketSend(private_path->socket_fd, &is_big_endian, sizeof(is_big_endian), false, timeout_nano_seconds, NULL, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to send is_big_endian: %s\n", error_message);
    return false;
  }
  // Send buffer count
  if (!socketSend(private_path->socket_fd, &buffer_count, sizeof(buffer_count), false, timeout_nano_seconds, NULL, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to send buffer_count: %s\n", error_message);
    return false;
  }

  for (uint32_t i=0; i<path->attrs.buffer_count; i++) {
    TakyonBuffer *buffer = &path->attrs.buffers[i];
    RdmaBuffer *rdma_buffer = (RdmaBuffer *)buffer->private;
    uint64_t bytes = buffer->bytes;
    uint64_t raddr = buffer->addr;
    uint32_t rkey = rdma_buffer->mr->rkey;

    // Send bytes
    if (!socketSend(private_path->socket_fd, &bytes, sizeof(bytes), false, timeout_nano_seconds, NULL, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to send bytes: %s\n", error_message);
      return false;
    }
    // Send raddr
    if (!socketSend(private_path->socket_fd, &raddr, sizeof(raddr), false, timeout_nano_seconds, NULL, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to send raddr: %s\n", error_message);
      return false;
    }
    // Send rkey
    if (!socketSend(private_path->socket_fd, &rkey, sizeof(rkey), false, timeout_nano_seconds, NULL, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to send rkey: %s\n", error_message);
      return false;
    }
  }

  return true;
}

static bool recvRdmaBufferInfo(TakyonPath *path, PrivateTakyonPath *private_path, int64_t timeout_nano_seconds) {
  bool is_big_endian = endianIsBig();
  char error_message[MAX_ERROR_MESSAGE_CHARS];

  // Recv endian
  bool remote_is_big_endian;
  if (!socketRecv(private_path->socket_fd, &remote_is_big_endian, sizeof(remote_is_big_endian), false, timeout_nano_seconds, NULL, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to recv remote_is_big_endian: %s\n", error_message);
    return false;
  }
  // Recv buffer count
  uint32_t buffer_count;
  if (!socketRecv(private_path->socket_fd, &buffer_count, sizeof(buffer_count), false, timeout_nano_seconds, NULL, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to recv buffer_count: %s\n", error_message);
    return false;
  }
  if (remote_is_big_endian != is_big_endian) endianSwap4Byte(&buffer_count, 1);

  // Store the info
  private_path->remote_buffer_count = buffer_count;
  private_path->remote_buffers = NULL;
  if (buffer_count = 0) return true;

  // Allocate the remote buffer list
  private_path->remote_buffers = (RemoteTakyonBuffer *)calloc(buffer_count, sizeof(RemoteTakyonBuffer));
  if (private_path->remote_buffers == NULL) {
    TAKYON_RECORD_ERROR(path->error_message, "Out of memory\n");
    return false;
  }

  for (uint32_t i=0; i<buffer_count; i++) {
    // Recv bytes
    uint64_t bytes;
    if (!socketRecv(private_path->socket_fd, &bytes, sizeof(bytes), false, timeout_nano_seconds, NULL, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to recv bytes: %s\n", error_message);
      return false;
    }
    if (remote_is_big_endian != is_big_endian) endianSwap8Byte(&bytes, 1);
    // Recv raddr
    uint64_t raddr;
    if (!socketRecv(private_path->socket_fd, &raddr, sizeof(raddr), false, timeout_nano_seconds, NULL, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to recv raddr: %s\n", error_message);
      return false;
    }
    if (remote_is_big_endian != is_big_endian) endianSwap8Byte(&raddr, 1);
    // Recv rkey
    uint32_t rkey;
    if (!socketRecv(private_path->socket_fd, &rkey, sizeof(rkey), false, timeout_nano_seconds, NULL, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to recv rkey: %s\n", error_message);
      return false;
    }
    if (remote_is_big_endian != is_big_endian) endianSwap4Byte(&rkey, 1);

    // Store the results
    private_path->remote_buffers[i].bytes = bytes;
    private_path->remote_buffers[i].raddr = raddr;
    private_path->remote_buffers[i].rkey = rkey;
  }

  return true;
}

bool rdmaUCCreate(TakyonPath *path, uint32_t post_recv_count, TakyonRecvRequest *recv_requests, double timeout_seconds) {
  TakyonComm *comm = (TakyonComm *)path->private;
  int64_t timeout_nano_seconds = (int64_t)(timeout_seconds * NANOSECONDS_PER_SECOND_DOUBLE);
  char error_message[MAX_ERROR_MESSAGE_CHARS];

  // Get the name of the provider
  char provider_name[TAKYON_MAX_PROVIDER_CHARS];
  if (!argGetProvider(path->attrs.provider, provider_name, TAKYON_MAX_PROVIDER_CHARS, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to get provider name: %s\n", error_message);
    return false;
  }

  // Get all posible flags and values
  bool is_client = argGetFlag(path->attrs.provider, "-client");
  bool is_server = argGetFlag(path->attrs.provider, "-server");
  bool allow_reuse = argGetFlag(path->attrs.provider, "-reuse");
  // -localIP=<ip_addr>|<hostname>|Any
  char local_ip_addr[TAKYON_MAX_PROVIDER_CHARS];
  bool local_ip_addr_found = false;
  bool ok = argGetText(path->attrs.provider, "-localIP=", local_ip_addr, TAKYON_MAX_PROVIDER_CHARS, &local_ip_addr_found, error_message, MAX_ERROR_MESSAGE_CHARS);
  if (!ok) {
    TAKYON_RECORD_ERROR(path->error_message, "provider argument -localIP=<ip_addr>|<hostname> is invalid: %s\n", error_message);
    return false;
  }
  if (!local_ip_addr_found) {
    TAKYON_RECORD_ERROR(path->error_message, "RdmaUC needs the argument: -localIP=<ip_addr>|<hostname>\n");
    return false;
  }
  // -remoteIP=<ip_addr>|<hostname>
  char remote_ip_addr[TAKYON_MAX_PROVIDER_CHARS];
  bool remote_ip_addr_found = false;
  ok = argGetText(path->attrs.provider, "-remoteIP=", remote_ip_addr, TAKYON_MAX_PROVIDER_CHARS, &remote_ip_addr_found, error_message, MAX_ERROR_MESSAGE_CHARS);
  if (!ok) {
    TAKYON_RECORD_ERROR(path->error_message, "provider argument -remoteIP=<ip_addr>|<hostname> is invalid: %s\n", error_message);
    return false;
  }
  // -port=<port_number>
  uint32_t port_number = 0;
  bool port_number_found = false;
  ok = argGetUInt(path->attrs.provider, "-port=", &port_number, &port_number_found, error_message, MAX_ERROR_MESSAGE_CHARS);
  if (!ok) {
    TAKYON_RECORD_ERROR(path->error_message, "provider spec -port=<port_number> is invalid: %s\n", error_message);
    return false;
  }
  if (!port_number_found) {
    TAKYON_RECORD_ERROR(path->error_message, "RdmaUC needs the argument: -port=<port_number>\n");
    return false;
  }
  if (port_number_found) {
    if ((port_number < 1024) || (port_number > 65535)) {
      TAKYON_RECORD_ERROR(path->error_message, "port numbers need to be between 1024 and 65535\n");
      return false;
    }
  }
  // -rdmaPort=<local_rdma_port_number>
  uint32_t rdma_port_number = 0;
  bool rdma_port_number_found = false;
  ok = argGetUInt(path->attrs.provider, "-rdmaPort=", &rdma_port_number, &rdma_port_number_found, error_message, MAX_ERROR_MESSAGE_CHARS);
  if (!ok) {
    TAKYON_RECORD_ERROR(path->error_message, "provider spec -rdmaPort=<local_rdma_port_number> is invalid: %s\n", error_message);
    return false;
  }
  if (!rdma_port_number_found) {
    TAKYON_RECORD_ERROR(path->error_message, "RdmaUC needs the argument: -rdmaPort=<local_rdma_port_number>\n");
    return false;
  }
  if (rdma_port_number_found) {
    if (rdma_port_number == 0) {
      TAKYON_RECORD_ERROR(path->error_message, "RDMA port needs to be greater than 0\n");
      return false;
    }
  }

  // Validate arguments
  int num_modes = (is_local ? 1 : 0) + (is_client ? 1 : 0) + (is_server ? 1 : 0);
  if (num_modes != 1) {
    TAKYON_RECORD_ERROR(path->error_message, "RdmaUC must specify one of -client or -server\n");
    return false;
  }

  // Make sure each buffer knows it's for this path: need for verifications later on
  for (uint32_t i=0; i<path->attrs.buffer_count; i++) {
    path->attrs.buffers[i].private = path;
  }

  // Allocate the private data
  PrivateTakyonPath *private_path = calloc(1, sizeof(PrivateTakyonPath));
  if (private_path == NULL) {
    TAKYON_RECORD_ERROR(path->error_message, "Out of memory\n");
    return false;
  }
  comm->data = private_path;

  // Create the socket connection that will be used to handshake to RDMA connection
  if (is_client) {
    // Client
    if (!socketCreateTcpClient(remote_ip_addr, (uint16_t)port_number, &private_path->socket_fd, timeout_nano_seconds, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to create TCP client socket needed for the RdmaUC handshake: %s\n", error_message);
      goto cleanup;
    }
  } else {
    // Server
    if (!socketCreateTcpServer(local_ip_addr, (uint16_t)port_number, allow_reuse, &private_path->socket_fd, timeout_nano_seconds, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to create TCP server socket needed for the RdmaUC handshake: %s\n", error_message);
      goto cleanup;
    }
  }

  // RDMA send requests and SGEs
  {
    if (path->attrs.verbosity & TAKYON_VERBOSITY_CREATE_DESTROY_MORE) printf("  Max send requests=%d, sub buffers per request=%d\n", path->attrs.max_pending_send_and_one_sided_requests, path->attrs.max_sub_buffers_per_send_request);
    if (path->attrs.max_pending_send_and_one_sided_requests == 0) {
      TAKYON_RECORD_ERROR(path->error_message, "path->attrs.max_pending_send_and_one_sided_requests must be > 0 for RdmaUC\n");
      goto failed;
    }
    private_path->rdma_send_request_list = (RdmaSendRequest *)calloc(path->attrs.max_pending_send_and_one_sided_requests, sizeof(RdmaSendRequest));
    if (private_path->rdma_send_request_list == NULL) {
      TAKYON_RECORD_ERROR(path->error_message, "Out of memory\n");
      goto failed;
    }
    uint32_t num_sges = path->attrs.max_pending_send_and_one_sided_requests * path->attrs.max_sub_buffers_per_send_request;
    if (num_sges > 0) {
      private_path->send_sge_list = (struct ibv_sge *)calloc(num_sges, sizeof(struct ibv_sge));
      if (private_path->send_sge_list == NULL) {
        TAKYON_RECORD_ERROR(path->error_message, "Out of memory\n");
        goto failed;
      }
      for (uint32_t i=0; i<path->attrs.max_pending_send_and_one_sided_requests; i++) {
        private_path->rdma_send_request_list[i].sges = &private_path->send_sge_list[i*path->attrs.max_sub_buffers_per_send_request];
      }
    }
  }

  // RDMA recv requests and SGEs
  {
    if (path->attrs.verbosity & TAKYON_VERBOSITY_CREATE_DESTROY_MORE) printf("  Max recv requests=%d, sub buffers per request=%d\n", path->attrs.max_pending_recv_requests, path->attrs.max_sub_buffers_per_recv_request);
    if (path->attrs.max_pending_recv_requests == 0) {
      TAKYON_RECORD_ERROR(path->error_message, "path->attrs.max_pending_recv_requests must be > 0 for RdmaUC\n");
      goto failed;
    }
    private_path->rdma_recv_request_list = (RdmaRecvRequest *)calloc(path->attrs.max_pending_recv_requests, sizeof(RdmaRecvRequest));
    if (private_path->rdma_recv_request_list == NULL) {
      TAKYON_RECORD_ERROR(path->error_message, "Out of memory\n");
      goto failed;
    }
    uint32_t num_sges = path->attrs.max_pending_recv_requests * path->attrs.max_sub_buffers_per_recv_request;
    if (num_sges > 0) {
      private_path->recv_sge_list = (struct ibv_sge *)calloc(num_sges, sizeof(struct ibv_sge));
      if (private_path->recv_sge_list == NULL) {
        TAKYON_RECORD_ERROR(path->error_message, "Out of memory\n");
        goto failed;
      }
      for (uint32_t i=0; i<path->attrs.max_pending_recv_requests; i++) {
        private_path->rdma_recv_request_list[i].sges = &private_path->recv_sge_list[i*path->attrs.max_sub_buffers_per_recv_request];
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
    path->attrs.buffers[i].private = &private_path->rdma_buffer_list[i];
  }

  // Prepare for endpoint creation to post recvs
  for (uint32_t i=0; i<post_recv_count; i++) {
    RdmaRecvRequest *rdma_request = &private_path->rdma_recv_request_list[private_path->curr_rdma_recv_request_index];
    private_path->curr_rdma_recv_request_index = (private_path->curr_rdma_recv_request_index + 1) % path->attrs.max_pending_recv_requests;
    recv_requests[i].private = rdma_request;
  }

  // Create the RDMA endpoint
  private_path->endpoint = rdmaCreateUCEndpoint(path, path->attrs.is_endpointA, private_path->socket_fd,
                                                path->attrs.max_pending_send_and_one_sided_requests, path->attrs.max_pending_recv_requests,
                                                path->attrs.max_sub_buffers_per_send_request, path->attrs.max_sub_buffers_per_recv_request,
                                                post_recv_count, recv_requests, timeout_seconds, error_message, MAX_ERROR_MESSAGE_CHARS);
  if (private_path->endpoint == NULL) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to create RDMA UC endpoint: %s\n", error_message);
    goto failed;
  }

  // Exchange buffer info
  if (path->attrs.is_endpointA) {
    if (!sendRdmaBufferInfo(path, private_path, timeout_nano_seconds)) goto failed;
    if (!recvRdmaBufferInfo(path, private_path, timeout_nano_seconds)) goto failed;
  } else {
    if (!recvRdmaBufferInfo(path, private_path, timeout_nano_seconds)) goto failed;
    if (!sendRdmaBufferInfo(path, private_path, timeout_nano_seconds)) goto failed;
  }

  // Start the thread to detect if the socket is disconnected
  int rc = pthread_create(&private_path->disconnect_detection_thread_id, NULL, disconnectDetectionThread, private_path);
  if (rc != 0) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to start disconnectDetectionThread(): rc=%d\n", rc);
    goto cleanup;
  }
  private_path->thread_started = true;

  // Ready to start transferring
  return true;

 failed:
  if (private_path->endpoint != NULL) (void)rdmaDestroyEndpoint(path, private_path->endpoint, timeout_seconds, error_message, MAX_ERROR_MESSAGE_CHARS);
  if (private_path->rdma_send_request_list != NULL) free(private_path->rdma_send_request_list);
  if (private_path->rdma_recv_request_list != NULL) free(private_path->rdma_recv_request_list);
  if (private_path->send_sge_list != NULL) free(private_path->send_sge_list);
  if (private_path->recv_sge_list != NULL) free(private_path->recv_sge_list);
  if (private_path->rdma_buffer_list != NULL) free(private_path->rdma_buffer_list);
  if (private_path->remote_buffers != NULL) free(private_path->remote_buffers);
  free(private_path);
  return false;
}

bool rdmaUCDestroy(TakyonPath *path, double timeout_seconds) {
  (void)timeout_seconds; // Quiet compiler checking
  TakyonComm *comm = (TakyonComm *)path->private;
  PrivateTakyonPath *private_path = (PrivateTakyonPath *)comm->data;
  RdmaEndpoint *endpoint = private_path->endpoint;
  int64_t timeout_nano_seconds = (int64_t)(timeout_seconds * NANOSECONDS_PER_SECOND_DOUBLE);

  // Wake up thread
  if (private_path->thread_started) {
    if (!private_path->connection_failed) {
      // Wake thread up so it will exit
      uint32_t dummy = 0;
      if (!socketSend(private_path->socket_fd, &dummy, sizeof(dummy), false, timeout_nano_seconds, NULL, error_message, MAX_ERROR_MESSAGE_CHARS)) {
        TAKYON_RECORD_ERROR(path->error_message, "Failed to wake up disconnectDetectionThread(): %s\n", error_message);
        private_path->connection_failed = true;
      }
    }
    // Wait for the thread to exit
    int rc = pthread_join(private_path->disconnect_detection_thread_id, NULL);
    if (rc != 0) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to join disconnectDetectionThread(): rc=%d\n", rc);
      private_path->connection_failed = true;
    }
  }

  // Socket barrier: to make sure pending transactions are complete
  bool barrier_ok = !private_path->connection_failed;
  if (barrier_ok) {
    if (path->attrs.is_endpointA) {
      uint32_t x = 23;
      // Send
      if (!socketSend(private_path->socket_fd, &x, sizeof(x), false, timeout_nano_seconds, NULL, error_message, MAX_ERROR_MESSAGE_CHARS)) {
        TAKYON_RECORD_ERROR(path->error_message, "Failed to send barriar value: %s\n", error_message);
        barrier_ok = false;
      }
      x = 33;
      // Recv
      if (barrier_ok && !socketRecv(private_path->socket_fd, &x, sizeof(x), false, timeout_nano_seconds, NULL, error_message, MAX_ERROR_MESSAGE_CHARS)) {
        TAKYON_RECORD_ERROR(path->error_message, "Failed to recv barrier value: %s\n", error_message);
        barrier_ok = false;
      }
      if (x != 23) {
        TAKYON_RECORD_ERROR(path->error_message, "Got incorrect barrier value: %s\n", error_message);
        barrier_ok = false;
      }
    } else {
      uint32_t x = 33;
      // Recv
      if (!socketRecv(private_path->socket_fd, &x, sizeof(x), false, timeout_nano_seconds, NULL, error_message, MAX_ERROR_MESSAGE_CHARS)) {
        TAKYON_RECORD_ERROR(path->error_message, "Failed to recv barrier value: %s\n", error_message);
        barrier_ok = false;
      }
      // Send
      if (barrier_ok && !socketSend(private_path->socket_fd, &x, sizeof(x), false, timeout_nano_seconds, NULL, error_message, MAX_ERROR_MESSAGE_CHARS)) {
        TAKYON_RECORD_ERROR(path->error_message, "Failed to send barrier value: %s\n", error_message);
        barrier_ok = false;
      }
    }
  }

  // Disconnect RDMA endpoint
  char error_message[MAX_ERROR_MESSAGE_CHARS];
  if (!rdmaDestroyEndpoint(path, endpoint, timeout_seconds, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to destroy RDMA endpoint: %s\n", error_message);
    return false;
  }

  // Free memory
  if (private_path->rdma_send_request_list != NULL) free(private_path->rdma_send_request_list);
  if (private_path->rdma_recv_request_list != NULL) free(private_path->rdma_recv_request_list);
  if (private_path->send_sge_list != NULL) free(private_path->send_sge_list);
  if (private_path->recv_sge_list != NULL) free(private_path->recv_sge_list);
  if (private_path->rdma_buffer_list != NULL) free(private_path->rdma_buffer_list);
  if (private_path->remote_buffers != NULL) free(private_path->remote_buffers);
  free(private_path);

  return true;
}

bool rdmaUCOneSided(TakyonPath *path, TakyonOneSidedRequest *request, double timeout_seconds, bool *timed_out_ret) {
  TakyonComm *comm = (TakyonComm *)path->private;
  PrivateTakyonPath *private_path = (PrivateTakyonPath *)comm->data;
  RdmaEndpoint *endpoint = private_path->endpoint;
  if (timed_out_ret != NULL) *timed_out_ret = false;
  char error_message[MAX_ERROR_MESSAGE_CHARS];

  // Verify connection is good
  if (private_path->connection_failed) {
    TAKYON_RECORD_ERROR(path->error_message, "Connection is broken\n");
    return false;
  }

#ifdef DEBUG_BUILD
  // Make sure at least one buffer
  if (request->sub_buffer_count == 0) {
    TAKYON_RECORD_ERROR(path->error_message, "One sided requests must have at least one sub buffer\n");
    return false;
  }
#endif

  // Get total bytes to transfer
  uint64_t total_local_bytes_to_transfer = 0;
  for (uint32_t i=0; i<request->sub_buffer_count; i++) {
    // Source info
    TakyonSubBuffer *sub_buffer = &request->sub_buffers[i];
#ifdef DEBUG_BUILD
    if (sub_buffer->buffer_index >= path->attrs.buffer_count) {
      TAKYON_RECORD_ERROR(path->error_message, "'sub_buffer->buffer_index == %d out of range\n", sub_buffer->buffer_index);
      return false;
    }
#endif
    TakyonBuffer *local_buffer = &path->attrs.buffers[sub_buffer->buffer_index];
#ifdef DEBUG_BUILD
    if (local_buffer->private != path) {
      TAKYON_RECORD_ERROR(path->error_message, "'sub_buffer[%d].buffer_index is not from this Takyon path\n", i);
      return false;
    }
#endif
    uint64_t local_bytes = sub_buffer->bytes;
#ifdef DEBUG_BUILD
    if (local_bytes > (local_buffer->bytes - sub_buffer->offset)) {
      TAKYON_RECORD_ERROR(path->error_message, "Bytes = %ju, offset = %ju exceeds local buffer (bytes = %ju)\n", local_bytes, sub_buffer->offset, local_buffer->bytes);
      return false;
    }
#endif
    total_local_bytes_to_transfer += local_bytes;
  }

  // Remote info
#ifdef DEBUG_BUILD
  if (request->remote_buffer_index >= private_path->remote_buffer_count) {
    TAKYON_RECORD_ERROR(path->error_message, "Remote buffer index = %d is out of range\n", request->remote_buffer_index);
    return false;
  }
#endif
  RemoteTakyonBuffer *remote_buffer = &private_path->remote_buffers[request->remote_buffer_index];
  uint64_t remote_addr = (uint64_t)remote_buffer->mmap_addr + request->remote_offset;
  uint64_t remote_max_bytes = remote_buffer->bytes - request->remote_offset;
  uint32_t rkey = remote_buffer->rkey;

  // Verify enough space in remote request
#ifdef DEBUG_BUILD
  if (total_local_bytes_to_transfer > remote_max_bytes) {
    TAKYON_RECORD_ERROR(path->error_message, "Not enough available remote bytes\n");
    return false;
  }
#endif

  // Get the next unused rdma_request
  RdmaSendRequest *rdma_request = &private_path->rdma_send_request_list[private_path->curr_rdma_request_index];
  private_path->curr_rdma_request_index = (private_path->curr_rdma_request_index + 1) % path->attrs.max_pending_send_and_one_sided_requests;

  // Start the 'read/write' RDMA transfer
  enum ibv_wr_opcode transfer_mode = (request->is_write_request) IBV_WR_RDMA_WRITE : IBV_WR_RDMA_READ;
  uint64_t transfer_id = (uint64_t)request;
  struct ibv_sge *sge_list = rdma_request->sge_list;
  uint32_t piggy_back_message = 0;
  if (!rdmaStartSend(path, endpoint, transfer_mode, transfer_id, request->sub_buffer_count, request->sub_buffers, sge_list, remote_addr, rkey, piggy_back_message, request->use_is_done_notification, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to start the RDMA %s: %s\n", request->is_write_request ? "write" : "read", error_message);
    return false;
  }

  return true;
}

bool rdmaUCIsOneSidedDone(TakyonPath *path, TakyonOneSidedRequest *request, double timeout_seconds, bool *timed_out_ret) {
  TakyonComm *comm = (TakyonComm *)path->private;
  PrivateTakyonPath *private_path = (PrivateTakyonPath *)comm->data;
  RdmaEndpoint *endpoint = private_path->endpoint;
  if (timed_out_ret != NULL) *timed_out_ret = false;
  char error_message[MAX_ERROR_MESSAGE_CHARS];

  // Verify connection is good
  if (private_path->connection_failed) {
    TAKYON_RECORD_ERROR(path->error_message, "Connection is broken\n");
    return false;
  }

  // See if the RDMA message is sent
  uint64_t expected_transfer_id = (uint64_t)request;
  if (!rdmaIsSent(endpoint, expected_transfer_id, request->use_polling_completion, request->usec_sleep_between_poll_attempts, timeout_seconds, timed_out_ret, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to wait for RDMA %s to complete: %s\n", request->is_write_request ? "write" : "read", error_message);
    return false;
  }

  return true;
}

bool rdmaUCSend(TakyonPath *path, TakyonSendRequest *request, uint32_t piggy_back_message, double timeout_seconds, bool *timed_out_ret) {
  TakyonComm *comm = (TakyonComm *)path->private;
  PrivateTakyonPath *private_path = (PrivateTakyonPath *)comm->data;
  RdmaEndpoint *endpoint = private_path->endpoint;
  if (timed_out_ret != NULL) *timed_out_ret = false;
  char error_message[MAX_ERROR_MESSAGE_CHARS];

  // Verify connection is good
  if (private_path->connection_failed) {
    TAKYON_RECORD_ERROR(path->error_message, "Connection is broken\n");
    return false;
  }

  // Validate message attributes
#ifdef DEBUG_BUILD
  for (uint32_t i=0; i<request->sub_buffer_count; i++) {
    TakyonSubBuffer *sub_buffer = &request->sub_buffers[i];
    if (sub_buffer->buffer_index >= path->attrs.buffer_count) {
      TAKYON_RECORD_ERROR(path->error_message, "'sub_buffer->buffer_index == %d out of range\n", sub_buffer->buffer_index);
      return false;
    }
    TakyonBuffer *src_buffer = &path->attrs.buffers[sub_buffer->buffer_index];
    if (src_buffer->private != path) {
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
  struct ibv_sge *sge_list = rdma_request->sge_list;
  uint64_t remote_addr = 0;
  uint32_t rkey = 0;
  if (!rdmaStartSend(path, endpoint, transfer_mode, transfer_id, request->sub_buffer_count, request->sub_buffers, sge_list, remote_addr, rkey, piggy_back_message, request->use_is_sent_notification, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to start the RDMA send: %s\n", error_message);
    return false;
  }

  return true;
}

bool rdmaUCIsSent(TakyonPath *path, TakyonSendRequest *request, double timeout_seconds, bool *timed_out_ret) {
  TakyonComm *comm = (TakyonComm *)path->private;
  PrivateTakyonPath *private_path = (PrivateTakyonPath *)comm->data;
  RdmaEndpoint *endpoint = private_path->endpoint;
  if (timed_out_ret != NULL) *timed_out_ret = false;
  char error_message[MAX_ERROR_MESSAGE_CHARS];

  // Verify connection is good
  if (private_path->connection_failed) {
    TAKYON_RECORD_ERROR(path->error_message, "Connection is broken\n");
    return false;
  }

  // See if the RDMA message is sent
  uint64_t expected_transfer_id = (uint64_t)request;
  if (!rdmaIsSent(endpoint, expected_transfer_id, request->use_polling_completion, request->usec_sleep_between_poll_attempts, timeout_seconds, timed_out_ret, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to wait for RDMA send to complete: %s\n", error_message);
    return false;
  }

  return true;
}

bool rdmaUCPostRecvs(TakyonPath *path, uint32_t request_count, TakyonRecvRequest *requests) {
  TakyonComm *comm = (TakyonComm *)path->private;
  PrivateTakyonPath *private_path = (PrivateTakyonPath *)comm->data;
  RdmaEndpoint *endpoint = private_path->endpoint;
  char error_message[MAX_ERROR_MESSAGE_CHARS];

  // Verify connection is good
  if (private_path->connection_failed) {
    TAKYON_RECORD_ERROR(path->error_message, "Connection is broken\n");
    return false;
  }

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
      if (dest_buffer->private != path) {
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
    RdmaRecvRequest *rdma_request = &private_path->rdma_recv_request_list[private_path->curr_rdma_recv_request_index];
    private_path->curr_rdma_recv_request_index = (private_path->curr_rdma_recv_request_index + 1) % path->attrs.max_pending_recv_requests;
    requests[i].private = rdma_request;
  }

  // Post the recv requests
  if (!rdmaPostRecvs(path, endpoint, request_count, requests, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to post RDMA recv requests: %s\n", error_message);
    return false;
  }

  return true;
}

bool rdmaUCIsRecved(TakyonPath *path, TakyonRecvRequest *request, double timeout_seconds, bool *timed_out_ret, uint64_t *bytes_received_ret, uint32_t *piggy_back_message_ret) {
  TakyonComm *comm = (TakyonComm *)path->private;
  PrivateTakyonPath *private_path = (PrivateTakyonPath *)comm->data;
  RdmaEndpoint *endpoint = private_path->endpoint;
  if (timed_out_ret != NULL) *timed_out_ret = false;
  char error_message[MAX_ERROR_MESSAGE_CHARS];

  // Verify connection is good
  if (private_path->connection_failed) {
    TAKYON_RECORD_ERROR(path->error_message, "Connection is broken\n");
    return false;
  }

  // Wait for the message
  uint64_t expected_transfer_id = (uint64_t)request;
  if (!rdmaIsRecved(endpoint, expected_transfer_id, request->use_polling_completion, request->usec_sleep_between_poll_attempts, timeout_seconds, timed_out_ret, error_message, MAX_ERROR_MESSAGE_CHARS, bytes_received_ret, piggy_back_message_ret)) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to recv RDMA message: %s\n", error_message);
    return false;
  }

  return true;
}
