// Takyon 1.x was originally developed by Michael Both at Abaco, and he is now continuing development independently
//
// Original copyright:
//     Copyright 2018,2020 Abaco Systems
//     Licensed under the Apache License, Version 2.0 (the "License");
//     you may not use this file except in compliance with the License.
//     You may obtain a copy of the License at
//         http://www.apache.org/licenses/LICENSE-2.0
//     Unless required by applicable law or agreed to in writing, software
//     distributed under the License is distributed on an "AS IS" BASIS,
//     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//     See the License for the specific language governing permissions and
//     limitations under the License.
//
// Changes for 2.0 (starting from Takyon 1.1.0):
//   - See comments in takyon.h for the bigger picture of the changes
//   - Redesigned to the 2.x API functionality
//
// Copyright for modifications:
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

#include "interconnect_TcpSocket.h"
#include "takyon_private.h"
#include "utils_socket.h"
#include "utils_arg_parser.h"
#include "utils_time.h"
#include <stdlib.h>
#include <stdio.h>

// Supported formats:
//
//   TCP point-to-point (inter-process and inter-processor)
//     - Max transfer size is 1 GB
//     - Data is gauranteed to arrive in the same order sent without any loss
//     - Use this if every bytes matters (reliable): e.g. downloading an applicaton
//   ------------------------------------------------------------------------
//   Local TCP Unix socket (both endpoints in the same OS, has better performance)
//     "TcpSocket -local -pathID=<non_negative_integer>"
//   User assigned port number:
//     "TcpSocket -client -remoteIP=<ip_addr>|<hostname> -port=<port_number>"
//     "TcpSocket -server -localIP=<ip_addr>|<hostname>|Any -port=<port_number> [-reuse]"
//   Ephemeral port number (assigned by system)
//     "TcpSocket -client -remoteIP=<ip_addr>|<hostname> -ephemeralID=<non_negative_integer>"
//     "TcpSocket -server -localIP=<ip_addr>|<hostname>|Any -ephemeralID=<non_negative_integer>"
//
//   Argument descriptions:
//     -port=<port_number> = [1024 .. 65535]

#define XFER_COMMAND 0x23feed33  // Used to sanity check the transfer header

typedef struct {
  TakyonSocket socket_fd;
  bool socket_is_in_polling_mode; // By default, a socket is in blocking mode
  bool using_ephemeral_port_manager;
  bool connection_failed;
} PrivateTakyonPath;

bool tcpSocketCreate(TakyonPath *path, uint32_t post_recv_count, TakyonRecvRequest *recv_requests, double timeout_seconds) {
  (void)post_recv_count; // Quiet compiler checking
  (void)recv_requests; // Quiet compiler checking
  TakyonComm *comm = (TakyonComm *)path->private;
  int64_t timeout_nano_seconds = (int64_t)(timeout_seconds * NANOSECONDS_PER_SECOND_DOUBLE);
  char error_message[MAX_ERROR_MESSAGE_CHARS];

  // Get the name of the interconnect
  char interconnect_name[TAKYON_MAX_INTERCONNECT_CHARS];
  if (!argGetInterconnect(path->attrs.interconnect, interconnect_name, TAKYON_MAX_INTERCONNECT_CHARS, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to get interconnect name: %s\n", error_message);
    return false;
  }

  // Get all posible flags and values
  bool is_local = argGetFlag(path->attrs.interconnect, "-local");
  bool is_client = argGetFlag(path->attrs.interconnect, "-client");
  bool is_server = argGetFlag(path->attrs.interconnect, "-server");
  bool allow_reuse = argGetFlag(path->attrs.interconnect, "-reuse");
  // -localIP=<ip_addr>|<hostname>|Any
  char local_ip_addr[TAKYON_MAX_INTERCONNECT_CHARS];
  bool local_ip_addr_found = false;
  bool ok = argGetText(path->attrs.interconnect, "-localIP=", local_ip_addr, TAKYON_MAX_INTERCONNECT_CHARS, &local_ip_addr_found, error_message, MAX_ERROR_MESSAGE_CHARS);
  if (!ok) {
    TAKYON_RECORD_ERROR(path->error_message, "interconnect argument -localIP=<ip_addr>|<hostname>|Any is invalid: %s\n", error_message);
    return false;
  }
  // -remoteIP=<ip_addr>|<hostname>
  char remote_ip_addr[TAKYON_MAX_INTERCONNECT_CHARS];
  bool remote_ip_addr_found = false;
  ok = argGetText(path->attrs.interconnect, "-remoteIP=", remote_ip_addr, TAKYON_MAX_INTERCONNECT_CHARS, &remote_ip_addr_found, error_message, MAX_ERROR_MESSAGE_CHARS);
  if (!ok) {
    TAKYON_RECORD_ERROR(path->error_message, "interconnect argument -remoteIP=<ip_addr>|<hostname> is invalid: %s\n", error_message);
    return false;
  }
  // -pathID=<non_negative_integer>
  uint32_t path_id = 0;
  bool path_id_found = false;
  ok = argGetUInt(path->attrs.interconnect, "-pathID=", &path_id, &path_id_found, error_message, MAX_ERROR_MESSAGE_CHARS);
  if (!ok) {
    TAKYON_RECORD_ERROR(path->error_message, "interconnect argument -pathID=<non_negative_integer> is invalid: %s\n", error_message);
    return false;
  }
  // -ephemeralID=<non_negative_integer>
  uint32_t ephemeral_id = 0;
  bool ephemeral_id_found = false;
  ok = argGetUInt(path->attrs.interconnect, "-ephemeralID=", &ephemeral_id, &ephemeral_id_found, error_message, MAX_ERROR_MESSAGE_CHARS);
  if (!ok) {
    TAKYON_RECORD_ERROR(path->error_message, "interconnect argument -ephemeralID=<non_negative_integer> is invalid: %s\n", error_message);
    return false;
  }

  // -port=<port_number>
  uint32_t port_number = 0;
  bool port_number_found = false;
  ok = argGetUInt(path->attrs.interconnect, "-port=", &port_number, &port_number_found, error_message, MAX_ERROR_MESSAGE_CHARS);
  if (!ok) {
    TAKYON_RECORD_ERROR(path->error_message, "interconnect spec -port=<port_number> is invalid: %s\n", error_message);
    return false;
  }
  if (port_number_found) {
    if ((port_number < 1024) || (port_number > 65535)) {
      TAKYON_RECORD_ERROR(path->error_message, "port numbers need to be between 1024 and 65535\n");
      return false;
    }
  }

  // Validate arguments
  int num_modes = (is_local ? 1 : 0) + (is_client ? 1 : 0) + (is_server ? 1 : 0);
  if (num_modes != 1) {
    TAKYON_RECORD_ERROR(path->error_message, "Interconnect must specifiy one of -local -client or -server\n");
    return false;
  }
  if (is_local) {
    if (!path_id_found) {
      TAKYON_RECORD_ERROR(path->error_message, "Local socket must specify -pathID=<non_negative_integer>\n");
      return false;
    }
  } else {
    num_modes = (ephemeral_id_found ? 1 : 0) + (port_number_found ? 1 : 0);
    if (num_modes != 1) {
      TAKYON_RECORD_ERROR(path->error_message, "Need to specify exactly one of -pathID=<non_negative_integer> or -port=<port_number>\n");
      return false;
    }
    if (is_client && !remote_ip_addr_found) {
      TAKYON_RECORD_ERROR(path->error_message, "Client/server sockets must specify -remoteIP=<ip_addr>|<hostname>\n");
      return false;
    }
    if (is_server && !local_ip_addr_found) {
      TAKYON_RECORD_ERROR(path->error_message, "Client/server sockets must specify -localIP=<ip_addr>|<hostname>|Any\n");
      return false;
    }
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
  private_path->connection_failed = false;
  private_path->using_ephemeral_port_manager = ephemeral_id_found;
  private_path->socket_fd = -1;
  private_path->socket_is_in_polling_mode = false;

  // See if the ephemeral port manager needs to get started
  if (ephemeral_id_found) {
    ephemeralPortManagerInit(path->attrs.verbosity);
  }

  // Create the socket and connect with remote endpoint
  if (is_local) {
    char local_socket_name[TAKYON_MAX_INTERCONNECT_CHARS];
    snprintf(local_socket_name, TAKYON_MAX_INTERCONNECT_CHARS, "TakyonSocket_%d", path_id);
    if (path->attrs.is_endpointA) {
      if (!socketCreateLocalClient(local_socket_name, &private_path->socket_fd, timeout_nano_seconds, error_message, MAX_ERROR_MESSAGE_CHARS)) {
        TAKYON_RECORD_ERROR(path->error_message, "Failed to create local client socket: %s\n", error_message);
        goto cleanup;
      }
    } else {
      if (!socketCreateLocalServer(local_socket_name, &private_path->socket_fd, timeout_nano_seconds, error_message, MAX_ERROR_MESSAGE_CHARS)) {
        TAKYON_RECORD_ERROR(path->error_message, "Failed to create local server socket: %s\n", error_message);
        goto cleanup;
      }
    }
  } else if (is_client) {
    // Client
    if (ephemeral_id_found) {
      if (!socketCreateEphemeralTcpClient(remote_ip_addr, interconnect_name, ephemeral_id, &private_path->socket_fd, timeout_nano_seconds, path->attrs.verbosity, error_message, MAX_ERROR_MESSAGE_CHARS)) {
        TAKYON_RECORD_ERROR(path->error_message, "Failed to create TCP client socket: %s\n", error_message);
        goto cleanup;
      }
    } else {
      if (!socketCreateTcpClient(remote_ip_addr, (uint16_t)port_number, &private_path->socket_fd, timeout_nano_seconds, error_message, MAX_ERROR_MESSAGE_CHARS)) {
        TAKYON_RECORD_ERROR(path->error_message, "Failed to create TCP client socket: %s\n", error_message);
        goto cleanup;
      }
    }
  } else {
    // Server
    if (ephemeral_id_found) {
      if (!socketCreateEphemeralTcpServer(local_ip_addr, interconnect_name, ephemeral_id, &private_path->socket_fd, timeout_nano_seconds, error_message, MAX_ERROR_MESSAGE_CHARS)) {
        TAKYON_RECORD_ERROR(path->error_message, "Failed to create TCP server socket: %s\n", error_message);
        goto cleanup;
      }
    } else {
      if (!socketCreateTcpServer(local_ip_addr, (uint16_t)port_number, allow_reuse, &private_path->socket_fd, timeout_nano_seconds, error_message, MAX_ERROR_MESSAGE_CHARS)) {
        TAKYON_RECORD_ERROR(path->error_message, "Failed to create TCP server socket: %s\n", error_message);
        goto cleanup;
      }
    }
  }

  // Ready to start transferring
  return true;

 cleanup:
  // An error ocurred so clean up all allocated resources
  if (private_path->using_ephemeral_port_manager) {
    ephemeralPortManagerFinalize();
  }
  free(private_path);

  return false;
}

bool tcpSocketDestroy(TakyonPath *path, double timeout_seconds) {
  (void)timeout_seconds; // Quiet compiler checking
  TakyonComm *comm = (TakyonComm *)path->private;
  PrivateTakyonPath *private_path = (PrivateTakyonPath *)comm->data;

  // Connection was made, so disconnect gracefully
  // NOTE: TCP_NODELAY is likely already active, so just need to provide some time for the remote side to get any in-transit data before a disconnect message is sent
  // NOTE: private_path->connection_failed may be true, but still want to provide time for remote side to handle arriving data
  clockSleepYield(MICROSECONDS_TO_SLEEP_BEFORE_DISCONNECTING);

  // Disconnect
  socketClose(private_path->socket_fd);

  // Stop the ephemeral port manager if no paths are using it
  if (private_path->using_ephemeral_port_manager) {
    ephemeralPortManagerFinalize();
  }

  free(private_path);

  return true;
}

bool tcpSocketSend(TakyonPath *path, TakyonSendRequest *request, uint32_t piggy_back_message, double timeout_seconds, bool *timed_out_ret) {
  TakyonComm *comm = (TakyonComm *)path->private;
  PrivateTakyonPath *private_path = (PrivateTakyonPath *)comm->data;
  int64_t timeout_nano_seconds = (int64_t)(timeout_seconds * NANOSECONDS_PER_SECOND_DOUBLE);
  if (timed_out_ret != NULL) *timed_out_ret = false;
  char error_message[MAX_ERROR_MESSAGE_CHARS];

  // Verify connection is good
  if (private_path->connection_failed) {
    TAKYON_RECORD_ERROR(path->error_message, "Connection is broken\n");
    return false;
  }

  // Make sure socket is in correct polling or event driven mode
  if (private_path->socket_is_in_polling_mode && request->use_polling_completion) {
    private_path->socket_is_in_polling_mode = request->use_polling_completion;
    if (!socketSetBlocking(private_path->socket_fd, !request->use_polling_completion, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Could not set the socket to '%s' mode: %s\n", request->use_polling_completion ? "polling" : "event driven", error_message);
      private_path->connection_failed = true;
      return false;
    }
  }

  // Get total bytes to send
  uint64_t total_bytes_to_send = 0;
  for (uint32_t i=0; i<request->sub_buffer_count; i++) {
    // Source info
    TakyonSubBuffer *sub_buffer = &request->sub_buffers[i];
    TakyonBuffer *src_buffer = sub_buffer->buffer;
    if (src_buffer->private != path) {
      private_path->connection_failed = true;
      TAKYON_RECORD_ERROR(path->error_message, "'sub_buffer[%d] is not from this Takyon path\n", i);
      return false;
    }
    uint64_t src_bytes = sub_buffer->bytes;
    if (src_bytes > (src_buffer->bytes - sub_buffer->offset)) {
      private_path->connection_failed = true;
      TAKYON_RECORD_ERROR(path->error_message, "Bytes = %ju exceeds src buffer\n", src_bytes);
      return false;
    }
    total_bytes_to_send += src_bytes;
  }

  // First, send the header
  uint64_t header[3];
  header[0] = XFER_COMMAND;
  header[1] = total_bytes_to_send;
  header[2] = piggy_back_message;
  if (!socketSend(private_path->socket_fd, header, sizeof(header), request->use_polling_completion, timeout_nano_seconds, timed_out_ret, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    private_path->connection_failed = true;
    TAKYON_RECORD_ERROR(path->error_message, "Failed to transfer header: %s\n", error_message);
    return false;
  }
  if ((timeout_nano_seconds >= 0) && (*timed_out_ret == true)) {
    // Timed out but no data was transfered yet
    return true;
  }

  // Send the sub buffers
  for (uint32_t i=0; i<request->sub_buffer_count; i++) {
    // Source info
    TakyonSubBuffer *sub_buffer = &request->sub_buffers[i];
    TakyonBuffer *src_buffer = sub_buffer->buffer;
    void *src_addr = (void *)((uint64_t)src_buffer->addr + sub_buffer->offset);
    uint64_t src_bytes = sub_buffer->bytes;
    if (src_bytes > 0) {
      if (!socketSend(private_path->socket_fd, src_addr, src_bytes, request->use_polling_completion, timeout_nano_seconds, timed_out_ret, error_message, MAX_ERROR_MESSAGE_CHARS)) {
        private_path->connection_failed = true;
        TAKYON_RECORD_ERROR(path->error_message, "Failed to transfer data: %s\n", error_message);
        return false;
      }
      if ((timeout_nano_seconds >= 0) && (*timed_out_ret == true)) {
        // Timed out but some data was already transfered
        private_path->connection_failed = true;
        TAKYON_RECORD_ERROR(path->error_message, "Timed out in the middle of a transfer\n");
        return false;
      }
    }
  }

  return true;
}

bool tcpSocketIsRecved(TakyonPath *path, TakyonRecvRequest *request, double timeout_seconds, bool *timed_out_ret, uint64_t *bytes_received_ret, uint32_t *piggy_back_message_ret) {
  TakyonComm *comm = (TakyonComm *)path->private;
  PrivateTakyonPath *private_path = (PrivateTakyonPath *)comm->data;
  int64_t timeout_nano_seconds = (int64_t)(timeout_seconds * NANOSECONDS_PER_SECOND_DOUBLE);
  if (timed_out_ret != NULL) *timed_out_ret = false;
  char error_message[MAX_ERROR_MESSAGE_CHARS];

  // Verify connection is good
  if (private_path->connection_failed) {
    TAKYON_RECORD_ERROR(path->error_message, "Connection is broken\n");
    return false;
  }

  // Make sure socket is in correct polling or event driven mode
  if (private_path->socket_is_in_polling_mode && request->use_polling_completion) {
    private_path->socket_is_in_polling_mode = request->use_polling_completion;
    if (!socketSetBlocking(private_path->socket_fd, !request->use_polling_completion, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Could not set the socket to '%s' mode: %s\n", request->use_polling_completion ? "polling" : "event driven", error_message);
      private_path->connection_failed = true;
      return false;
    }
  }

  // Get the header
  uint64_t header[3];
  if (!socketRecv(private_path->socket_fd, header, sizeof(header), request->use_polling_completion, timeout_nano_seconds, timed_out_ret, error_message, MAX_ERROR_MESSAGE_CHARS)) {
    private_path->connection_failed = true;
    TAKYON_RECORD_ERROR(path->error_message, "failed to receive header: %s\n", error_message);
    return false;
  }
  if ((timeout_nano_seconds >= 0) && (*timed_out_ret == true)) {
    // Timed out but no data was transfered yet
    return true;
  }

  // Process the header
  if (header[0] != XFER_COMMAND) { 
    private_path->connection_failed = true;
    TAKYON_RECORD_ERROR(path->error_message, "got unexpected header\n");
    return false;
  }
  uint64_t total_bytes_sent = header[1];
  uint32_t piggy_back_message = header[2];

  // Recv the data
  if (total_bytes_sent > 0) {
    // Get total available recv bytes
    uint64_t total_available_recv_bytes = 0;
    for (uint32_t i=0; i<request->sub_buffer_count; i++) {
      TakyonSubBuffer *sub_buffer = &request->sub_buffers[i];
      TakyonBuffer *buffer = sub_buffer->buffer;
      if (buffer->private != path) {
        private_path->connection_failed = true;
        TAKYON_RECORD_ERROR(path->error_message, "'sub_buffers[%d] is not from the remote Takyon path\n", i);
        return false;
      }
      uint64_t max_bytes = sub_buffer->bytes;
      if (max_bytes < (buffer->bytes - sub_buffer->offset)) {
        private_path->connection_failed = true;
        TAKYON_RECORD_ERROR(path->error_message, "Bytes = %ju and exceeds buffer\n", max_bytes);
        return false;
      }
      total_available_recv_bytes += max_bytes;
    }

    // Verify enough space in remote request
    if (total_bytes_sent > total_available_recv_bytes) {
      private_path->connection_failed = true;
      TAKYON_RECORD_ERROR(path->error_message, "Not enough available bytes to receive message\n");
      return false;
    }

    // Recv bytes
    uint64_t bytes_to_receive = total_bytes_sent;
    for (uint32_t i=0; i<request->sub_buffer_count; i++) {
      TakyonSubBuffer *sub_buffer = &request->sub_buffers[i];
      TakyonBuffer *buffer = sub_buffer->buffer;
      uint64_t max_bytes = sub_buffer->bytes;
      void *recv_addr = (void *)((uint64_t)buffer->addr + sub_buffer->offset);
      uint64_t recv_bytes = (bytes_to_receive < max_bytes) ? bytes_to_receive : max_bytes;
      if (!socketRecv(private_path->socket_fd, recv_addr, recv_bytes, request->use_polling_completion, timeout_nano_seconds, timed_out_ret, error_message, MAX_ERROR_MESSAGE_CHARS)) {
        private_path->connection_failed = true;
        TAKYON_RECORD_ERROR(path->error_message, "failed to receive data: %s\n", error_message);
        return false;
      }
      if ((timeout_nano_seconds >= 0) && (*timed_out_ret == true)) {
        // Timed out but some data already arrived
        private_path->connection_failed = true;
        TAKYON_RECORD_ERROR(path->error_message, "timed out in the middle of a transfer\n");
        return false;
      }
      bytes_to_receive -= recv_bytes;
      if (bytes_to_receive == 0) break;
    }
  }

  // Return results
  *bytes_received_ret = total_bytes_sent;
  *piggy_back_message_ret = piggy_back_message;

  return true;
}
