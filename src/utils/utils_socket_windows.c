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
//   - Minor changes in this file
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

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <winsock2.h> // Needed for Socket
#include <ws2tcpip.h> // Need for inet_pton()
#include "utils_socket.h"
#include "utils_time.h"
#include "pthread.h"
#include "takyon.h" // For TAKYON_MAX_PROVIDER_CHARS
#include <stdio.h>
#include <stdlib.h>

#define MAX_IP_ADDR_CHARS 40 // Good for IPv4 or IPv6
#define MAX_TEMP_FILENAME_CHARS (TAKYON_MAX_PROVIDER_CHARS+MAX_PATH+1)
#define MICROSECONDS_BETWEEN_CONNECT_ATTEMPTS 10000

static pthread_once_t L_once_control = PTHREAD_ONCE_INIT;

static bool hostnameToIpaddr(int address_family, const char *hostname, char *resolved_ip_addr, int max_chars, char *error_message, int max_error_message_chars) {
  struct addrinfo hints;
  memset(&hints, 0, sizeof hints); // Zeroing the structure sets most of the flags to open ended
//#define TEST_NAME_RESOLUTION
#ifdef TEST_NAME_RESOLUTION
  hints.ai_family = AF_UNSPEC;
#else
  hints.ai_family = address_family; // AF_INET6 to force IPv6, AF_INET to force IPv4, or AF_UNSPEC to allow for either
#endif
  hints.ai_socktype = 0;  // 0 = any type, Typical: SOCK_STREAM or SOCK_DGRAM
  hints.ai_protocol = 0;  // 0 = any protocol
  struct addrinfo *servinfo;
  int status;
  if ((status = getaddrinfo(hostname, "http", &hints, &servinfo)) != 0)  {
    // Failed
#ifdef TEST_NAME_RESOLUTION
    printf("FAILED to find ip addr info for hostname '%s': %s\n", hostname, gai_strerror(status));
#else
    snprintf(error_message, max_error_message_chars, "Could not convert hostname '%s' to an IP address: %s", hostname, gai_strerror(status));
#endif
    return false;
  }
  // loop through all the results and connect to the first we can
  bool found = false;
  int index = 0;
#ifdef TEST_NAME_RESOLUTION
  printf("IP addresses for host: %s\n", hostname);
#endif
  for (struct addrinfo *p = servinfo; p != NULL; p = p->ai_next, index++) {
    void *addr;
    // Get the pointer to the address itself, different fields in IPv4 and IPv6
    if (p->ai_family == AF_INET) {
      // IPv4
      struct sockaddr_in *ipv4 = (struct sockaddr_in *)p->ai_addr;
      addr = &(ipv4->sin_addr);
    } else {
      // IPv6
      struct sockaddr_in6 *ipv6 = (struct sockaddr_in6 *)p->ai_addr;
      addr = &(ipv6->sin6_addr);
    }
    if (inet_ntop(p->ai_family, addr, resolved_ip_addr, max_chars) == NULL) {
#ifdef TEST_NAME_RESOLUTION
      printf("  %d: %s FAILED TO GET IP ADDR: errno=%d\n", index, p->ai_family == AF_INET6 ? "IPv6" : "IPv4", errno);
#else
      snprintf(error_message, max_error_message_chars, "Could not convert hostname '%s' to an IP address: errno=%d", hostname, errno);
#endif
    } else {
#ifdef TEST_NAME_RESOLUTION
      printf("  %d: %s [%s]\n", index, p->ai_family == AF_INET6 ? "IPv6" : "IPv4", resolved_ip_addr);
#else
      found = true;
      break;
#endif
    }
  }
  freeaddrinfo(servinfo); // all done with this structure
  // NOTE: only the last IP address found is returned
  return found;
}

static void windows_socket_finalize(void) {
  if (WSACleanup() != 0) {
    fprintf(stderr, "Sockt finalize failed.\n");
  }
}

static void windows_socket_init_once(void) {
  // Initialize Winsock
  WSADATA wsaData;
  if (WSAStartup(MAKEWORD(2,2), &wsaData) != NO_ERROR) {
    fprintf(stderr, "Could not initialize the Winsock socket interface.\n");
    abort();
  }
  // This will get called if the app calls exit() or if main does a normal return.
  if (atexit(windows_socket_finalize) == -1) {
    fprintf(stderr, "Failed to call atexit() in the mutex manager initializer\n");
    abort();
  }
}

static int windows_socket_manager_init(char *error_message, int max_error_message_chars) {
  // Call this to make sure Windows inits sockets: This can be called multiple times, but it's garanteed to atomically run the only the first time called.
  if (pthread_once(&L_once_control, windows_socket_init_once) != 0) {
    snprintf(error_message, max_error_message_chars, "Failed to start Windows sockets");
    return false;
  }
#ifdef TEST_NAME_RESOLUTION
  char resolved_ip_addr[MAX_IP_ADDR_CHARS];
  hostnameToIpaddr(AF_INET, "10.3.45.234", resolved_ip_addr, MAX_IP_ADDR_CHARS, error_message, max_error_message_chars);
  hostnameToIpaddr(AF_INET, "127.0.0.1", resolved_ip_addr, MAX_IP_ADDR_CHARS, error_message, max_error_message_chars);
  hostnameToIpaddr(AF_INET, "localhost", resolved_ip_addr, MAX_IP_ADDR_CHARS, error_message, max_error_message_chars);
  hostnameToIpaddr(AF_INET, "fakehost", resolved_ip_addr, MAX_IP_ADDR_CHARS, error_message, max_error_message_chars);
  hostnameToIpaddr(AF_INET, "www.google.com", resolved_ip_addr, MAX_IP_ADDR_CHARS, error_message, max_error_message_chars);
  hostnameToIpaddr(AF_INET, "www.nba.com", resolved_ip_addr, MAX_IP_ADDR_CHARS, error_message, max_error_message_chars);
  hostnameToIpaddr(AF_INET, "www.comcast.net", resolved_ip_addr, MAX_IP_ADDR_CHARS, error_message, max_error_message_chars);
#endif
  return true;
}

static bool setSocketTimeout(TakyonSocket socket_fd, int timeout_name, int64_t timeout_ns, char *error_message, int max_error_message_chars) {
  DWORD timeout_ms;
  if (timeout_ns < 0) {
    timeout_ms = INT_MAX;
  } else if (timeout_ns == 0) {
    timeout_ms = 0;
  } else {
    timeout_ms = (DWORD)(timeout_ns / 1000000);
  }
  if (setsockopt(socket_fd, SOL_SOCKET, timeout_name, (char *)&timeout_ms, sizeof(timeout_ms)) != 0) {
    snprintf(error_message, max_error_message_chars, "Failed to set the timeout = %lld. sock_error=%d", timeout_ns, WSAGetLastError());
    return false;
  }
  return true;
}

static bool socket_send_event_driven(TakyonSocket socket_fd, void *addr, size_t total_bytes_to_write, int64_t timeout_ns, bool *timed_out_ret, char *error_message, int max_error_message_chars) {
  *timed_out_ret = false;
  // Set the timeout on the socket
  if (!setSocketTimeout(socket_fd, SO_SNDTIMEO, timeout_ns, error_message, max_error_message_chars)) return false;
  bool got_some_data = false;

  // Write the data
  size_t total_bytes_sent = 0;
  while (total_bytes_sent < total_bytes_to_write) {
    size_t bytes_to_write = total_bytes_to_write - total_bytes_sent;
    int bytes_to_write2 = (bytes_to_write > INT_MAX) ? INT_MAX : (int)bytes_to_write;
    int bytes_written = send(socket_fd, addr, bytes_to_write2, 0);
    if (bytes_written == bytes_to_write) {
      // Done
      break;
    }
    if (bytes_written == SOCKET_ERROR) {
      int sock_error = WSAGetLastError();
      if (sock_error == WSAEWOULDBLOCK) {
        // Timed out
        if (total_bytes_sent == 0) {
          // No problem, timed out
          *timed_out_ret = true;
          return true;
        } else {
          // This is bad... the connection might have gone down while in the middle of sending
          snprintf(error_message, max_error_message_chars, "Timed out in the middle of a send transfer: total_bytes_sent=%ju", total_bytes_sent);
          return false;
        }
      } else if (sock_error == WSAEINTR) {
        // Interrupted by external signal. Just try again
        bytes_written = 0;
      } else {
        snprintf(error_message, max_error_message_chars, "Failed to write to socket (are you writing to GPU memory?): sock_error=%d", sock_error);
        return false;
      }
    } else {
      // There was some progress, so keep trying
      total_bytes_sent += bytes_written;
      addr = (void *)((char *)addr + bytes_written);
    }
  }

  return true;
}

static bool socket_send_polling(TakyonSocket socket, void *addr, size_t total_bytes_to_write, int64_t timeout_ns, bool *timed_out_ret, char *error_message, int max_error_message_chars) {
  *timed_out_ret = false;
  int64_t start_time = clockTimeNanoseconds();
  bool got_some_data = false;

  size_t total_bytes_sent = 0;
  while (total_bytes_sent < total_bytes_to_write) {
    size_t bytes_to_write = total_bytes_to_write - total_bytes_sent;
    int bytes_to_write2 = (bytes_to_write > INT_MAX) ? INT_MAX : (int)bytes_to_write;
    int bytes_written = send(socket, addr, bytes_to_write2, 0);
    if (bytes_written == bytes_to_write) {
      // Done
      break;
    }
    if (bytes_written == SOCKET_ERROR) {
      int sock_error = WSAGetLastError();
      if (sock_error == WSAEWOULDBLOCK) {
        // Nothing written, but no errors
        if (timeout_ns >= 0) {
          int64_t ellapsed_time = clockTimeNanoseconds() - start_time;
          if (ellapsed_time >= timeout_ns) {
            // Timed out
            if (total_bytes_sent == 0) {
              *timed_out_ret = true;
              return true;
            } else {
              // This is bad... the connection might have gone down while in the middle of sending
              snprintf(error_message, max_error_message_chars, "Timed out in the middle of a send transfer: total_bytes_sent=%ju", total_bytes_sent);
              return false;
            }
          }
        }
        bytes_written = 0;
      } else if (sock_error == WSAEINTR) { 
        // Interrupted by external signal. Just try again
        bytes_written = 0;
      } else {
        snprintf(error_message, max_error_message_chars, "Failed to write to socket (are you writing to GPU memory?): sock_error=%d", sock_error);
        return false;
      }
    } else {
      // There was some progress, so keep trying
      total_bytes_sent += bytes_written;
      addr = (void *)((char *)addr + bytes_written);
      // See if the timeout needs to be updated
      if (bytes_written > 0 && !got_some_data) {
        start_time = clockTimeNanoseconds();
        got_some_data = true;
      }
    }
  }

  return true;
}

bool socketSend(TakyonSocket socket, void *addr, size_t bytes_to_write, bool is_polling, int64_t timeout_ns, bool *timed_out_ret, char *error_message, int max_error_message_chars) {
  *timed_out_ret = false;
  if (is_polling) {
    return socket_send_polling(socket, addr, bytes_to_write, timeout_ns, timed_out_ret, error_message, max_error_message_chars);
  } else {
    return socket_send_event_driven(socket, addr, bytes_to_write, timeout_ns, timed_out_ret, error_message, max_error_message_chars);
  }
}

static bool socket_recv_event_driven(TakyonSocket socket_fd, void *data_ptr, size_t bytes_to_read, int64_t timeout_ns, bool *timed_out_ret, char *error_message, int max_error_message_chars) {
  *timed_out_ret = false;

  // Set the timeout on the socket
  if (!setSocketTimeout(socket_fd, SO_RCVTIMEO, timeout_ns, error_message, max_error_message_chars)) return false;
  bool got_some_data = false;

  char *data = data_ptr;
  size_t total_bytes_read = 0;
  while (total_bytes_read < bytes_to_read) {
    size_t left_over = bytes_to_read - total_bytes_read;
    int bytes_to_read2 = left_over > INT_MAX ? INT_MAX : (int)left_over;
    int read_bytes = recv(socket_fd, data+total_bytes_read, bytes_to_read2, 0);
    if (read_bytes == 0) {
      // The socket was gracefully close
      snprintf(error_message, max_error_message_chars, "Remote side of socket has been closed");
      return false;
    } else if (read_bytes == SOCKET_ERROR) {
      int sock_error = WSAGetLastError();
      if (sock_error == WSAEWOULDBLOCK) {
        // Timed out
        if (total_bytes_read == 0) {
          // No problem, timed out
          *timed_out_ret = true;
          return true;
        } else {
          // This is bad... the connection might have gone down while in the middle of receiving
          snprintf(error_message, max_error_message_chars, "Timed out in the middle of a recv transfer");
          return false;
        }
      } else if (sock_error == WSAEINTR) {
        // Interrupted by external signal. Just try again
        read_bytes = 0;
      } else {
        snprintf(error_message, max_error_message_chars, "Failed to read from socket: sock_error=%d", sock_error);
        return false;
      }
    }
    total_bytes_read += read_bytes;
  }

  return true;
}

static bool socket_recv_polling(TakyonSocket socket_fd, void *data_ptr, size_t bytes_to_read, int64_t timeout_ns, bool *timed_out_ret, char *error_message, int max_error_message_chars) {
  *timed_out_ret = false;
  int64_t start_time = clockTimeNanoseconds();
  bool got_some_data = false;

  char *data = data_ptr;
  size_t total_bytes_read = 0;
  while (total_bytes_read < bytes_to_read) {
    size_t left_over = bytes_to_read - total_bytes_read;
    int bytes_to_read2 = left_over > INT_MAX ? INT_MAX : (int)left_over;
    int read_bytes = recv(socket_fd, data+total_bytes_read, bytes_to_read2, 0);
    if (read_bytes == 0) {
      // The socket was gracefully closed
      snprintf(error_message, max_error_message_chars, "Remote side of socket has been closed");
      return false;
    } else if (read_bytes == SOCKET_ERROR) {
      int sock_error = WSAGetLastError();
      if (sock_error == WSAEWOULDBLOCK) {
        // Nothing read, but no errors
        if (timeout_ns >= 0) {
          int64_t ellapsed_time = clockTimeNanoseconds() - start_time;
          if (ellapsed_time >= timeout_ns) {
            // Timed out
            if (total_bytes_read == 0) {
              *timed_out_ret = true;
              return true;
            } else {
              // This is bad... the connection might have gone down while in the middle of receiving
              snprintf(error_message, max_error_message_chars, "Timed out in the middle of a recv transfer");
              return false;
            }
          }
        }
        read_bytes = 0;
      } else if (sock_error == WSAEINTR) {
        // Interrupted by external signal. Just try again
        read_bytes = 0;
      } else {
        snprintf(error_message, max_error_message_chars, "Failed to read from socket: sock_error=%d", sock_error);
        return false;
      }
    }
    total_bytes_read += read_bytes;
    // See if the timeout should be updated
    if (read_bytes > 0 && !got_some_data) {
      start_time = clockTimeNanoseconds();
      got_some_data = true;
    }
  }

  return true;
}

bool socketRecv(TakyonSocket socket_fd, void *data_ptr, size_t bytes_to_read, bool is_polling, int64_t timeout_ns, bool *timed_out_ret, char *error_message, int max_error_message_chars) {
  *timed_out_ret = false;
  if (is_polling) {
    return socket_recv_polling(socket_fd, data_ptr, bytes_to_read, timeout_ns, timed_out_ret, error_message, max_error_message_chars);
  } else {
    return socket_recv_event_driven(socket_fd, data_ptr, bytes_to_read, timeout_ns, timed_out_ret, error_message, max_error_message_chars);
  }
}

static bool wait_for_socket_read_activity(TakyonSocket socket, int64_t timeout_ns, char *error_message, int max_error_message_chars) {
  int timeout_in_milliseconds;
  int num_fds_with_activity;
  struct pollfd poll_fd_list[1];
  unsigned long num_fds;

  // Will come back here if EINTR is detected while waiting
 restart_wait:

  if (timeout_ns < 0) {
    timeout_in_milliseconds = -1;
  } else {
    timeout_in_milliseconds = (int)(timeout_ns / 1000000);
  }

  poll_fd_list[0].fd = socket;
  poll_fd_list[0].events = POLLIN;
  poll_fd_list[0].revents = 0;
  num_fds = 1;

  // Wait for activity on the socket
  num_fds_with_activity = WSAPoll(poll_fd_list, num_fds, timeout_in_milliseconds);
  if (num_fds_with_activity == 0) {
    // Can't report the time out
    snprintf(error_message, max_error_message_chars, "Failed to listen for connection. Timed out waiting for socket read activity");
    return false;
  } else if (num_fds_with_activity != 1) {
    // Select has an error
    int sock_error = WSAGetLastError();
    if (sock_error == WSAEINTR) {
      // Interrupted by external signal. Just try again
      goto restart_wait;
    }
    snprintf(error_message, max_error_message_chars, "Failed to listen for connection. Error while waiting for socket read activity: sock_error=%d", sock_error);
    return false;
  } else {
    // Got activity.
    return true;
  }
}

static bool get_port_number_filename(const char *socket_name, char *port_number_filename, int max_chars, char *error_message, int max_error_message_chars) {
  char system_folder[MAX_PATH+1];
  snprintf(system_folder, MAX_PATH, "%s", ".\\"); // Use the current working folder
  unsigned int folder_bytes = (unsigned int)strlen(system_folder);
  if ((folder_bytes == 0) || (folder_bytes > MAX_PATH)) {
    snprintf(error_message, max_error_message_chars, "System folder name is bigger than %d chars", MAX_PATH);
    return false;
  }
  if (system_folder[folder_bytes-1] == '\\') {
    snprintf(port_number_filename, max_chars, "%stemp_%s.txt", system_folder, socket_name);
  } else {
    snprintf(port_number_filename, max_chars, "%s\\temp_%s.txt", system_folder, socket_name);
  }
  return true;
}

static bool socketSetNoDelay(TakyonSocket socket_fd, bool use_it, char *error_message, int max_error_message_chars) {
  // This will allow small messages to be sent right away instead of letting the socket wait to buffer up 8k of data.
  BOOL use_no_delay = use_it ? 1 : 0;
  if (setsockopt(socket_fd, IPPROTO_TCP, TCP_NODELAY, (char *)&use_no_delay, sizeof(use_no_delay)) != 0) {
    snprintf(error_message, max_error_message_chars, "Failed to set socket to TCP_NODELAY. sock_error=%d", WSAGetLastError());
    return false;
  }
  return true;
}

bool socketSetKernelRecvBufferingSize(TakyonSocket socket_fd, int bytes, char *error_message, int max_error_message_chars) {
  // Allow datagrams to be buffered in the OS to avoid dropping packets
  if (setsockopt(socket_fd, SOL_SOCKET, SO_RCVBUF, (char *)&bytes, sizeof(bytes)) != 0) {
    snprintf(error_message, max_error_message_chars, "Failed to set socket to SO_RCVBUF. sock_error=%d", WSAGetLastError());
    return false;
  }
  return true;
}

bool socketCreateLocalClient(const char *socket_name, TakyonSocket *socket_fd_ret, int64_t timeout_ns, char *error_message, int max_error_message_chars) {
  if (!windows_socket_manager_init(error_message, max_error_message_chars)) {
    return false;
  }

  TakyonSocket socket_fd = 0;

  // Keep trying until a connection is made or timeout
  int64_t start_time = clockTimeNanoseconds();
  char *ip_addr = "127.0.0.1"; // Local socket
  while (1) {
    unsigned short port_number = 0;
    socket_fd = 0;

    // Get the published port number
    {
      char port_number_filename[MAX_TEMP_FILENAME_CHARS];
      if (!get_port_number_filename(socket_name, port_number_filename, MAX_TEMP_FILENAME_CHARS, error_message, max_error_message_chars)) {
        // Error message already set
        return false;
      }
      FILE *fp = fopen(port_number_filename, "r");
      if (fp == NULL) {
        if (errno != ENOENT) {
          snprintf(error_message, max_error_message_chars, "Failed to open file '%s' to get published port number. errno=%d", port_number_filename, errno);
          return false;
        }
        // Remote side not ready yet, Port number is still 0, so a retry will occur
      } else {
        // Read the port number
        int tokens = fscanf(fp, "%hu", &port_number);
        if ((tokens != 1) || (port_number <= 0)) {
          // The file might be in the act of being created or destroyed.
          port_number = 0;
        }
        fclose(fp);
      }
    }

    if (port_number > 0) {
      socket_fd = socket(AF_INET, SOCK_STREAM, 0);
      if (socket_fd == INVALID_SOCKET) {
        snprintf(error_message, max_error_message_chars, "Could not create local TCP socket. sock_error=%d", WSAGetLastError());
        return false;
      }

      // Set up server connection info
      struct sockaddr_in server_addr;
      memset(&server_addr, 0, sizeof(server_addr));
      server_addr.sin_family = AF_INET;
      server_addr.sin_port = htons(port_number);
      int status = inet_pton(AF_INET, ip_addr, &server_addr.sin_addr);
      if (status == 0) {
        snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. Invalid name.", ip_addr);
        closesocket(socket_fd);
        return false;
      } else if (status == -1) {
        snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. sock_error=%d", ip_addr, WSAGetLastError());
        closesocket(socket_fd);
        return false;
      }

      if (connect(socket_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) != SOCKET_ERROR) {
        // Connection was made
        break;
      }
      int sock_error = WSAGetLastError();
      /*+ check for EBADF? Test with reduce example: graph_mp.txt */
      if (sock_error != WSAECONNREFUSED) {
        snprintf(error_message, max_error_message_chars, "Could not connect local socket. sock_error=%d", sock_error);
        closesocket(socket_fd);
        return false;
      }
      closesocket(socket_fd);
    }

    // Check if time to timeout
    int64_t ellapsed_time_ns = clockTimeNanoseconds() - start_time;
    if (timeout_ns >= 0) {
      if (ellapsed_time_ns >= timeout_ns) {
        // Timed out
        snprintf(error_message, max_error_message_chars, "Timed out waiting for connection on local client socket");
        return false;
      }
    }
    // Context switch out for a little time to allow remote side to get going.
    clockSleepYield(MICROSECONDS_BETWEEN_CONNECT_ATTEMPTS);
  }

  // TCP sockets use the Nagle algorithm, so need to turn it off to get good performance.
  if (!socketSetNoDelay(socket_fd, 1, error_message, max_error_message_chars)) {
    // Error message already set
    closesocket(socket_fd);
    return false;
  }

  *socket_fd_ret = socket_fd;

  return true;
}

bool socketCreateTcpClient(const char *ip_addr, uint16_t port_number, TakyonSocket *socket_fd_ret, int64_t timeout_ns, char *error_message, int max_error_message_chars) {
  if (!windows_socket_manager_init(error_message, max_error_message_chars)) {
    return false;
  }

  // Resolve the IP address
  char resolved_ip_addr[MAX_IP_ADDR_CHARS];
  if (!hostnameToIpaddr(AF_INET, ip_addr, resolved_ip_addr, MAX_IP_ADDR_CHARS, error_message, max_error_message_chars)) {
    // Error message already set
    return false;
  }

  // Keep trying until a connection is made or timeout
  TakyonSocket socket_fd = 0;
  int64_t start_time = clockTimeNanoseconds();
  while (1) {
    // Create a socket file descriptor
    socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd == INVALID_SOCKET) {
      snprintf(error_message, max_error_message_chars, "Could not create TCP socket. sock_error=%d", WSAGetLastError());
      return false;
    }

    // Set up server connection info
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port_number);
    int status = inet_pton(AF_INET, resolved_ip_addr, &server_addr.sin_addr);
    if (status == 0) {
      snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. Invalid name.", ip_addr);
      closesocket(socket_fd);
      return false;
    } else if (status == -1) {
      snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. sock_error=%d", ip_addr, WSAGetLastError());
      closesocket(socket_fd);
      return false;
    }

    // Wait for the connection
    if (connect(socket_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) == SOCKET_ERROR) {
      int sock_error = WSAGetLastError();
      closesocket(socket_fd);
      /*+ check for EBADF? Test with reduce example: graph_mp.txt */
      if (sock_error == WSAECONNREFUSED) {
        // Server side is not ready yet
        int64_t ellapsed_time_ns = clockTimeNanoseconds() - start_time;
        if (timeout_ns >= 0) {
          if (ellapsed_time_ns >= timeout_ns) {
            // Timed out
            snprintf(error_message, max_error_message_chars, "Timed out waiting for connection");
            return false;
          }
        }
        // Context switch out for a little time to allow remote side to get going.
        clockSleepYield(MICROSECONDS_BETWEEN_CONNECT_ATTEMPTS);
        // Try connecting again now
      } else {
        snprintf(error_message, max_error_message_chars, "Could not connect socket. sock_error=%d", sock_error);
        return false;
      }
    } else {
      break;
    }
  }

  // TCP sockets use the Nagle algorithm, so need to turn it off to get good performance.
  if (!socketSetNoDelay(socket_fd, 1, error_message, max_error_message_chars)) {
    // Error message already set
    closesocket(socket_fd);
    return false;
  }

  *socket_fd_ret = socket_fd;

  return true;
}

#ifdef ENABLE_EPHEMERAL_PORT_MANAGER
bool socketCreateEphemeralTcpClient(const char *ip_addr, const char *provider_name, uint32_t path_id, TakyonSocket *socket_fd_ret, int64_t timeout_ns, uint64_t verbosity, char *error_message, int max_error_message_chars) {
  if (!windows_socket_manager_init(error_message, max_error_message_chars)) {
    return false;
  }

  int64_t start_time = clockTimeNanoseconds();

  // Resolve the IP address
  char resolved_ip_addr[MAX_IP_ADDR_CHARS];
  if (!hostnameToIpaddr(AF_INET, ip_addr, resolved_ip_addr, MAX_IP_ADDR_CHARS, error_message, max_error_message_chars)) {
    // Error message already set
    return false;
  }

  // Keep trying until a connection is made or timed out
  TakyonSocket socket_fd = 0;
  uint16_t ephemeral_port_number = 0;
  while (1) {
    // Wait for the ephemeral port number to get multicasted (this is in the while loop incase a stale port number is initially grabbed, and allows it to refresh)
    bool timed_out = false;
    ephemeral_port_number = ephemeralPortManagerGet(provider_name, path_id, timeout_ns, &timed_out, verbosity, error_message, max_error_message_chars);
    if (ephemeral_port_number == 0) {
      if (timed_out) {
        snprintf(error_message, max_error_message_chars, "TCP client socket timed out waiting for the ephemeral port number");
      } else {
        // Error message already set
      }
      return false;
    }

    // Create a socket file descriptor
    socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd == INVALID_SOCKET) {
      snprintf(error_message, max_error_message_chars, "Could not create TCP socket. sock_error=%d", WSAGetLastError());
      return false;
    }

    // Set up server connection info
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(ephemeral_port_number);
    int status = inet_pton(AF_INET, resolved_ip_addr, &server_addr.sin_addr);
    if (status == 0) {
      snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. Invalid name.", ip_addr);
      closesocket(socket_fd);
      return false;
    } else if (status == -1) {
      snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. sock_error=%d", ip_addr, WSAGetLastError());
      closesocket(socket_fd);
      return false;
    }

    // Try to connect. If the server side is ready, then the connection will be made, otherwise the function will return with an error.
    // NOTE: Even though the ephemeral port number has arrived, it does not guarantee the server is in the connect state
    if (connect(socket_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) == SOCKET_ERROR) {
      int sock_error = WSAGetLastError();
      closesocket(socket_fd);
      /*+ check for EBADF? Test with reduce example: graph_mp.txt */
      if (sock_error == WSAECONNREFUSED) {
        // Server side is not ready yet
        int64_t ellapsed_time_ns = clockTimeNanoseconds() - start_time;
        if (timeout_ns >= 0) {
          if (ellapsed_time_ns >= timeout_ns) {
            // Timed out
            snprintf(error_message, max_error_message_chars, "Timed out waiting for connection");
            return false;
          }
        }
        // Context switch out for a little time to allow remote side to get going.
        clockSleepYield(MICROSECONDS_BETWEEN_CONNECT_ATTEMPTS);
        // Try connecting again now
      } else {
        snprintf(error_message, max_error_message_chars, "Could not connect socket. sock_error=%d", sock_error);
        return false;
      }
    } else {
      break;
    }
  }

  // The connect is made

  // Send out a message to let the other nodes know the path no longer needs the ephemeral port number
  ephemeralPortManagerRemove(provider_name, path_id, ephemeral_port_number);

  // TCP sockets use the Nagle algorithm, so need to turn it off to get good latency (at the expense of added network traffic and under utilized TCP packets).
  if (!socketSetNoDelay(socket_fd, 1, error_message, max_error_message_chars)) {
    // Error message already set
    closesocket(socket_fd);
    return false;
  }

  *socket_fd_ret = socket_fd;

  return true;
}
#endif

bool socketCreateLocalServer(const char *socket_name, TakyonSocket *socket_fd_ret, int64_t timeout_ns, char *error_message, int max_error_message_chars) {
  if (!windows_socket_manager_init(error_message, max_error_message_chars)) {
    return false;
  }

  TakyonSocket listening_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (listening_fd == INVALID_SOCKET) {
    snprintf(error_message, max_error_message_chars, "Could not create TCP listening socket. sock_error=%d", WSAGetLastError());
    return false;
  }

  // Create server connection info
  unsigned short port_number = 0; // This will tell the OS to find an available port number
  const char *ip_addr = "127.0.0.1"; // This will force interprocess communication
  struct sockaddr_in server_addr;
  memset((char*)&server_addr, 0, sizeof(server_addr));
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(port_number); // NOTE: Use 0 to have the system find an unused port number
  // NOTE: htonl(INADDR_ANY) would allow any interface to listen
  int status = inet_pton(AF_INET, ip_addr, &server_addr.sin_addr);
  if (status == 0) {
    snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. Invalid name.", ip_addr);
    closesocket(listening_fd);
    return false;
  } else if (status == -1) {
    snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. sock_error=%d", ip_addr, WSAGetLastError());
    closesocket(listening_fd);
    return false;
  }

  // IMPORTANT: SO_REUSEADDR should not be used with ephemeral port numbers

  // This is when the port number will be determined
  if (bind(listening_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) == SOCKET_ERROR) {
    int sock_error = WSAGetLastError();
    if (sock_error == WSAEADDRINUSE) {
      snprintf(error_message, max_error_message_chars, "Could not bind TCP socket. The address is already in use or in a time wait state. May need to use the option '-reuse'");
    } else {
      snprintf(error_message, max_error_message_chars, "Could not bind TCP socket. sock_error=%d", WSAGetLastError());
    }
    closesocket(listening_fd);
    return false;
  }

  // If using an ephemeral port number, then use this to know the actual port number and pass it to the rest of the system
  {
    SOCKADDR_IN socket_info;
    int len = sizeof(socket_info);
    memset(&socket_info, 0, len);
    if (getsockname(listening_fd, (struct sockaddr *)&socket_info, &len) == SOCKET_ERROR) {
      snprintf(error_message, max_error_message_chars, "Could not get the socket name info. sock_error=%d", WSAGetLastError());
      closesocket(listening_fd);
      return false;
    }
    port_number = htons(socket_info.sin_port);
  }

  // Write the port number to a temporary file
  char port_number_filename[MAX_TEMP_FILENAME_CHARS];
  {
    if (!get_port_number_filename(socket_name, port_number_filename, MAX_TEMP_FILENAME_CHARS, error_message, max_error_message_chars)) {
      // Error message already set
      closesocket(listening_fd);
      return false;
    }
    FILE *fp = fopen(port_number_filename, "w");
    if (fp == NULL) {
      snprintf(error_message, max_error_message_chars, "Failed to open file '%s' to temporarily publish port number. errno=%d", port_number_filename, errno);
      closesocket(listening_fd);
      return false;
    }
    fprintf(fp, "%d\n", port_number);
    fclose(fp);
  }
  
  // Set the number of simultaneous connections that can be made
  if (listen(listening_fd, 1) == SOCKET_ERROR) {
    snprintf(error_message, max_error_message_chars, "Could not listen on TCP socket. sock_error=%d", WSAGetLastError());
    remove(port_number_filename);
    closesocket(listening_fd);
    return false;
  }

  // Wait for a client to ask for a connection
  if (timeout_ns >= 0) {
    if (!wait_for_socket_read_activity(listening_fd, timeout_ns, error_message, max_error_message_chars)) {
      // Error message already set
      remove(port_number_filename);
      closesocket(listening_fd);
      return false;
    }
  }

  // This blocks until a connection is acctual made
  TakyonSocket socket_fd = accept(listening_fd, NULL, NULL);
  if (socket_fd == INVALID_SOCKET) {
    snprintf(error_message, max_error_message_chars, "Could not accept TCP socket. sock_error=%d", WSAGetLastError());
    remove(port_number_filename);
    closesocket(listening_fd);
    return false;
  }

  // TCP sockets use the Nagle algorithm, so need to turn it off to get good performance.
  if (!socketSetNoDelay(socket_fd, 1, error_message, max_error_message_chars)) {
    // Error message already set
    remove(port_number_filename);
    closesocket(listening_fd);
    closesocket(socket_fd);
    return false;
  }

  closesocket(listening_fd);
  remove(port_number_filename);
  *socket_fd_ret = socket_fd;

  return true;
}

bool socketCreateTcpServer(const char *ip_addr, uint16_t port_number, bool allow_reuse, TakyonSocket *socket_fd_ret, int64_t timeout_ns, char *error_message, int max_error_message_chars) {
  if (!windows_socket_manager_init(error_message, max_error_message_chars)) {
    return false;
  }

  TakyonSocket listening_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (listening_fd == INVALID_SOCKET) {
    snprintf(error_message, max_error_message_chars, "Could not create TCP listening socket. sock_error=%d", WSAGetLastError());
    return false;
  }

  // Create server connection info
  struct sockaddr_in server_addr;
  memset((char*)&server_addr, 0, sizeof(server_addr));
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(port_number); // NOTE: Use 0 to have the system find an unused port number
  if (strcmp(ip_addr, "Any") == 0) {
    // NOTE: htonl(INADDR_ANY) will allow any IP interface to listen
    server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
  } else {
    // Resolve the IP address
    char resolved_ip_addr[MAX_IP_ADDR_CHARS];
    if (!hostnameToIpaddr(AF_INET, ip_addr, resolved_ip_addr, MAX_IP_ADDR_CHARS, error_message, max_error_message_chars)) {
      // Error message already set
      closesocket(listening_fd);
      return false;
    }
    // Listen on a specific IP interface
    int status = inet_pton(AF_INET, resolved_ip_addr, &server_addr.sin_addr);
    if (status == 0) {
      snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. Invalid name.", ip_addr);
      closesocket(listening_fd);
      return false;
    } else if (status == -1) {
      snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. sock_error=%d", ip_addr, WSAGetLastError());
      closesocket(listening_fd);
      return false;
    }
  }

  if (allow_reuse) {
    // This will allow a previously closed socket, that is still in the TIME_WAIT stage, to be used.
    // This may accur if the application did not exit gracefully on a previous run.
    BOOL allow = 1;
    if (setsockopt(listening_fd, SOL_SOCKET, SO_REUSEADDR, (char *)&allow, sizeof(allow)) != 0) {
      snprintf(error_message, max_error_message_chars, "Failed to set socket to SO_REUSEADDR. sock_error=%d", WSAGetLastError());
      closesocket(listening_fd);
      return false;
    }
  }

  if (bind(listening_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) == SOCKET_ERROR) {
    int sock_error = WSAGetLastError();
    if (sock_error == WSAEADDRINUSE) {
      snprintf(error_message, max_error_message_chars, "Could not bind TCP socket. The address is already in use or in a time wait state. May need to use the option '-reuse'");
    } else {
      snprintf(error_message, max_error_message_chars, "Could not bind TCP socket. sock_error=%d", WSAGetLastError());
    }
    closesocket(listening_fd);
    return false;
  }

  // If using an ephemeral port number, then use this to know the actual port number and pass it to the rest of the system
  /*
  socklen_t len = sizeof(sin);
  if (getsockname(listen_sock, (struct sockaddr *)&sin, &len) < 0) {
  // Handle error here
  }
  // You can now get the port number with ntohs(sin.sin_port).
  */

  // Set the number of simultaneous connections that can be made
  if (listen(listening_fd, 1) == SOCKET_ERROR) {
    snprintf(error_message, max_error_message_chars, "Could not listen on TCP socket. sock_error=%d", WSAGetLastError());
    closesocket(listening_fd);
    return false;
  }

  // Wait for a client to ask for a connection
  if (timeout_ns >= 0) {
    if (!wait_for_socket_read_activity(listening_fd, timeout_ns, error_message, max_error_message_chars)) {
      // Error message already set
      closesocket(listening_fd);
      return false;
    }
  }

  // This blocks until a connection is acctual made
  TakyonSocket socket_fd = accept(listening_fd, NULL, NULL);
  if (socket_fd == INVALID_SOCKET) {
    snprintf(error_message, max_error_message_chars, "Could not accept TCP socket. sock_error=%d", WSAGetLastError());
    closesocket(listening_fd);
    return false;
  }

  // TCP sockets use the Nagle algorithm, so need to turn it off to get good performance.
  if (!socketSetNoDelay(socket_fd, 1, error_message, max_error_message_chars)) {
    // Error message already set
    closesocket(listening_fd);
    closesocket(socket_fd);
    return false;
  }

  closesocket(listening_fd);
  *socket_fd_ret = socket_fd;

  return true;
}

#ifdef ENABLE_EPHEMERAL_PORT_MANAGER
bool socketCreateEphemeralTcpServer(const char *ip_addr, const char *provider_name, uint32_t path_id, TakyonSocket *socket_fd_ret, int64_t timeout_ns, char *error_message, int max_error_message_chars) {
  if (!windows_socket_manager_init(error_message, max_error_message_chars)) {
    return false;
  }

  TakyonSocket listening_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (listening_fd == INVALID_SOCKET) {
    snprintf(error_message, max_error_message_chars, "Could not create TCP listening socket. sock_error=%d", WSAGetLastError());
    return false;
  }

  // Create server connection info
  struct sockaddr_in server_addr;
  memset((char*)&server_addr, 0, sizeof(server_addr));
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(0); // NOTE: Use 0 to have the system find an unused port number
  if (strcmp(ip_addr, "Any") == 0) {
    // NOTE: htonl(INADDR_ANY) will allow any IP interface to listen
    server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
  } else {
    // Resolve the IP address
    char resolved_ip_addr[MAX_IP_ADDR_CHARS];
    if (!hostnameToIpaddr(AF_INET, ip_addr, resolved_ip_addr, MAX_IP_ADDR_CHARS, error_message, max_error_message_chars)) {
      // Error message already set
      closesocket(listening_fd);
      return false;
    }
    // Listen on a specific IP interface
    int status = inet_pton(AF_INET, resolved_ip_addr, &server_addr.sin_addr);
    if (status == 0) {
      snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. Invalid name.", ip_addr);
      closesocket(listening_fd);
      return false;
    } else if (status == -1) {
      snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. sock_error=%d", ip_addr, WSAGetLastError());
      closesocket(listening_fd);
      return false;
    }
  }

  // IMPORTANT: SO_REUSEADDR should not be used with ephemeral port numbers

  if (bind(listening_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) == SOCKET_ERROR) {
    int sock_error = WSAGetLastError();
    if (sock_error == WSAEADDRINUSE) {
      snprintf(error_message, max_error_message_chars, "Could not bind TCP socket. The address is already in use or in a time wait state. May need to use the option '-reuse'");
    } else {
      snprintf(error_message, max_error_message_chars, "Could not bind TCP socket. sock_error=%d", WSAGetLastError());
    }
    closesocket(listening_fd);
    return false;
  }

  // If using an ephemeral port number, then use this to know the actual port number and pass it to the rest of the system
  struct sockaddr_in sin;
  socklen_t len = sizeof(sin);
  if (getsockname(listening_fd, (struct sockaddr *)&sin, &len) == SOCKET_ERROR) {
    snprintf(error_message, max_error_message_chars, "Failed to get ephemeral port number from listening socket for IP addr (%s). sock_error=%d", ip_addr, WSAGetLastError());
    closesocket(listening_fd);
    return false;
  }
  uint16_t ephemeral_port_number = ntohs(sin.sin_port);

  // Set the number of simultaneous connections that can be made
  if (listen(listening_fd, 1) == SOCKET_ERROR) {
    snprintf(error_message, max_error_message_chars, "Could not listen on TCP socket. sock_error=%d", WSAGetLastError());
    closesocket(listening_fd);
    return false;
  }

  // Multicast this out to all active Takyon endpoints
  ephemeralPortManagerSet(provider_name, path_id, ephemeral_port_number);

  // Wait for a client to ask for a connection
  if (timeout_ns >= 0) {
    if (!wait_for_socket_read_activity(listening_fd, timeout_ns, error_message, max_error_message_chars)) {
      // Error message already set
      ephemeralPortManagerRemoveLocally(provider_name, path_id);
      closesocket(listening_fd);
      return false;
    }
  }

  // This blocks until a connection is acctual made
  TakyonSocket socket_fd = accept(listening_fd, NULL, NULL);
  ephemeralPortManagerRemoveLocally(provider_name, path_id);
  if (socket_fd == INVALID_SOCKET) {
    snprintf(error_message, max_error_message_chars, "Could not accept TCP socket. sock_error=%d", WSAGetLastError());
    closesocket(listening_fd);
    return false;
  }

  // TCP sockets use the Nagle algorithm, so need to turn it off to get good performance.
  if (!socketSetNoDelay(socket_fd, 1, error_message, max_error_message_chars)) {
    // Error message already set
    closesocket(listening_fd);
    closesocket(socket_fd);
    return false;
  }

  closesocket(listening_fd);
  *socket_fd_ret = socket_fd;

  return true;
}
#endif

bool pipeCreate(int *read_pipe_fd_ret, int *write_pipe_fd_ret, char *error_message, int max_error_message_chars) {
  // IMPORTANT: Windows does not have pipes, so may need to poll on connection to see if it's disconnected
  *read_pipe_fd_ret = -1;
  *write_pipe_fd_ret = -1;
  return true;
}

bool socketWaitForDisconnectActivity(TakyonSocket socket_fd, int read_pipe_fd, bool *got_socket_activity_ret, char *error_message, int max_error_message_chars) {
  // NOTE: This function is used to detect when a socket closes gracefull or not.
  //       A pipe is not use here since on Windows, the thread listening for disconnect will
  //       wake up after the socket if gracefully closed, and then the thread can quit.

  int timeout_in_milliseconds = -1; // Wait forever
  int num_fds_with_activity;
  struct pollfd poll_fd_list[1];
  unsigned long num_fds = 1;

  // Will come back here if EINTR is detected while waiting
 restart_wait:

  poll_fd_list[0].fd = socket_fd;
  poll_fd_list[0].events = POLLIN;
  poll_fd_list[0].revents = 0;

  // Wait for activity on the socket
  num_fds_with_activity = WSAPoll(poll_fd_list, num_fds, timeout_in_milliseconds);
  if (num_fds_with_activity == 0) {
    // Can't report the time out
    snprintf(error_message, max_error_message_chars, "Timed out waiting for socket activity");
    return false;
  } else if (num_fds_with_activity < 0) {
    // Select has an error */
    int sock_error = WSAGetLastError();
    if (sock_error == WSAEINTR) {
      // Interrupted by external signal. Just try again
      goto restart_wait;
    }
    snprintf(error_message, max_error_message_chars, "Error while waiting for socket activity: sock_error=%d", sock_error);
    return false;
  } else {
    // Got activity.
    *got_socket_activity_ret = 1;
    return true;
  }
}

bool pipeWakeUpPollFunction(int write_pipe_fd, char *error_message, int max_error_message_chars) {
  // IMPORTANT: Nothing to do since WSAPoll function should bewoken up periodically to check if the thread should shut down */
  return true;
}

void pipeDestroy(int read_pipe_fd, int write_pipe_fd) {
  // Nothing to do
}

bool socketSetBlocking(TakyonSocket socket_fd, bool is_blocking, char *error_message, int max_error_message_chars) {
  unsigned long blocking_mode = is_blocking ? 0 : 1;
  if (ioctlsocket(socket_fd, FIONBIO, &blocking_mode) == SOCKET_ERROR) {
    snprintf(error_message, max_error_message_chars, "Failed to set socket to %s. sock_error=%d", is_blocking ? "blocking" : "non-blocking", WSAGetLastError());
    return false;
  }
  return true;
}

void socketClose(TakyonSocket socket_fd) {
  closesocket(socket_fd);
}

static bool datagram_send_polling(TakyonSocket socket_fd, void *sock_in_addr, void *addr, size_t bytes_to_write, int64_t timeout_ns, bool *timed_out_ret, char *error_message, int max_error_message_chars) {
  *timed_out_ret = false;
  int64_t start_time = clockTimeNanoseconds();

  if (bytes_to_write > INT_MAX) {
    snprintf(error_message, max_error_message_chars, "Data size is greater than what an 'int' can hold.");
    return false;
  }

  while (1) {
    int flags = 0;
    int bytes_written = sendto(socket_fd, addr, (int)bytes_to_write, flags, (struct sockaddr *)sock_in_addr, sizeof(struct sockaddr_in));
    if (bytes_written == bytes_to_write) {
      // Transfer completed
      break;
    }
    if (bytes_written == SOCKET_ERROR) {
      int sock_error = WSAGetLastError();
      if (sock_error == WSAEWOULDBLOCK) {
        // Nothing written, but no errors
        if (timeout_ns >= 0) {
          int64_t ellapsed_time = clockTimeNanoseconds() - start_time;
          if (ellapsed_time >= timeout_ns) {
            // Timed out
            *timed_out_ret = true;
            return true;
          }
        }
        bytes_written = 0;
      } else if (sock_error == WSAEMSGSIZE) {
        snprintf(error_message, max_error_message_chars, "Failed to send datagram message. %lld bytes exceeds the datagram size", (long long)bytes_to_write);
        return false;
      } else if (sock_error == WSAEINTR) {
        // Interrupted by external signal. Just try again
        bytes_written = 0;
      } else {
        snprintf(error_message, max_error_message_chars, "Failed to write to socket (are you writing to GPU memory?): error=%d", sock_error);
        return false;
      }
    } else if (bytes_written == 0) {
      // Keep trying
    } else {
      snprintf(error_message, max_error_message_chars, "Wrote a partial datagram: bytes_sent=%d, bytes_to_send=%lld", bytes_written, bytes_to_write);
      return false;
    }
  }

  return true;
}

static bool datagram_send_event_driven(TakyonSocket socket_fd, void *sock_in_addr, void *addr, size_t bytes_to_write, int64_t timeout_ns, bool *timed_out_ret, char *error_message, int max_error_message_chars) {
  *timed_out_ret = false;

  if (bytes_to_write > INT_MAX) {
    snprintf(error_message, max_error_message_chars, "Data size is greater than what an 'int' can hold.");
    return false;
  }

  // Set the timeout on the socket
  if (!setSocketTimeout(socket_fd, SO_SNDTIMEO, timeout_ns, error_message, max_error_message_chars)) return false;

  while (1) {
    // Write the data
    int flags = 0;
    int bytes_written = sendto(socket_fd, addr, (int)bytes_to_write, flags, (struct sockaddr *)sock_in_addr, sizeof(struct sockaddr_in));
    if (bytes_written == bytes_to_write) {
      // Transfer completed
      return true;
    }

    // Something went wrong
    if (bytes_written == SOCKET_ERROR) {
      int sock_error = WSAGetLastError();
      if (sock_error == WSAEMSGSIZE) {
        snprintf(error_message, max_error_message_chars, "Failed to send datagram message. %lld bytes exceeds the allowable datagram size", (long long)bytes_to_write);
        return false;
      } else if (sock_error == WSAEWOULDBLOCK) { /*+ may need to also check WSAENETUNREACH which seems to be temporary, but probably still need to report as error */
        // Timed out
        *timed_out_ret = true;
        return true;
      } else if (sock_error == WSAEINTR) {
        // Interrupted by external signal. Just try again
      } else {
        snprintf(error_message, max_error_message_chars, "Failed to write to socket (are you writing to GPU memory?): error=%d", sock_error);
        return false;
      }
    } else if (bytes_written == 0) {
      // Keep trying
    } else {
      snprintf(error_message, max_error_message_chars, "Wrote a partial datagram: bytes_sent=%d, bytes_to_send=%lld", bytes_written, bytes_to_write);
      return false;
    }
  }

  return true;
}

static bool datagram_recv_event_driven(TakyonSocket socket_fd, void *data_ptr, size_t buffer_bytes, uint64_t *bytes_read_ret, int64_t timeout_ns, bool *timed_out_ret, char *error_message, int max_error_message_chars) {
  *timed_out_ret = false;

  if (buffer_bytes > INT_MAX) {
    snprintf(error_message, max_error_message_chars, "Buffer bytes is greater than what an 'int' can hold.");
    return false;
  }

  // Set the timeout on the socket
  if (!setSocketTimeout(socket_fd, SO_RCVTIMEO, timeout_ns, error_message, max_error_message_chars)) return false;

  while (1) {
    int flags = 0;
    int bytes_received = recvfrom(socket_fd, data_ptr, (int)buffer_bytes, flags, NULL, NULL);
    if (bytes_received > 0) {
      // Got a datagram
      *bytes_read_ret = (uint64_t)bytes_received;
      return true;
    }
    if (bytes_received == 0) {
      // Socket was probably closed by the remote side
      snprintf(error_message, max_error_message_chars, "Remote side of socket has been closed");
      return false;
    }
    if (bytes_received == SOCKET_ERROR) {
      int sock_error = WSAGetLastError();
      //*+*/printf("WSAEWOULDBLOCK=%d, WSAETIMEDOUT=%d, sock_error=%d\n", WSAEWOULDBLOCK, WSAETIMEDOUT, sock_error);
      if (sock_error == WSAEWOULDBLOCK || sock_error == WSAETIMEDOUT) {
        // Timed out
        *timed_out_ret = true;
        return true;
      } else if (sock_error == WSAEINTR) {
        // Interrupted by external signal. Just try again
      } else {
        snprintf(error_message, max_error_message_chars, "Failed to read from socket: error=%d", sock_error);
        return false;
      }
    }
  }
}

static bool datagram_recv_polling(TakyonSocket socket_fd, void *data_ptr, size_t buffer_bytes, uint64_t *bytes_read_ret, int64_t timeout_ns, bool *timed_out_ret, char *error_message, int max_error_message_chars) {
  *timed_out_ret = false;
  int64_t start_time = clockTimeNanoseconds();

  if (buffer_bytes > INT_MAX) {
    snprintf(error_message, max_error_message_chars, "Buffer bytes is greater than what an 'int' can hold.");
    return false;
  }

  while (1) {
    int flags = 0;
    int bytes_received = recvfrom(socket_fd, data_ptr, (int)buffer_bytes, flags, NULL, NULL);
    if (bytes_received > 0) {
      // Got a datagram
      *bytes_read_ret = (uint64_t)bytes_received;
      return true;
    }
    if (bytes_received == 0) {
      // Socket was probably closed by the remote side
      snprintf(error_message, max_error_message_chars, "Remote side of socket has been closed");
      return false;
    }
    if (bytes_received == SOCKET_ERROR) {
      int sock_error = WSAGetLastError();
      if (sock_error == WSAEWOULDBLOCK) {
        // Nothing read, but no errors
        if (timeout_ns >= 0) {
          int64_t ellapsed_time = clockTimeNanoseconds() - start_time;
          if (ellapsed_time >= timeout_ns) {
            // Timed out
            *timed_out_ret = true;
            return true;
          }
        }
      } else if (sock_error == WSAEINTR) {
        // Interrupted by external signal. Just try again
      } else {
        snprintf(error_message, max_error_message_chars, "Failed to read from socket: error=%d", sock_error);
        return false;
      }
    }
  }
}

bool socketCreateUnicastSender(const char *ip_addr, uint16_t port_number, TakyonSocket *socket_fd_ret, void **sock_in_addr_ret, char *error_message, int max_error_message_chars) {
  if (!windows_socket_manager_init(error_message, max_error_message_chars)) {
    return false;
  }

  // Resolve the IP address
  char resolved_ip_addr[MAX_IP_ADDR_CHARS];
  if (!hostnameToIpaddr(AF_INET, ip_addr, resolved_ip_addr, MAX_IP_ADDR_CHARS, error_message, max_error_message_chars)) {
    // Error message already set
    return false;
  }

  // Create a datagram socket on which to send.
  TakyonSocket socket_fd = socket(AF_INET, SOCK_DGRAM, 0);
  if (socket_fd == INVALID_SOCKET) {
    snprintf(error_message, max_error_message_chars, "Failed to create datagram sender socket: error=%d", WSAGetLastError());
    return false;
  }

  // Allocate structure to hold socket address information
  struct sockaddr_in *sock_in_addr = malloc(sizeof(struct sockaddr_in));
  if (sock_in_addr == NULL) {
    snprintf(error_message, max_error_message_chars, "Out of memory");
    closesocket(socket_fd);
    return false;
  }

  // Fill in the structure describing the destination
  memset((char *)sock_in_addr, 0, sizeof(struct sockaddr_in));
#ifdef __APPLE__
  sock_in_addr->sin_len = sizeof(struct sockaddr_in);
#endif
  sock_in_addr->sin_family = AF_INET;
  int status = inet_pton(AF_INET, resolved_ip_addr, &sock_in_addr->sin_addr);
  if (status == 0) {
    snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. Invalid name.", ip_addr);
    closesocket(socket_fd);
    free(sock_in_addr);
    return false;
  } else if (status == -1) {
    snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. sock_error=%d", ip_addr, WSAGetLastError());
    closesocket(socket_fd);
    free(sock_in_addr);
    return false;
  }
  sock_in_addr->sin_port = htons(port_number);
  *sock_in_addr_ret = sock_in_addr;

  *socket_fd_ret = socket_fd;
  return true;
}

bool socketDatagramSend(TakyonSocket socket_fd, void *sock_in_addr, void *addr, size_t bytes_to_write, bool is_polling, int64_t timeout_ns, bool *timed_out_ret, char *error_message, int max_error_message_chars) {
  *timed_out_ret = false;
  if (is_polling) {
    return datagram_send_polling(socket_fd, sock_in_addr, addr, bytes_to_write, timeout_ns, timed_out_ret, error_message, max_error_message_chars);
  } else {
    return datagram_send_event_driven(socket_fd, sock_in_addr, addr, bytes_to_write, timeout_ns, timed_out_ret, error_message, max_error_message_chars);
  }
}

bool socketCreateUnicastReceiver(const char *ip_addr, uint16_t port_number, bool allow_reuse, TakyonSocket *socket_fd_ret, char *error_message, int max_error_message_chars) {
  if (!windows_socket_manager_init(error_message, max_error_message_chars)) {
    return false;
  }

  // Create a datagram socket on which to receive.
  TakyonSocket socket_fd = socket(AF_INET, SOCK_DGRAM, 0);
  if (socket_fd == INVALID_SOCKET) {
    snprintf(error_message, max_error_message_chars, "Failed to create datagram socket: error=%d", WSAGetLastError());
    return false;
  }

  // Bind to the proper port number with the IP address specified as INADDR_ANY.
  // I.e. only allow packets received on the specified port number
  struct sockaddr_in localSock;
  memset((char *) &localSock, 0, sizeof(localSock));
#ifdef __APPLE__
  localSock.sin_len = sizeof(localSock);
#endif
  localSock.sin_family = AF_INET;
  localSock.sin_port = htons(port_number);
  if (strcmp(ip_addr, "Any") == 0) {
    // NOTE: htonl(INADDR_ANY) will allow any IP interface to listen
    localSock.sin_addr.s_addr = htonl(INADDR_ANY);
  } else {
    // Resolve the IP address
    char resolved_ip_addr[MAX_IP_ADDR_CHARS];
    if (!hostnameToIpaddr(AF_INET, ip_addr, resolved_ip_addr, MAX_IP_ADDR_CHARS, error_message, max_error_message_chars)) {
      // Error message already set
      closesocket(socket_fd);
      return false;
    }
    // Listen on a specific IP interface
    int status = inet_pton(AF_INET, resolved_ip_addr, &localSock.sin_addr);
    if (status == 0) {
      snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. Invalid name.", ip_addr);
      closesocket(socket_fd);
      return false;
    } else if (status == -1) {
      snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. sock_error=%d", ip_addr, WSAGetLastError());
      closesocket(socket_fd);
      return false;
    }
  }

  if (allow_reuse) {
    // This will allow a previously closed socket, that is still in the TIME_WAIT stage, to be used.
    // This may accur if the application did not exit gracefully on a previous run.
    BOOL allow = 1;
    if (setsockopt(socket_fd, SOL_SOCKET, SO_REUSEADDR, (char *)&allow, sizeof(allow)) != 0) {
      snprintf(error_message, max_error_message_chars, "Failed to set datagram socket to SO_REUSEADDR. error=%d", WSAGetLastError());
      closesocket(socket_fd);
      return false;
    }
  }

  // Bind the port number and socket
  if (bind(socket_fd, (struct sockaddr*)&localSock, sizeof(localSock))) {
    int sock_error = WSAGetLastError();
    if (sock_error == WSAEADDRINUSE) {
      snprintf(error_message, max_error_message_chars, "Could not bind datagram socket. The address is already in use or in a time wait state. May need to use the option '-reuse'");
    } else {
      snprintf(error_message, max_error_message_chars, "Failed to bind datagram socket: error=%d", sock_error);
    }
    closesocket(socket_fd);
    return false;
  }

  *socket_fd_ret = socket_fd;
  return true;
}

bool socketDatagramRecv(TakyonSocket socket_fd, void *data_ptr, size_t buffer_bytes, uint64_t *bytes_read_ret, bool is_polling, int64_t timeout_ns, bool *timed_out_ret, char *error_message, int max_error_message_chars) {
  *timed_out_ret = false;
  *bytes_read_ret = 0;
  if (is_polling) {
    return datagram_recv_polling(socket_fd, data_ptr, buffer_bytes, bytes_read_ret, timeout_ns, timed_out_ret, error_message, max_error_message_chars);
  } else {
    return datagram_recv_event_driven(socket_fd, data_ptr, buffer_bytes, bytes_read_ret, timeout_ns, timed_out_ret, error_message, max_error_message_chars);
  }
}

bool socketCreateMulticastSender(const char *ip_addr, const char *multicast_group, uint16_t port_number, bool disable_loopback, int ttl_level, TakyonSocket *socket_fd_ret, void **sock_in_addr_ret, char *error_message, int max_error_message_chars) {
  if (!windows_socket_manager_init(error_message, max_error_message_chars)) {
    return false;
  }

  // Resolve the IP address
  char resolved_ip_addr[MAX_IP_ADDR_CHARS];
  if (!hostnameToIpaddr(AF_INET, ip_addr, resolved_ip_addr, MAX_IP_ADDR_CHARS, error_message, max_error_message_chars)) {
    // Error message already set
    return false;
  }

  // Create a datagram socket on which to send.
  TakyonSocket socket_fd = socket(AF_INET, SOCK_DGRAM, 0);
  if (socket_fd == INVALID_SOCKET) {
    snprintf(error_message, max_error_message_chars, "Failed to create datagram sender socket: error=%d", WSAGetLastError());
    return false;
  }

  // Disable loopback so you do not receive your own datagrams.
  {
    char loopch = 0;
    int loopch_length = sizeof(loopch);
    if (getsockopt(socket_fd, IPPROTO_IP, IP_MULTICAST_LOOP, (char *)&loopch, &loopch_length) < 0) {
      snprintf(error_message, max_error_message_chars, "Getting IP_MULTICAST_LOOP error when trying to determine loopback value");
      closesocket(socket_fd);
      return false;
    }
    char expected_loopch = disable_loopback ? 0 : 1;
    if (loopch != expected_loopch) {
      loopch = expected_loopch;
      if (setsockopt(socket_fd, IPPROTO_IP, IP_MULTICAST_LOOP, &loopch, sizeof(loopch)) < 0) {
        snprintf(error_message, max_error_message_chars, "Getting IP_MULTICAST_LOOP error when trying to %s loopback", expected_loopch ? "enable" : "disable");
        closesocket(socket_fd);
        return false;
      }
    }
  }

  // Set local interface for outbound multicast datagrams.
  // The IP address specified must be associated with a local, multicast capable interface.
  {
    struct in_addr localInterface;
    int status = inet_pton(AF_INET, resolved_ip_addr, &localInterface.s_addr);
    if (status == 0) {
      snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. Invalid name.", ip_addr);
      closesocket(socket_fd);
      return false;
    } else if (status == -1) {
      snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. sock_error=%d", ip_addr, WSAGetLastError());
      closesocket(socket_fd);
      return false;
    }
    if (setsockopt(socket_fd, IPPROTO_IP, IP_MULTICAST_IF, (char *)&localInterface, sizeof(localInterface)) < 0) {
      snprintf(error_message, max_error_message_chars, "Failed to set local interface for multicast");
      closesocket(socket_fd);
      return false;
    }
  }

  // Set the Time-to-Live of the multicast to 1 so it's only on the local subnet.
  {
    // Supported levels:
    // 0:   Are restricted to the same host
    // 1:   Are restricted to the same subnet
    // 32:  Are restricted to the same site
    // 64:  Are restricted to the same region
    // 128: Are restricted to the same continent
    // 255: Are unrestricted in scope
    int router_depth = ttl_level;
    if (setsockopt(socket_fd, IPPROTO_IP, IP_MULTICAST_TTL, (char *)&router_depth, sizeof(router_depth)) < 0) {
      snprintf(error_message, max_error_message_chars, "Failed to set IP_MULTICAST_TTL for multicast");
      closesocket(socket_fd);
      return false;
    }
  }

  // Allocate the multicast address structure
  struct sockaddr_in *sock_in_addr = malloc(sizeof(struct sockaddr_in));
  if (sock_in_addr == NULL) {
    snprintf(error_message, max_error_message_chars, "Out of memory");
    closesocket(socket_fd);
    return false;
  }

  // Fill in the structure
  memset((char *)sock_in_addr, 0, sizeof(struct sockaddr_in));
#ifdef __APPLE__
  sock_in_addr->sin_len = sizeof(struct sockaddr_in);
#endif
  sock_in_addr->sin_family = AF_INET;
  int status = inet_pton(AF_INET, multicast_group, &sock_in_addr->sin_addr);
  if (status == 0) {
    snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. Invalid name.", ip_addr);
    closesocket(socket_fd);
    free(sock_in_addr);
    return false;
  } else if (status == -1) {
    snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. sock_error=%d", ip_addr, WSAGetLastError());
    closesocket(socket_fd);
    free(sock_in_addr);
    return false;
  }
  sock_in_addr->sin_port = htons(port_number);
  *sock_in_addr_ret = sock_in_addr;

  *socket_fd_ret = socket_fd;
  return true;
}

bool socketCreateMulticastReceiver(const char *ip_addr, const char *multicast_group, uint16_t port_number, bool allow_reuse, TakyonSocket *socket_fd_ret, char *error_message, int max_error_message_chars) {
  if (!windows_socket_manager_init(error_message, max_error_message_chars)) {
    return false;
  }

  // Resolve the IP address
  char resolved_ip_addr[MAX_IP_ADDR_CHARS];
  if (!hostnameToIpaddr(AF_INET, ip_addr, resolved_ip_addr, MAX_IP_ADDR_CHARS, error_message, max_error_message_chars)) {
    // Error message already set
    return false;
  }

  // Create a datagram socket on which to receive.
  TakyonSocket socket_fd = socket(AF_INET, SOCK_DGRAM, 0);
  if (socket_fd == INVALID_SOCKET) {
    snprintf(error_message, max_error_message_chars, "Failed to create multicast socket: error=%d", WSAGetLastError());
    return false;
  }

  // Bind to the proper port number with the IP address specified as INADDR_ANY.
  // I.e. only allow packets received on the specified port number
  struct sockaddr_in localSock;
  memset((char *) &localSock, 0, sizeof(localSock));
#ifdef __APPLE__
  localSock.sin_len = sizeof(localSock);
#endif
  localSock.sin_family = AF_INET;
  localSock.sin_port = htons(port_number);
  // NOTE: htonl(INADDR_ANY) will allow any IP interface to listen
  // IMPORTANT: can't use an actual IP address here or else won't work. The interface address is set when the group is set (below).
  localSock.sin_addr.s_addr = htonl(INADDR_ANY);

  if (allow_reuse) {
    // This will allow a previously closed socket, that is still in the TIME_WAIT stage, to be used.
    // This may accur if the application did not exit gracefully on a previous run.
    BOOL allow = 1;
    if (setsockopt(socket_fd, SOL_SOCKET, SO_REUSEADDR, (char *)&allow, sizeof(allow)) != 0) {
      snprintf(error_message, max_error_message_chars, "Failed to set multicast socket to SO_REUSEADDR. error=%d", WSAGetLastError());
      closesocket(socket_fd);
      return false;
    }
  }

  // Bind the port number and socket
  if (bind(socket_fd, (struct sockaddr*)&localSock, sizeof(localSock))) {
    int sock_error = WSAGetLastError();
    if (sock_error == WSAEADDRINUSE) {
      snprintf(error_message, max_error_message_chars, "Could not bind multicast socket. The address is already in use or in a time wait state. May need to use the option '-reuse'");
    } else {
      snprintf(error_message, max_error_message_chars, "Failed to bind multicast socket: error=%d", sock_error);
    }
    closesocket(socket_fd);
    return false;
  }

  // Join the multicast group <multicast_group_addr> on the local network interface <reader_network_interface_addr> interface.
  // Note that this IP_ADD_MEMBERSHIP option must be called for each local interface over which the multicast
  // datagrams are to be received.
  {
    struct ip_mreq group;
    int status = inet_pton(AF_INET, multicast_group, &group.imr_multiaddr);
    if (status == 0) {
      snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. Invalid name.", multicast_group);
      closesocket(socket_fd);
      return false;
    } else if (status == -1) {
      snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. sock_error=%d", multicast_group, WSAGetLastError());
      closesocket(socket_fd);
      return false;
    }
    status = inet_pton(AF_INET, resolved_ip_addr, &group.imr_interface);
    if (status == 0) {
      snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. Invalid name.", ip_addr);
      closesocket(socket_fd);
      return false;
    } else if (status == -1) {
      snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. sock_error=%d", ip_addr, WSAGetLastError());
      closesocket(socket_fd);
      return false;
    }
    if (setsockopt(socket_fd, IPPROTO_IP, IP_ADD_MEMBERSHIP, (char *)&group, sizeof(group)) < 0) {
      snprintf(error_message, max_error_message_chars, "Failed to add socket to the multicast group: error=%d", WSAGetLastError());
      closesocket(socket_fd);
      return false;
    }
  }

  *socket_fd_ret = socket_fd;
  return true;
}
