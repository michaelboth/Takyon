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

#include "utils_socket.h"
#include "utils_time.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netdb.h>
#include <sys/un.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <poll.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#ifdef VXWORKS_7
#include <sys/fcntlcom.h>
#include <sys/time.h>
#endif
#ifdef __APPLE__
#include <signal.h>
#endif

#define MAX_IP_ADDR_CHARS 40 // Good for IPv4 or IPv6
#define MICROSECONDS_BETWEEN_CONNECT_ATTEMPTS 10000

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

static bool setSocketTimeout(int socket_fd, int timeout_name, int64_t timeout_ns, char *error_message, int max_error_message_chars) {
  struct timeval timeout;
  if (timeout_ns < 0) {
    timeout.tv_sec = INT_MAX;
    timeout.tv_usec = 0;
  } else if (timeout_ns == 0) {
    timeout.tv_sec = 0;
    timeout.tv_usec = 1;
  } else {
    timeout.tv_sec = timeout_ns / 1000000000LL;
    timeout.tv_usec = (timeout_ns % 1000000000LL) / 1000;
  }
  if (setsockopt(socket_fd, SOL_SOCKET, timeout_name, &timeout, sizeof(timeout)) != 0) {
    snprintf(error_message, max_error_message_chars, "Failed to set the timeout = %lld", (long long)timeout_ns);
    return false;
  }
  return true;
}

static ssize_t socket_write_private(int socket, void *addr, size_t bytes_to_write) {
  // The send() or write() can crash due to a SIGPIPE, so need a way to mask that signal while transferring, and instead return a graceful failure code.
#if defined(__APPLE__) || defined (VXWORKS_7)
  // NOTE: VxWorks does not seem to support MSG_NOSIGNAL, use Apple's method. TCP Sockets seem to work, but not Local Unix sockets.
  // NOTE: APPLEs OSX does not support MSG_NOSIGNAL so can't use send()
  // The write() can abort if the socket is no longer valid. To avoid this, need to catch the SIGPIPE signal
  struct sigaction new_actn, old_actn;
  new_actn.sa_handler = SIG_IGN;
  sigemptyset(&new_actn.sa_mask);
  new_actn.sa_flags = 0;
  sigaction(SIGPIPE, &new_actn, &old_actn);
  // Send
  ssize_t bytes_written = write(socket, addr, bytes_to_write);
  // Enable the SIGPIPE signal again
  sigaction(SIGPIPE, &old_actn, NULL);
#else
  // This method works on Linux
  // NOTE: Will stop SIGPIPE from causing an abort if the socket is dicsonnecting
  ssize_t bytes_written = send(socket, addr, bytes_to_write, MSG_NOSIGNAL);
#endif
  return bytes_written;
}

static bool socket_send_event_driven(int socket_fd, void *addr, size_t total_bytes_to_write, int64_t timeout_ns, bool *timed_out_ret, char *error_message, int max_error_message_chars) {
  *timed_out_ret = false;

  // Set the timeout on the socket
  if (!setSocketTimeout(socket_fd, SO_SNDTIMEO, timeout_ns, error_message, max_error_message_chars)) return false;

  // Write the data
  size_t total_bytes_sent = 0;
  while (total_bytes_sent < total_bytes_to_write) {
    size_t bytes_to_write = total_bytes_to_write - total_bytes_sent;
    ssize_t bytes_written = socket_write_private(socket_fd, addr, bytes_to_write);
    if (bytes_written > 0 && (size_t)bytes_written == bytes_to_write) {
      // Done
      break;
    }
    if (bytes_written == -1) {
      if (errno == EWOULDBLOCK) {
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
      } else if (errno == EINTR) {
        // Interrupted by external signal. Just try again
        bytes_written = 0;
      } else {
        snprintf(error_message, max_error_message_chars, "Failed to write to socket (are you writing to GPU memory?): errno=%d", errno);
        return false;
      }
    } else {
      // There was some progress, so keep trying
      total_bytes_sent += bytes_written;
      addr += bytes_written;
    }
  }

  return true;
}

static bool socket_send_polling(int socket, void *addr, size_t total_bytes_to_write, int64_t timeout_ns, bool *timed_out_ret, char *error_message, int max_error_message_chars) {
  *timed_out_ret = false;
  int64_t start_time = clockTimeNanoseconds();
  bool got_some_data = false;

  size_t total_bytes_sent = 0;
  while (total_bytes_sent < total_bytes_to_write) {
    size_t bytes_to_write = total_bytes_to_write - total_bytes_sent;
    ssize_t bytes_written = socket_write_private(socket, addr, bytes_to_write);
    if (bytes_written > 0 && (size_t)bytes_written == bytes_to_write) {
      // Done
      break;
    }
    if (bytes_written == -1) {
      if (errno == EAGAIN) {
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
      } else if (errno == EINTR) { 
        // Interrupted by external signal. Just try again
        bytes_written = 0;
      } else {
        snprintf(error_message, max_error_message_chars, "Failed to write to socket (are you writing to GPU memory?): errno=%d", errno);
        return false;
      }
    } else {
      // There was some progress, so keep trying
      total_bytes_sent += bytes_written;
      addr += bytes_written;
      // See if the timeout needs to be updated
      if (bytes_written > 0 && !got_some_data) {
        start_time = clockTimeNanoseconds();
        got_some_data = true;
      }
    }
  }

  return true;
}

bool socketSend(int socket, void *addr, size_t bytes_to_write, bool is_polling, int64_t timeout_ns, bool *timed_out_ret, char *error_message, int max_error_message_chars) {
  *timed_out_ret = false;
  if (is_polling) {
    return socket_send_polling(socket, addr, bytes_to_write, timeout_ns, timed_out_ret, error_message, max_error_message_chars);
  } else {
    return socket_send_event_driven(socket, addr, bytes_to_write, timeout_ns, timed_out_ret, error_message, max_error_message_chars);
  }
}

static bool socket_recv_event_driven(int socket_fd, void *data_ptr, size_t bytes_to_read, int64_t timeout_ns, bool *timed_out_ret, char *error_message, int max_error_message_chars) {
  *timed_out_ret = false;

  // Set the timeout on the socket
  if (!setSocketTimeout(socket_fd, SO_RCVTIMEO, timeout_ns, error_message, max_error_message_chars)) return false;

  char *data = data_ptr;
  size_t total_bytes_read = 0;
  while (total_bytes_read < bytes_to_read) {
    size_t left_over = bytes_to_read - total_bytes_read;
    ssize_t read_bytes = read(socket_fd, data+total_bytes_read, left_over); 
    if (read_bytes == 0) {
      // Socket was probably closed by the remote side
      snprintf(error_message, max_error_message_chars, "Remote side of socket has been closed");
      return false;
    }
    if (read_bytes == -1) {
      if (errno == EWOULDBLOCK) {
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
      } else if (errno == EINTR) {
        // Interrupted by external signal. Just try again
        read_bytes = 0;
      } else {
        snprintf(error_message, max_error_message_chars, "Failed to read from socket: errno=%d", errno);
        return false;
      }
    }
    total_bytes_read += read_bytes;
  }

  return true;
}

static bool socket_recv_polling(int socket_fd, void *data_ptr, size_t bytes_to_read, int64_t timeout_ns, bool *timed_out_ret, char *error_message, int max_error_message_chars) {
  *timed_out_ret = false;
  int64_t start_time = clockTimeNanoseconds();
  bool got_some_data = false;

  char *data = data_ptr;
  size_t total_bytes_read = 0;
  while (total_bytes_read < bytes_to_read) {
    size_t left_over = bytes_to_read - total_bytes_read;
    ssize_t read_bytes = read(socket_fd, data+total_bytes_read, left_over); 
    if (read_bytes == 0) {
      // Socket was probably closed by the remote side
      snprintf(error_message, max_error_message_chars, "Remote side of socket has been closed");
      return false;
    }
    if (read_bytes == -1) {
      if (errno == EAGAIN) {
        // Nothing read, but no errors
        if (timeout_ns >= 0) {
          int64_t ellapsed_time = clockTimeNanoseconds() - start_time;
          if (ellapsed_time >= timeout_ns) {
            // Timed out
            if (total_bytes_read == 0) {
              // Timed out
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
      } else if (errno == EINTR) {
        // Interrupted by external signal. Just try again
        read_bytes = 0;
      } else {
        snprintf(error_message, max_error_message_chars, "Failed to read from socket: errno=%d", errno);
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

bool socketRecv(int socket_fd, void *data_ptr, size_t bytes_to_read, bool is_polling, int64_t timeout_ns, bool *timed_out_ret, char *error_message, int max_error_message_chars) {
  *timed_out_ret = false;
  if (is_polling) {
    return socket_recv_polling(socket_fd, data_ptr, bytes_to_read, timeout_ns, timed_out_ret, error_message, max_error_message_chars);
  } else {
    return socket_recv_event_driven(socket_fd, data_ptr, bytes_to_read, timeout_ns, timed_out_ret, error_message, max_error_message_chars);
  }
}

static bool wait_for_socket_read_activity(int socket, int64_t timeout_ns, char *error_message, int max_error_message_chars) {
  int timeout_in_milliseconds;
  int num_fds_with_activity;
  struct pollfd poll_fd_list[1];
  nfds_t num_fds;

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
  num_fds_with_activity = poll(poll_fd_list, num_fds, timeout_in_milliseconds);
  if (num_fds_with_activity == 0) {
    // Can't report the time out
    snprintf(error_message, max_error_message_chars, "Timed out waiting for socket read activity");
    return false;
  } else if (num_fds_with_activity != 1) {
    // Select has an error
    if (errno == EINTR) {
      // Interrupted by external signal. Just try again
      goto restart_wait;
    }
    snprintf(error_message, max_error_message_chars, "Error while waiting for socket read activity: errno=%d", errno);
    return false;
  } else {
    // Got activity.
    return true;
  }
}

bool socketCreateLocalClient(const char *socket_name, int *socket_fd_ret, int64_t timeout_ns, char *error_message, int max_error_message_chars) {
  int64_t start_time = clockTimeNanoseconds();

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

  // Keep trying until a connection is made or timed out
  while (1) {
    int socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (socket_fd == -1) {
      snprintf(error_message, max_error_message_chars, "Could not create local unix socket. errno=%d", errno);
      return false;
    }

    struct sockaddr_un server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
#ifdef __APPLE__
    server_addr.sun_len = sizeof(server_addr);
#endif
    server_addr.sun_family = AF_UNIX;
    strncpy(server_addr.sun_path, socket_name, sizeof(server_addr.sun_path)-1);

    // Wait for the connection
    bool retry = false;
    // Try to connect. If the server side is ready, then the connection will be made, otherwise the function will return with an error.
    if (connect(socket_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) == -1) {
      if ((errno == ECONNREFUSED) || (errno == ENOENT/*No file or directory (only useful with local unix sockets)*/) || (errno == EBADF/* bad file number*/)) {
        // Server side is not ready yet
        retry = true;
        close(socket_fd);
      } else {
        // Failed in a non recoverable way
        snprintf(error_message, max_error_message_chars, "Could not connect local client socket. errno=%d", errno);
        close(socket_fd);
        return false;
      }
    }

    if (retry) {
      // Server side is not ready yet

      // Close the socket to avoid running out of resources
      close(socket_fd);

      // Test for timeout
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

    } else {
      // The connect is made
      *socket_fd_ret = socket_fd;

      return true;
    }
  }
}

static bool socketSetNoDelay(int socket_fd, bool use_it, char *error_message, int max_error_message_chars) {
  // NOTE: Makes no difference on OSX. Maybe it will on Linux.
  // This will allow small messages to be sent right away instead of letting the socket wait to buffer up 8k of data.
  int use_no_delay = use_it ? 1 : 0;
  if (setsockopt(socket_fd, IPPROTO_TCP, TCP_NODELAY, &use_no_delay, sizeof(use_no_delay)) != 0) {
    snprintf(error_message, max_error_message_chars, "Failed to set socket to TCP_NODELAY. errno=%d", errno);
    return false;
  }
  return true;
}

bool socketSetKernelRecvBufferingSize(int socket_fd, int bytes, char *error_message, int max_error_message_chars) {
  // Allow datagrams to be buffered in the OS to avoid dropping packets
  if (setsockopt(socket_fd, SOL_SOCKET, SO_RCVBUF, &bytes, sizeof(bytes)) != 0) {
    snprintf(error_message, max_error_message_chars, "Failed to set socket to SO_RCVBUF. errno=%d", errno);
    return false;
  }
  return true;
}

bool socketCreateTcpClient(const char *ip_addr, uint16_t port_number, int *socket_fd_ret, int64_t timeout_ns, char *error_message, int max_error_message_chars) {
  int64_t start_time = clockTimeNanoseconds();

  // Resolve the IP address
  char resolved_ip_addr[MAX_IP_ADDR_CHARS];
  if (!hostnameToIpaddr(AF_INET, ip_addr, resolved_ip_addr, MAX_IP_ADDR_CHARS, error_message, max_error_message_chars)) {
    // Error message already set
    return false;
  }

  // Keep trying until a connection is made or timed out
  while (1) {
    // Create a socket file descriptor
    int socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd == -1) {
      snprintf(error_message, max_error_message_chars, "Could not create TCP socket. errno=%d", errno);
      return false;
    }

    // Set up server connection info
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
#ifdef __APPLE__
    server_addr.sin_len = sizeof(server_addr);
#endif
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port_number);
    int status = inet_pton(AF_INET, resolved_ip_addr, &server_addr.sin_addr);
    if (status == 0) {
      snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. Invalid name.", ip_addr);
      close(socket_fd);
      return false;
    } else if (status == -1) {
      snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. errno=%d", ip_addr, errno);
      close(socket_fd);
      return false;
    }

    // Wait for the connection
    bool retry = false;
    // Try to connect. If the server side is ready, then the connection will be made, otherwise the function will return with an error.
    if (connect(socket_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) == -1) {
      if ((errno == ECONNREFUSED) || (errno == EBADF/* bad file number*/)) {
        // Server side is not ready yet
        retry = true;
      } else {
        // Failed in a non recoverable way
        snprintf(error_message, max_error_message_chars, "Could not connect TCP client socket. errno=%d", errno);
        close(socket_fd);
        return false;
      }
    }

    if (retry) {
      // Server side is not ready yet

      // Close the socket to avoid running out of resources
      close(socket_fd);

      // Test for timeout
      int64_t ellapsed_time_ns = clockTimeNanoseconds() - start_time;
      if (timeout_ns >= 0) {
        if (ellapsed_time_ns >= timeout_ns) {
          // Timed out
          snprintf(error_message, max_error_message_chars, "Timed out waiting for connection on TCP client socket");
          return false;
        }
      }
      // Context switch out for a little time to allow remote side to get going.
      clockSleepYield(MICROSECONDS_BETWEEN_CONNECT_ATTEMPTS);

    } else {
      // The connect is made

      // TCP sockets on Linux use the Nagle algorithm, so need to turn it off to get good latency (at the expense of added network traffic and under utilized TCP packets).
      if (!socketSetNoDelay(socket_fd, 1, error_message, max_error_message_chars)) {
        // Message already set
        close(socket_fd);
        return false;
      }

      *socket_fd_ret = socket_fd;

      return true;
    }
  }
}

bool socketCreateLocalServer(const char *socket_name, int *socket_fd_ret, int64_t timeout_ns, char *error_message, int max_error_message_chars) {
  int listening_fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (listening_fd == -1) {
    snprintf(error_message, max_error_message_chars, "Could not create unix socket. errno=%d", errno);
    return false;
  }

  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
#ifdef __APPLE__
  addr.sun_len = sizeof(addr);
#endif
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, socket_name, sizeof(addr.sun_path)-1);

  unlink(socket_name); // Unlink the socket name will remove the file, since it's no longer needed at this point.

  // IMPORTANT: SO_REUSEADDR is not relevant for AF_UNIX

  if (bind(listening_fd, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
    if (errno == EADDRINUSE) {
      snprintf(error_message, max_error_message_chars, "Could not bind unix socket. The address is already in use or in a time wait state. May need to use the option '-reuse'");
    } else {
      snprintf(error_message, max_error_message_chars, "Could not bind unix socket. errno=%d", errno);
    }
    close(listening_fd);
    return false;
  }

  // Set the number of simultaneous connections that can be made
  if (listen(listening_fd, 1) == -1) {
    snprintf(error_message, max_error_message_chars, "Could not listen on unix socket. errno=%d", errno);
    close(listening_fd);
    return false;
  }

  // Wait for a client to ask for a connection
  if (timeout_ns >= 0) {
    if (!wait_for_socket_read_activity(listening_fd, timeout_ns, error_message, max_error_message_chars)) {
      // Error message already set
      close(listening_fd);
      return false;
    }
  }

  // This blocks until a connection is actually made
  int socket_fd = accept(listening_fd, NULL, NULL);
  if (socket_fd == -1) {
    snprintf(error_message, max_error_message_chars, "Could not accept unix socket. errno=%d", errno);
    close(listening_fd);
    return false;
  }

  close(listening_fd);
  unlink(socket_name); // Unlink the socket name will remove the file, since it's no longer needed at this point.
  *socket_fd_ret = socket_fd;

  return true;
}

bool socketCreateTcpServer(const char *ip_addr, uint16_t port_number, bool allow_reuse, int *socket_fd_ret, int64_t timeout_ns, char *error_message, int max_error_message_chars) {
  int listening_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (listening_fd == -1) {
    snprintf(error_message, max_error_message_chars, "Could not create TCP socket. errno=%d", errno);
    return false;
  }

  // Create server connection info
  struct sockaddr_in server_addr;
  memset((char*)&server_addr, 0, sizeof(server_addr));
#ifdef __APPLE__
  server_addr.sin_len = sizeof(server_addr);
#endif
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(port_number); // NOTE: Use 0 to have the system find an unused port number
  if (strcmp(ip_addr, "Any") == 0) {
    // NOTE: htonl(INADDR_ANY) will allow any IP interface to listen
    server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
  } else {
    // Listen on a specific IP interface
    // Resolve the IP address
    char resolved_ip_addr[MAX_IP_ADDR_CHARS];
    if (!hostnameToIpaddr(AF_INET, ip_addr, resolved_ip_addr, MAX_IP_ADDR_CHARS, error_message, max_error_message_chars)) {
      // Error message already set
      close(listening_fd);
      return false;
    }
    int status = inet_pton(AF_INET, resolved_ip_addr, &server_addr.sin_addr);
    if (status == 0) {
      snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. Invalid name.", ip_addr);
      close(listening_fd);
      return false;
    } else if (status == -1) {
      snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. errno=%d", ip_addr, errno);
      close(listening_fd);
      return false;
    }
  }

  if (allow_reuse) {
    // This will allow a previously closed socket, that is still in the TIME_WAIT stage, to be used.
    // This may accur if the application did not exit gracefully on a previous run.
    int allow = 1;
    if (setsockopt(listening_fd, SOL_SOCKET, SO_REUSEADDR, &allow, sizeof(allow)) != 0) {
      snprintf(error_message, max_error_message_chars, "Failed to set socket to SO_REUSEADDR. errno=%d", errno);
      close(listening_fd);
      return false;
    }
  }

  if (bind(listening_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) == -1) {
    if (errno == EADDRINUSE) {
      snprintf(error_message, max_error_message_chars, "Could not bind TCP server socket. The address is already in use or in a time wait state. May need to use the option '-reuse'");
    } else {
      snprintf(error_message, max_error_message_chars, "Could not bind TCP server socket. Make sure the IP addr (%s) is defined for this machine. errno=%d", ip_addr, errno);
    }
    close(listening_fd);
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
  if (listen(listening_fd, 1) == -1) {
    snprintf(error_message, max_error_message_chars, "Could not listen on TCP socket. errno=%d", errno);
    close(listening_fd);
    return false;
  }

  // Wait for a client to ask for a connection
  if (timeout_ns >= 0) {
    if (!wait_for_socket_read_activity(listening_fd, timeout_ns, error_message, max_error_message_chars)) {
      // Error message already set
      close(listening_fd);
      return false;
    }
  }

  // This blocks until a connection is actually made
  int socket_fd = accept(listening_fd, NULL, NULL);
  if (socket_fd == -1) {
    snprintf(error_message, max_error_message_chars, "Could not accept TCP socket. errno=%d", errno);
    close(listening_fd);
    return false;
  }

  // TCP sockets on Linux use the Nagle algorithm, so need to turn it off to get good latency.
  if (!socketSetNoDelay(socket_fd, 1, error_message, max_error_message_chars)) {
    // Error message already set
    close(listening_fd);
    close(socket_fd);
    return false;
  }

  close(listening_fd);
  *socket_fd_ret = socket_fd;

  return true;
}

#ifdef ENABLE_EPHEMERAL_PORT_MANAGER
bool socketCreateEphemeralTcpClient(const char *ip_addr, const char *provider_name, uint32_t path_id, int *socket_fd_ret, int64_t timeout_ns, uint64_t verbosity, char *error_message, int max_error_message_chars) {
  int64_t start_time = clockTimeNanoseconds();

  // Resolve the IP address
  char resolved_ip_addr[MAX_IP_ADDR_CHARS];
  if (!hostnameToIpaddr(AF_INET, ip_addr, resolved_ip_addr, MAX_IP_ADDR_CHARS, error_message, max_error_message_chars)) {
    // Error message already set
    return false;
  }

  // Keep trying until a connection is made or timed out
  int socket_fd = 0;
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
    if (socket_fd == -1) {
      snprintf(error_message, max_error_message_chars, "Could not create TCP socket. errno=%d", errno);
      return false;
    }

    // Set up server connection info
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
#ifdef __APPLE__
    server_addr.sin_len = sizeof(server_addr);
#endif
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(ephemeral_port_number);
    int status = inet_pton(AF_INET, resolved_ip_addr, &server_addr.sin_addr);
    if (status == 0) {
      snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. Invalid name.", ip_addr);
      close(socket_fd);
      return false;
    } else if (status == -1) {
      snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. errno=%d", ip_addr, errno);
      close(socket_fd);
      return false;
    }

    // Try to connect. If the server side is ready, then the connection will be made, otherwise the function will return with an error.
    // NOTE: Even though the ephemeral port number has arrived, it does not guarantee the server is in the connect state
    if (connect(socket_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) == -1) {
      if ((errno == ECONNREFUSED) || (errno == EBADF/* bad file number*/)) {
        // Server side is not ready yet
        close(socket_fd);
      } else {
        // Failed in a non recoverable way
        snprintf(error_message, max_error_message_chars, "Could not connect TCP client socket. errno=%d", errno);
        close(socket_fd);
        return false;
      }
      // Test for timeout
      int64_t ellapsed_time_ns = clockTimeNanoseconds() - start_time;
      if (timeout_ns >= 0) {
        if (ellapsed_time_ns >= timeout_ns) {
          // Timed out
          snprintf(error_message, max_error_message_chars, "Timed out waiting for connection on TCP client socket");
          return false;
        }
      }
      // Context switch out for a little time to allow remote side to get going.
      clockSleepYield(MICROSECONDS_BETWEEN_CONNECT_ATTEMPTS);
    } else {
      // Got the connection
      break;
    }
  }

  // The connect is made

  // Send out a message to let the other nodes know the path no longer needs the ephemeral port number
  ephemeralPortManagerRemove(provider_name, path_id, ephemeral_port_number);

  // TCP sockets on Linux use the Nagle algorithm, so need to turn it off to get good latency (at the expense of added network traffic and under utilized TCP packets).
  if (!socketSetNoDelay(socket_fd, 1, error_message, max_error_message_chars)) {
    // Error message already set
    close(socket_fd);
    return false;
  }

  *socket_fd_ret = socket_fd;

  return true;
}

bool socketCreateEphemeralTcpServer(const char *ip_addr, const char *provider_name, uint32_t path_id, int *socket_fd_ret, int64_t timeout_ns, char *error_message, int max_error_message_chars) {
  int listening_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (listening_fd == -1) {
    snprintf(error_message, max_error_message_chars, "Could not create TCP socket. errno=%d", errno);
    return false;
  }

  // Create server connection info
  struct sockaddr_in server_addr;
  memset((char*)&server_addr, 0, sizeof(server_addr));
#ifdef __APPLE__
  server_addr.sin_len = sizeof(server_addr);
#endif
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(0); // NOTE: Use 0 to have the system find an unused ephemeral port number
  if (strcmp(ip_addr, "Any") == 0) {
    // NOTE: htonl(INADDR_ANY) will allow any IP interface to listen
    server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
  } else {
    // Listen on a specific IP interface
    // Resolve the IP address
    char resolved_ip_addr[MAX_IP_ADDR_CHARS];
    if (!hostnameToIpaddr(AF_INET, ip_addr, resolved_ip_addr, MAX_IP_ADDR_CHARS, error_message, max_error_message_chars)) {
      // Error message already set
      close(listening_fd);
      return false;
    }
    int status = inet_pton(AF_INET, resolved_ip_addr, &server_addr.sin_addr);
    if (status == 0) {
      snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. Invalid name.", ip_addr);
      close(listening_fd);
      return false;
    } else if (status == -1) {
      snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. errno=%d", ip_addr, errno);
      close(listening_fd);
      return false;
    }
  }

  // IMPORTANT: SO_REUSEADDR should not be used with ephemeral port numbers

  if (bind(listening_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) == -1) {
    if (errno == EADDRINUSE) {
      snprintf(error_message, max_error_message_chars, "Could not bind TCP server socket. The address is already in use or in a time wait state. May need to use the option '-reuse'");
    } else {
      snprintf(error_message, max_error_message_chars, "Could not bind TCP server socket. Make sure the IP addr (%s) is defined for this machine. errno=%d", ip_addr, errno);
    }
    close(listening_fd);
    return false;
  }

  // If using an ephemeral port number, then use this to know the actual port number and pass it to the rest of the system
  struct sockaddr_in sin;
  socklen_t len = sizeof(sin);
  if (getsockname(listening_fd, (struct sockaddr *)&sin, &len) < 0) {
    snprintf(error_message, max_error_message_chars, "Failed to get ephemeral port number from listening socket for IP addr (%s). errno=%d", ip_addr, errno);
    close(listening_fd);
    return false;
  }
  uint16_t ephemeral_port_number = ntohs(sin.sin_port);

  // Set the number of simultaneous connections that can be made
  if (listen(listening_fd, 1) == -1) {
    snprintf(error_message, max_error_message_chars, "Could not listen on TCP socket. errno=%d", errno);
    close(listening_fd);
    return false;
  }

  // Multicast this out to all active Takyon endpoints
  ephemeralPortManagerSet(provider_name, path_id, ephemeral_port_number);

  // Wait for a client to ask for a connection
  if (timeout_ns >= 0) {
    if (!wait_for_socket_read_activity(listening_fd, timeout_ns, error_message, max_error_message_chars)) {
      // Error message already set
      ephemeralPortManagerRemoveLocally(provider_name, path_id);
      close(listening_fd);
      return false;
    }
  }

  // This typically blocks until a connection is actually made, but the above function will do the same but with a timeout
  int socket_fd = accept(listening_fd, NULL, NULL);
  ephemeralPortManagerRemoveLocally(provider_name, path_id);
  if (socket_fd == -1) {
    snprintf(error_message, max_error_message_chars, "Could not accept TCP socket. errno=%d", errno);
    close(listening_fd);
    return false;
  }

  // TCP sockets on Linux use the Nagle algorithm, so need to turn it off to get good latency.
  if (!socketSetNoDelay(socket_fd, 1, error_message, max_error_message_chars)) {
    // Error message already set
    close(listening_fd);
    close(socket_fd);
    return false;
  }

  close(listening_fd);
  *socket_fd_ret = socket_fd;

  return true;
}
#endif

bool pipeCreate(int *read_pipe_fd_ret, int *write_pipe_fd_ret, char *error_message, int max_error_message_chars) {
  int fds[2];
  if (pipe(fds) != 0) {
    snprintf(error_message, max_error_message_chars, "Could not create synchronization pipe.");
    return false;
  }
  *read_pipe_fd_ret = fds[0];
  *write_pipe_fd_ret = fds[1];
  return true;
}

bool socketWaitForDisconnectActivity(int socket_fd, int read_pipe_fd, bool *got_socket_activity_ret, char *error_message, int max_error_message_chars) {
  int timeout_in_milliseconds = -1; // Wait forever
  int num_fds_with_activity;
  struct pollfd poll_fd_list[2];
  nfds_t num_fds = 2;

  // Will come back here if EINTR is detected while waiting
 restart_wait:

  poll_fd_list[0].fd = socket_fd;
  poll_fd_list[0].events = POLLIN;
  poll_fd_list[0].revents = 0;
  poll_fd_list[1].fd = read_pipe_fd;
  poll_fd_list[1].events = POLLIN;
  poll_fd_list[1].revents = 0;

  // Wait for activity on the socket
  num_fds_with_activity = poll(poll_fd_list, num_fds, timeout_in_milliseconds);
  if (num_fds_with_activity == 0) {
    // Can't report the time out
    snprintf(error_message, max_error_message_chars, "Timed out waiting for socket activity");
    return false;
  } else if (num_fds_with_activity < 0) {
    // Select has an error */
    if (errno == EINTR) {
      // Interrupted by external signal. Just try again
      goto restart_wait;
    }
    snprintf(error_message, max_error_message_chars, "Error while waiting for socket activity: errno=%d", errno);
    return false;
  } else {
    // Got activity.
    bool got_read_pipe_activity = (poll_fd_list[1].revents != 0);
    *got_socket_activity_ret = !got_read_pipe_activity;
    // Either the socket got activity (data or a disconnect), or the pipe got activity indicating time to gracefully shut the socket down.
    return true;
  }
}

bool pipeWakeUpPollFunction(int write_pipe_fd, char *error_message, int max_error_message_chars) {
  // Goal is to write to the right pipe, and then the read pipe will get activity from the poll() function, waking it up
  int data = 1;
  size_t msg_size = sizeof(int);
  // NOTE: use this instead of can be broken from a sigpipe: size_t bytes_written = send(write_pipe_fd, (void *)&data, msg_size, MSG_NOSIGNAL);
  // NOTE: could use the following, but can crash due to SIGPIPE:
  ssize_t bytes_written = write(write_pipe_fd, (void *)&data, msg_size);
  if (bytes_written < 0 || (size_t)bytes_written != msg_size) {
    snprintf(error_message, max_error_message_chars, "Could not wake up select with pipe.");
    return false;
  }
  return true;
}

void pipeDestroy(int read_pipe_fd, int write_pipe_fd) {
  close(read_pipe_fd);
  close(write_pipe_fd);
}

bool socketSetBlocking(int socket_fd, bool is_blocking, char *error_message, int max_error_message_chars) {
  int flags = fcntl(socket_fd, F_GETFL);
  int rc;
  if (is_blocking) {
    rc = fcntl(socket_fd, F_SETFL, flags & (~O_NONBLOCK));
  } else {
    rc = fcntl(socket_fd, F_SETFL, flags | O_NONBLOCK);
  }
  if (rc == -1) {
    snprintf(error_message, max_error_message_chars, "Failed to set socket to %s. errno=%d", is_blocking ? "blocking" : "non-blocking", errno);
    return false;
  }
  return true;
}

void socketClose(int socket_fd) {
  close(socket_fd);
}

static bool datagram_send_polling(int socket_fd, void *sock_in_addr, void *addr, size_t bytes_to_write, int64_t timeout_ns, bool *timed_out_ret, char *error_message, int max_error_message_chars) {
  *timed_out_ret = false;
  int64_t start_time = clockTimeNanoseconds();

  while (1) {
    int flags = 0;
    ssize_t bytes_written = sendto(socket_fd, addr, bytes_to_write, flags, (struct sockaddr *)sock_in_addr, sizeof(struct sockaddr_in));
    if (bytes_written > 0 && (size_t)bytes_written == bytes_to_write) {
      // Transfer completed
      break;
    }
    if (bytes_written == -1) {
      if (errno == EAGAIN) {
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
      } else if (errno == EMSGSIZE) {
        snprintf(error_message, max_error_message_chars, "Failed to send datagram message. %lld bytes exceeds the datagram size", (long long)bytes_to_write);
        return false;
      } else if (errno == EINTR) { 
        // Interrupted by external signal. Just try again
        bytes_written = 0;
#if defined(__APPLE__)
      } else if (errno == ENOBUFS) {
        // BUG: OSX apparently does not actually block if there is no local send buffering in the OS. Not consistent with Windows or Linux
        //      Could mayber try setsockopt(s, SOL_SOCKET, SO_SNDBUF) but that seems like a kludge
        // Solution: since this is an unrelaible interconnect, just let the message quietly drop here
        return true;
#endif
      } else {
        snprintf(error_message, max_error_message_chars, "Failed to write to socket (are you writing to GPU memory?): errno=%d", errno);
        return false;
      }
    } else if (bytes_written == 0) {
      // Keep trying
    } else {
      snprintf(error_message, max_error_message_chars, "Wrote a partial datagram: bytes_sent=%ld, bytes_to_send=%ld", bytes_written, bytes_to_write);
      return false;
    }
  }

  return true;
}

static bool datagram_send_event_driven(int socket_fd, void *sock_in_addr, void *addr, size_t bytes_to_write, int64_t timeout_ns, bool *timed_out_ret, char *error_message, int max_error_message_chars) {
  *timed_out_ret = false;
  // Set the timeout on the socket
  if (!setSocketTimeout(socket_fd, SO_SNDTIMEO, timeout_ns, error_message, max_error_message_chars)) return false;

  while (1) {
    // Write the data
    int flags = 0;
    ssize_t bytes_written = sendto(socket_fd, addr, bytes_to_write, flags, (struct sockaddr *)sock_in_addr, sizeof(struct sockaddr_in));
    if (bytes_written > 0 && (size_t)bytes_written == bytes_to_write) {
      // Transfer completed
      return true;
    }

    // Something went wrong
    if (bytes_written == -1) {
      if (errno == EMSGSIZE) {
        snprintf(error_message, max_error_message_chars, "Failed to send datagram message. %lld bytes exceeds the allowable datagram size", (long long)bytes_to_write);
        return false;
      } else if (errno == EWOULDBLOCK) {
        // Timed out
        *timed_out_ret = true;
        return true;
      } else if (errno == EINTR) {
        // Interrupted by external signal. Just try again
#if defined(__APPLE__)
      } else if (errno == ENOBUFS) {
        // BUG: OSX apparently does not actually block if there is no local send buffering in the OS. Not consistent with Windows or Linux
        //      Could mayber try setsockopt(s, SOL_SOCKET, SO_SNDBUF) but that seems like a kludge
        // Solution: since this is an unrelaible interconnect, just let the message quietly drop here
        return true;
#endif
      } else {
        snprintf(error_message, max_error_message_chars, "Failed to write to socket (are you writing to GPU memory?): errno=%d", errno);
        return false;
      }
    } else if (bytes_written == 0) {
      // Keep trying
    } else {
      snprintf(error_message, max_error_message_chars, "Wrote a partial datagram: bytes_sent=%ld, bytes_to_send=%ld", bytes_written, bytes_to_write);
      return false;
    }
  }

  return true;
}

static bool datagram_recv_event_driven(int socket_fd, void *data_ptr, size_t buffer_bytes, uint64_t *bytes_read_ret, int64_t timeout_ns, bool *timed_out_ret, char *error_message, int max_error_message_chars) {
  *timed_out_ret = false;
  // Set the timeout on the socket
  if (!setSocketTimeout(socket_fd, SO_RCVTIMEO, timeout_ns, error_message, max_error_message_chars)) return false;

  while (1) {
    int flags = 0;
    ssize_t bytes_received = recvfrom(socket_fd, data_ptr, buffer_bytes, flags, NULL, NULL);
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
    if (bytes_received == -1) {
      if (errno == EWOULDBLOCK) {
        // Timed out
        *timed_out_ret = true;
        return true;
      } else if (errno == EINTR) {
        // Interrupted by external signal. Just try again
      } else {
        snprintf(error_message, max_error_message_chars, "Failed to read from socket: errno=%d", errno);
        return false;
      }
    }
  }
}

static bool datagram_recv_polling(int socket_fd, void *data_ptr, size_t buffer_bytes, uint64_t *bytes_read_ret, int64_t timeout_ns, bool *timed_out_ret, char *error_message, int max_error_message_chars) {
  *timed_out_ret = false;
  int64_t start_time = clockTimeNanoseconds();

  while (1) {
    int flags = 0;
    ssize_t bytes_received = recvfrom(socket_fd, data_ptr, buffer_bytes, flags, NULL, NULL);
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
    if (bytes_received == -1) {
      if (errno == EAGAIN) {
        // Nothing read, but no errors
        if (timeout_ns >= 0) {
          int64_t ellapsed_time = clockTimeNanoseconds() - start_time;
          if (ellapsed_time >= timeout_ns) {
            // Timed out
            *timed_out_ret = true;
            return true;
          }
        }
      } else if (errno == EINTR) {
        // Interrupted by external signal. Just try again
      } else {
        snprintf(error_message, max_error_message_chars, "Failed to read from socket: errno=%d", errno);
        return false;
      }
    }
  }
}

bool socketCreateUnicastSender(const char *ip_addr, uint16_t port_number, TakyonSocket *socket_fd_ret, void **sock_in_addr_ret, char *error_message, int max_error_message_chars) {
  // Resolve the IP address
  char resolved_ip_addr[MAX_IP_ADDR_CHARS];
  if (!hostnameToIpaddr(AF_INET, ip_addr, resolved_ip_addr, MAX_IP_ADDR_CHARS, error_message, max_error_message_chars)) {
    // Error message already set
    return false;
  }

  // Create a datagram socket on which to send.
  int socket_fd = socket(AF_INET, SOCK_DGRAM, 0);
  if (socket_fd < 0) {
    snprintf(error_message, max_error_message_chars, "Failed to create datagram sender socket: errno=%d", errno);
    return false;
  }

  // Allocate structure to hold socket address information
  struct sockaddr_in *sock_in_addr = malloc(sizeof(struct sockaddr_in));
  if (sock_in_addr == NULL) {
    snprintf(error_message, max_error_message_chars, "Out of memory");
    close(socket_fd);
    return false;
  }

  // Fill in the structure describing the destination
  memset((char *)sock_in_addr, 0, sizeof(struct sockaddr_in));
#ifdef __APPLE__
  sock_in_addr->sin_len = sizeof(struct sockaddr_in);
#endif
  sock_in_addr->sin_family = AF_INET;
  sock_in_addr->sin_port = htons(port_number);
  int status = inet_pton(AF_INET, resolved_ip_addr, &sock_in_addr->sin_addr);
  if (status == 0) {
    snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. Invalid name.", ip_addr);
    free(sock_in_addr);
    close(socket_fd);
    return false;
  } else if (status == -1) {
    snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. errno=%d", ip_addr, errno);
    free(sock_in_addr);
    close(socket_fd);
    return false;
  }

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
  // Create a datagram socket on which to receive.
  int socket_fd = socket(AF_INET, SOCK_DGRAM, 0);
  if (socket_fd < 0) {
    snprintf(error_message, max_error_message_chars, "Failed to create datagram socket: errno=%d", errno);
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
    // Listen on a specific IP interface
    // Resolve the IP address
    char resolved_ip_addr[MAX_IP_ADDR_CHARS];
    if (!hostnameToIpaddr(AF_INET, ip_addr, resolved_ip_addr, MAX_IP_ADDR_CHARS, error_message, max_error_message_chars)) {
      // Error message already set
      close(socket_fd);
      return false;
    }
    int status = inet_pton(AF_INET, resolved_ip_addr, &localSock.sin_addr);
    if (status == 0) {
      snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. Invalid name.", ip_addr);
      close(socket_fd);
      return false;
    } else if (status == -1) {
      snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. errno=%d", ip_addr, errno);
      close(socket_fd);
      return false;
    }
  }

  if (allow_reuse) {
    // This will allow a previously closed socket, that is still in the TIME_WAIT stage, to be used.
    // This may accur if the application did not exit gracefully on a previous run.
    int allow = 1;
    if (setsockopt(socket_fd, SOL_SOCKET, SO_REUSEADDR, &allow, sizeof(allow)) != 0) {
      snprintf(error_message, max_error_message_chars, "Failed to set datagram socket to SO_REUSEADDR. errno=%d", errno);
      close(socket_fd);
      return false;
    }
  }

  // Bind the port number and socket
  if (bind(socket_fd, (struct sockaddr*)&localSock, sizeof(localSock))) {
    if (errno == EADDRINUSE) {
      snprintf(error_message, max_error_message_chars, "Could not bind datagram socket. The address is already in use or in a time wait state. May need to use the option '-reuse'");
    } else {
      snprintf(error_message, max_error_message_chars, "Failed to bind datagram socket: errno=%d", errno);
    }
    close(socket_fd);
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
  // Resolve the IP address
  char resolved_ip_addr[MAX_IP_ADDR_CHARS];
  if (!hostnameToIpaddr(AF_INET, ip_addr, resolved_ip_addr, MAX_IP_ADDR_CHARS, error_message, max_error_message_chars)) {
    // Error message already set
    return false;
  }

  // Create a datagram socket on which to send.
  int socket_fd = socket(AF_INET, SOCK_DGRAM, 0);
  if (socket_fd < 0) {
    snprintf(error_message, max_error_message_chars, "Failed to create datagram sender socket: errno=%d", errno);
    return false;
  }

  // Disable loopback so you do not receive your own datagrams.
  {
    char loopch = 0;
    socklen_t loopch_length = sizeof(loopch);
    if (getsockopt(socket_fd, IPPROTO_IP, IP_MULTICAST_LOOP, (char *)&loopch, &loopch_length) < 0) {
      snprintf(error_message, max_error_message_chars, "Getting IP_MULTICAST_LOOP error when trying to determine loopback value");
      close(socket_fd);
      return false;
    }
    char expected_loopch = disable_loopback ? 0 : 1;
    if (loopch != expected_loopch) {
      loopch = expected_loopch;
      if (setsockopt(socket_fd, IPPROTO_IP, IP_MULTICAST_LOOP, &loopch, sizeof(loopch)) < 0) {
        snprintf(error_message, max_error_message_chars, "Getting IP_MULTICAST_LOOP error when trying to %s loopback", expected_loopch ? "enable" : "disable");
        close(socket_fd);
        return false;
      }
    }
  }

  // Set local interface for outbound multicast datagrams.
  // The IP address specified must be associated with a local, multicast capable interface.
  {
    struct in_addr localInterface;
    int status = inet_pton(AF_INET, resolved_ip_addr, &localInterface);
    if (status == 0) {
      snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. Invalid name.", ip_addr);
      close(socket_fd);
      return false;
    } else if (status == -1) {
      snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. errno=%d", ip_addr, errno);
      close(socket_fd);
      return false;
    }
    if (setsockopt(socket_fd, IPPROTO_IP, IP_MULTICAST_IF, (char *)&localInterface, sizeof(localInterface)) < 0) {
      snprintf(error_message, max_error_message_chars, "Failed to set local interface for multicast");
      close(socket_fd);
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
      close(socket_fd);
      return false;
    }
  }

  // Allocate the multicast address structure
  struct sockaddr_in *sock_in_addr = malloc(sizeof(struct sockaddr_in));
  if (sock_in_addr == NULL) {
    snprintf(error_message, max_error_message_chars, "Out of memory");
    close(socket_fd);
    return false;
  }

  // Fill in the structure
  memset((char *)sock_in_addr, 0, sizeof(struct sockaddr_in));
#ifdef __APPLE__
  sock_in_addr->sin_len = sizeof(struct sockaddr_in);
#endif
  sock_in_addr->sin_family = AF_INET;
  sock_in_addr->sin_port = htons(port_number);
  sock_in_addr->sin_addr.s_addr = inet_addr(multicast_group);
  *sock_in_addr_ret = sock_in_addr;

  *socket_fd_ret = socket_fd;
  return true;
}

bool socketCreateMulticastReceiver(const char *ip_addr, const char *multicast_group, uint16_t port_number, bool allow_reuse, TakyonSocket *socket_fd_ret, char *error_message, int max_error_message_chars) {
  // Resolve the IP address
  char resolved_ip_addr[MAX_IP_ADDR_CHARS];
  if (!hostnameToIpaddr(AF_INET, ip_addr, resolved_ip_addr, MAX_IP_ADDR_CHARS, error_message, max_error_message_chars)) {
    // Error message already set
    return false;
  }

  // Create a datagram socket on which to receive.
  int socket_fd = socket(AF_INET, SOCK_DGRAM, 0);
  if (socket_fd < 0) {
    snprintf(error_message, max_error_message_chars, "Failed to create multicast socket: errno=%d", errno);
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
  // The bind address needs to be the multicast group IP. Could use INADDR_ANY, but that would connect to any interface
  //localSock.sin_addr.s_addr = htonl(INADDR_ANY);
  {
    int status = inet_pton(AF_INET, multicast_group, &localSock.sin_addr);
    if (status == 0) {
      snprintf(error_message, max_error_message_chars, "Could not get the IP address from the multicast group '%s'. Invalid name.", multicast_group);
      close(socket_fd);
      return false;
    } else if (status == -1) {
      snprintf(error_message, max_error_message_chars, "Could not get the IP address from the multicast group '%s'. errno=%d", multicast_group, errno);
      close(socket_fd);
      return false;
    }
  }

  if (allow_reuse) {
    // This will allow a previously closed socket, that is still in the TIME_WAIT stage, to be used.
    // This may accur if the application did not exit gracefully on a previous run.
    int allow = 1;
    if (setsockopt(socket_fd, SOL_SOCKET, SO_REUSEADDR, &allow, sizeof(allow)) != 0) {
      snprintf(error_message, max_error_message_chars, "Failed to set multicast socket to SO_REUSEADDR. errno=%d", errno);
      close(socket_fd);
      return false;
    }
  }

  // Bind the port number and socket
  if (bind(socket_fd, (struct sockaddr*)&localSock, sizeof(localSock))) {
    if (errno == EADDRINUSE) {
      snprintf(error_message, max_error_message_chars, "Could not bind multicast socket %s:%d. The address is already in use or in a time wait state. May need to use the option '-reuse'", ip_addr, port_number);
    } else {
      snprintf(error_message, max_error_message_chars, "Failed to bind multicast socket: errno=%d", errno);
    }
    close(socket_fd);
    return false;
  }

  // Join the multicast group <multicast_group_addr> on the local network interface <reader_network_interface_addr> interface.
  // Note that this IP_ADD_MEMBERSHIP option must be called for each local interface over which the multicast
  // datagrams are to be received.
  {
    struct ip_mreq group;
    group.imr_multiaddr.s_addr = inet_addr(multicast_group);
    int status = inet_pton(AF_INET, resolved_ip_addr, &group.imr_interface);
    if (status == 0) {
      snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. Invalid name.", ip_addr);
      close(socket_fd);
      return false;
    } else if (status == -1) {
      snprintf(error_message, max_error_message_chars, "Could not get the IP address from the hostname '%s'. errno=%d", ip_addr, errno);
      close(socket_fd);
      return false;
    }
    if (setsockopt(socket_fd, IPPROTO_IP, IP_ADD_MEMBERSHIP, (char *)&group, sizeof(group)) < 0) {
      snprintf(error_message, max_error_message_chars, "Failed to add socket to the multicast group: errno=%d", errno);
      close(socket_fd);
      return false;
    }
    /*+ when the socket is closed, should setsockopt(IP_DROP_MEMBERSHIP) be used? */
  }

  *socket_fd_ret = socket_fd;
  return true;
}
