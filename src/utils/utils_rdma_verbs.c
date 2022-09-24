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

#include "utils_rdma_verbs.h"

/* RDMA Notes:
  - RDMA Protocols:
    - UD Multicast: Using the CM (connection manager) since it's convenient and no hand shake information is needed
    - UD Unicast:   /*+ ? use TCP socket to handshake and detect disconnect, pass socket to rdmaCreateUnicast
    - UC:           CM doesn't support UC so using raw verbs with an external TCP socket to do the init handshake and lifetime disconnect detection
    - RC:           /*+ ? 
  - Avoiding use of inline byte when sending since it only works with CPU memory, and adds complixity to track memory type
*/

/*+ design ideas:
  - For bidirectional, the QP could have two different CQs and CCs to avoid send/recv completion detection complexity
*/

/*+ valgrind */
/*+ test zero-byte messages with UD */

static struct rdma_event_channel *createConnectionManagerEventChannel(char *error_message, int max_error_message_chars) {
  struct rdma_event_channel *ch = rdma_create_event_channel();
  if (ch == NULL) {
    snprintf(error_message, max_error_message_chars, "rdma_create_event_channel() failed: errno=%d", errno);
    return NULL;
  }
  return ch;
}

static struct rdma_cm_event *waitForConnectionManagerEvent(struct rdma_event_channel *event_ch, char *error_message, int max_error_message_chars) {
  struct rdma_cm_event *event = NULL;
  static rc = rdma_get_cm_event(event_ch, &event);
  if (rc != 0) {
    snprintf(error_message, max_error_message_chars, "rdma_get_cm_event() failed: errno=%d", errno);
    return NULL;
  }
  return event;
}

static struct rdma_cm_id *createConnectionManagerId(struct rdma_event_channel *event_ch, enum rdma_port_space port_space, char *error_message, int max_error_message_chars) {
  // Port space options: RDMA_PS_UDP, RDMA_PS_TCP
  struct rdma_cm_id *id = NULL;
  void *app_data = NULL;
  int rc = rdma_create_id(event_ch, &id, app_data, port_space);
  if (rc != 0) {
    snprintf(error_message, max_error_message_chars, "rdma_create_id() failed: errno=%d", errno);
    return NULL;
  }
  return id;
}

// Needed for event driven transfer completion
static struct ibv_comp_channel *createCompletionChannel(struct ibv_context *context, char *error_message, int max_error_message_chars) {
  struct ibv_comp_channel *ch = ibv_create_comp_channel(context);
  if (ch == NULL) {
    snprintf(error_message, max_error_message_chars, "ibv_create_comp_channel() failed");
    return NULL;
  }
  return ch;
}

// Needed for event driven transfer completion
static struct ibv_cq *createCompletionQueue(struct ibv_context *context, int min_completions, struct ibv_comp_channel *comp_ch, char *error_message, int max_error_message_chars) {
  void *app_data = NULL;
  int comp_vector = 0;
  struct ibv_cq *cq = ibv_create_cq(context, min_completions, app_data, comp_ch, comp_vector)
  if (cq == NULL) {
    snprintf(error_message, max_error_message_chars, "ibv_create_cq() failed");
    return NULL;
  }
  return pd;
}

// Needed for event driven transfer completion
static bool armCompletionQueue(struct ibv_cq *cq, char *error_message, int max_error_message_chars) {
  int solicited_only = 0; // Allow unsolicited completion events
  int rc = ibv_req_notify_cq(cq, solicited_only);
  if (rc != 0) {
    snprintf(error_message, max_error_message_chars, "ibv_req_notify_cq() failed: errno=%d", errno);
    return false;
  }
  return true;
}

static bool createConnectionManagerQueuePair(struct rdma_cm_id *id, struct ibv_pd *pd, struct ibv_cq *cq, enum ibv_qp_type qp_type,
                                             uint32_t max_send_wr, uint32_t max_recv_wr, uint32_t max_send_sge, uint32_t max_recv_sge,
                                             char *error_message, int max_error_message_chars) {
  struct ibv_qp_init_attr attrs = {}; // Zero structure
  attrs.qp_context = NULL; // app_data
  attrs.send_cq = cq;
  attrs.recv_cq = cq;
  attrs.srq = NULL;
  attrs.cap.max_send_wr = max_send_wr;
  attrs.cap.max_recv_wr = max_recv_wr;
  attrs.cap.max_send_sge = max_send_sge;
  attrs.cap.max_recv_sge = max_recv_sge;
  attrs.cap.max_inline_data = 0;
  attrs.qp_type = qp_type; // One of IBV_QPT_RC, IBV_QPT_UD, IBV_QPT_UC
  attrs.sq_sig_all = 0; // If 0, then preparing a request must set if signaled or not

  int rc = rdma_create_qp(id, pd, &attrs);
  if (rc != 0) {
    snprintf(error_message, max_error_message_chars, "rdma_create_qp() failed: errno=%d", errno);
    return false;
  }

  return true;
}

static struct ibv_pd *createProtectionDomain(struct ibv_context *context, char *error_message, int max_error_message_chars) {
  struct ibv_pd *pd = ibv_alloc_pd(context);
  if (pd == NULL) {
    snprintf(error_message, max_error_message_chars, "ibv_alloc_pd() failed");
    return NULL;
  }
  return pd;
}

static struct ibv_mr *registerMemoryRegion(struct ibv_pd *pd, void *addr, size_t bytes, enum ibv_access_flags access) {
  // Access flags:
  //   IBV_ACCESS_LOCAL_WRITE   Allow local host write access
  //   IBV_ACCESS_REMOTE_WRITE  Allow remote hosts write access
  //   IBV_ACCESS_REMOTE_READ   Allow remote hosts read access
  //   IBV_ACCESS_REMOTE_ATOMIC Allow remote hosts atomic access
  //   IBV_ACCESS_MW_BIND       Allow memory windows on this MR
  struct ibv_mr *mr = ibv_reg_mr(pd, addr, bytes, access);
  if (mr == NULL) {
    snprintf(error_message, max_error_message_chars, "ibv_reg_mr(addr=0x%lx, bytes=%ju, access=0x%x) failed", addr, bytes, access);
    return NULL;
  }
  return mr;
}

static bool getMulticastAddr(struct rdma_cm_id *id, struct rdma_event_channel *event_ch, const char *local_NIC_ip_addr, const char *multicast_group_ip_addr, int timeout_ms, char *error_message, int max_error_message_chars, struct sockaddr *multicast_addr_out) {
  // NOTE: RoCE UD implicitly uses network port number 4791, so no need to define a port number through the service text
  bool got_addr = false;

  // Find the local NIC info
  struct rdma_addrinfo *local_NIC_info = NULL;
  {
    struct rdma_addrinfo requested_info = {}; // Zero structure
    requested_info.ai_port_space = RDMA_PS_UDP;
    requested_info.ai_flags = RAI_PASSIVE; // Server side
    const char *service = NULL;
    int rc = rdma_getaddrinfo(local_NIC_ip_addr, service, &requested_info, &local_NIC_info);
    if (rc != 0) {
      snprintf(error_message, max_error_message_chars, "rdma_getaddrinfo(localNIC=%s) failed: errno=%d", local_NIC_ip_addr, errno);
      goto cleanup;
    }
  }

  // Get a handle to the multicast group
  struct rdma_addrinfo *multicast_group_info = NULL;
  {
    struct rdma_addrinfo requested_info = {}; // Zero structure
    requested_info.ai_port_space = RDMA_PS_UDP;
    const char *service = NULL;
    int rc = rdma_getaddrinfo(multicast_group_ip_addr, service, &requested_info, &multicast_group_info);
    if (rc != 0) {
      snprintf(error_message, max_error_message_chars, "rdma_getaddrinfo(groupIP=%s) failed: errno=%d", multicast_group_ip_addr, errno);
      goto cleanup;
    }
  }

  // Bind to the local NIC
  int rc = rdma_bind_addr(id, local_NIC_info->ai_src_addr);
  if (rc != 0) {
    snprintf(error_message, max_error_message_chars, "rdma_bind_addr(localNIC=%s) failed: errno=%d", local_NIC_ip_addr, errno);
    goto cleanup;
  }

  // Initiate the request to get the resolved multicast addr
  {
    int rc = rdma_resolve_addr(id, local_NIC_info, multicast_group_info, timeout_ms);
    if (rc != 0) {
      snprintf(error_message, max_error_message_chars, "rdma_resolve_addr(localNIC=%s, groupIP=%s) failed: errno=%d", local_NIC_ip_addr, multicast_group_ip_addr, errno);
      goto cleanup;
    }
    // Wait for the event
    struct rdma_cm_event *event = waitForConnectionManagerEvent(event_ch, error_message, max_error_message_chars);
    if (event == NULL) {
      // Error already filled in
      goto cleanup;
    }
    // Verify
    if (event->event != RDMA_CM_EVENT_ADDR_RESOLVED) {
      snprintf(error_message, max_error_message_chars, "Failed to get correct event for rdma_resolve_addr(localNIC=%s, groupIP=%s), got '%s'", local_NIC_ip_addr, multicast_group_ip_addr, rdma_event_str(event->event));
      goto cleanup;
    }
    // Ack the event
    rc = rdma_ack_cm_event(event);
    if (rc != 0) {
      snprintf(error_message, max_error_message_chars, "Failed to ACK event from rdma_resolve_addr(localNIC=%s, groupIP=%s): errno=%d", local_NIC_ip_addr, multicast_group_ip_addr, errno);
      goto cleanup;
    }
  }

  // Multicast address is resolved
  *multicast_addr_out = multicast_group_info->ai_dst_addr;
  got_addr = true;

  // NOTE: At this point the connection manager now has a valid protection domain: id->pd, and knows it RDMA MTU bytes

 cleanup:
  if (multicast_group_info != NULL) rdma_freeaddrinfo(multicast_group_info);
  if (local_NIC_info != NULL) rdma_freeaddrinfo(local_NIC_info);
  return got_addr;
}

static uint32_t getRdmaMTU(struct ibv_context *context, uint8_t rdma_port_num, char *error_message, int max_error_message_chars) {
  struct ibv_port_attr attr;
  int rc = ibv_query_port(context, rdma_port_num, &attr);
  if (rc != 0) {
    snprintf(error_message, max_error_message_chars, "ibv_query_port() failed: errno=%d", errno);
    return 0;
  }
  switch (attr.active_mtu) {
  case IBV_MTU_256 : return 256;
  case IBV_MTU_512 : return 512;
  case IBV_MTU_1024 : return 1024;
  case IBV_MTU_2048 : return 2048;
  case IBV_MTU_4096 : return 4096;
  default :
    snprintf(error_message, max_error_message_chars, "Failed to determine RDMA MTU bytes");
    return 0;
  };
}

static struct ibv_ah *joinMulticastGroup(struct rdma_cm_id *id, struct sockaddr *multicast_addr, char *error_message, int max_error_message_chars, uint32_t *qp_num_out, uint32_t *qkey_out) {
  // Initiate the request to join the multicast group
  void *app_data = NULL;
  int rc = rdma_join_multicast(id, multicast_addr, app_data);
  if (rc != 0) {
    snprintf(error_message, max_error_message_chars, "rdma_join_multicast() failed: errno=%d", errno);
    return NULL;
  }

  // Wait for the event
  struct rdma_cm_event *event = waitForConnectionManagerEvent(event_ch, error_message, max_error_message_chars);
  if (event == NULL) {
    // Error already filled in
    return NULL;
  }

  // Verify event
  if (event->event != RDMA_CM_EVENT_MULTICAST_JOIN) {
    snprintf(error_message, max_error_message_chars, "Failed to get correct event for joining multicast group, got '%s'", rdma_event_str(event->event));
    return NULL;
  }

  // Create the multicast addr handle
  struct ibv_ah *ah = ibv_create_ah(id->pd, &event->param.ud.ah_attr);
  if (ah == NULL) {
    snprintf(error_message, max_error_message_chars, "Join multicast ibv_create_ah() failed");
    return NULL;
  }

  // Return the QP values
  *qp_num_out = event->param.ud.qp_num;
  *qkey_out = event->param.ud.qkey;

  // Ack the event
  rc = rdma_ack_cm_event(event);
  if (rc != 0) {
    snprintf(error_message, max_error_message_chars, "Failed to ACK event from rdma_resolve_addr(localNIC=%s, groupIP=%s): errno=%d", local_NIC_ip_addr, multicast_group_ip_addr, errno);
    ibv_destroy_ah(ah);
    return NULL;
  }

  return ah;
}

bool rdmaPostRecvs(TakyonPath *path, RdmaEndpoint *endpoint, uint32_t request_count, TakyonRecvRequest *requests, char *error_message, int max_error_message_chars) {
  // Prepare WR chain
  struct ibv_recv_wr *first_wr = NULL;
  struct ibv_recv_wr *curr_wr = NULL;
  for (uint32_t i=0; i<request_count; i++) {
    TakyonRecvRequest *takyon_request = &requests[i];
    RdmaRecvRequest *rdma_request = (RdmaRecvRequest *)takyon_request->private;
    // Add to wr chain
    if (first_wr == NULL) {
      first_wr = &rdma_request->recv_wr;
      curr_wr = first_wr;
    } else {
      curr_wr->next = &rdma_request->recv_wr;
      curr_wr = curr_wr->next;
    }
    // Fill in WR info
    curr_wr->next = NULL;
    curr_wr->wr_id = takyon_request;
    curr_wr->num_sge = takyon_request->sub_buffer_count;
    curr_wr->sg_list = rdma_request->sges;
    for (uint32_t j=0; j<takyon_request->sub_buffer_count; j++) {
      TakyonSubBuffer *sub_buffer = takyon_request->sub_buffers[j];
      TakyonBuffer *buffer = path->attrs.buffers[sub_buffer->buffer_index];
      RdmaBuffer *rdma_buffer = (RdmaBuffer *)buffer->private;
      struct ibv_sge *sge = &curr_wr->sg_list[j];
      sge->addr = (uint64_t)buffer + sub_buffer->offset;
      sge->length = sub_buffer->bytes;
      sge->lkey = rdma_buffer->mr->lkey;
    }
  }

  // Post WR chain
  struct ibv_recv_wr *bad_wr;
  struct ibv_qp *qp = (endpoint->protocol == RDMA_PROTOCOL_UD_MULTICAST) ? endpoint->id->qp : endpoint->qp;
  int rc = ibv_post_recv(qp, first_wr, &bad_wr);
  if (rc != 0) {
    snprintf(error_message, max_error_message_chars, "Failed to post recvs: errno=%d", errno);
    return false;
  }
  return true;
}

RdmaEndpoint *rdmaCreateMulticastEndpoint(TakyonPath *path, const char *local_NIC_ip_addr, const char *multicast_group_ip_addr, bool is_sender,
                                          uint32_t max_send_wr, uint32_t max_recv_wr, uint32_t max_send_sge, uint32_t max_recv_sge,
                                          uint32_t recv_request_count, TakyonRecvRequest *recv_requests,
                                          double timeout_seconds, char *error_message, int max_error_message_chars) {
  struct rdma_event_channel *event_ch = NULL;
  struct rdma_cm_id *id = NULL;
  struct ibv_comp_channel *comp_ch = NULL;
  struct ibv_cq *cq = NULL;
  struct ibv_ah *multicast_ah = NULL;
  struct sockaddr multicast_addr;
  bool qp_created = false;
  uint32_t multicast_qp_num;
  uint32_t multicast_qkey;

  // Event channel
  event_ch = createConnectionManagerEventChannel(error_message, max_error_message_chars);
  if (event_ch == NULL) goto failed;

  // Connection manager ID
  enum rdma_port_space port_space = RDMA_PS_UDP;
  id = createConnectionManagerId(event_ch, port_space, error_message, max_error_message_chars);
  if (id == NULL) goto failed;

  // Multicast addr
  int timeout_ms = (timeout_seconds < 0) ? MAX_INT : (int)(timeout_seconds * 1000);
  if (!getMulticastAddr(id, event_ch, local_NIC_ip_addr, multicast_group_ip_addr, timeout_ms, error_message, max_error_message_chars, &multicast_addr)) goto failed;

  // RDMA MTU bytes
  uint32_t mtu_bytes = getRdmaMTU(id->verbs, id->port_num, error_message, max_error_message_chars);
  if (mtu_bytes == 0) goto failed;
  if (path->attrs.verbosity & TAKYON_VERBOSITY_CREATE_DESTROY_MORE) {
    printf("%-15s (%s:%s) RDMA Multicast MTU bytes=%d\n", __FUNCTION__, path->attrs.is_endpointA ? "A" : "B", path->attrs.provider, mtu_bytes);
  }

  // IMPORTANT: Transfers might be event driven so create the completion channel just in case
  // Completion channel
  comp_ch = createCompletionChannel(id->verbs, error_message, max_error_message_chars);
  if (comp_ch == NULL) goto failed;

  // Completion queue
  int min_completions = is_sender ? max_send_wr : max_recv_wr;
  cq = createCompletionQueue(id->verbs, min_completions, comp_ch, error_message, max_error_message_chars);
  if (cq != NULL) goto failed;
  if (!armCompletionQueue(cq, error_message, max_error_message_chars)) goto failed;

  // Queue pair
  enum ibv_qp_type qp_type = IBV_QPT_UD;
  if (!createConnectionManagerQueuePair(id, id->pd, cq, qp_type, max_send_wr, max_recv_wr, max_send_sge, max_recv_sge, error_message, max_error_message_chars)) goto failed;
  qp_created = true;

  // Register buffers
  enum ibv_access_flags access = is_sender ? 0 : IBV_ACCESS_LOCAL_WRITE;
  for (uint32_t i=0; i<path->attrs.buffer_count; i++) {
    TakyonBuffer *takyon_buffer = path->attrs.buffers[i];
    RdmaBuffer *rdma_buffer = (RdmaBuffer *)takyon_buffer->private;
    // IMPORTANT: assuming buffers are zeroed at init
    rdma_buffer->mr = registerMemoryRegion(id->pd, takyon_buffer->addr, takyon_buffer->bytes, access);
    if (rdma_buffer->mr == NULL) goto failed;
  }

  // Post recvs
  if (!is_sender) {
    if (!postRecvs(id->qp, path, recv_request_count, recv_requests, error_message, max_error_message_chars)) goto failed;
  }

  // Join multicast group
  multicast_ah = joinMulticastGroup(id, &multicast_addr, error_message, max_error_message_chars, &multicast_qp_num, &multicast_qkey);
  if (multicast_ah == NULL) goto failed;

  // Return details
  RdmaEndpoint *endpoint = calloc(1, sizeof(RdmaEndpoint));
  if (endpoint == NULL) goto failed;
  endpoint->protocol = RDMA_PROTOCOL_UD_MULTICAST;
  endpoint->is_sender = is_sender;
  endpoint->event_ch = event_ch;
  endpoint->id = id;
  endpoint->comp_ch = comp_ch;
  endpoint->cq = cq;
  endpoint->multicast_ah = multicast_ah;
  endpoint->multicast_addr = multicast_addr;
  endpoint->multicast_qp_num = multicast_qp_num;
  endpoint->multicast_qkey = multicast_qkey;
  return endpoint;

 failed:
  if (multicast_ah != NULL) {
    rdma_leave_multicast(id, &multicast_addr);
    ibv_destroy_ah(multicast_ah);
  }
  if (qp_created) rdma_destroy_qp(id);
  if (cq != NULL) ibv_destroy_cq(cq);
  for (uint32_t i=0; i<path->attrs.buffer_count; i++) {
    TakyonBuffer *takyon_buffer = path->attrs.buffers[i];
    RdmaBuffer *rdma_buffer = (RdmaBuffer *)takyon_buffer->private;
    if (rdma_buffer->mr != NULL) ibv_dereg_mr(rdma_buffer->mr);
  }
  if (comp_ch != NULL) ibv_destroy_comp_channel(comp_ch);
  if (id != NULL) rdma_destroy_id(id);
  if (event_ch != NULL) rdma_destroy_event_channel(event_ch);
}

bool rdmaDestroyEndpoint(TakyonPath *path, RdmaEndpoint *endpoint, double timeout_seconds, char *error_message, int max_error_message_chars) {
  // Ack events to avoid deadlocking the destructions of the QP
  if (endpoint->nevents_to_ack > 0) {
    ibv_ack_cq_events(endpoint->cq, endpoint->nevents_to_ack);
  }

  // Leave the multicast group
  if (endpoint->multicast_ah != NULL) {
    int rc = rdma_leave_multicast(id, &endpoint->multicast_addr);
    if (rc != 0) {
      snprintf(error_message, max_error_message_chars, "Failed to leave multicast group: errno=%d", errno);
      return false;
    }
    rc = ibv_destroy_ah(endpoint->multicast_ah);
    if (rc != 0) {
      snprintf(error_message, max_error_message_chars, "Failed to destroy multicast address handle: errno=%d", errno);
      return false;
    }
  }

  // Destroy verb resources
  if (endpoint->protocol == RDMA_PROTOCOL_UD_MULTICAST) {
    rdma_destroy_qp(endpoint->id);
  }
  //*+*/endpoint->qp
  if (endpoint->cq != NULL) {
    int rc = ibv_destroy_cq(endpoint->cq);
    if (rc != 0) {
      snprintf(error_message, max_error_message_chars, "Failed to destroy completion queue: errno=%d", errno);
      return false;
    }
  }
  for (uint32_t i=0; i<path->attrs.buffer_count; i++) {
    TakyonBuffer *takyon_buffer = path->attrs.buffers[i];
    RdmaBuffer *rdma_buffer = (RdmaBuffer *)takyon_buffer->private;
    if (rdma_buffer->mr != NULL) {
      int rc = ibv_dereg_mr(rdma_buffer->mr);
      if (rc != 0) {
        snprintf(error_message, max_error_message_chars, "Failed to unregister memory region: errno=%d", errno);
        return false;
      }
    }
  }
  if (endpoint->comp_ch != NULL) {
    int rc = ibv_destroy_comp_channel(endpoint->comp_ch);
    if (rc != 0) {
      snprintf(error_message, max_error_message_chars, "Failed to destroy completion channel: errno=%d", errno);
      return false;
    }
  }
  if (endpoint->id != NULL) {
    int rc = rdma_destroy_id(endpoint->id);
    if (rc != 0) {
      snprintf(error_message, max_error_message_chars, "Failed to destroy connection manager ID: errno=%d", errno);
      return false;
    }
  }
  if (endpoint->event_ch != NULL) {
    rdma_destroy_event_channel(endpoint->event_ch);
  }
}

static bool eventDrivenCompletionWait(RdmaEndpoint *endpoint, double timeout_seconds, char *error_message, int max_error_message_chars) {
  if (timeout_seconds >= 0) {
    // Verbs does not have an API to do a timed wait, so need to use an underlying socket in the completion channel to do it
  retry:
    struct pollfd poll_fd;
    poll_fd.fd      = endpoint->comp_ch->fd; // RDMA's completion channel socket
    poll_fd.events  = POLLIN; // Requested events: POLLIN means there is data to read
    poll_fd.revents = 0;      // Returned events that will be filled in after the call to poll()
    int timeout_ms = (int)(timeout_seconds * 1000);
    nfds_t fd_count = 1;    
    int count = poll(poll_fd_list, fd_count, timeout_ms);
    if (count <= 0) {
      // Failed
      if (count == -1 && errno == EINTR) {
        // Nothing is wrong, OS woke this process up due to some other signal, so try again
        goto retry;
      } else if (count == -1) {
        snprintf(error_message, max_error_message_chars, "poll() failed to wait for an RDMA complete event: errno=%d", errno);
        return false;
      } else {
        snprintf(error_message, max_error_message_chars, "poll() failed to wait for an RDMA complete event: unknown error");
        return false;
      }
    }
    // Got activity
    if (poll_fd.revents != POLLIN) {
      snprintf(error_message, max_error_message_chars, "poll() returned but did not get read 'POLLIN' activity");
      return false;
    }
  }

  // Block waiting for a completion
  // If the timeout was not wait forever, then at this point there should be a completion if the socket poll() did its job
  struct ibv_cq *cq; // Since the completion channel only has one CQ, can ignore the returned values
  void *cq_app_data;
  int rc = ibv_get_cq_event(endpoint->comp_ch, &cq, &cq_app_data);
  if (rc != 0) {
    snprintf(error_message, max_error_message_chars, "ibv_get_cq_event() failed: errno=%d", errno);
    return false;
  }

  // Need to ack the completion event counter if at the limit
  endpoint->nevents_to_ack++;
  if (endpoint->nevents_to_ack == UINT_MAX) {
    ibv_ack_cq_events(/*endpoint->*/cq, endpoint->nevents_to_ack);
    endpoint->nevents_to_ack = 0;
  }

  // Need to re-arm the CQ so completion events can be detected
  /*+ Even if the path only uses polling, since the completion channel was created, does this need to be called? If so, maybe create a different CQ for polling? */
  if (!armCompletionQueue(/*endpoint->*/cq, error_message, max_error_message_chars)) return false;

  return true;
}

static const char *wcErrorToText(enum ibv_wc_status) {
  switch (ibv_wc_status) {
  case IBV_WC_LOC_LEN_ERR : return "IBV_WC_LOC_LEN_ERR";
  case IBV_WC_LOC_QP_OP_ERR : return "IBV_WC_LOC_QP_OP_ERR";
  case IBV_WC_LOC_EEC_OP_ERR : return "IBV_WC_LOC_EEC_OP_ERR";
  case IBV_WC_LOC_PROT_ERR : return "IBV_WC_LOC_PROT_ERR";
  case IBV_WC_WR_FLUSH_ERR : return "IBV_WC_WR_FLUSH_ERR";
  case IBV_WC_MW_BIND_ERR : return "IBV_WC_MW_BIND_ERR";
  case IBV_WC_BAD_RESP_ERR : return "IBV_WC_BAD_RESP_ERR";
  case IBV_WC_LOC_ACCESS_ERR : return "IBV_WC_LOC_ACCESS_ERR";
  case IBV_WC_REM_INV_REQ_ERR : return "IBV_WC_REM_INV_REQ_ERR";
  case IBV_WC_REM_ACCESS_ERR : return "IBV_WC_REM_ACCESS_ERR";
  case IBV_WC_REM_OP_ERR : return "IBV_WC_REM_OP_ERR";
  case IBV_WC_RETRY_EXC_ERR : return "IBV_WC_RETRY_EXC_ERR";
  case IBV_WC_RNR_RETRY_EXC_ERR : return "IBV_WC_RNR_RETRY_EXC_ERR";
  case IBV_WC_LOC_RDD_VIOL_ERR : return "IBV_WC_LOC_RDD_VIOL_ERR";
  case IBV_WC_REM_INV_RD_REQ_ERR : return "IBV_WC_REM_INV_RD_REQ_ERR";
  case IBV_WC_REM_ABORT_ERR : return "IBV_WC_REM_ABORT_ERR";
  case IBV_WC_INV_EECN_ERR : return "IBV_WC_INV_EECN_ERR";
  case IBV_WC_INV_EEC_STATE_ERR : return "IBV_WC_INV_EEC_STATE_ERR";
  case IBV_WC_FATAL_ERR : return "IBV_WC_FATAL_ERR";
  case IBV_WC_RESP_TIMEOUT_ERR : return "IBV_WC_RESP_TIMEOUT_ERR";
  case IBV_WC_GENERAL_ERR : return "IBV_WC_GENERAL_ERR";
  default : return "Unknown IBV_WC error";
}

static const char *wcOpcodeToText(enum ibv_wc_opcode) {
  switch (ibv_wc_opcode) {
  case IBV_WC_SEND : return "IBV_WC_SEND";
  case IBV_WC_RDMA_WRITE : return "IBV_WC_RDMA_WRITE";
  case IBV_WC_RDMA_READ : return "IBV_WC_RDMA_READ";
  case IBV_WC_COMP_SWAP : return "IBV_WC_COMP_SWAP";
  case IBV_WC_FETCH_ADD : return "IBV_WC_FETCH_ADD";
  case IBV_WC_BIND_MW : return "IBV_WC_BIND_MW";
  case IBV_WC_RECV : return "IBV_WC_RECV";
  case IBV_WC_RECV_RDMA_WITH_IMM : return "IBV_WC_RECV_RDMA_WITH_IMM";
  default : return "Unknown IBV_WC opcode";
}

static bool waitForCompletion(bool is_send, RdmaEndpoint *endpoint, uint64_t expected_wr_id, bool use_polling_completion, uint32_t usec_sleep_between_poll_attempts, double timeout_seconds, bool *timed_out_ret, char *error_message, int max_error_message_chars, uint64_t *bytes_received_ret, uint32_t *piggy_back_message_ret) {
  bool got_start_time = false;
  double start_time = 0;

 reetry:
  // Just try to get one completion, even if many are ready
  struct ibv_wc wc;
  int completion_count = ibv_poll_cq(endpoint->cq, 1, &wc);
  if (completion_count == -1) {
    snprintf(error_message, max_error_message_chars, "ibv_poll_cq() failed: errno=%d", errno);
    return false;
  }

  // See if need to wait for the completion
  if (completion_count == 0) {
    if (timeout_seconds == 0) {
      // Don't wait just return
      *timed_out_ret = true;
      return true;
    }
    // Need to wait
    if (use_polling_completion) {
      // See if time to return
      if (timeout_seconds > 0) {
        if (!got_start_time) {
          got_start_time = true;
          start_time = clockTimeSeconds();
        }
        double elapsed_seconds = clockTimeSeconds() - start_time;
        if (elapsed_seconds >= timeout_seconds) {
          // Timed out
          *timed_out_ret = true;
          return true;
        }
      }
      // Data not ready. In polling mode, so sleep a little to avoid buring up CPU core
      if (usec_sleep_between_poll_attempts > 0) clockSleepUsecs(usec_sleep_between_poll_attempts);
    } else {
      // Use completion channel to sleep
      if (!eventDrivenCompletionWait(endpoint, timeout_seconds, error_message, max_error_message_chars)) return false;
    }
    goto retry;
  }

  // Got a completion, process it
  if (wc.status != IBV_WC_SUCCESS) {
    snprintf(error_message, max_error_message_chars, "ibv_poll_cq() has work complete error: '%s'", wcErrorToText(wc.status));
    return false;
  }
  if (wc.wr_id != expected_wr_id) {
    if (is_send) {
      snprintf(error_message, max_error_message_chars, "Work completion does not match expected send_request. Was takyonIsSent() called in a different order from takyonSend()?");
    } else {
      snprintf(error_message, max_error_message_chars, "Work completion does not match expected recv_request. Was takyonIsRecved() called in a different order from takyonPostRecvs()?");
    }
    return false;
  }
  if (is_send) {
    if (wc.opcode != IBV_WC_SEND) {
      snprintf(error_message, max_error_message_chars, "Work completion was for '%s' but expected 'IBV_WC_SEND'", wcOpcodeToText(wc.opcode));
      return false;
    }
  } else {
    if (wc.opcode != IBV_WC_RECV) {
      snprintf(error_message, max_error_message_chars, "Work completion was for '%s' but expected 'IBV_WC_RECV'", wcOpcodeToText(wc.opcode));
      return false;
    }
    if (!(wc.wc_flags & IBV_WC_WITH_IMM)) {
      snprintf(error_message, max_error_message_chars, "Work completion did not contain an IMM");
      return false;
    }
  }

  // Success: return info
  if (!is_send) {
    *bytes_received_ret = wc.byte_len;
    *piggy_back_message_ret = ntohl(wc.imm_data);
  }

  return true;
}

bool rdmaStartSend(TakyonPath *path, RdmaEndpoint *endpoint, TakyonSendRequest *request, uint32_t piggy_back_message, double timeout_seconds, bool *timed_out_ret, char *error_message, int max_error_message_chars) {
  RdmaSendRequest *rdma_request = (RdmaSendRequest *)request->private;
  struct ibv_send_wr send_wr;

  // Fill in message to be sent
  send_wr.next = NULL;
  send_wr.wr_id = request;
  send_wr.num_sge = request->sub_buffer_count;
  send_wr.sg_list = rdma_request->sges;
  send_wr.opcode = IBV_WR_SEND_WITH_IMM;
  send_wr.imm_data = htonl(piggy_back_message);
  for (uint32_t j=0; j<request->sub_buffer_count; j++) {
    TakyonSubBuffer *sub_buffer = request->sub_buffers[j];
    TakyonBuffer *buffer = path->attrs.buffers[sub_buffer->buffer_index];
    RdmaBuffer *rdma_buffer = (RdmaBuffer *)buffer->private;
    struct ibv_sge *sge = &send_wr.sg_list[j];
    sge->addr = (uint64_t)buffer + sub_buffer->offset;
    sge->length = sub_buffer->bytes;
    sge->lkey = rdma_buffer->mr->lkey;
  }

  // Protocal specific stuff
  if (endpoint->protocol == RDMA_PROTOCOL_UD_MULTICAST) {
    send_wr.wr.ud.ah = endpoint->multicast_ah;
    send_wr.wr.ud.remote_qpn = endpoint->multicast_qpn;
    send_wr.wr.ud.remote_qkey = endpoint->multicast_qkey;
  }

  // Signaling
  send_wr.send_flags = 0;
  if (request->use_is_sent_notification) {
    send_wr.send_flags |= IBV_SEND_SIGNALED; /*+ does signaling need to eventually occur? */
  }

  // Start the send transfer
  struct ibv_send_wr *bad_wr;
  struct ibv_qp *qp = (endpoint->protocol == RDMA_PROTOCOL_UD_MULTICAST) ? endpoint->id->qp : endpoint->qp;
  int rc = ibv_post_send(qp, send_wr, &bad_wr);
  if (rc != 0) {
    snprintf(error_message, max_error_message_chars, "Failed to start send: errno=%d", errno);
    return false;
  }
  return true;
}

bool rdmaIsRecved(RdmaEndpoint *endpoint, TakyonRecvRequest *request, double timeout_seconds, bool *timed_out_ret, char *error_message, int max_error_message_chars, uint64_t *bytes_received_ret, uint32_t *piggy_back_message_ret) {
  bool is_send = false;
  return waitForCompletion(is_send, endpoint, (uint64_t)request, request->use_polling_completion, request->usec_sleep_between_poll_attempts, timeout_seconds, timed_out_ret, error_message, max_error_message_chars, bytes_received_ret, piggy_back_message_ret);
}

bool rdmaIsSent(RdmaEndpoint *endpoint, TakyonSendRequest *request, double timeout_seconds, bool *timed_out_ret, char *error_message, int max_error_message_chars) {
  bool is_send = true;
  return waitForCompletion(is_send, endpoint, (uint64_t)request, request->use_polling_completion, request->usec_sleep_between_poll_attempts, timeout_seconds, timed_out_ret, error_message, max_error_message_chars, NULL, NULL);
}
