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
#include "utils_time.h"
#include "utils_socket.h"
#include "takyon_private.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <poll.h>
#include <arpa/inet.h>

/* RDMA Notes:
  - RDMA Protocols:
    - UD Multicast: Using the CM (connection manager) since it's convenient and no hand shake information is needed
    - UD Unicast:   Using raw verbs with an external TCP socket to do the 'create' handshake and lifetime disconnect detection
    - UC:           CM doesn't support UC so using raw verbs with an external TCP socket to do the 'create' handshake and lifetime disconnect detection
    - RC:           Using raw verbs with an external TCP socket to do the 'create' handshake and lifetime disconnect detection
  - Avoiding use of inline bytes when sending since it only works with CPU memory, and adds complexity to track memory type
  - Posted UD recv requests need to have an extra sizeof(struct ibv_grh) bytes (40 bytes), which is the IB's Global Routing Header placed at the beginning of any UD received message

  - Currently not tested with RoCE v1.x since that may be a dead RDMA variant
  - Currently not tested with iWarp since that may be a dead RDMA variant
  - Currently not implemented the Network Direct SPI v2 variant as may be not any demand
*/

/*+ MELLANOX questions:
   - How to detect max MTU for RoCEv2 across switches?
   - CUDA transfer need at least 33 bytes or ibv_poll_cq crashes (Segmentation fault (core dumped)):
     GDB back trace:
      #0  __memcpy_generic () at ../sysdeps/aarch64/multiarch/../memcpy.S:120
   	 #1  0x0000fffff007dcd4 in ?? () from /usr/lib/aarch64-linux-gnu/libibverbs/libmlx5-rdmav34.so
   	 #2  0x0000fffff004f8b8 in ?? () from /usr/lib/aarch64-linux-gnu/libibverbs/libmlx5-rdmav34.so
   	 #3  0x0000aaaaaaac1ed4 in ibv_poll_cq (cq=0xaaaaab099ac0, num_entries=1, wc=0xffffffffe568) at /usr/include/infiniband/verbs.h:2873
   - r3u04 is duplicating RDMA (not using sockets) multicast packets (is this an RDMA loopback issue? If so, how to turn off?). This does not occur on mercuryRDP.
*/

#define MY_MAX(_a, _b) ((_a)>(_b) ? (_a) : (_b))

#define PORT_INFO_TEXT_BYTES 200

typedef struct  {
  int lid;           // For infiniband
  int qpn;
  int psn;
  union ibv_gid gid; // For RoCE
  char gid_text[33]; // For RoCE
  enum ibv_mtu mtu_mode;
  uint32_t max_pending_read_and_atomic_requests;
  uint32_t unicast_qkey;
  char socket_data[PORT_INFO_TEXT_BYTES]; // Use for transferring to remote side regardless of endian
} PortInfo;

static struct ibv_context *getRdmaContextFromNamedDevice(TakyonPath *path, const char *rdma_device_name, char *error_message, int max_error_message_chars, int *max_qp_wr_ret, int *max_qp_rd_atom_ret) {
  // IMPORTANT: valgrind is reporting this memory is never freed even with ibv_free_device_list(dev_list) being called. Mellanox has been informed
  struct ibv_device **dev_list = ibv_get_device_list(NULL);
  if (dev_list == NULL) {
    snprintf(error_message, max_error_message_chars, "ibv_get_device_list() failed");
    return NULL;
  }
  uint32_t index = 0;
  while (dev_list[index] != NULL) {
    if (strcmp(ibv_get_device_name(dev_list[index]), rdma_device_name) == 0) {
      // Found the device, so open it
      struct ibv_context *context = ibv_open_device(dev_list[index]);
      if (context == NULL) {
        snprintf(error_message, max_error_message_chars, "ibv_open_device() failed for device '%s': errno=%d", rdma_device_name, errno);
        return NULL;
      }

      // Determine the device attributes
      struct ibv_device_attr device_attr;
      int rc = ibv_query_device(context, &device_attr);
      if (rc != 0) {
	snprintf(error_message, max_error_message_chars, "ibv_query_device() failed for device '%s': errno=%d", rdma_device_name, errno);
	return NULL;
      }
      *max_qp_rd_atom_ret = device_attr.max_qp_rd_atom;
      *max_qp_wr_ret = device_attr.max_qp_wr;
      if (path->attrs.verbosity & TAKYON_VERBOSITY_CREATE_DESTROY_MORE) {
	printf("  Device info:\n");
	printf("    max_qp_wr = %d\n", device_attr.max_qp_wr);
	printf("    max_qp_rd_atom = %d\n", device_attr.max_qp_rd_atom);
	printf("    max_qp_init_rd_atom = %d\n", device_attr.max_qp_init_rd_atom);
	printf("    atomic_cap = %d, IBV_ATOMIC_NONE=%d, IBV_ATOMIC_HCA=%d, IBV_ATOMIC_GLOB=%d\n", device_attr.atomic_cap, IBV_ATOMIC_NONE, IBV_ATOMIC_HCA, IBV_ATOMIC_GLOB);
      }

      ibv_free_device_list(dev_list);
      return context;
    }
    index++;
  }

  ibv_free_device_list(dev_list);
  snprintf(error_message, max_error_message_chars, "Failed to find the RDMA device '%s'. Run the command line program 'ibv_devinfo' to see the devices.", rdma_device_name);
  return NULL;
}

static void gidToText(const union ibv_gid *gid, char *gid_text) {
  uint32_t gid_int_array[4];
  memcpy(gid_int_array, gid, sizeof(gid_int_array));
  for (int i=0; i<4; i++) {
    sprintf(&gid_text[i * 8], "%08x", htobe32(gid_int_array[i]));
  }
}

static void textToGid(const char *gid_text, union ibv_gid *gid) {
  uint32_t gid_int_array[4];
  char tmp[9];
  tmp[8] = 0;
  for (int i = 0; i < 4; i++) {
    memcpy(tmp, gid_text + i * 8, 8);
    __be32 v32;
    sscanf(tmp, "%x", &v32);
    gid_int_array[i] = be32toh(v32);
  }
  memcpy(gid, gid_int_array, sizeof(*gid));
}

static const char *portStateToText(enum ibv_port_state state) {
  switch (state) {
  case IBV_PORT_NOP : return "IBV_PORT_NOP";
  case IBV_PORT_DOWN : return "IBV_PORT_DOWN";
  case IBV_PORT_INIT : return "IBV_PORT_INIT";
  case IBV_PORT_ARMED : return "IBV_PORT_ARMED";
  case IBV_PORT_ACTIVE : return "IBV_PORT_ACTIVE";
  case IBV_PORT_ACTIVE_DEFER : return "IBV_PORT_ACTIVE_DEFER";
  default : return "unknown";
  }
}

static uint32_t mtuModeToBytes(enum ibv_mtu mtu_mode) {
  switch (mtu_mode) {
  case IBV_MTU_256 : return 256;
  case IBV_MTU_512 : return 512;
  case IBV_MTU_1024 : return 1024;
  case IBV_MTU_2048 : return 2048;
  case IBV_MTU_4096 : return 4096;
  default : return 0;
  }
}

static const char *mtuModeToText(enum ibv_mtu mtu_mode) {
  switch (mtu_mode) {
  case IBV_MTU_256 : return "IBV_MTU_256";
  case IBV_MTU_512 : return "IBV_MTU_512";
  case IBV_MTU_1024 : return "IBV_MTU_1024";
  case IBV_MTU_2048 : return "IBV_MTU_2048";
  case IBV_MTU_4096 : return "IBV_MTU_4096";
  default : return "unknown";
  }
}

static const char *linkLayerToText(uint8_t link_layer) {
  switch (link_layer) {
  case IBV_LINK_LAYER_ETHERNET : return "IBV_LINK_LAYER_ETHERNET";
  case IBV_LINK_LAYER_INFINIBAND : return "IBV_LINK_LAYER_INFINIBAND";
  case IBV_LINK_LAYER_UNSPECIFIED : return "IBV_LINK_LAYER_UNSPECIFIED";
  default : return "unknown";
  }
}

static bool getLocalPortInfo(TakyonPath *path, struct ibv_context *context, struct ibv_qp *qp, uint32_t rdma_port_id, uint32_t max_pending_read_and_atomic_requests, uint32_t unicast_local_qkey, int gid_index, PortInfo *port_info_ret, char *error_message, int max_error_message_chars) {
  PortInfo port_info = {}; // Zero the structure

  struct ibv_port_attr port_attrs;
  int rc = ibv_query_port(context, rdma_port_id, &port_attrs);
  if (rc != 0) {
    snprintf(error_message, max_error_message_chars, "ibv_query_port() failed for RDMA port ID %d: errno=%d", rdma_port_id, errno);
    return false;
  }

  // Provide some info
  if (path->attrs.verbosity & TAKYON_VERBOSITY_CREATE_DESTROY_MORE) {
    printf("  RDMA Port %u Info:\n", rdma_port_id);
    printf("    State:            %s\n", portStateToText(port_attrs.state));
    printf("    Max MTU:          %s\n", mtuModeToText(port_attrs.max_mtu));
    printf("    Active MTU:       %s\n", mtuModeToText(port_attrs.active_mtu));
    printf("    Max message size: %u\n", port_attrs.max_msg_sz);
    printf("    Link Layer:       %s\n", linkLayerToText(port_attrs.link_layer));
  }

  port_info.max_pending_read_and_atomic_requests = max_pending_read_and_atomic_requests;
  port_info.unicast_qkey = unicast_local_qkey;
  port_info.mtu_mode = port_attrs.active_mtu;
  port_info.qpn = qp->qp_num;
  srand48((long int)clockTimeNanoseconds());
  port_info.psn = lrand48() & 0xffffff;
  port_info.lid = port_attrs.lid;
  if (port_attrs.link_layer != IBV_LINK_LAYER_ETHERNET && port_attrs.lid == 0) {
    snprintf(error_message, max_error_message_chars, "Could not determine the RDMA port's LID value");
    return false;
  }

  if (gid_index >= 0) {
    rc = ibv_query_gid(context, rdma_port_id, gid_index, &port_info.gid);
    if (rc != 0) {
      snprintf(error_message, max_error_message_chars, "Could not determine the RDMA port's GID value");
      return false;
    }
  }

  if (inet_ntop(AF_INET6, &port_info.gid, port_info.gid_text, sizeof(port_info.gid_text)) == NULL) {
    snprintf(error_message, max_error_message_chars, "inet_ntop() failed, need to get the RDMA port's GID info: errno=%d", errno);
    return false;
  }
  if (path->attrs.verbosity & TAKYON_VERBOSITY_CREATE_DESTROY_MORE) {
    printf("  Local connection info: LID 0x%04x, QPN 0x%06x, PSN 0x%06x, MTU %s, MaxPendingReadAndAtomics %u, UnicastQKEY 0x%06x, GID '%s'\n",
	   port_info.lid, port_info.qpn, port_info.psn, mtuModeToText(port_info.mtu_mode), port_info.max_pending_read_and_atomic_requests, port_info.unicast_qkey, port_info.gid_text);
  }

  gidToText(&port_info.gid, port_info.gid_text);
  sprintf(port_info.socket_data, "%04x:%06x:%06x:%04x:%06x:%06x:%s", port_info.lid, port_info.qpn, port_info.psn, port_info.mtu_mode, port_info.max_pending_read_and_atomic_requests, port_info.unicast_qkey, port_info.gid_text);

  *port_info_ret = port_info;

  return true;
}


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
  int rc = rdma_get_cm_event(event_ch, &event);
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

static struct ibv_cq *createCompletionQueue(struct ibv_context *context, int min_completions, struct ibv_comp_channel *comp_ch, char *error_message, int max_error_message_chars) {
  void *app_data = NULL;
  int comp_vector = 0;
  if (min_completions == 0) min_completions = 1; // This may occur for RC or UC where the endpoint only does one of 'send' or 'recv', but QPs need value send and recv completion queues.
  struct ibv_cq *cq = ibv_create_cq(context, min_completions, app_data, comp_ch, comp_vector);
  if (cq == NULL) {
    snprintf(error_message, max_error_message_chars, "ibv_create_cq() failed");
    return NULL;
  }
  return cq;
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

static struct ibv_qp *createQueuePair(TakyonPath *path, struct ibv_pd *pd, struct ibv_cq *send_cq, struct ibv_cq *recv_cq, enum ibv_qp_type qp_type, uint32_t rdma_port_id, uint32_t unicast_local_qkey,
                                      uint32_t max_send_wr, uint32_t max_recv_wr, uint32_t max_send_sge, uint32_t max_recv_sge, char *error_message, int max_error_message_chars) {
  struct ibv_qp_init_attr init_attrs = {}; // Zero structure
  init_attrs.qp_context = NULL; // app_data
  init_attrs.send_cq = send_cq;
  init_attrs.recv_cq = recv_cq;
  init_attrs.srq = NULL;
  init_attrs.cap.max_send_wr = max_send_wr;
  init_attrs.cap.max_recv_wr = max_recv_wr;
  init_attrs.cap.max_send_sge = max_send_sge;
  init_attrs.cap.max_recv_sge = max_recv_sge;
  init_attrs.cap.max_inline_data = 0;
  init_attrs.qp_type = qp_type; // One of IBV_QPT_RC, IBV_QPT_UD, IBV_QPT_UC
  init_attrs.sq_sig_all = 0; // If 0, then preparing a request must set if signaled or not
  struct ibv_qp *qp = ibv_create_qp(pd, &init_attrs);
  if (qp == NULL) {
    snprintf(error_message, max_error_message_chars, "ibv_create_qp() failed");
    return NULL;
  }

  // Ask for the max inline byte size
  if (path->attrs.verbosity & TAKYON_VERBOSITY_CREATE_DESTROY_MORE) {
    struct ibv_qp_attr attrs;
    int rc = ibv_query_qp(qp, &attrs, IBV_QP_CAP, &init_attrs);
    if (rc != 0) {
      snprintf(error_message, max_error_message_chars, "ibv_query_qp() failed. Could not determine max_inline_bytes: errno=%d", errno);
      ibv_destroy_qp(qp);
      return false;
    }
    printf("  Max Inline Bytes = %u, but currently set to 0 in the case GPU memory is used\n", init_attrs.cap.max_inline_data);
  }

  // Move the QP state to INIT
  int qp_access_flags = 0;
  if (qp_type == IBV_QPT_UC) {
    qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
  } else if (qp_type == IBV_QPT_RC) {
    qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
  }
  struct ibv_qp_attr attrs = { .qp_state        = IBV_QPS_INIT,
                               .pkey_index      = 0,
                               .port_num        = rdma_port_id,
                               .qp_access_flags = qp_access_flags };
  if (qp_type == IBV_QPT_UD) {
    attrs.qkey = unicast_local_qkey;
  }

  enum ibv_qp_attr_mask attr_mask = 0;
  if (qp_type == IBV_QPT_UD) {
    attr_mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY;
  } else {
    attr_mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
  }
  int rc = ibv_modify_qp(qp, &attrs, attr_mask);
  if (rc != 0) {
    snprintf(error_message, max_error_message_chars, "ibv_modify_qp() failed. Could not change QP state to INIT: errno=%d", errno);
    ibv_destroy_qp(qp);
    return false;
  }

  return qp;
}

static bool moveQpStateToRTS(struct ibv_qp *qp, struct ibv_pd *pd, enum ibv_qp_type qp_type, bool is_UD_sender, uint32_t rdma_port_id, PortInfo *local_port_info, PortInfo *remote_port_info, RdmaAppOptions app_options, enum ibv_mtu mtu_mode, char *error_message, int max_error_message_chars, struct ibv_ah **unicast_sender_ah_ret) {
  struct ibv_qp_attr attrs = { .qp_state    = IBV_QPS_RTR,
                               .path_mtu    = mtu_mode, //*+ how to auto detect where it also accounts for intermediate switch MTUs?
                               .dest_qp_num = remote_port_info->qpn,
                               .rq_psn      = remote_port_info->psn,
			       .max_dest_rd_atomic = MY_MAX(1,remote_port_info->max_pending_read_and_atomic_requests), // RC only: Number of pending read or atomic operations with this endpoint as the destination. If more are posted, an error may occur or transfers will be stalled.
			       .min_rnr_timer = app_options.min_rnr_timer,  // RC only: Defines index into timeout table. Index is 0 .. 31, 0 = 665 msecs, 1 = 0.01 msecs, 31 = 491 msecs.
                               .ah_attr     = { .is_global     = 0,
                                                .dlid          = remote_port_info->lid,
                                                .sl            = app_options.service_level,
                                                .src_path_bits = 0,
                                                .port_num      = rdma_port_id
                                               }};
  if (remote_port_info->gid.global.interface_id != 0) {
    attrs.ah_attr.is_global      = 1;
    attrs.ah_attr.grh.hop_limit  = app_options.hop_limit; // Max routers to travel through before being dropped
    attrs.ah_attr.grh.dgid       = remote_port_info->gid;
    attrs.ah_attr.grh.sgid_index = app_options.gid_index;
  }

  // Move to RTR
  enum ibv_qp_attr_mask attr_mask = 0;
  if (qp_type == IBV_QPT_UC) {
    attr_mask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN;
  } else if (qp_type == IBV_QPT_RC) {
    attr_mask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
  } else if (qp_type == IBV_QPT_UD) {
    attr_mask = IBV_QP_STATE;
  }
  int rc = ibv_modify_qp(qp, &attrs, attr_mask);
  if (rc != 0) {
    snprintf(error_message, max_error_message_chars, "ibv_modify_qp() failed. Could not change QP state to RTR: errno=%d", errno);
    return false;
  }

  // Move to RTS
  if (qp_type != IBV_QPT_UD || is_UD_sender) {
    attrs.qp_state = IBV_QPS_RTS;
    attrs.sq_psn   = local_port_info->psn;
    if (qp_type == IBV_QPT_RC) {
      // Set additional attrs for RC (reliable connection)
      attrs.timeout       = app_options.retransmit_timeout; // RC Only: Retransmit timeout. Defines index into timeout table. Index is 0 .. 31, 0 = infinite, 1 = 8.192 usecs, 14 = .0671 secs, 31 = 8800 secs
      attrs.retry_cnt     = app_options.retry_cnt;          // RC Only: Max re-transmits before erroring (without remote NACK). Max is 7
      attrs.rnr_retry     = app_options.rnr_retry;          // RC Only: Max re-transmits before erroring (with remote NACK). Max is 6, but 7 is infinit
      attrs.max_rd_atomic = MY_MAX(1,local_port_info->max_pending_read_and_atomic_requests); // RC Only: Number of pending read or atomic operations initiated by this endpoint. If more are posted, an error may occur or transfers will be stalled.
    }
    attr_mask = IBV_QP_STATE | IBV_QP_SQ_PSN;
    if (qp_type == IBV_QPT_RC) {
      attr_mask |= IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC;
    }
    rc = ibv_modify_qp(qp, &attrs, attr_mask);
    if (rc != 0) {
      snprintf(error_message, max_error_message_chars, "ibv_modify_qp() failed. Could not change QP state to RTS: errno=%d", errno);
      return false;
    }

    if (qp_type == IBV_QPT_UD) {
      // UD unicast sender: need an address handle
      struct ibv_ah *unicast_sender_ah = ibv_create_ah(pd, &attrs.ah_attr);
      if (unicast_sender_ah == NULL) {
	snprintf(error_message, max_error_message_chars, "ibv_create_ah() failed. Could not create UD unicast sender address handle");
	return false;
      }
      *unicast_sender_ah_ret = unicast_sender_ah;
    }
  }

  return true;
}

static bool createConnectionManagerQueuePair(struct rdma_cm_id *id, struct ibv_pd *pd, struct ibv_cq *send_cq, struct ibv_cq *recv_cq, enum ibv_qp_type qp_type,
                                             uint32_t max_send_wr, uint32_t max_recv_wr, uint32_t max_send_sge, uint32_t max_recv_sge,
                                             char *error_message, int max_error_message_chars) {
  struct ibv_qp_init_attr attrs = {}; // Zero structure
  attrs.qp_context = NULL; // app_data
  attrs.send_cq = send_cq;
  attrs.recv_cq = recv_cq;
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

static struct ibv_mr *registerMemoryRegion(struct ibv_pd *pd, void *addr, size_t bytes, enum ibv_access_flags access, char *error_message, int max_error_message_chars) {
  // Access flags:
  //   IBV_ACCESS_LOCAL_WRITE   Allow local host write access
  //   IBV_ACCESS_REMOTE_WRITE  Allow remote hosts write access
  //   IBV_ACCESS_REMOTE_READ   Allow remote hosts read access
  //   IBV_ACCESS_REMOTE_ATOMIC Allow remote hosts atomic access
  //   IBV_ACCESS_MW_BIND       Allow memory windows on this MR
  struct ibv_mr *mr = ibv_reg_mr(pd, addr, bytes, access);
  if (mr == NULL) {
    snprintf(error_message, max_error_message_chars, "ibv_reg_mr(addr=0x%jx, bytes=%ju, access=0x%x) failed", (uint64_t)addr, bytes, access);
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
    int rc = rdma_resolve_addr(id, local_NIC_info->ai_src_addr, multicast_group_info->ai_dst_addr, timeout_ms);
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
  *multicast_addr_out = *multicast_group_info->ai_dst_addr;
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

static struct ibv_ah *joinMulticastGroup(struct rdma_cm_id *id, struct rdma_event_channel *event_ch, struct sockaddr *multicast_addr, char *error_message, int max_error_message_chars, uint32_t *qp_num_out, uint32_t *qkey_out) {
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
    snprintf(error_message, max_error_message_chars, "Failed to ACK event: errno=%d", errno);
    ibv_destroy_ah(ah);
    return NULL;
  }

  return ah;
}

bool rdmaEndpointPostRecvs(TakyonPath *path, RdmaEndpoint *endpoint, uint32_t request_count, TakyonRecvRequest *requests, char *error_message, int max_error_message_chars) {
  // Prepare WR chain
  struct ibv_recv_wr *first_wr = NULL;
  struct ibv_recv_wr *curr_wr = NULL;
  for (uint32_t i=0; i<request_count; i++) {
    TakyonRecvRequest *takyon_request = &requests[i];
    RdmaRecvRequest *rdma_request = (RdmaRecvRequest *)takyon_request->private_data;
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
    curr_wr->wr_id = (uint64_t)takyon_request;
    curr_wr->num_sge = takyon_request->sub_buffer_count;
    curr_wr->sg_list = rdma_request->sges;
    for (uint32_t j=0; j<takyon_request->sub_buffer_count; j++) {
      TakyonSubBuffer *sub_buffer = &takyon_request->sub_buffers[j];
      TakyonBuffer *buffer = &path->attrs.buffers[sub_buffer->buffer_index];
      RdmaBuffer *rdma_buffer = (RdmaBuffer *)buffer->private_data;
      struct ibv_sge *sge = &curr_wr->sg_list[j];
      sge->addr = (uint64_t)buffer->addr + sub_buffer->offset;
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
                                          uint32_t max_send_wr, uint32_t max_recv_wr, uint32_t max_send_sges, uint32_t max_recv_sges,
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
  RdmaEndpoint *endpoint = NULL;

  if (path->attrs.verbosity & TAKYON_VERBOSITY_CREATE_DESTROY_MORE) printf("  Create RDMA multicast %s, max_send_wr=%u, max_recv_wr=%u, max_send_sges=%u, max_recv_sges=%u\n", is_sender ? "sender" : "recver", max_send_wr, max_recv_wr, max_send_sges, max_recv_sges);

  // Event channel
  event_ch = createConnectionManagerEventChannel(error_message, max_error_message_chars);
  if (event_ch == NULL) goto failed;

  // Connection manager ID
  enum rdma_port_space port_space = RDMA_PS_UDP;
  id = createConnectionManagerId(event_ch, port_space, error_message, max_error_message_chars);
  if (id == NULL) goto failed;

  // Multicast addr
  int timeout_ms = (timeout_seconds < 0) ? INT_MAX : (int)(timeout_seconds * 1000);
  if (!getMulticastAddr(id, event_ch, local_NIC_ip_addr, multicast_group_ip_addr, timeout_ms, error_message, max_error_message_chars, &multicast_addr)) goto failed;

  // RDMA MTU bytes
  uint32_t mtu_bytes = getRdmaMTU(id->verbs, id->port_num, error_message, max_error_message_chars);
  if (mtu_bytes == 0) goto failed;
  if (path->attrs.verbosity & TAKYON_VERBOSITY_CREATE_DESTROY_MORE) printf("  RDMA Multicast MTU bytes=%d on RDMA port %d\n", mtu_bytes, id->port_num);

  // IMPORTANT: Transfers might be event driven so create the completion channel just in case
  // Completion channel
  comp_ch = createCompletionChannel(id->verbs, error_message, max_error_message_chars);
  if (comp_ch == NULL) goto failed;

  // Completion queue
  int min_completions = is_sender ? max_send_wr : max_recv_wr;
  cq = createCompletionQueue(id->verbs, min_completions, comp_ch, error_message, max_error_message_chars);
  if (cq == NULL) goto failed;
  if (!armCompletionQueue(cq, error_message, max_error_message_chars)) goto failed;

  // Queue pair
  enum ibv_qp_type qp_type = IBV_QPT_UD;
  if (!createConnectionManagerQueuePair(id, id->pd, cq, cq, qp_type, max_send_wr, max_recv_wr, max_send_sges, max_recv_sges, error_message, max_error_message_chars)) goto failed;
  qp_created = true;

  // Register buffers
  enum ibv_access_flags access = IBV_ACCESS_LOCAL_WRITE; // Need write access even for the sender since CUDA send memory needs write access
  for (uint32_t i=0; i<path->attrs.buffer_count; i++) {
    TakyonBuffer *takyon_buffer = &path->attrs.buffers[i];
    RdmaBuffer *rdma_buffer = (RdmaBuffer *)takyon_buffer->private_data;
    // IMPORTANT: assuming buffers are zeroed at init
    rdma_buffer->mr = registerMemoryRegion(id->pd, takyon_buffer->addr, takyon_buffer->bytes, access, error_message, max_error_message_chars);
    if (rdma_buffer->mr == NULL) goto failed;
    if (path->attrs.verbosity & TAKYON_VERBOSITY_CREATE_DESTROY_MORE) printf("  Registered memory region: addr=0x%jx, bytes=%ju, lkey=%u, rkey=%u\n", (uint64_t)takyon_buffer->addr, takyon_buffer->bytes, rdma_buffer->mr->lkey, rdma_buffer->mr->rkey);
  }

  // Build the endpoint structure before posting recvs
  endpoint = calloc(1, sizeof(RdmaEndpoint));
  if (endpoint == NULL) goto failed;
  endpoint->protocol = RDMA_PROTOCOL_UD_MULTICAST;
  endpoint->is_sender = is_sender;
  endpoint->event_ch = event_ch;
  endpoint->id = id;
  if (is_sender) {
    endpoint->send_comp_ch = comp_ch;
    endpoint->send_cq = cq;
  } else {
    endpoint->recv_comp_ch = comp_ch;
    endpoint->recv_cq = cq;
  }
  endpoint->multicast_addr = multicast_addr;
  endpoint->mtu_bytes = mtu_bytes;

  // Post recvs
  if (!is_sender && recv_request_count > 0) {
    if (!rdmaEndpointPostRecvs(path, endpoint, recv_request_count, recv_requests, error_message, max_error_message_chars)) goto failed;
  }

  // Join multicast group
  multicast_ah = joinMulticastGroup(id, event_ch, &multicast_addr, error_message, max_error_message_chars, &multicast_qp_num, &multicast_qkey);
  if (multicast_ah == NULL) goto failed;
  if (path->attrs.verbosity & TAKYON_VERBOSITY_CREATE_DESTROY_MORE) printf("  Joined RDMA Multicast group: qpn=%u, qkey=%u\n", multicast_qp_num, multicast_qkey);
  endpoint->multicast_ah = multicast_ah;
  endpoint->multicast_qp_num = multicast_qp_num;
  endpoint->multicast_qkey = multicast_qkey;

  // Ready to start transfering
  return endpoint;

 failed:
  if (multicast_ah != NULL) {
    rdma_leave_multicast(id, &multicast_addr);
    ibv_destroy_ah(multicast_ah);
  }
  if (qp_created) rdma_destroy_qp(id);
  if (cq != NULL) ibv_destroy_cq(cq);
  for (uint32_t i=0; i<path->attrs.buffer_count; i++) {
    TakyonBuffer *takyon_buffer = &path->attrs.buffers[i];
    RdmaBuffer *rdma_buffer = (RdmaBuffer *)takyon_buffer->private_data;
    if (rdma_buffer->mr != NULL) ibv_dereg_mr(rdma_buffer->mr);
  }
  if (comp_ch != NULL) ibv_destroy_comp_channel(comp_ch);
  if (id != NULL) rdma_destroy_id(id);
  if (event_ch != NULL) rdma_destroy_event_channel(event_ch);
  if (endpoint != NULL) free(endpoint);
  return NULL;
}

RdmaEndpoint *rdmaCreateEndpoint(TakyonPath *path, bool is_endpointA, int read_pipe_fd, enum ibv_qp_type qp_type, bool is_UD_sender, const char *rdma_device_name, uint32_t rdma_port_id,
				 uint32_t max_send_wr, uint32_t max_recv_wr, uint32_t max_send_sges, uint32_t max_recv_sges, uint32_t max_pending_read_and_atomic_requests,
				 uint32_t recv_request_count, TakyonRecvRequest *recv_requests,
				 RdmaAppOptions app_options, double timeout_seconds, char *error_message, int max_error_message_chars) {
  struct ibv_context *context = NULL;
  struct ibv_comp_channel *send_comp_ch = NULL;
  struct ibv_comp_channel *recv_comp_ch = NULL;
  struct ibv_cq *send_cq = NULL;
  struct ibv_cq *recv_cq = NULL;
  struct ibv_pd *pd = NULL;
  struct ibv_qp *qp = NULL;
  struct ibv_ah *unicast_sender_ah = NULL;
  RdmaEndpoint *endpoint = NULL;
  int64_t timeout_nano_seconds = (int64_t)(timeout_seconds * NANOSECONDS_PER_SECOND_DOUBLE);
  bool timed_out = false;

  if (path->attrs.verbosity & TAKYON_VERBOSITY_CREATE_DESTROY_MORE) {
    printf("  Create RDMA UC, max_send_wr=%u, max_recv_wr=%u, max_send_sges=%u, max_recv_sges=%u, max_pending_read_and_atomic_requests=%u\n",
           max_send_wr, max_recv_wr, max_send_sges, max_recv_sges, max_pending_read_and_atomic_requests);
  }

  // Get handle to RDMA device
  int max_qp_wr;
  int max_qp_rd_atom;
  context = getRdmaContextFromNamedDevice(path, rdma_device_name, error_message, max_error_message_chars, &max_qp_wr, &max_qp_rd_atom);
  if (context == NULL) return NULL;
  if (max_qp_wr < (int)(max_send_wr + max_recv_wr)) {
    TAKYON_RECORD_ERROR(path->error_message, "Total send/recv/read/write/atomic requests must be <= %d\n", max_qp_wr);
    goto failed;
  }
  if (max_qp_rd_atom < (int)max_pending_read_and_atomic_requests) {
    TAKYON_RECORD_ERROR(path->error_message, "Total read and atomic requests must be <= %d\n", max_qp_rd_atom);
    goto failed;
  }

  // IMPORTANT: Transfers might be event driven so create the completion channel just in case
  // Completion channels
  send_comp_ch = createCompletionChannel(context, error_message, max_error_message_chars);
  if (send_comp_ch == NULL) goto failed;
  recv_comp_ch = createCompletionChannel(context, error_message, max_error_message_chars);
  if (recv_comp_ch == NULL) goto failed;

  // Completion queues
  send_cq = createCompletionQueue(context, max_send_wr, send_comp_ch, error_message, max_error_message_chars);
  if (send_cq == NULL) goto failed;
  if (!armCompletionQueue(send_cq, error_message, max_error_message_chars)) goto failed;
  recv_cq = createCompletionQueue(context, max_recv_wr, recv_comp_ch, error_message, max_error_message_chars);
  if (recv_cq == NULL) goto failed;
  if (!armCompletionQueue(recv_cq, error_message, max_error_message_chars)) goto failed;

  // Protection domain
  pd = createProtectionDomain(context, error_message, max_error_message_chars);
  if (pd == NULL) goto failed;

  // Queue pair
  srand48((long int)clockTimeNanoseconds());
  uint32_t unicast_local_qkey = lrand48() & 0xffffff;
  qp = createQueuePair(path, pd, send_cq, recv_cq, qp_type, rdma_port_id, unicast_local_qkey, max_send_wr, max_recv_wr, max_send_sges, max_recv_sges, error_message, max_error_message_chars);
  if (qp == NULL) goto failed;

  // Register buffers
  enum ibv_access_flags access = IBV_ACCESS_LOCAL_WRITE;
  if (qp_type == IBV_QPT_UC) {
    access |= IBV_ACCESS_REMOTE_WRITE;
  } else if (qp_type == IBV_QPT_RC) {
    access |= IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
  }
  for (uint32_t i=0; i<path->attrs.buffer_count; i++) {
    TakyonBuffer *takyon_buffer = &path->attrs.buffers[i];
    RdmaBuffer *rdma_buffer = (RdmaBuffer *)takyon_buffer->private_data;
    // IMPORTANT: assuming buffers are zeroed at init
    rdma_buffer->mr = registerMemoryRegion(pd, takyon_buffer->addr, takyon_buffer->bytes, access, error_message, max_error_message_chars);
    if (rdma_buffer->mr == NULL) goto failed;
    if (path->attrs.verbosity & TAKYON_VERBOSITY_CREATE_DESTROY_MORE) printf("  Registered memory region: addr=0x%jx, bytes=%ju, lkey=%u, rkey=%u\n", (uint64_t)takyon_buffer->addr, takyon_buffer->bytes, rdma_buffer->mr->lkey, rdma_buffer->mr->rkey);
  }

  // Build the endpoint structure before posting recvs
  endpoint = calloc(1, sizeof(RdmaEndpoint));
  if (endpoint == NULL) goto failed;
  endpoint->protocol = (qp_type == IBV_QPT_RC) ? RDMA_PROTOCOL_RC : (qp_type == IBV_QPT_UC) ? RDMA_PROTOCOL_UC : RDMA_PROTOCOL_UD_UNICAST;
  endpoint->context = context;
  endpoint->send_comp_ch = send_comp_ch;
  endpoint->recv_comp_ch = recv_comp_ch;
  endpoint->send_cq = send_cq;
  endpoint->recv_cq = recv_cq;
  endpoint->pd = pd;
  endpoint->qp = qp;

  // Post recvs
  if (recv_request_count > 0) {
    if (!rdmaEndpointPostRecvs(path, endpoint, recv_request_count, recv_requests, error_message, max_error_message_chars)) goto failed;
  }

  // Get the local endpoint info that will be sent to the remote endpoint
  PortInfo local_port_info;
  if (!getLocalPortInfo(path, context, qp, rdma_port_id, max_pending_read_and_atomic_requests, unicast_local_qkey, app_options.gid_index, &local_port_info, error_message, max_error_message_chars)) goto failed;

  // Exchange QP and port info with the remote endpoint
  PortInfo remote_port_info;
  if (is_endpointA) {
    if (!socketSend(read_pipe_fd, local_port_info.socket_data, sizeof(local_port_info.socket_data), false, timeout_nano_seconds, &timed_out, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to send RDMA port info text: %s\n", error_message);
      goto failed;
    }
    if (!socketRecv(read_pipe_fd, remote_port_info.socket_data, sizeof(remote_port_info.socket_data), false, timeout_nano_seconds, &timed_out, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to recv remote RDMA port info text: %s\n", error_message);
      goto failed;
    }
  } else {
    if (!socketRecv(read_pipe_fd, remote_port_info.socket_data, sizeof(remote_port_info.socket_data), false, timeout_nano_seconds, &timed_out, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to recv remote RDMA port info text: %s\n", error_message);
      goto failed;
    }
    if (!socketSend(read_pipe_fd, local_port_info.socket_data, sizeof(local_port_info.socket_data), false, timeout_nano_seconds, &timed_out, error_message, MAX_ERROR_MESSAGE_CHARS)) {
      TAKYON_RECORD_ERROR(path->error_message, "Failed to send RDMA port info text: %s\n", error_message);
      goto failed;
    }
  }

  // Parse the remote port info
  int tokens = sscanf(remote_port_info.socket_data, "%x:%x:%x:%x:%x:%x:%s", &remote_port_info.lid, &remote_port_info.qpn, &remote_port_info.psn, &remote_port_info.mtu_mode, &remote_port_info.max_pending_read_and_atomic_requests, &remote_port_info.unicast_qkey, remote_port_info.gid_text);
  if (tokens != 7) {
    TAKYON_RECORD_ERROR(path->error_message, "Failed to parse the remote RDMA port info text\n");
    goto failed;
  }
  textToGid(remote_port_info.gid_text, &remote_port_info.gid);
  if (path->attrs.verbosity & TAKYON_VERBOSITY_CREATE_DESTROY_MORE) {
    printf("  Remote connection info: LID 0x%04x, QPN 0x%06x, PSN 0x%06x, MTU %s, MaxPendingReadAndAtomics %u, UnicastQKEY 0x%06x, GID '%s'\n",
           remote_port_info.lid, remote_port_info.qpn, remote_port_info.psn, mtuModeToText(remote_port_info.mtu_mode), remote_port_info.max_pending_read_and_atomic_requests, remote_port_info.unicast_qkey, remote_port_info.gid_text);
  }

  // Move the QP state to RTS (ready to send)
  enum ibv_mtu mtu_mode = (remote_port_info.mtu_mode < local_port_info.mtu_mode) ? remote_port_info.mtu_mode : local_port_info.mtu_mode; // Use the minimum
  if (app_options.mtu_bytes != 0) {
    if (app_options.mtu_bytes == 256) mtu_mode = IBV_MTU_256;
    else if (app_options.mtu_bytes == 512) mtu_mode = IBV_MTU_512;
    else if (app_options.mtu_bytes == 1024) mtu_mode = IBV_MTU_1024;
    else if (app_options.mtu_bytes == 2048) mtu_mode = IBV_MTU_2048;
    else if (app_options.mtu_bytes == 4096) mtu_mode = IBV_MTU_4096;
  }
  endpoint->mtu_bytes = mtuModeToBytes(mtu_mode);
  if (endpoint->mtu_bytes == 0) {
    TAKYON_RECORD_ERROR(path->error_message, "Could not determine MTU bytes\n");
    goto failed;
  }
  if (path->attrs.verbosity & TAKYON_VERBOSITY_CREATE_DESTROY_MORE) printf("  Will use %s between endpoints\n", mtuModeToText(mtu_mode));
  if (!moveQpStateToRTS(qp, pd, qp_type, is_UD_sender, rdma_port_id, &local_port_info, &remote_port_info, app_options, mtu_mode, error_message, MAX_ERROR_MESSAGE_CHARS, &unicast_sender_ah)) goto failed;
  if (qp_type == IBV_QPT_UD) {
    endpoint->unicast_sender_ah = unicast_sender_ah;
    endpoint->unicast_remote_qp_num = remote_port_info.qpn;
    endpoint->unicast_remote_qkey = remote_port_info.unicast_qkey;
  }

  // Ready to start transfering
  return endpoint;

 failed:
  if (unicast_sender_ah != NULL) ibv_destroy_ah(unicast_sender_ah);
  if (qp != NULL) ibv_destroy_qp(qp);
  if (send_cq != NULL) ibv_destroy_cq(send_cq);
  if (recv_cq != NULL) ibv_destroy_cq(recv_cq);
  for (uint32_t i=0; i<path->attrs.buffer_count; i++) {
    TakyonBuffer *takyon_buffer = &path->attrs.buffers[i];
    RdmaBuffer *rdma_buffer = (RdmaBuffer *)takyon_buffer->private_data;
    if (rdma_buffer->mr != NULL) ibv_dereg_mr(rdma_buffer->mr);
  }
  if (pd != NULL) ibv_dealloc_pd(pd);
  if (send_comp_ch != NULL) ibv_destroy_comp_channel(send_comp_ch);
  if (recv_comp_ch != NULL) ibv_destroy_comp_channel(recv_comp_ch);
  if (context != NULL) ibv_close_device(context);
  if (endpoint != NULL) free(endpoint);
  return NULL;
}

bool rdmaDestroyEndpoint(TakyonPath *path, RdmaEndpoint *endpoint, char *error_message, int max_error_message_chars) {
  // Ack events to avoid deadlocking the destructions of the QP
  if (endpoint->num_send_events_to_ack > 0) {
    ibv_ack_cq_events(endpoint->send_cq, endpoint->num_send_events_to_ack);
  }
  if (endpoint->num_recv_events_to_ack > 0) {
    ibv_ack_cq_events(endpoint->recv_cq, endpoint->num_recv_events_to_ack);
  }

  // Leave the multicast group
  if (endpoint->multicast_ah != NULL) {
    int rc = rdma_leave_multicast(endpoint->id, &endpoint->multicast_addr);
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
  if (endpoint->unicast_sender_ah != NULL) {
    int rc = ibv_destroy_ah(endpoint->unicast_sender_ah);
    if (rc != 0) {
      snprintf(error_message, max_error_message_chars, "Failed to destroy unicast sender address handle: errno=%d", errno);
      return false;
    }
  }
  if (endpoint->protocol == RDMA_PROTOCOL_UD_MULTICAST) {
    rdma_destroy_qp(endpoint->id);
  }
  if (endpoint->qp != NULL) {
    int rc = ibv_destroy_qp(endpoint->qp);
    if (rc != 0) {
      snprintf(error_message, max_error_message_chars, "Failed to destroy queue pair: errno=%d", errno);
      return false;
    }
  }
  if (endpoint->send_cq != NULL) {
    int rc = ibv_destroy_cq(endpoint->send_cq);
    if (rc != 0) {
      snprintf(error_message, max_error_message_chars, "Failed to destroy completion queue: errno=%d", errno);
      return false;
    }
  }
  if (endpoint->recv_cq != NULL) {
    int rc = ibv_destroy_cq(endpoint->recv_cq);
    if (rc != 0) {
      snprintf(error_message, max_error_message_chars, "Failed to destroy recv completion queue: errno=%d", errno);
      return false;
    }
  }
  for (uint32_t i=0; i<path->attrs.buffer_count; i++) {
    TakyonBuffer *takyon_buffer = &path->attrs.buffers[i];
    RdmaBuffer *rdma_buffer = (RdmaBuffer *)takyon_buffer->private_data;
    if (rdma_buffer->mr != NULL) {
      int rc = ibv_dereg_mr(rdma_buffer->mr);
      if (rc != 0) {
        snprintf(error_message, max_error_message_chars, "Failed to unregister memory region: errno=%d", errno);
        return false;
      }
    }
  }
  if (endpoint->pd != NULL) {
    int rc = ibv_dealloc_pd(endpoint->pd);
    if (rc != 0) {
      snprintf(error_message, max_error_message_chars, "Failed to dealloc protection domain: errno=%d", errno);
      return false;
    }
  }
  if (endpoint->send_comp_ch != NULL) {
    int rc = ibv_destroy_comp_channel(endpoint->send_comp_ch);
    if (rc != 0) {
      snprintf(error_message, max_error_message_chars, "Failed to destroy completion channel: errno=%d", errno);
      return false;
    }
  }
  if (endpoint->recv_comp_ch != NULL) {
    int rc = ibv_destroy_comp_channel(endpoint->recv_comp_ch);
    if (rc != 0) {
      snprintf(error_message, max_error_message_chars, "Failed to destroy recv completion channel: errno=%d", errno);
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
  if (endpoint->context != NULL) {
    int rc = ibv_close_device(endpoint->context);
    if (rc != 0) {
      snprintf(error_message, max_error_message_chars, "Failed to destroy RDMA context/device handle: errno=%d", errno);
      return false;
    }
  }
  free(endpoint);

  return true;
}

static bool eventDrivenCompletionWait(RdmaEndpoint *endpoint, struct ibv_comp_channel *comp_ch, int read_pipe_fd, unsigned int *num_events_to_ack_inout, double timeout_seconds, bool *timed_out_ret, char *error_message, int max_error_message_chars) {
  // Verbs does not have an API to do a timed wait, so need to use an underlying socket in the completion channel to do it
  struct pollfd poll_fds[2];
 retry:
  poll_fds[0].fd      = comp_ch->fd; // RDMA's completion channel socket
  poll_fds[0].events  = POLLIN;      // Requested events: POLLIN means there is data to read
  poll_fds[0].revents = 0;           // Returned events that will be filled in after the call to poll()
  poll_fds[1].fd      = read_pipe_fd; // The Takyon connection socket (to detect disconnects and then do a shut down barrier)
  poll_fds[1].events  = POLLIN;    // Requested events: POLLIN means there is activity
  poll_fds[1].revents = 0;         // Returned events that will be filled in after the call to poll()
  nfds_t fd_count = 1;
  if (read_pipe_fd > 0) {
    fd_count = 2;
  }
  int timeout_ms = (timeout_seconds < 0) ? -1 : (int)(timeout_seconds * 1000);
  if (endpoint->connection_broken) {
    snprintf(error_message, max_error_message_chars, "Looks like the remote endpoint may have disconnected");
    return false;
  }
  int count = poll(poll_fds, fd_count, timeout_ms);
  if (count == 0) {
    // Timed out
    *timed_out_ret = true;
    return true;
  }
  if (count < 0) {
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
  if (poll_fds[1].revents != 0) {
    snprintf(error_message, max_error_message_chars, "Looks like the remote endpoint may have disconnected: revents=%d, POLLIN=%d, POLLOUT=%d", poll_fds[1].revents, POLLIN, POLLOUT);
    return false;
  }
  if (poll_fds[0].revents != POLLIN) {
    snprintf(error_message, max_error_message_chars, "poll() returned but did not get read 'POLLIN' activity");
    return false;
  }

  // Block waiting for a completion
  //   NOTE: Because of using poll() above, at this point there should be a completion if the socket poll() did its job, without actually need to wait any further
  struct ibv_cq *cq; // Since the completion channel only has one CQ, can ignore the returned values
  void *cq_app_data;
  int rc = ibv_get_cq_event(comp_ch, &cq, &cq_app_data);
  if (rc != 0) {
    snprintf(error_message, max_error_message_chars, "ibv_get_cq_event() failed: errno=%d", errno);
    return false;
  }

  // Need to ack the completion event counter if at the limit
  unsigned int num_events_to_ack = *num_events_to_ack_inout;
  num_events_to_ack++;
  if (num_events_to_ack == UINT_MAX) {
    ibv_ack_cq_events(cq, num_events_to_ack);
    num_events_to_ack = 0;
  }
  *num_events_to_ack_inout = num_events_to_ack;

  // Need to re-arm the CQ so completion events can be detected
  if (!armCompletionQueue(cq, error_message, max_error_message_chars)) return false;

  return true;
}

static const char *wcErrorToText(enum ibv_wc_status status) {
  switch (status) {
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
}

#ifdef EXTRA_ERROR_CHECKING
static const char *wcOpcodeToText(enum ibv_wc_opcode opcode_val) {
  switch (opcode_val) {
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
}
#endif

static bool waitForCompletion(bool is_send, RdmaEndpoint *endpoint, uint64_t expected_wr_id, enum ibv_wc_opcode expected_opcode, int read_pipe_fd, bool use_polling_completion, uint32_t usec_sleep_between_poll_attempts, double timeout_seconds, bool *timed_out_ret, char *error_message, int max_error_message_chars, uint64_t *bytes_received_ret, uint32_t *piggyback_message_ret) {
  (void)expected_opcode; // Quiet compiler
  bool got_start_time = false;
  double start_time = 0;
  struct ibv_wc wc;
  int completion_count;
  struct ibv_cq *cq = is_send ? endpoint->send_cq : endpoint->recv_cq;

 retry:
  // Just try to get one completion, even if many are ready
  completion_count = ibv_poll_cq(cq, 1, &wc);
  if (completion_count == -1) {
    snprintf(error_message, max_error_message_chars, "ibv_poll_cq() failed: errno=%d", errno);
    return false;
  }
  if (endpoint->connection_broken) {
    snprintf(error_message, max_error_message_chars, "Looks like the remote side disconnected");
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
      bool timed_out = false;
      struct ibv_comp_channel *comp_ch = is_send ? endpoint->send_comp_ch : endpoint->recv_comp_ch;
      unsigned int *num_events_to_ack_inout = is_send ? &endpoint->num_send_events_to_ack : &endpoint->num_recv_events_to_ack;
      if (!eventDrivenCompletionWait(endpoint, comp_ch, read_pipe_fd, num_events_to_ack_inout, timeout_seconds, &timed_out, error_message, max_error_message_chars)) return false;
      if (timed_out) {
	// Timed out
	*timed_out_ret = true;
	return true;
      }
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
#ifdef EXTRA_ERROR_CHECKING
  if (is_send) {
    if (wc.opcode != expected_opcode) {
      snprintf(error_message, max_error_message_chars, "Work completion was for '%s' but expected '%s'", wcOpcodeToText(wc.opcode), wcOpcodeToText(expected_opcode));
      return false;
    }
  } else {
    if (wc.opcode != expected_opcode) {
      snprintf(error_message, max_error_message_chars, "Work completion was for '%s' but expected '%s'", wcOpcodeToText(wc.opcode), wcOpcodeToText(expected_opcode));
      return false;
    }
    if (!(wc.wc_flags & IBV_WC_WITH_IMM)) {
      snprintf(error_message, max_error_message_chars, "Work completion did not contain an IMM");
      return false;
    }
  }
#endif

  // Success: return info
  if (!is_send) {
    *bytes_received_ret = wc.byte_len;
    *piggyback_message_ret = ntohl(wc.imm_data);
  }

  return true;
}

bool rdmaEndpointStartSend(TakyonPath *path, RdmaEndpoint *endpoint, enum ibv_wr_opcode transfer_mode, uint64_t transfer_id, uint32_t sub_buffer_count, TakyonSubBuffer *sub_buffers, struct ibv_sge *sge_list, uint64_t *atomics, uint64_t remote_addr, uint32_t rkey, uint32_t piggyback_message, bool invoke_fence, bool use_is_sent_notification, char *error_message, int max_error_message_chars) {
  struct ibv_send_wr send_wr;

  // Fill in message to be sent
  send_wr.next = NULL;
  send_wr.wr_id = transfer_id;
  send_wr.num_sge = sub_buffer_count;
  send_wr.sg_list = sge_list;
  send_wr.opcode = transfer_mode;
  if (transfer_mode == IBV_WR_SEND_WITH_IMM) {
    send_wr.imm_data = htonl(piggyback_message);
  }

  // SGEs
  uint32_t total_bytes = 0;
  for (uint32_t j=0; j<sub_buffer_count; j++) {
    TakyonSubBuffer *sub_buffer = &sub_buffers[j];
    TakyonBuffer *buffer = &path->attrs.buffers[sub_buffer->buffer_index];
    RdmaBuffer *rdma_buffer = (RdmaBuffer *)buffer->private_data;
    struct ibv_sge *sge = &send_wr.sg_list[j];
    sge->addr = (uint64_t)buffer->addr + sub_buffer->offset;
    sge->length = sub_buffer->bytes;
    sge->lkey = rdma_buffer->mr->lkey;
    total_bytes += sub_buffer->bytes;
#ifdef EXTRA_ERROR_CHECKING
    if (path->attrs.verbosity & TAKYON_VERBOSITY_TRANSFERS_MORE) printf("    SGE[%d]: addr=0x%jx, bytes=%u, lkey=%u\n", j, (uint64_t)sge->addr, sge->length, sge->lkey);
#endif
  }
  if (endpoint->protocol == RDMA_PROTOCOL_UD_MULTICAST || endpoint->protocol == RDMA_PROTOCOL_UD_UNICAST) {
    if (total_bytes > endpoint->mtu_bytes) {
      snprintf(error_message, max_error_message_chars, "Message bytes=%u is larger than MTU bytes=%u", total_bytes, endpoint->mtu_bytes);
      return false;
    }
  }

#ifdef EXTRA_ERROR_CHECKING
  if (transfer_mode == IBV_WR_SEND_WITH_IMM) {
    if (path->attrs.verbosity & TAKYON_VERBOSITY_TRANSFERS_MORE) printf("  Posting send: IMM=%u, nSGEs=%d\n", piggyback_message, send_wr.num_sge);
  } else if (transfer_mode == IBV_WR_RDMA_WRITE || transfer_mode == IBV_WR_RDMA_READ) {
    if (path->attrs.verbosity & TAKYON_VERBOSITY_TRANSFERS_MORE) printf("  Posting %s: nSGEs=%d\n", (transfer_mode == IBV_WR_RDMA_WRITE) ? "write" : "read", send_wr.num_sge);
  } else if (transfer_mode == IBV_WR_ATOMIC_CMP_AND_SWP) {
    uint64_t *value_ptr = (uint64_t *)send_wr.sg_list[0].addr;
    if (path->attrs.verbosity & TAKYON_VERBOSITY_TRANSFERS_MORE) printf("  Posting atomic compare and swap: comapre=%ju, swap=%ju\n", value_ptr[1], value_ptr[2]);
  } else if (transfer_mode == IBV_WR_ATOMIC_FETCH_AND_ADD) {
    uint64_t *value_ptr = (uint64_t *)send_wr.sg_list[0].addr;
    if (path->attrs.verbosity & TAKYON_VERBOSITY_TRANSFERS_MORE) printf("  Posting atomic fetch and add: value=%ju\n", value_ptr[1]);
  }
#endif

  // Protocal specific stuff
  if (endpoint->protocol == RDMA_PROTOCOL_UD_MULTICAST) {
#ifdef EXTRA_ERROR_CHECKING
    if (path->attrs.verbosity & TAKYON_VERBOSITY_TRANSFERS_MORE) printf("  Send to multicast addr: qpn=%u, qkey=%u\n", endpoint->multicast_qp_num, endpoint->multicast_qkey);
#endif
    send_wr.wr.ud.ah = endpoint->multicast_ah;
    send_wr.wr.ud.remote_qpn = endpoint->multicast_qp_num;
    send_wr.wr.ud.remote_qkey = endpoint->multicast_qkey;
  } else if (endpoint->protocol == RDMA_PROTOCOL_UD_UNICAST) {
#ifdef EXTRA_ERROR_CHECKING
    if (path->attrs.verbosity & TAKYON_VERBOSITY_TRANSFERS_MORE) printf("  Send to unicast addr: qpn=%u, qkey=%u\n", endpoint->unicast_remote_qp_num, endpoint->unicast_remote_qkey);
#endif
    send_wr.wr.ud.ah = endpoint->unicast_sender_ah;
    send_wr.wr.ud.remote_qpn = endpoint->unicast_remote_qp_num;
    send_wr.wr.ud.remote_qkey = endpoint->unicast_remote_qkey;
  } else if (endpoint->protocol == RDMA_PROTOCOL_UC || endpoint->protocol == RDMA_PROTOCOL_RC) {
    if (transfer_mode == IBV_WR_RDMA_WRITE || transfer_mode == IBV_WR_RDMA_READ) {
      // Read, write
#ifdef EXTRA_ERROR_CHECKING
      if (path->attrs.verbosity & TAKYON_VERBOSITY_TRANSFERS_MORE) printf("  RDMA %s: rkey=%u, raddr=0x%jx\n", (transfer_mode == IBV_WR_RDMA_WRITE) ? "write" : "read", rkey, remote_addr);
#endif
      send_wr.wr.rdma.rkey = rkey;
      send_wr.wr.rdma.remote_addr = remote_addr;
    } else if (transfer_mode == IBV_WR_ATOMIC_CMP_AND_SWP || transfer_mode == IBV_WR_ATOMIC_FETCH_AND_ADD) {
      // Atomics
#ifdef EXTRA_ERROR_CHECKING
      if (path->attrs.verbosity & TAKYON_VERBOSITY_TRANSFERS_MORE) printf("  RDMA atomic %s: rkey=%u, raddr=0x%jx\n", (transfer_mode == IBV_WR_ATOMIC_CMP_AND_SWP) ? "CAS" : "add", rkey, remote_addr);
#endif
      send_wr.wr.atomic.compare_add = atomics[0];
      send_wr.wr.atomic.swap = atomics[1];
      send_wr.wr.atomic.rkey = rkey;
      send_wr.wr.atomic.remote_addr = remote_addr;
    }
  }

  // Signaling
  send_wr.send_flags = 0;
  if (use_is_sent_notification) {
#ifdef EXTRA_ERROR_CHECKING
    if (path->attrs.verbosity & TAKYON_VERBOSITY_TRANSFERS_MORE) printf("  Send signaled\n");
#endif
    send_wr.send_flags |= IBV_SEND_SIGNALED; // Can only do this for the QP's max number of pending send requests before a signal is needed to avoid overrunning the request buffer
  }
  if (invoke_fence && endpoint->protocol == RDMA_PROTOCOL_RC) {
    // This won't block submiting the send request, but will eventually block the DMA engine until all preceding transfer are complete
    // This is typically only needed if a 'read' or 'atomic' operation is done (changes local memory) just before sending the results of either of those two operations
    // Only supported with RC
    send_wr.send_flags |= IBV_SEND_FENCE;
  }

  // Start the send transfer
  struct ibv_send_wr *bad_wr;
  struct ibv_qp *qp = (endpoint->protocol == RDMA_PROTOCOL_UD_MULTICAST) ? endpoint->id->qp : endpoint->qp;
  int rc = ibv_post_send(qp, &send_wr, &bad_wr);
  if (rc != 0) {
    snprintf(error_message, max_error_message_chars, "Failed to start send: errno=%d", errno);
    return false;
  }
  return true;
}

bool rdmaEndpointIsRecved(RdmaEndpoint *endpoint, uint64_t expected_transfer_id, int read_pipe_fd, bool use_polling_completion, uint32_t usec_sleep_between_poll_attempts, double timeout_seconds, bool *timed_out_ret, char *error_message, int max_error_message_chars, uint64_t *bytes_received_ret, uint32_t *piggyback_message_ret) {
  bool is_send = false;
  enum ibv_wc_opcode expected_opcode = IBV_WC_RECV;
  return waitForCompletion(is_send, endpoint, expected_transfer_id, expected_opcode, read_pipe_fd, use_polling_completion, usec_sleep_between_poll_attempts, timeout_seconds, timed_out_ret, error_message, max_error_message_chars, bytes_received_ret, piggyback_message_ret);
}

bool rdmaEndpointIsSent(RdmaEndpoint *endpoint, uint64_t expected_transfer_id, enum ibv_wc_opcode expected_opcode, int read_pipe_fd, bool use_polling_completion, uint32_t usec_sleep_between_poll_attempts, double timeout_seconds, bool *timed_out_ret, char *error_message, int max_error_message_chars) {
  bool is_send = true;
  return waitForCompletion(is_send, endpoint, expected_transfer_id, expected_opcode, read_pipe_fd, use_polling_completion, usec_sleep_between_poll_attempts, timeout_seconds, timed_out_ret, error_message, max_error_message_chars, NULL, NULL);
}
