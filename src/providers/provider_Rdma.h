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

#ifndef _provider_Rdma_h_
#define _provider_Rdma_h_

#include "takyon.h"

#ifdef __cplusplus
extern "C"
{
#endif

extern bool rdmaCreate(TakyonPath *path, uint32_t post_recv_count, TakyonRecvRequest *recv_requests, double timeout_seconds);
extern bool rdmaDestroy(TakyonPath *path, double timeout_seconds);
extern bool rdmaOneSided(TakyonPath *path, TakyonOneSidedRequest *request, double timeout_seconds, bool *timed_out_ret);
extern bool rdmaIsOneSidedDone(TakyonPath *path, TakyonOneSidedRequest *request, double timeout_seconds, bool *timed_out_ret);
extern bool rdmaSend(TakyonPath *path, TakyonSendRequest *request, uint32_t piggyback_message, double timeout_seconds, bool *timed_out_ret);
extern bool rdmaIsSent(TakyonPath *path, TakyonSendRequest *request, double timeout_seconds, bool *timed_out_ret);
extern bool rdmaPostRecvs(TakyonPath *path, uint32_t request_count, TakyonRecvRequest *requests);
extern bool rdmaIsRecved(TakyonPath *path, TakyonRecvRequest *request, double timeout_seconds, bool *timed_out_ret, uint64_t *bytes_received_ret, uint32_t *piggyback_message_ret);

#ifdef __cplusplus
}
#endif

#endif
