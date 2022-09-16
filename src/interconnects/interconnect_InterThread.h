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

#ifndef _interconnect_InterThread_h_
#define _interconnect_InterThread_h_

#include "takyon.h"

#ifdef __cplusplus
extern "C"
{
#endif

extern bool interThreadCreate(TakyonPath *path, uint32_t post_recv_count, TakyonRecvRequest *recv_requests, double timeout_seconds);
extern bool interThreadDestroy(TakyonPath *path, double timeout_seconds);
extern bool interThreadOneSided(TakyonPath *path, TakyonOneSidedRequest *request, double timeout_seconds, bool *timed_out_ret);
extern bool interThreadSend(TakyonPath *path, TakyonSendRequest *request, uint32_t piggy_back_message, double timeout_seconds, bool *timed_out_ret);
extern bool interThreadPostRecvs(TakyonPath *path, uint32_t request_count, TakyonRecvRequest *requests);
extern bool interThreadIsRecved(TakyonPath *path, TakyonRecvRequest *request, double timeout_seconds, bool *timed_out_ret, uint64_t *bytes_received_ret, uint32_t *piggy_back_message_ret);

#ifdef __cplusplus
}
#endif

#endif
