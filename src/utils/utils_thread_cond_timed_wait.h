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

#ifndef _utils_thread_cond_timed_wait_h_
#define _utils_thread_cond_timed_wait_h_

#include <stdbool.h>
#include <stdint.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C"
{
#endif

extern bool threadCondWait(pthread_mutex_t *mutex, pthread_cond_t *cond_var, int64_t timeout_ns, bool *timed_out_ret, char *error_message, int max_error_message_chars);

#ifdef __cplusplus
}
#endif

#endif