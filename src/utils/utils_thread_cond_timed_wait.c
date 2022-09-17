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

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
  #include <sys/timeb.h>
#else
  #include <sys/time.h>
#endif
#include <stdio.h>
#include <errno.h>
#include "utils_thread_cond_timed_wait.h"
#include "utils_time.h"

static struct timespec getCurrentAbsoluteTime(int64_t timeout_ns) {
  struct timespec future_time;
#define NANOSECONDS_PER_SECOND 1000000000
  unsigned timeout_secs = (unsigned)(timeout_ns / NANOSECONDS_PER_SECOND);
  unsigned timeout_nsecs = (unsigned)(timeout_ns % NANOSECONDS_PER_SECOND);

#ifdef _WIN32
  struct _timeb curr_sys_time;
  // Get the current time since 1970
  _ftime(&curr_sys_time);
  future_time.tv_sec = timeout_secs + (unsigned)curr_sys_time.time;
  future_time.tv_nsec = timeout_nsecs + curr_sys_time.millitm * 1000000;
#else
  struct timeval ctp;
  gettimeofday(&ctp, NULL);
  future_time.tv_sec  = (unsigned)(timeout_secs + ctp.tv_sec);
  future_time.tv_nsec = (unsigned)(timeout_nsecs + ctp.tv_usec*1000);
#endif

  // Do some overlap correction if needed
  while (future_time.tv_nsec >= NANOSECONDS_PER_SECOND) {
    future_time.tv_nsec -= NANOSECONDS_PER_SECOND;
    future_time.tv_sec++;
  }

  return future_time;
}

bool threadCondWait(pthread_mutex_t *mutex, pthread_cond_t *cond_var, int64_t timeout_ns, bool *timed_out_ret, char *error_message, int max_error_message_chars) {
  int rc;

  if (timed_out_ret != NULL) *timed_out_ret = false;

  if (timeout_ns < 0) {
#ifdef __APPLE__
    int retry_count = 0;
  retry:
#endif
    rc = pthread_cond_wait(cond_var, mutex);
    if (rc != 0) {
#ifdef __APPLE__
      if ((errno == ETIMEDOUT) || (errno == ENOENT)) {
        if (retry_count < 10) {
          int64_t microseconds = 1;
          clockSleepYield(microseconds);
          retry_count++;
          goto retry;
        }
      }
#endif
      snprintf(error_message, max_error_message_chars, "Failed to call pthread_cond_wait(), rc=%d, errno=%d", rc, errno);
      return false;
    }
    return true;

  } else {
    struct timespec future_time = getCurrentAbsoluteTime(timeout_ns);
    // Wait until signaled or timeout
    rc = pthread_cond_timedwait(cond_var, mutex, &future_time);
    if (rc == 0) {
      // Returned succesfully
      return true;
    } else if (rc == ETIMEDOUT) {
      if (timed_out_ret != NULL) *timed_out_ret = true;
      return true;
    } else {
      snprintf(error_message, max_error_message_chars, "Failed to call pthread_cond_timedwait()");
      return false;
    }
  }
}
