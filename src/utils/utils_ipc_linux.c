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

#include "utils_ipc.h"
#include "takyon.h" // Just need TAKYON_MAX_BUFFER_NAME_CHARS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/syscall.h>
#include <sys/file.h>

#define MEMORY_MAP_PREFIX   "/"  // Prefix required by Posix

struct _MmapHandle {
  bool is_creator;
  void *mapped_addr;
  uint64_t total_bytes;
  char map_name[TAKYON_MAX_BUFFER_NAME_CHARS];
};

bool mmapAlloc(const char *map_name, uint64_t bytes, void **addr_ret, void **mmap_handle_ret, char *error_message, int max_error_message_chars) {
  char full_map_name[TAKYON_MAX_BUFFER_NAME_CHARS];
  int mapping_fd = 0;
  void *mapped_addr = NULL;
  unsigned int max_name_length = TAKYON_MAX_BUFFER_NAME_CHARS - (unsigned int)strlen(MEMORY_MAP_PREFIX) - 1;
  int rc;
  struct _MmapHandle *mmap_handle = NULL;

  if (map_name == NULL) {
    snprintf(error_message, max_error_message_chars, "map_name is NULL");
    return false;
  }
  if (strlen(map_name) > max_name_length) {
    snprintf(error_message, max_error_message_chars, "map_name '%s' has too many characters. Limit is %d", map_name, max_name_length);
    return false;
  }
  if (addr_ret == NULL) {
    snprintf(error_message, max_error_message_chars, "addr_ret is NULL");
    return false;
  }

  /* Create full map name */
  snprintf(full_map_name, TAKYON_MAX_BUFFER_NAME_CHARS, "%s%s", MEMORY_MAP_PREFIX, map_name);

  /* Verify mapping not already in use */
  mapping_fd = shm_open(full_map_name, O_RDWR, S_IRUSR | S_IWUSR);
  if (mapping_fd != -1) {
    /* This must be lingering around from a previous run */
    /* Close the handle */
    close(mapping_fd);
    /* Unlink the old */
    rc = shm_unlink(full_map_name);
    if (rc == -1) {
      snprintf(error_message, max_error_message_chars, "Could not unlink old memory map '%s'", full_map_name);
      return false;
    }
    //printf("Shared mem '%s' already exists, and was safely unlinked.\n", full_map_name);
  }

  /* Create memory map handle */
  mapping_fd = shm_open(full_map_name, O_CREAT | O_RDWR | O_EXCL, S_IRUSR | S_IWUSR);
  if (mapping_fd == -1) {
    snprintf(error_message, max_error_message_chars, "could not create shared memory '%s' (errno=%d)", full_map_name, errno);
    return false;
  }

  /* Set the size of the shared memory */
  rc = ftruncate(mapping_fd, bytes);
  if (rc == -1) {
    close(mapping_fd);
    snprintf(error_message, max_error_message_chars, "could not set shared memory size '%s' to %lu bytes", full_map_name, bytes);
    return false;
  }

  /* Get a handle to mapped memory */
  mapped_addr = mmap(NULL/*addr*/, bytes, PROT_READ | PROT_WRITE, MAP_SHARED /* | MAP_HASSEMAPHORE */, mapping_fd, 0/*offset*/);
  if (mapped_addr == (void *)-1)  {
    snprintf(error_message, max_error_message_chars, "could not obtain mapped address for '%s': errno=%d", full_map_name, errno);
    close(mapping_fd);
    return false;
  }

  /* Close the file descriptor. This will not un map the the shared memory */
  close(mapping_fd);

  *addr_ret = (void *)mapped_addr;

  /* Pin memory so it doesn't get swapped out */
  mlock(*addr_ret, bytes);

  /* IMPORTANT: Don't set any of the memory values here since there might be a race condition. The memory values needed to be coordinated properly by the processes ussing it. */

  /* Create handle and store info */
  mmap_handle = (struct _MmapHandle *)calloc(1, sizeof(struct _MmapHandle));
  if (mmap_handle == NULL) {
    snprintf(error_message, max_error_message_chars, "Failed to allocate the shared mem handle. Out of memory.");
    munmap(mapped_addr, bytes);
    return false;
  }

  /* Store the map info */
  strncpy(mmap_handle->map_name, full_map_name, TAKYON_MAX_BUFFER_NAME_CHARS);
  mmap_handle->is_creator = true;
  mmap_handle->mapped_addr = mapped_addr;
  mmap_handle->total_bytes = bytes;

  /* Set returned handle */
  *mmap_handle_ret = mmap_handle;

  return true;
}

bool mmapGet(const char *map_name, uint64_t bytes, void **addr_ret, bool *got_it_ret, void **mmap_handle_ret, char *error_message, int max_error_message_chars) {
  int mapping_fd = 0;
  void *mapped_addr = NULL;
  unsigned int max_name_length = TAKYON_MAX_BUFFER_NAME_CHARS - (unsigned int)strlen(MEMORY_MAP_PREFIX) - 1;
  char full_map_name[TAKYON_MAX_BUFFER_NAME_CHARS];
  struct _MmapHandle *mmap_handle = NULL;

  if (map_name == NULL) {
    snprintf(error_message, max_error_message_chars, "map_name is NULL");
    return false;
  }
  if (strlen(map_name) > max_name_length) {
    snprintf(error_message, max_error_message_chars, "map_name '%s' has too many characters. Limit is %d", map_name, max_name_length);
    return false;
  }
  if (addr_ret == NULL) {
    snprintf(error_message, max_error_message_chars, "addr_ret is NULL");
    return false;
  }
  if (got_it_ret == NULL) {
    snprintf(error_message, max_error_message_chars, "got_it_ret is NULL");
    return false;
  }

  /* Create full map name */
  snprintf(full_map_name, TAKYON_MAX_BUFFER_NAME_CHARS, "%s%s", MEMORY_MAP_PREFIX, map_name);

  /* Obtain the existing mapped file */
  mapping_fd = shm_open(full_map_name, O_RDWR, S_IRUSR | S_IWUSR);
  if (mapping_fd == -1) {
    /* The memory does not exists, but this is not an error. The caller can just see if the return address is null to know it doesn't exist. */
    *addr_ret = NULL;
    *got_it_ret = false;
    return true;
  }

  /* Get a handle to remote mapped memory */
  mapped_addr = mmap(NULL/*addr*/, bytes, PROT_READ | PROT_WRITE, MAP_SHARED /* | MAP_HASSEMAPHORE */, mapping_fd, 0/*offset*/);
  if (mapped_addr == (void *)-1) {
    close(mapping_fd);
    snprintf(error_message, max_error_message_chars, "Could not obtain address mapping for '%s'", map_name);
    return false;
  }
  *addr_ret = (void *)mapped_addr;

  /* Close the file descriptor. This will not un map the the shared memory */
  close(mapping_fd);

  /* Create handle and store info */
  mmap_handle = (struct _MmapHandle *)calloc(1, sizeof(struct _MmapHandle));
  if (mmap_handle == NULL) {
    snprintf(error_message, max_error_message_chars, "Failed to allocate the shared mem handle. Out of memory.");
    return false;
  }
  strncpy(mmap_handle->map_name, full_map_name, TAKYON_MAX_BUFFER_NAME_CHARS);
  mmap_handle->is_creator = false;
  mmap_handle->mapped_addr = mapped_addr;
  mmap_handle->total_bytes = bytes;

  /* Set returned handle */
  *got_it_ret = true;
  *mmap_handle_ret = mmap_handle;

  return true;
}

bool mmapFree(void *mmap_handle_opaque, char *error_message, int max_error_message_chars) {
  struct _MmapHandle *mmap_handle = (struct _MmapHandle *)mmap_handle_opaque;

  if (mmap_handle == NULL) {
    snprintf(error_message, max_error_message_chars, "The shared mem is NULL");
    return false;
  }

  /* Unmap memory address */
  int rc = munmap(mmap_handle->mapped_addr, mmap_handle->total_bytes);
  if (rc == -1) {
    snprintf(error_message, max_error_message_chars, "Could not un map memory map '%s'", mmap_handle->map_name);
    return false;
  }

  /* Delete memory map */
  if (mmap_handle->is_creator) {
    /* Unpin memory */
    munlock(mmap_handle->mapped_addr, mmap_handle->total_bytes);
    /* Unlink */
    rc = shm_unlink(mmap_handle->map_name);
    if (rc == -1) {
      snprintf(error_message, max_error_message_chars, "Could not unlink memory map '%s', errno=%d", mmap_handle->map_name, errno);
      return false;
    }
  }

  /* Free the handle */
  mmap_handle->map_name[0] = '\0';
  mmap_handle->mapped_addr = NULL;
  mmap_handle->is_creator = false;
  mmap_handle->total_bytes = 0;
  free(mmap_handle);

  return true;
}
