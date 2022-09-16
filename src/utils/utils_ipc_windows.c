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

#define MEMORY_MAP_PREFIX "_" /* This follows the Windows way: no prefix characters really needed */

struct _MmapHandle {
  HANDLE mapping_handle; // Does double duty as 'is_creator'
  LPCTSTR mapped_addr;
  uint64_t total_bytes;
  char map_name[TAKYON_MAX_BUFFER_NAME_CHARS];
};

bool mmapAlloc(const char *map_name, uint64_t bytes, void **addr_ret, void **mmap_handle_ret, char *error_message, int max_error_message_chars) {
  char full_map_name[TAKYON_MAX_BUFFER_NAME_CHARS];
  HANDLE mapping_handle = NULL;
  LPCTSTR mapped_addr = NULL;
  size_t max_name_length = TAKYON_MAX_BUFFER_NAME_CHARS - strlen(MEMORY_MAP_PREFIX) - 1;
  struct _MmapHandle *mmap_handle = NULL;

  if (map_name == NULL) {
    snprintf(error_message, max_error_message_chars, "map_name is NULL");
    return false;
  }
  if (strlen(map_name) > max_name_length) {
    snprintf(error_message, max_error_message_chars, "map_name '%s' has too many characters. Limit is %lld", map_name, max_name_length);
    return false;
  }
  if (addr_ret == NULL) {
    snprintf(error_message, max_error_message_chars, "addr_ret is NULL");
    return false;
  }

  // Create full map name
  snprintf(full_map_name, TAKYON_MAX_BUFFER_NAME_CHARS, "%s%s", MEMORY_MAP_PREFIX, map_name);

  /* Verify mapping not already in use */
  mapping_handle = OpenFileMapping(FILE_MAP_ALL_ACCESS,   /* read/write access */
                                   FALSE,                 /* do not inherit the name */
                                   (TCHAR *)full_map_name);
  if (mapping_handle != NULL) {
    // This was around from a previous run, need to remove it
    // Close the handle. This should hopefully remove the underlying file
    CloseHandle(mapping_handle);
    //printf("Shared mem '%s' already exists, and was safely unlinked.\n", full_map_name);
  }

  /* Create memory map handle */
  mapping_handle = CreateFileMapping(INVALID_HANDLE_VALUE,    /* use paging file */
                                     NULL,                    /* default security */
                                     PAGE_READWRITE,          /* read/write access */
                                     0,                       /* max. object size */
                                     (DWORD)bytes,            /* buffer size */
                                     (TCHAR *)full_map_name);
  if (mapping_handle == NULL) {
    LPVOID lpMsgBuf;
    FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                  NULL,
                  GetLastError(),
                  MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), // Default language
                  (LPTSTR) &lpMsgBuf,
                  0,
                  NULL);
    snprintf(error_message, max_error_message_chars, "could not create shared memory '%s'. Error: %s", map_name, (char *)lpMsgBuf);
    LocalFree(lpMsgBuf);
    return false;
  }

  /* Get a handle to mapped memory */
  mapped_addr = (LPTSTR)MapViewOfFile(mapping_handle, FILE_MAP_ALL_ACCESS, 0, 0, bytes);
  if (mapped_addr == NULL) {
    CloseHandle(mapping_handle);
    snprintf(error_message, max_error_message_chars, "could not obtain mapped address for '%s'", map_name);
    return false;
  }

  /* IMPORTANT: Don't set any of the memory values here since there might be a race condition. The memory values needed to be coordinated properly by the processes ussing it. */
  *addr_ret = (void *)mapped_addr;
  /* NOTE: Not pinning memory like with Linux, but may not be an issue */

  /* Create handle and store info */
  mmap_handle = (struct _MmapHandle *)calloc(1, sizeof(struct _MmapHandle));
  if (mmap_handle == NULL) {
    UnmapViewOfFile(mapped_addr);
    CloseHandle(mapping_handle);
    snprintf(error_message, max_error_message_chars, "Failed to allocate the mmap handle. Out of memory.");
    return false;
  }

  /* Store the map info */
  strncpy(mmap_handle->map_name, full_map_name, TAKYON_MAX_BUFFER_NAME_CHARS);
  mmap_handle->mapping_handle = mapping_handle;
  mmap_handle->mapped_addr = mapped_addr;
  mmap_handle->total_bytes = bytes;

#ifdef DEBUG_MESSAGE
  printf("CREATED: '%s' %d bytes\n", full_map_name, bytes);
#endif

  /* Set returned handle */
  *mmap_handle_ret = mmap_handle;

  return true;
}

bool mmapGet(const char *map_name, uint64_t bytes, void **addr_ret, bool *got_it_ret, void **mmap_handle_ret, char *error_message, int max_error_message_chars) {
  HANDLE mapping_handle = NULL;
  LPCTSTR mapped_addr = NULL;
  size_t max_name_length = TAKYON_MAX_BUFFER_NAME_CHARS - strlen(MEMORY_MAP_PREFIX) - 1;
  char full_map_name[TAKYON_MAX_BUFFER_NAME_CHARS];
  struct _MmapHandle *mmap_handle = NULL;

  if (map_name == NULL) {
    snprintf(error_message, max_error_message_chars, "map_name is NULL");
    return false;
  }
  if (strlen(map_name) > max_name_length) {
    snprintf(error_message, max_error_message_chars, "map_name '%s' has too many characters. Limit is %lld", map_name, max_name_length);
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

  /* Verify mapping not already in use */
  mapping_handle = OpenFileMapping(FILE_MAP_ALL_ACCESS,   /* read/write access */
                                   FALSE,                 /* do not inherit the name */
                                   (TCHAR *)full_map_name);
  if (mapping_handle == NULL) {
    /* The memory does not exists, but this is not an error. The caller can just see if the return address is null to know it doesn't exist. */
    *addr_ret = NULL;
    *got_it_ret = 0;
    return true;
  }

  /* Get a handle to remote mapped memory */
  mapped_addr = (LPTSTR)MapViewOfFile(mapping_handle, FILE_MAP_ALL_ACCESS, 0, 0, bytes);
  if (mapped_addr == NULL) {
    CloseHandle(mapping_handle);
    snprintf(error_message, max_error_message_chars, "Could not obtain address mapping for '%s'", map_name);
    return false;
  }
  *addr_ret = (void *)mapped_addr;

  /* Close the file descriptor. This will not un map the the shared memory */
  CloseHandle(mapping_handle);

  /* Create handle and store info */
  mmap_handle = (struct _MmapHandle *)calloc(1, sizeof(struct _MmapHandle));
  if (mmap_handle == NULL) {
    snprintf(error_message, max_error_message_chars, "Failed to allocate the mmap handle. Out of memory.");
    return false;
  }
  strncpy(mmap_handle->map_name, full_map_name, TAKYON_MAX_BUFFER_NAME_CHARS);
  mmap_handle->mapping_handle = NULL;
  mmap_handle->mapped_addr = mapped_addr;
  mmap_handle->total_bytes = bytes;

#ifdef DEBUG_MESSAGE
  printf("FOUND: '%s' %d bytes\n", full_map_name, bytes);
#endif

  *got_it_ret = 1;
  *mmap_handle_ret = mmap_handle;

  return true;
}

bool mmapFree(void *mmap_handle_opaque, char *error_message, int max_error_message_chars) {
  struct _MmapHandle *mmap_handle = (struct _MmapHandle *)mmap_handle_opaque;

  if (mmap_handle == NULL) {
    snprintf(error_message, max_error_message_chars, "The mmap is NULL");
    return false;
  }

  /* Unmap memory address */
  UnmapViewOfFile(mmap_handle->mapped_addr);

  /* Delete memory map */
  if (mmap_handle->mapping_handle != NULL) {
    /* Remove map */
    CloseHandle(mmap_handle->mapping_handle);
  }

  /* Free the handle */
  mmap_handle->map_name[0] = '\0';
  mmap_handle->mapped_addr = NULL;
  mmap_handle->mapping_handle = NULL;
  mmap_handle->total_bytes = 0;
  free(mmap_handle);

  return true;
}
