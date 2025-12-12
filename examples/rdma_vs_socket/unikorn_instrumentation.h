//     Copyright 2021..2025 Michael Both
//     Licensed under the Apache License, Version 2.0 (the "License");
//     you may not use this file except in compliance with the License.
//     You may obtain a copy of the License at
//         http://www.apache.org/licenses/LICENSE-2.0
//     Unless required by applicable law or agreed to in writing, software
//     distributed under the License is distributed on an "AS IS" BASIS,
//     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//     See the License for the specific language governing permissions and
//     limitations under the License.

#ifndef _UNIKORN_INSTRUMENTATION_H_
#define _UNIKORN_INSTRUMENTATION_H_

// NOTE: Include this header file in any source file that will use unikorn event intrumenting
#ifdef ENABLE_UNIKORN_RECORDING
#include "unikorn.h"
#include "unikorn_clock.h"       // Provide your own clock functionality if you don't want to use one of the clocks provided with the Unikorn distribution
#include "unikorn_file_flush.h"  // Provide your own flush functionality (e.g. socket) here if you don't want to flush to a file

// ------------------------------------------------
// Define the unique IDs for the folders and events
// ------------------------------------------------
enum {
  // IMPORTANT, IDs must start with 1 since 0 is reserved for 'close folder'
  // Folders   (not required to have any folders)
  //  FOLDER1_ID=1,
  //  FOLDER2_ID,
  // Events   (must have at least one start/end ID combo)
  INIT_START_ID = 1,
  INIT_END_ID,
  LATENCY_TEST_START_ID,
  LATENCY_TEST_END_ID,
  LATENCY_ITERATION_START_ID,
  LATENCY_ITERATION_END_ID,
  THROUGHPUT_TEST_START_ID,
  THROUGHPUT_TEST_END_ID,
  SEND_BUFFERS_START_ID,
  SEND_BUFFERS_END_ID,
  WAIT_FOR_ACK_START_ID,
  WAIT_FOR_ACK_END_ID,
  RECV_BUFFERS_START_ID,
  RECV_BUFFERS_END_ID,
  POST_RECVS_START_ID,
  POST_RECVS_END_ID,
  SEND_ACK_START_ID,
  SEND_ACK_END_ID,
  FINALIZE_START_ID,
  FINALIZE_END_ID
};

// IMPORTANT: Call #define ENABLE_UNIKORN_SESSION_CREATION, just before #include "unikorn_instrumentation.h", in the file that calls UK_CREATE()
#ifdef ENABLE_UNIKORN_SESSION_CREATION

// ------------------------------------------------
// Define custom folders
// ------------------------------------------------
//static UkFolderRegistration L_unikorn_folders[] = {
//    Name        ID
//  { "Folder 1", FOLDER1_ID},
//  { "Folder 2", FOLDER2_ID}
//   IMPORTANT: This folder registration list must be in the same order as the folder ID enumerations above
//};
//#define NUM_UNIKORN_FOLDER_REGISTRATIONS (sizeof(L_unikorn_folders) / sizeof(UkFolderRegistration))
#define L_unikorn_folders NULL
#define NUM_UNIKORN_FOLDER_REGISTRATIONS 0

// ------------------------------------------------
// Define custom events
// ------------------------------------------------
static UkEventRegistration L_unikorn_events[] = {
  // Name                  Color      Start ID                    End ID                    Start Value Name  End Value Name
  { "Init",                UK_PURPLE, INIT_START_ID,              INIT_END_ID,              "",               ""},
  { "Latency Test",        UK_BLACK,  LATENCY_TEST_START_ID,      LATENCY_TEST_END_ID,      "",               ""},
  { "Latency Iteration",   UK_GREEN,  LATENCY_ITERATION_START_ID, LATENCY_ITERATION_END_ID, "",               ""},
  { "Throughput Test",     UK_BLACK,  THROUGHPUT_TEST_START_ID,   THROUGHPUT_TEST_END_ID,   "",               ""},
  { "Send all Buffers",    UK_BLUE,   SEND_BUFFERS_START_ID,      SEND_BUFFERS_END_ID,      "",               ""},
  { "Wait for ACK",        UK_RED,    WAIT_FOR_ACK_START_ID,      WAIT_FOR_ACK_END_ID,      "",               ""},
  { "Recv all Buffers",    UK_GREEN,  RECV_BUFFERS_START_ID,      RECV_BUFFERS_END_ID,      "",               ""},
  { "Post all Recvs",      UK_BLACK,  POST_RECVS_START_ID,        POST_RECVS_END_ID,        "",               ""},
  { "Send ACK",            UK_ORANGE, SEND_ACK_START_ID,          SEND_ACK_END_ID,          "",               ""},
  { "Finalize",            UK_PURPLE, FINALIZE_START_ID,          FINALIZE_END_ID,          "",               ""}
  // IMPORTANT: This event registration list must be in the same order as the event ID enumerations above
};
#define NUM_UNIKORN_EVENT_REGISTRATIONS (sizeof(L_unikorn_events) / sizeof(UkEventRegistration))

#endif // ENABLE_UNIKORN_SESSION_CREATION

// Helpful macros
#define UK_DESTROY(_session, _flush_info) ukDestroy(_session)
#define UK_FLUSH(_session) ukFlush(_session)
#define UK_OPEN_FOLDER(_session, _folder_id) ukOpenFolder(_session, _folder_id)
#define UK_CLOSE_FOLDER(_session) ukCloseFolder(_session)
#define UK_RECORD_EVENT(_session, _event_id, _value) ukRecordEvent(_session, _event_id, _value, __FILE__, __FUNCTION__, __LINE__)

#else

// Helpful macros to strip out Unikorn instrumentation
#define UK_DESTROY(_session, _flush_info)
#define UK_FLUSH(_session)
#define UK_OPEN_FOLDER(_session, _folder_id)
#define UK_CLOSE_FOLDER(_session)
#define UK_RECORD_EVENT(_session, _event_id, _value)

#endif // ENABLE_UNIKORN_RECORDING

#endif // _UNIKORN_INSTRUMENTATION_H_
