//     Copyright 2025 Michael Both
//     Licensed under the Apache License, Version 2.0 (the "License");
//     you may not use this file except in compliance with the License.
//     You may obtain a copy of the License at
//         http://www.apache.org/licenses/LICENSE-2.0
//     Unless required by applicable law or agreed to in writing, software
//     distributed under the License is distributed on an "AS IS" BASIS,
//     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//     See the License for the specific language governing permissions and
//     limitations under the License.

/*+ valgrind */

#include "LatencyTest.hpp"
#include "ThroughputTest.hpp"
#include <cstring>

// For Unikorn instrumentation
#define ENABLE_UNIKORN_SESSION_CREATION
#include "unikorn_instrumentation.h"
#include "unikorn_macros.h" /*+ merge into unikorn_instrumentation.h to avoid requireing Unikorn for customer */

static void printUsageAndExit(const char *_program) {
  printf("usage: %s <lat|tp> <params_file> <provider> <A|B> [-once] [-iters=<uint32>] [-bytes=<uint64>] [-nbufs=<uint32>] [-poll] [-validate] [-verbose]\n", _program);
  printf("   lat or tp        : If 'lat', then run latency test. If 'tp' then run throughput test.\n");
  printf("   params_file      : Text file containing the Takyon parameters for each provider\n");
  printf("   provider         : One of TCP, UDP, RC, US, UD\n");
  printf("   A or B           : A is the sender, B is the receiver. Only provide one of them.\n");
  printf("   -once            : Run the number of iteration once. Default is to run forever repeating the number of iterations.\n");
  printf("   -iters=<uint32>  : Number of times to transfer using all message buffers. Default is %d.\n", Common::DEFAULT_NITERS);
  printf("   -nbytes=<uint64> : Bytes per message. Default is to run a set of sizes from %lu bytes to %lu MB.\n", Common::MIN_BYTES, Common::MAX_BYTES/(1024*1024));
  printf("   -nbufs=<uint32>  : Number of transport buffers (must be an even number), where each buffer can hold a single message. Default is %u.\n", Common::DEFAULT_NBUFS);
  printf("   -poll            : Use polling (low latency, high cpu usage) instead of the default event driven (high latency, low cpu usage).\n");
  printf("   -validate        : Validate the messages. Default is no validation\n");
  printf("   -verbose         : Print extra helpful messages. Default is off\n");
  printf("   -h               : Print this help message.\n");
  exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
  if (argc < 5) {
    printUsageAndExit(argv[0]);
  }

  // Validate required parameters
  Common::AppParams app_params;
  std::string test_mode = argv[1];
  if (test_mode != "lat" && test_mode != "tp") {
    EXIT_WITH_MESSAGE(std::string("test mode (arg2) must be one of lat (latency) or tp (throughput)"));
  }
  if (test_mode == "lat") {
    app_params.nbytes = Common::MIN_BYTES;
  } else if (test_mode == "tp") {
    app_params.nbytes = Common::MAX_BYTES;
  }
  std::map<std::string, std::string> connection_params = Common::loadProviderParamsFile(argv[2]);
  app_params.provider = argv[3];
  if (app_params.provider != "TCP" && app_params.provider != "UDP" && app_params.provider != "RC" && app_params.provider != "UC" && app_params.provider != "UD") {
    EXIT_WITH_MESSAGE(std::string("provider (arg2) must be one of TCP, UDP, RC, UC, or UD"));
  }
  if (app_params.provider == "UDP" || app_params.provider == "UD") {
    /*+ remove after it's implemented */EXIT_WITH_MESSAGE(std::string("UDP and UD not yet implemented"));
  }
  std::string side = argv[4];
  if (side != "A" && side != "B") {
    EXIT_WITH_MESSAGE(std::string("side (arg3) must be either A or B"));
  }
  bool is_sender = (side == "A");
  std::string provider_args_key = app_params.provider + "_" + side;
  app_params.provider_params = connection_params.at(provider_args_key);

  // Check for optional params
  for (int i=5; i<argc; i++) {
    if (strcmp(argv[i], "-h") == 0) {
      printUsageAndExit(argv[0]);
    } else if (strcmp(argv[i], "-poll") == 0) {
      app_params.use_polling = true;
    } else if (strcmp(argv[i], "-validate") == 0) {
      app_params.validate = true;
    } else if (strcmp(argv[i], "-verbose") == 0) {
      app_params.verbose = true;
    } else if (strcmp(argv[i], "-once") == 0) {
      app_params.run_forever = false;
    } else if (strncmp(argv[i], "-iters=", 7) == 0) {
      int tokens = sscanf(argv[i], "-iters=%u", &app_params.iters);
      if (tokens != 1) { EXIT_WITH_MESSAGE(std::string("Arg '" + std::string(argv[i]) + "' is invalid.")); }
      if (app_params.iters == 0) { EXIT_WITH_MESSAGE(std::string("Arg '" + std::string(argv[i]) + "' must be greater than zero.")); }
    } else if (strncmp(argv[i], "-nbufs=", 7) == 0) {
      int tokens = sscanf(argv[i], "-nbufs=%u", &app_params.nbufs);
      if (tokens != 1) { EXIT_WITH_MESSAGE(std::string("Arg '" + std::string(argv[i]) + "' is invalid.")); }
      if (app_params.nbufs == 0) { EXIT_WITH_MESSAGE(std::string("Arg '" + std::string(argv[i]) + "' must be greater than zero.")); }
      if ((app_params.nbufs % 2) != 0) { EXIT_WITH_MESSAGE(std::string("Arg '" + std::string(argv[i]) + "' must be an even number.")); }
    } else if (strncmp(argv[i], "-nbytes=", 8) == 0) {
      int tokens = sscanf(argv[i], "-nbytes=%lu", &app_params.nbytes);
      if (tokens != 1) { EXIT_WITH_MESSAGE(std::string("Arg '" + std::string(argv[i]) + "' is invalid.")); }
      if (app_params.nbytes < Common::MIN_BYTES) { EXIT_WITH_MESSAGE(std::string("Arg '" + std::string(argv[i]) + "' must be at least " + std::to_string(Common::MIN_BYTES) + ".")); }
      if ((app_params.nbytes % 4) != 0) { EXIT_WITH_MESSAGE(std::string("Arg '" + std::string(argv[i]) + "' must be aligned to 4 bytes to keep the algorithm simple.")); }
    } else {
      EXIT_WITH_MESSAGE(std::string("Arg '" + std::string(argv[i]) + "' is not supported."));
    }
  }

  // Print some helpful info
  if (app_params.verbose) {
    printf("Parameters\n");
    printf("  Size: %s\n", is_sender ? "sender" : "receiver");
    printf("  Provider: %s\n", app_params.provider.c_str());
    printf("  Provider args: %s\n", app_params.provider_params.c_str());
    printf("  NBufs: %u\n", app_params.nbufs);
    printf("  NBytes: %lu\n", app_params.nbytes);
    printf("  Iterations: %u\n", app_params.iters);
    printf("  Number of time to repeat iterations: %s\n", app_params.run_forever ? "infinite" : "once");
    printf("  Completion Check: %s\n", app_params.use_polling ? "polling" : "event driven");
    printf("  Validation: %s\n", app_params.validate ? "on" : "off");
    printf("\n");
  }

  // Setup the Unikorn instrumentation
#ifdef ENABLE_UNIKORN_RECORDING
  char filename[100];
  snprintf(filename, 100, "timing_%s_%s_%lu_bytes_%u_bufs.events", app_params.provider.c_str(), side.c_str(), app_params.nbytes, app_params.nbufs);
  UkFileFlushInfo flush_info; // Needs to be persistent for life of session
#endif
  UK_CREATE(filename, 1000000, true, false, true, true, true, NUM_UNIKORN_FOLDER_REGISTRATIONS, L_unikorn_folders, NUM_UNIKORN_EVENT_REGISTRATIONS, L_unikorn_events, &flush_info, &app_params.unikorn_session);

  // Run test
  if (test_mode == "lat") {
    LatencyTest::runLatencyTest(is_sender, app_params);
  } else if (test_mode == "tp") {
    ThroughputTest::runThroughputTest(is_sender, app_params);
  }

  // Finalize
  printf("Done.\n");

  // Finish the Unikorn session
  UK_FLUSH(app_params.unikorn_session);
  UK_DESTROY(app_params.unikorn_session, &flush_info);

  return EXIT_SUCCESS;
}
