<img src="docs/Takyon_Logo.png" alt="Logo" style="width:400px;"/>

The future of high performance heterogeneous point-to-point communication for the eHPC industry (embedded High Performance Computing)<br><br>
Takyon is currently the only proposal for a new Khronos open standard for communication: www.khronos.org/exploratory/heterogeneous-communication/<br>

### Takyon Presentation
This shows the Takyon features and reasoning behind creating a new point-to-point communication API<br>
<a href="Takyon_Introduction.pdf">
  <img src="docs/presentation_icon.png" alt="Takyon Introduction" width="256" height="149">
</a>
<br>

# Key Features
- Point-to-point message passing communication API (reliable and unreliable)
- Heterogeneous: unifies RDMA, sockets, and many other communication interconnects
- Takyon's abstration does not compromise performance
- Only 8 functions... intuitive and can learn in a few days
- Supports unreliable unicast & multicast
- Supports blocking and non-blocking transfers
- Supports one-sided (read or write) and two-sided transfers (send -> recv)
- Designed for zero-copy one-way (can't get any faster than that)
- Supports multiple memory blocks in a single transfer (even mixing CPU and CUDA memeory)
- Designed to allow your app to be fault tolerant (via timeouts, disconnect detection, and dynamic path creation)
- Supports GPU memory via CUDA's cudaMemcpy, CUDA IPC, and GPUDirect
- Tested on Windows, Mac, & Linux

### Supported Providers
A Takyon Provider gives access to a specific communication interconnect.<br>
Each Takyon Provider has it's own provider specification. To know the details see ```src/providers/provider_<interconnect>.c```<br>
Add your own providers as needed (see ```src/providers/supported_providers.h```).
Provider       | Type       | Message Bytes | Non-Blocking | Includes One-Sided | Supports CUDA | Supports 32 bit Piggyback Message | Platforms
---------------|------------|---------------|--------------|--------------------|---------------|-----------------------------------|----------
InterThread    | Reliable   | 0 .. >4 GB    | Recv         | Read, Write        | Yes           | Yes                               | All
InterThread UC | Uneliable  | 0 .. >4 GB    | Recv         |                    | Yes           | Yes                               | All
InterProcess   | Reliable   | 0 .. >4 GB    | Recv         | Read, Write        | Yes           | Yes                               | All
InterProcess UC| Unreliable | 0 .. >4 GB    | Recv         |                    | Yes           | Yes                               | All
Socket Tcp     | Reliable   | 0 .. 1 GB     |              |                    |               | Yes                               | All
Socket Udp     | Unreliable | Unicast:<br>1 .. 64 KB<br>Multicast:<br>1 .. MTU  |     |   |     |                                   | All
Rdma RC        | Reliable   | 0 .. 1 GB     | Send, Recv, Read, Write  | Read, Write, Atomics | Yes | Yes                           | Linux
Rdma UC        | Unreliable | 0 .. 1 GB     | Send, Recv, Write        | Write  | Yes           | Yes                               | Linux
Rdma UD        | Unreliable | 0 .. 4 KB     | Send, Recv               |        | Yes           | Yes                               | Linux

# The API
The 8 Takyon functions and most of the details are in the single header file: ```inc/takyon.h```<br>

# Building and Testing
Takyon and its examples are provided as C code. Takyon is compiled into a library with only the providers you want, and the examples are linked with the Takyon library.<br>
<br>
You can build and run with the direct approach or the more automated approach:
1. **Direct**: To compile the library, see ```lib/README.txt```. To build an example, see ```examples/<example>/README.txt```.
2. **Automated**: One script will build the Library and all the examples, then a second script will run many variations of the examples to validate the build and providers. See ```testing/README.txt```

### Examples
The examples cover most of Takyon's features.
Example | Description
--------|------------
hello-two_sided | Transfer a simple greeting between two endpoints using 'send' -> 'recv'.<br>Supports all Providers, CUDA, MMAPs, relialbe, unrelaible, and multiple memory blocks.
hello-one_sided | Transfer a simple greeting between two endpoints using a one-sided read, write, and atomic transfers.<br>Optionally supports CUDA & MMAPs. Only works with reliable Providers that support one sided.
throughput | Determine transfer speed of a Provider; 'send' -> 'recv' or 'write','read'.<br>Supports all Providers, CUDA, and MMAPs.<br>Run with '-h' to see all the options.
