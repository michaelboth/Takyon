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
- Point-to-point message passing communication API
- Heterogeneous: unifies RDMA, sockets, and many other communication interconnects
- Supports reliable and unreliable (unicast and multicast)
- Supports blocking and non-blocking (DMA based) transfers
- Supports one-sided (read or write) and two-sided transfers (send -> recv)
- Designed for zero-copy AND (not 'or') one-way (can't get any faster than that)
- Supports multiple memory blocks in a single transfer (even mixing CPU and CUDA memory)
- Designed to allow your app to be fault tolerant (via timeouts, disconnect detection, and dynamic path creation)
- Supports GPU memory via NVIDIA's cudaMemcpy, CUDA IPC, and GPUDirect
- Only 8 functions... intuitive and can learn in a few days
- Tested on Windows, Mac, & Linux

# Takyon Providers
A Takyon Provider gives the application access to a specific communication interconnect (see table below).<br>
Each Takyon Provider supports a set of properties. See the details in ```src/providers/provider_<interconnect>.c```<br>
Add your own Takyon Providers as needed (see ```src/providers/supported_providers.h```).<br><br>Currently supported Takyon Providers:
Provider       | Type       | Message Bytes | Non-Blocking | Includes One-Sided | Supports CUDA | Supports 32 bit Piggyback Message | Platforms
---------------|------------|---------------|--------------|--------------------|---------------|-----------------------------------|----------
InterThread    | Reliable   | 0 .. >4 GB    | Recv         | Read, Write        | Yes           | Yes                               | All
InterThread UC | Uneliable  | 0 .. >4 GB    | Recv         | Read, Write        | Yes           | Yes                               | All
InterProcess   | Reliable   | 0 .. >4 GB    | Recv         | Read, Write        | Yes           | Yes                               | All
InterProcess UC| Unreliable | 0 .. >4 GB    | Recv         | Read, Write        | Yes           | Yes                               | All
Socket Tcp     | Reliable   | 0 .. 1 GB     |              |                    |               | Yes                               | All
Socket Udp     | Unreliable | Unicast:<br>1 .. 64 KB<br>Multicast:<br>1 .. MTU  |     |   |     |                                   | All
RDMA RC        | Reliable   | 0 .. 1 GB     | Send, Recv, Read, Write, Atomics | Read, Write, Atomics | Yes | Yes                   | Linux
RDMA UC        | Unreliable | 0 .. 1 GB     | Send, Recv, Write        | Write  | Yes           | Yes                               | Linux
RDMA UD        | Unreliable | 0 .. 4 KB<br>Unicast Multicast | Send, Recv   |   | Yes           | Yes                               | Linux
RDMA tested on Nvidia's Mellanox platforms and Intel platforms

# The API
Takyon's API only contains 8 functions which essentially defines the basis of all communication. For eHPC engineers, there's no need to bog them down in further gory details of the underlying communication.
Grouping | Functions | Description
---------|-----------|------------------
Creation | takyonCreate()<br>takyonDestroy() | Create and destroy endpoints of a point-to-point communication path. The remote side is not required to be a Takyon endpoint (e.g. a sensor sending out UDP packets)
Two-Sided | takyonSend()<br>takyonIsSent()<br>takyonPostRecvs()<br>takyonIsRecved() | Sending and receiving is a coordinated effort by both endpoints. The source side sends, and the destination side receives.
One-Sided | takyonOneSided()<br>takyonIsOneSidedDone() | Only one endpoint does all the work, and the other endpoint doesn't even know it's happening (i.e. it's not involved). This includes 'read', 'write' and atomics.

The Takyon functions and data structures are in the header file: ```inc/takyon.h```<br>The header file explains most of the usage details, and the <a href="Takyon_Introduction.pdf">Takyon Introduction</a> presentation is a good starting point to understand the general features.

# Building and Testing
Takyon and its examples are provided as C code. Takyon is compiled into a library with only the Takyon Providers you want, and the examples are linked with the Takyon library.<br>
<br>
You can build and run with the direct approach or the more automated approach:
1. **Direct**: To compile the library, see ```lib/README.txt```. To build an example, see ```examples/<example>/README.txt```.
2. **Automated**: One script will build the library and all the examples, then a second script will run many variations of the examples to validate the build and Takyon Providers. See ```testing/README.txt```

### Examples
The examples cover most of Takyon's features. Use these as the basis for your application and strip out all the features you don't need.
Example | Description
--------|------------
hello-two_sided | Transfer a simple greeting between two endpoints using 'send' -> 'recv'.<br>Supports all Takyon Providers, CUDA, MMAPs, reliable, unreliable, and multiple memory blocks.
hello-one_sided | Transfer a simple greeting between two endpoints using one-sided read, write, and/or atomic transfers.<br>Optionally supports CUDA & MMAPs. Even though some unreliable Takyon Providers can support one sided, this example only works with reliable Providers.
throughput | Determines the transfer speed of a Provider; 'send' -> 'recv' or 'write','read'.<br>Supports all Takyon Providers, CUDA, MMAPs, reliable, and unreliable.<br>Run with '-h' to see all the options.
