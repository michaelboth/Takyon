<img src="docs/Takyon_Logo.png" alt="Logo" style="width:400px;"/>

The future of high performance heterogeneous point-to-point communication for the eHPC industry (embedded High Performance Communication)<br><br>
Takyon is currently the only proposal for a new Khronos open standard for communication: www.khronos.org/exploratory/heterogeneous-communication/<br>

## Takyon Presentation
<a href="Takyon_Introduction.pdf">
  <img src="docs/presentation_icon.png" alt="Takyon Introduction" width="256" height="149">
</a>
<br>
View this presentation to learn about Takyon
<br>

# Takyon's Key Features
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

## Takyon's Currently Supported Providers
A Takyon Provider gives access to a specific communication interconnect.<br>
Add your own providers as needed (see ```src/providers/supported_providers.h```).
Provider      | Type       | Message Bytes | Non-Blocking | Supports One-Sided | Supports CUDA | Platforms
--------------|------------|---------------|--------------|--------------------|---------------|----------
Inter-Thread  | Reliable   | 0 .. >4 GB    | No           | Yes                | Yes           | All
Inter-Process | Reliable   | 0 .. >4 GB    | No           | Yes                | Yes           | All
Socket Tcp    | Reliable   | 0 .. 1 GB     | No           | No                 | No            | All
Socket Udp    | Unreliable | 1 .. 64 KB    | No           | No                 | No            | All
Rdma RC       | Reliable   | 0 .. 1 GB     | Yes          | Yes                | Yes           | Linux
Rdma UC       | Unreliable | 0 .. 1 GB     | Yes          | Yes                | Yes           | Linux
Rdma UD       | Unreliable | 0 .. MTU      | Yes          | No                 | Yes           | Linux

## Prepare the Build Environment
To build the Takyon examples, the following is needed:

OS | Requirements
--------|------------
Linux | gcc <br> make
Mac | xCode
Windows | Visual Studio<br><br> Takyon requires Posix threads (not supported in Visual Studio). To download and build it:<br> 1. Get the source code from: https://sourceforge.net/projects/pthreads4w/ <br> 2. Unzip, rename to 'pthreads4w' and put in the C:\ folder <br> 3. Start a Visual Studio x64 native shell <br> ```> cd c:\pthreads4w``` <br> ```> nmake VC VC-debug VC-static VC-static-debug install DESTROOT=.\install```

To easily build and run all the examples to do some thorough testing, go into the ```testing``` folder and see ```README.txt```

## Examples
To help you get started, some examples are provided. Each example has a ```README.txt``` to know how to build and run the example. The examples cover most of Takyon's features.
Example | Description
--------|------------
hello-two_sided | Transfer a simple greeting between two endpoints using 'send' -> 'recv'.<br>Supports all Providers, CUDA, MMAPs, relialbe, unrelaible, and multiple memory blocks.
hello-one_sided | Transfer a simple greeting between two endpoints using a one-sided 'read','write'.<br>Optionally supports CUDA & MMAPs. Only works will Proivers that support read and write.
throughput | Determine transfer speed of a provider; 'send' -> 'recv' or 'write','read'.<br>Supports all Providers, CUDA, and MMAPs.<br>Run with '-h' to see all the options.
