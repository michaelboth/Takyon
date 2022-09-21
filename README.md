<img src="docs/Takyon_Logo.png" alt="Logo" style="width:400px;"/>

The only proposal for a new Khronos open standard: www.khronos.org/exploratory/heterogeneous-communication/<br>
The future of high performance heterogeneous point-to-point communication<br>

## Takyon Presentation
<a href="Takyon_Introduction.pdf">
  <img src="docs/presentation_icon.png" alt="Takyon Introduction" width="256" height="149">
</a>
<br>
View this presentation to learn the about Takyon
<br>

# Takyon's Key Features
- Point-to-point message passing communication API (reliable and unreliable)
- Heterogeneous: unifies RDMA, sockets, and many other communication interconnects
- Only 8 functions... can learn in a few days
- Supports connect and unconnected (unicast & multicast)
- Supports blocking and non-blocking transfers
- Supports one-sided and two-sided transfers
- Designed for zero-copy one-way
- Designed to allow your app to be fault tolerant
- Tested on Windows, Mac, & Linux

## Takyon's Currently Supported Providers
A provider gives access to a specific communication interconnect
Provider      | Type                   | Max Message Bytes | Non-Blocking | Supports One-Sided | Zero Byte Messages | GPU
--------------|------------------------|-------------------|--------------|--------------------|--------------------|----
InterThread   | Reliable-Connected     | >4 GB             | No           | Yes                | Yes                | Yes
InterProcess  | Reliable-Connected     | >4 GB             | No           | Yes                | Yes                | Yes
SocketTcp     | Reliable-Connected     | 1 GB              | No           | No                 | Yes                | No
SocketUdp     | Unreliable-Unconnected | 64 KB             | No           | No                 | No                 | No
RdmaRC (soon) | Reliable-Connected     | 1 GB              | Yes          | Yes                | Yes                | Yes
RdmaUC (soon) | Unreliable-Connected   | 1 GB              | Yes          | Yes                | Yes                | Yes
RdmaUD (soon) | Unreliable-Unconnected | MTU               | Yes          | No                 | Yes                | Yes

## Prepare the OS Environment
To build Takyon into your application, the following is needed:

OS | Requirements
--------|------------
Linux | gcc <br> make
Mac | xCode
Windows | Visual Studio<br><br> Takyon requires Posix threads (not supported in Visual Studio). To download and build it:<br> 1. Get the source code from: https://sourceforge.net/projects/pthreads4w/ <br> 2. Unzip, rename to 'pthreads4w' and put in the C:\ folder <br> 3. Start a Visual Studio x64 native shell <br> ```> cd c:\pthreads4w``` <br> ```> nmake VC VC-debug VC-static VC-static-debug install DESTROOT=.\install```

## Examples
To help you get started, some examples are provided. Each example has a ```README.txt``` to know how to build and run the example
Example | Description
--------|------------
hello-two_sided | Transfer a simple greeting between two endpoints using send() and recv()
hello-one_sided | Transfer a simple greeting between two endpoints using a one-sided read/write
