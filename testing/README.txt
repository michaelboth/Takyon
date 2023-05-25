Linux & Mac
  Build
    usage: ./buildExamples.bash [debug] [mmap] [rdma] [cuda]
    > ./buildExamples.bash debug mmap rdma cuda
  Run Inter-thread
    > ./runInterThread.bash
  Run Inter-process
    usage: ./runInterProcess.bash <A|B> [mmap] [socket] [multicast] [ephemeral]
      A> ./runInterProcess.bash A mmap socket multicast ephemeral
      B> ./runInterProcess.bash B mmap socket multicast ephemeral
    If using CUDA:
      A> ./runInterProcess.bash A mmap
      B> ./runInterProcess.bash B mmap
  Run Inter-processor
    usage: ./runInterProcessor.bash <A|B> <local_ip> <remote_ip> [socket] [multicast] [ephemeral] [rdma]
      A> ./runInterProcessor.bash A 192.168.0.210 192.168.0.211 socket multicast ephemeral rdma
      B> ./runInterProcessor.bash B 192.168.0.210 192.168.0.211 socket multicast ephemeral rdma
    If using CUDA:
      A> ./runInterProcessor.bash A 192.168.0.210 192.168.0.211 rdma multicast
      B> ./runInterProcessor.bash B 192.168.0.210 192.168.0.211 rdma multicast
  Clean
    > ./buildExamples.bash clean

Windows
  Build
    usage: buildExamples.bat [debug] [mmap] [cuda]
    > buildExamples.bat debug mmap cuda
  Run Inter-thread
    > runInterThread.bat
  Run Inter-process
    usage: runInterProcess.bat <A|B> [mmap] [socket] [multicast] [ephemeral]
      A> runInterProcess.bat A mmap socket multicast ephemeral
      B> runInterProcess.bat B mmap socket multicast ephemeral
    If using CUDA:
      A> runInterProcess.bat A mmap
      B> runInterProcess.bat B mmap
  Run Inter-processor
    usage: runInterProcessor.bat <A|B> <local_ip> <remote_ip> [socket] [multicast] [ephemeral]
      A> runInterProcessor.bat A 192.168.0.210 192.168.0.211 socket multicast ephemeral
      B> runInterProcessor.bat B 192.168.0.210 192.168.0.211 socket multicast ephemeral
  Clean
    > buildExamples.bat clean
