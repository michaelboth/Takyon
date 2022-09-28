Linux & Mac
  Build
    usage: ./buildExamples.bash [debug] [mmap] [rdma] [cuda]
    > ./buildExamples.bash debug mmap rdma cuda
  Run Inter-thread
    > ./runInterThread.bash
  Run Inter-process
    usage: ./runInterProcess.bash <A|B> [mmap] [socket] [ephemeral]
      A> ./runInterProcess.bash A mmap socket ephemeral
      B> ./runInterProcess.bash B mmap socket ephemeral
    If using CUDA:
      A> ./runInterProcess.bash A mmap
      B> ./runInterProcess.bash B mmap
  Run Inter-processor
    usage: ./runInterProcessor.bash <A|B> <local_ip> <remote_ip> [socket] [ephemeral] [rdma]
      A> ./runInterProcessor.bash A 192.168.0.210 192.168.0.211 socket ephemeral rdma
      B> ./runInterProcessor.bash B 192.168.0.210 192.168.0.211 socket ephemeral rdma
    If using CUDA:
      A> ./runInterProcessor.bash A 192.168.0.210 192.168.0.211 rdma
      B> ./runInterProcessor.bash B 192.168.0.210 192.168.0.211 rdma
  Clean
    > ./buildExamples.bash clean

Windows
  Build
    usage: buildExamples.bat [debug] [mmap] [cuda]
    > buildExamples.bat debug mmap cuda
  Run Inter-thread
    > runInterThread.bat
  Run Inter-process
    usage: runInterProcess.bat <A|B> [mmap] [socket] [ephemeral]
      A> runInterProcess.bat A mmap socket ephemeral
      B> runInterProcess.bat B mmap socket ephemeral
    If using CUDA:
      A> runInterProcess.bat A mmap
      B> runInterProcess.bat B mmap
  Run Inter-processor
    usage: runInterProcessor.bat <A|B> <local_ip> <remote_ip> [socket] [ephemeral]
      A> runInterProcessor.bat A 192.168.0.210 192.168.0.211 socket ephemeral
      B> runInterProcessor.bat B 192.168.0.210 192.168.0.211 socket ephemeral
  Clean
    > ./buildExamples.bat clean
