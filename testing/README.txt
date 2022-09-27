Linux & Mac
  Build
    usage: ./buildExamples.bash [debug] [mmap] [rdma] [cuda]
    > ./buildExamples.bash debug mmap rdma cuda
  Run
    > ./runInterThread.bash
    usage: ./runInterProcess.bash <A|B> [mmap] [socket] [ephemeral]
    A> ./runInterProcess.bash A mmap socket ephemeral
    B> ./runInterProcess.bash B mmap socket ephemeral
    A> ./runInterProcess.bash A mmap cuda
    B> ./runInterProcess.bash B mmap cuda
  Clean
    > ./buildExamples.bash clean

Windows
  Build
    usage: buildExamples.bat [debug] [mmap] [cuda]
    > buildExamples.bat debug mmap cuda
  Run
    > runInterThread.bat
    usage: runInterProcess.bat <A|B> [mmap] [socket] [ephemeral]
    A> runInterProcess.bat A mmap socket ephemeral
    B> runInterProcess.bat B mmap socket ephemeral
    A> runInterProcess.bat A mmap cuda
    B> runInterProcess.bat B mmap cuda
  Clean
    > ./buildExamples.bat clean
