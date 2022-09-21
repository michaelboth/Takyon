Linux & Mac
  Build
    > ./buildExamples.bash [debug] [mmap] [cuda]
  Run
    > ./runInterThread.bash
    usage: ./runInterProcess.bash <A|B> [mmap] [socket] [ephemeral]
    A> ./runInterProcess.bash A mmap socket ephemeral
    B> ./runInterProcess.bash B mmap socket ephemeral
  Clean
    > ./buildExamples.bash clean

Windows
  Build
    > buildExamples.bat [debug] [mmap] [cuda]
  Run
    > runInterThread.bat
    usage: runInterProcess.bat <A|B> [mmap] [socket] [ephemeral]
    A> runInterProcess.bat A mmap socket ephemeral
    B> runInterProcess.bat B mmap socket ephemeral
  Clean
    > ./buildExamples.bat clean
