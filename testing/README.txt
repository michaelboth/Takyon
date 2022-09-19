Linux & Mac
  Build
    > ./buildExamples.bash debug mmap cuda
  Run
    > ./runInterThread.bash
    A> ./runInterProcess.bash A mmap socket ephemeral
    B> ./runInterProcess.bash B mmap socket ephemeral
  Clean
    > ./buildExamples.bash clean
