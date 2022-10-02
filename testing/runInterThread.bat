@echo off

rem Keep variables local to this script
setlocal EnableDelayedExpansion

rem hello-two_sided
cd ..\examples\hello-two_sided
if ERRORLEVEL 1 ( echo "Failed to cd ..\examples\hello-two_sided" & GOTO:done )
hello_mt "InterThreadRC -pathID=1" 0
if ERRORLEVEL 1 ( echo "Failed to run hello-two_sided" & GOTO:done )
hello_mt "InterThreadRC -pathID=1" 1
if ERRORLEVEL 1 ( echo "Failed to run hello-two_sided" & GOTO:done )
hello_mt "InterThreadRC -pathID=1" 10
if ERRORLEVEL 1 ( echo "Failed to run hello-two_sided" & GOTO:done )
hello_mt "InterThreadUC -pathID=1" 0
if ERRORLEVEL 1 ( echo "Failed to run hello-two_sided" & GOTO:done )
hello_mt "InterThreadUC -pathID=1" 1
if ERRORLEVEL 1 ( echo "Failed to run hello-two_sided" & GOTO:done )
hello_mt "InterThreadUC -pathID=1" 10
if ERRORLEVEL 1 ( echo "Failed to run hello-two_sided" & GOTO:done )

rem hello-one_sided
cd ..\hello-one_sided
if ERRORLEVEL 1 ( echo "Failed to cd ..\examples\hello-one_sided" & GOTO:done )
hello_mt "InterThreadRC -pathID=1" 0
if ERRORLEVEL 1 ( echo "Failed to run hello-one_sided" & GOTO:done )
hello_mt "InterThreadRC -pathID=1" 1
if ERRORLEVEL 1 ( echo "Failed to run hello-one_sided" & GOTO:done )
hello_mt "InterThreadRC -pathID=1" 10
if ERRORLEVEL 1 ( echo "Failed to run hello-one_sided" & GOTO:done )

rem throughput
cd ..\throughput
if ERRORLEVEL 1 ( echo "Failed to cd ..\examples\throughput" & GOTO:done )
throughput_mt "InterThreadRC -pathID=1" -b=32768 -v
if ERRORLEVEL 1 ( echo "Failed to run throughput_mt" & GOTO:done )
throughput_mt "InterThreadRC -pathID=1" -b=32768
if ERRORLEVEL 1 ( echo "Failed to run throughput_mt" & GOTO:done )
throughput_mt "InterThreadRC -pathID=1" -b=32768 -v -o
if ERRORLEVEL 1 ( echo "Failed to run throughput_mt" & GOTO:done )
throughput_mt "InterThreadRC -pathID=1" -b=32768 -o
if ERRORLEVEL 1 ( echo "Failed to run throughput_mt" & GOTO:done )

throughput_mt "InterThreadRC -pathID=1" -b=32768 -v -e
if ERRORLEVEL 1 ( echo "Failed to run throughput_mt" & GOTO:done )
throughput_mt "InterThreadRC -pathID=1" -b=32768 -e
if ERRORLEVEL 1 ( echo "Failed to run throughput_mt" & GOTO:done )
throughput_mt "InterThreadRC -pathID=1" -b=32768 -v -o -e
if ERRORLEVEL 1 ( echo "Failed to run throughput_mt" & GOTO:done )
throughput_mt "InterThreadRC -pathID=1" -b=32768 -o -e
if ERRORLEVEL 1 ( echo "Failed to run throughput_mt" & GOTO:done )

throughput_mt "InterThreadUC -pathID=1" -b=32768 -v
if ERRORLEVEL 1 ( echo "Failed to run throughput_mt" & GOTO:done )
throughput_mt "InterThreadUC -pathID=1" -b=32768
if ERRORLEVEL 1 ( echo "Failed to run throughput_mt" & GOTO:done )

throughput_mt "InterThreadUC -pathID=1" -b=32768 -v -e
if ERRORLEVEL 1 ( echo "Failed to run throughput_mt" & GOTO:done )
throughput_mt "InterThreadUC -pathID=1" -b=32768 -e
if ERRORLEVEL 1 ( echo "Failed to run throughput_mt" & GOTO:done )

GOTO:done

:done
endlocal
GOTO:eof
