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
throughput_mt "InterThreadRC -pathID=1" -bytes=32768 -V
if ERRORLEVEL 1 ( echo "Failed to run throughput_mt" & GOTO:done )
throughput_mt "InterThreadRC -pathID=1" -bytes=32768
if ERRORLEVEL 1 ( echo "Failed to run throughput_mt" & GOTO:done )
throughput_mt "InterThreadRC -pathID=1" -bytes=32768 -V -write
if ERRORLEVEL 1 ( echo "Failed to run throughput_mt" & GOTO:done )
throughput_mt "InterThreadRC -pathID=1" -bytes=32768 -write
if ERRORLEVEL 1 ( echo "Failed to run throughput_mt" & GOTO:done )

throughput_mt "InterThreadRC -pathID=1" -bytes=32768 -V -e
if ERRORLEVEL 1 ( echo "Failed to run throughput_mt" & GOTO:done )
throughput_mt "InterThreadRC -pathID=1" -bytes=32768 -e
if ERRORLEVEL 1 ( echo "Failed to run throughput_mt" & GOTO:done )
throughput_mt "InterThreadRC -pathID=1" -bytes=32768 -V -write -e
if ERRORLEVEL 1 ( echo "Failed to run throughput_mt" & GOTO:done )
throughput_mt "InterThreadRC -pathID=1" -bytes=32768 -write -e
if ERRORLEVEL 1 ( echo "Failed to run throughput_mt" & GOTO:done )

throughput_mt "InterThreadUC -pathID=1" -bytes=32768 -i=100000 -V
if ERRORLEVEL 1 ( echo "Failed to run throughput_mt" & GOTO:done )
throughput_mt "InterThreadUC -pathID=1" -bytes=32768
if ERRORLEVEL 1 ( echo "Failed to run throughput_mt" & GOTO:done )

throughput_mt "InterThreadUC -pathID=1" -bytes=32768 -i=100000 -V -e
if ERRORLEVEL 1 ( echo "Failed to run throughput_mt" & GOTO:done )
throughput_mt "InterThreadUC -pathID=1" -bytes=32768 -e
if ERRORLEVEL 1 ( echo "Failed to run throughput_mt" & GOTO:done )

GOTO:done

:done
endlocal
GOTO:eof
