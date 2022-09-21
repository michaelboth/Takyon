@echo off

rem Keep variables local to this script
setlocal EnableDelayedExpansion

set endpoint=no_value
set mmap=no
set socket=no
set ephemeral=no

rem Parse args
for %%a in (%*) do (
  if %%a == A set endpoint=A
  if %%a == B set endpoint=B
  if %%a == mmap set mmap=yes
  if %%a == socket set socket=yes
  if %%a == ephemeral set ephemeral=yes
)

if %endpoint% == no_value (
  echo "USAGE: runInterProcess.bat <A|B> [mmap] [socket] [ephemeral]"
  GOTO:done
)

echo "endpoint  = %endpoint%"
echo "mmap      = %mmap%"
echo "socket    = %socket%"
echo "ephemeral = %ephemeral%"

rem hello-two_sided
cd ..\examples\hello-two_sided
if ERRORLEVEL 1 ( echo "Failed to cd ..\examples\hello-two_sided" & GOTO:done )
if %mmap% == yes (
  hello_mp %endpoint% "InterProcess -pathID=1" 0
  if ERRORLEVEL 1 ( echo "Failed to run hello-two_sided" & GOTO:done )
  hello_mp %endpoint% "InterProcess -pathID=2" 1
  if ERRORLEVEL 1 ( echo "Failed to run hello-two_sided" & GOTO:done )
  hello_mp %endpoint% "InterProcess -pathID=3" 10
  if ERRORLEVEL 1 ( echo "Failed to run hello-two_sided" & GOTO:done )
)
if %socket% == yes (
  hello_mp %endpoint% "SocketTcp -local -pathID=1" 0
  if ERRORLEVEL 1 ( echo "Failed to run hello-two_sided" & GOTO:done )
  hello_mp %endpoint% "SocketTcp -local -pathID=2" 10
  if ERRORLEVEL 1 ( echo "Failed to run hello-two_sided" & GOTO:done )
  if %endpoint% == A (
    hello_mp %endpoint% "SocketTcp -client -remoteIP=127.0.0.1 -port=23456" 0
    if ERRORLEVEL 1 ( echo "Failed to run hello-two_sided" & GOTO:done )
    hello_mp %endpoint% "SocketTcp -client -remoteIP=127.0.0.1 -port=23456" 10
    if ERRORLEVEL 1 ( echo "Failed to run hello-two_sided" & GOTO:done )
  )
  if %endpoint% == B (
    hello_mp %endpoint% "SocketTcp -server -localIP=127.0.0.1 -port=23456 -reuse" 0
    if ERRORLEVEL 1 ( echo "Failed to run hello-two_sided" & GOTO:done )
    hello_mp %endpoint% "SocketTcp -server -localIP=127.0.0.1 -port=23456 -reuse" 10
    if ERRORLEVEL 1 ( echo "Failed to run hello-two_sided" & GOTO:done )
  )
  if %ephemeral% == yes (
    if %endpoint% == A (
      hello_mp %endpoint% "SocketTcp -client -remoteIP=127.0.0.1 -ephemeralID=1" 0
      if ERRORLEVEL 1 ( echo "Failed to run hello-two_sided" & GOTO:done )
      hello_mp %endpoint% "SocketTcp -client -remoteIP=127.0.0.1 -ephemeralID=1" 10
      if ERRORLEVEL 1 ( echo "Failed to run hello-two_sided" & GOTO:done )
    )
    if %endpoint% == B (
      hello_mp %endpoint% "SocketTcp -server -localIP=127.0.0.1 -ephemeralID=1" 0
      if ERRORLEVEL 1 ( echo "Failed to run hello-two_sided" & GOTO:done )
      hello_mp %endpoint% "SocketTcp -server -localIP=127.0.0.1 -ephemeralID=1" 10
      if ERRORLEVEL 1 ( echo "Failed to run hello-two_sided" & GOTO:done )
    )
  )
  if %endpoint% == A (
    rem sleep 1
    CHOICE /N /C YN /T 1 /D Y >NUL
    hello_mp %endpoint% "SocketUdpSend -unicast -remoteIP=127.0.0.1 -port=23456" 10
    if ERRORLEVEL 1 ( echo "Failed to run hello-two_sided" & GOTO:done )
  )
  if %endpoint% == B (
    hello_mp %endpoint% "SocketUdpRecv -unicast -localIP=127.0.0.1 -port=23456 -reuse" 10
    if ERRORLEVEL 1 ( echo "Failed to run hello-two_sided" & GOTO:done )
  )
  if %endpoint% == A (
    rem sleep 1
    CHOICE /N /C YN /T 1 /D Y >NUL
    hello_mp %endpoint% "SocketUdpSend -multicast -localIP=127.0.0.1 -groupIP=233.23.33.56 -port=23456" 10
    if ERRORLEVEL 1 ( echo "Failed to run hello-two_sided" & GOTO:done )
  )
  if %endpoint% == B (
    hello_mp %endpoint% "SocketUdpRecv -multicast -localIP=127.0.0.1 -groupIP=233.23.33.56 -port=23456 -reuse" 10
    if ERRORLEVEL 1 ( echo "Failed to run hello-two_sided" & GOTO:done )
  )
)

rem hello-one_sided
cd ..\hello-one_sided
if ERRORLEVEL 1 ( echo "Failed to cd ..\examples\hello-one_sided" & GOTO:done )
if %mmap% == yes (
  hello_mp %endpoint% "InterProcess -pathID=1" 0
  if ERRORLEVEL 1 ( echo "Failed to run hello-one_sided" & GOTO:done )
  hello_mp %endpoint% "InterProcess -pathID=2" 1
  if ERRORLEVEL 1 ( echo "Failed to run hello-one_sided" & GOTO:done )
  hello_mp %endpoint% "InterProcess -pathID=3" 10
  if ERRORLEVEL 1 ( echo "Failed to run hello-one_sided" & GOTO:done )
)

GOTO:done

:done
endlocal
GOTO:eof
