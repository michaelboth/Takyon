@echo off

rem Keep variables local to this script
setlocal EnableDelayedExpansion

rem Count args
set arg_count=0
for %%i in (%*) do set /A arg_count+=1
if "%arg_count%" lss "3" (
  echo "USAGE: runInterProcess.bat <A|B> <local_ip> <remote_ip> [socket] [ephemeral] [multicast]"
  GOTO:done
)

set endpoint=%1
set local_ip=%2
set remote_ip=%3
set socket=no
set ephemeral=no
set multicast=no

rem Parse args
for %%a in (%*) do (
  if %%a == socket set socket=yes
  if %%a == ephemeral set ephemeral=yes
  if %%a == multicast set multicast=yes
)

if %endpoint% == no_value (
  echo "USAGE: runInterProcess.bat <A|B> <local_ip> <remote_ip> [socket] [ephemeral] [multicast]"
  GOTO:done
)

echo "endpoint  = %endpoint%"
echo "local_ip  = %local_ip%"
echo "remote_ip = %remote_ip%"
echo "socket    = %socket%"
echo "ephemeral = %ephemeral%"
echo "multicast = %multicast%"

rem hello-two_sided
cd ..\examples\hello-two_sided
if ERRORLEVEL 1 ( echo "Failed to cd ..\examples\hello-two_sided" & GOTO:done )
if %socket% == yes (
  if %endpoint% == A (
    hello_mp %endpoint% "SocketTcp -client -remoteIP=%remote_ip% -port=23456" 0
    if ERRORLEVEL 1 ( echo "Failed to run hello-two_sided" & GOTO:done )
    hello_mp %endpoint% "SocketTcp -client -remoteIP=%remote_ip% -port=23457" 10
    if ERRORLEVEL 1 ( echo "Failed to run hello-two_sided" & GOTO:done )
  )
  if %endpoint% == B (
    hello_mp %endpoint% "SocketTcp -server -localIP=%local_ip% -port=23456 -reuse" 0
    if ERRORLEVEL 1 ( echo "Failed to run hello-two_sided" & GOTO:done )
    hello_mp %endpoint% "SocketTcp -server -localIP=%local_ip% -port=23457 -reuse" 10
    if ERRORLEVEL 1 ( echo "Failed to run hello-two_sided" & GOTO:done )
  )
  if %ephemeral% == yes (
    if %endpoint% == A (
      hello_mp %endpoint% "SocketTcp -client -remoteIP=%remote_ip% -ephemeralID=1" 0
      if ERRORLEVEL 1 ( echo "Failed to run hello-two_sided" & GOTO:done )
      hello_mp %endpoint% "SocketTcp -client -remoteIP=%remote_ip% -ephemeralID=2" 10
      if ERRORLEVEL 1 ( echo "Failed to run hello-two_sided" & GOTO:done )
    )
    if %endpoint% == B (
      hello_mp %endpoint% "SocketTcp -server -localIP=%local_ip% -ephemeralID=1" 0
      if ERRORLEVEL 1 ( echo "Failed to run hello-two_sided" & GOTO:done )
      hello_mp %endpoint% "SocketTcp -server -localIP=%local_ip% -ephemeralID=2" 10
      if ERRORLEVEL 1 ( echo "Failed to run hello-two_sided" & GOTO:done )
    )
  )
  if %endpoint% == A (
    rem sleep 1
    CHOICE /N /C YN /T 1 /D Y >NUL
    hello_mp %endpoint% "SocketUdpSend -unicast -remoteIP=%remote_ip% -port=23458" 50
    if ERRORLEVEL 1 ( echo "Failed to run hello-two_sided" & GOTO:done )
  )
  if %endpoint% == B (
    hello_mp %endpoint% "SocketUdpRecv -unicast -localIP=%local_ip% -port=23458 -reuse" 50
    if ERRORLEVEL 1 ( echo "Failed to run hello-two_sided" & GOTO:done )
  )
  if %multicast% == yes (
    if %endpoint% == A (
      rem sleep 1
      CHOICE /N /C YN /T 1 /D Y >NUL
      hello_mp %endpoint% "SocketUdpSend -multicast -localIP=%local_ip% -groupIP=233.23.33.56 -port=23459" 50
      if ERRORLEVEL 1 ( echo "Failed to run hello-two_sided" & GOTO:done )
    )
    if %endpoint% == B (
      hello_mp %endpoint% "SocketUdpRecv -multicast -localIP=%local_ip% -groupIP=233.23.33.56 -port=23459 -reuse" 50
      if ERRORLEVEL 1 ( echo "Failed to run hello-two_sided" & GOTO:done )
    )
  )
)

rem hello-one_sided
cd ..\hello-one_sided
if ERRORLEVEL 1 ( echo "Failed to cd ..\examples\hello-one_sided" & GOTO:done )

rem throughput
cd ..\throughput
if ERRORLEVEL 1 ( echo "Failed to cd ..\examples\throughput" & GOTO:done )
if %socket% == yes (
  if %endpoint% == A (
    throughput_mp %endpoint% "SocketTcp -client -remoteIP=%remote_ip% -port=23456" -n=10000 -b=1024 -v
    if ERRORLEVEL 1 ( echo "Failed to run throughput" & GOTO:done )
    throughput_mp %endpoint% "SocketTcp -client -remoteIP=%remote_ip% -port=23457" -n=10000 -b=1024
    if ERRORLEVEL 1 ( echo "Failed to run throughput" & GOTO:done )
    throughput_mp %endpoint% "SocketTcp -client -remoteIP=%remote_ip% -port=23458" -n=10000 -b=1024 -v -e
    if ERRORLEVEL 1 ( echo "Failed to run throughput" & GOTO:done )
    throughput_mp %endpoint% "SocketTcp -client -remoteIP=%remote_ip% -port=23458" -n=10000 -b=1024 -e
    if ERRORLEVEL 1 ( echo "Failed to run throughput" & GOTO:done )
  )
  if %endpoint% == B (
    throughput_mp %endpoint% "SocketTcp -server -localIP=%local_ip% -port=23456 -reuse" -n=10000 -b=1024 -v
    if ERRORLEVEL 1 ( echo "Failed to run throughput" & GOTO:done )
    throughput_mp %endpoint% "SocketTcp -server -localIP=%local_ip% -port=23457 -reuse" -n=10000 -b=1024
    if ERRORLEVEL 1 ( echo "Failed to run throughput" & GOTO:done )
    throughput_mp %endpoint% "SocketTcp -server -localIP=%local_ip% -port=23458 -reuse" -n=10000 -b=1024 -v -e
    if ERRORLEVEL 1 ( echo "Failed to run throughput" & GOTO:done )
    throughput_mp %endpoint% "SocketTcp -server -localIP=%local_ip% -port=23458 -reuse" -n=10000 -b=1024 -e
    if ERRORLEVEL 1 ( echo "Failed to run throughput" & GOTO:done )
  )

  if %ephemeral% == yes (
    if %endpoint% == A (
      throughput_mp %endpoint% "SocketTcp -client -remoteIP=%remote_ip% -ephemeralID=1" -n=10000 -b=1024 -v
      if ERRORLEVEL 1 ( echo "Failed to run throughput" & GOTO:done )
      throughput_mp %endpoint% "SocketTcp -client -remoteIP=%remote_ip% -ephemeralID=2" -n=10000 -b=1024
      if ERRORLEVEL 1 ( echo "Failed to run throughput" & GOTO:done )
      throughput_mp %endpoint% "SocketTcp -client -remoteIP=%remote_ip% -ephemeralID=3" -n=10000 -b=1024 -v -e
      if ERRORLEVEL 1 ( echo "Failed to run throughput" & GOTO:done )
      throughput_mp %endpoint% "SocketTcp -client -remoteIP=%remote_ip% -ephemeralID=3" -n=10000 -b=1024 -e
      if ERRORLEVEL 1 ( echo "Failed to run throughput" & GOTO:done )
    )
    if %endpoint% == B (
      throughput_mp %endpoint% "SocketTcp -server -localIP=%local_ip% -ephemeralID=1" -n=10000 -b=1024 -v
      if ERRORLEVEL 1 ( echo "Failed to run throughput" & GOTO:done )
      throughput_mp %endpoint% "SocketTcp -server -localIP=%local_ip% -ephemeralID=2" -n=10000 -b=1024
      if ERRORLEVEL 1 ( echo "Failed to run throughput" & GOTO:done )
      throughput_mp %endpoint% "SocketTcp -server -localIP=%local_ip% -ephemeralID=3" -n=10000 -b=1024 -v -e
      if ERRORLEVEL 1 ( echo "Failed to run throughput" & GOTO:done )
      throughput_mp %endpoint% "SocketTcp -server -localIP=%local_ip% -ephemeralID=3" -n=10000 -b=1024 -e
      if ERRORLEVEL 1 ( echo "Failed to run throughput" & GOTO:done )
    )
  )

  if %endpoint% == A (
    rem sleep 1
    CHOICE /N /C YN /T 1 /D Y >NUL
    throughput_mp %endpoint% "SocketUdpSend -unicast -remoteIP=%remote_ip% -port=23456" -n=10000 -b=1024 -v
    if ERRORLEVEL 1 ( echo "Failed to run throughput" & GOTO:done )
    rem sleep 1
    CHOICE /N /C YN /T 1 /D Y >NUL
    throughput_mp %endpoint% "SocketUdpSend -unicast -remoteIP=%remote_ip% -port=23457" -n=10000 -b=1024
    if ERRORLEVEL 1 ( echo "Failed to run throughput" & GOTO:done )
    rem sleep 1
    CHOICE /N /C YN /T 1 /D Y >NUL
    throughput_mp %endpoint% "SocketUdpSend -unicast -remoteIP=%remote_ip% -port=23458" -n=10000 -b=1024 -v -e
    if ERRORLEVEL 1 ( echo "Failed to run throughput" & GOTO:done )
    rem sleep 1
    CHOICE /N /C YN /T 1 /D Y >NUL
    throughput_mp %endpoint% "SocketUdpSend -unicast -remoteIP=%remote_ip% -port=23458" -n=10000 -b=1024 -e
    if ERRORLEVEL 1 ( echo "Failed to run throughput" & GOTO:done )
  )
  if %endpoint% == B (
    throughput_mp %endpoint% "SocketUdpRecv -unicast -localIP=%local_ip% -port=23456 -reuse" -n=10000 -b=1024 -v
    if ERRORLEVEL 1 ( echo "Failed to run throughput" & GOTO:done )
    throughput_mp %endpoint% "SocketUdpRecv -unicast -localIP=%local_ip% -port=23457 -reuse" -n=10000 -b=1024
    if ERRORLEVEL 1 ( echo "Failed to run throughput" & GOTO:done )
    throughput_mp %endpoint% "SocketUdpRecv -unicast -localIP=%local_ip% -port=23458 -reuse" -n=10000 -b=1024 -v -e
    if ERRORLEVEL 1 ( echo "Failed to run throughput" & GOTO:done )
    throughput_mp %endpoint% "SocketUdpRecv -unicast -localIP=%local_ip% -port=23458 -reuse" -n=10000 -b=1024 -e
    if ERRORLEVEL 1 ( echo "Failed to run throughput" & GOTO:done )
  )

  if %multicast% == yes (
    if %endpoint% == A (
      rem sleep 1
      CHOICE /N /C YN /T 1 /D Y >NUL
      throughput_mp %endpoint% "SocketUdpSend -multicast -localIP=%local_ip% -groupIP=233.23.33.56 -port=23456" -n=10000 -b=1024 -v
      if ERRORLEVEL 1 ( echo "Failed to run throughput" & GOTO:done )
      rem sleep 1
      CHOICE /N /C YN /T 1 /D Y >NUL
      throughput_mp %endpoint% "SocketUdpSend -multicast -localIP=%local_ip% -groupIP=233.23.33.57 -port=23457" -n=10000 -b=1024
      if ERRORLEVEL 1 ( echo "Failed to run throughput" & GOTO:done )
      rem sleep 1
      CHOICE /N /C YN /T 1 /D Y >NUL
      throughput_mp %endpoint% "SocketUdpSend -multicast -localIP=%local_ip% -groupIP=233.23.33.58 -port=23458" -n=10000 -b=1024 -v -e
      if ERRORLEVEL 1 ( echo "Failed to run throughput" & GOTO:done )
      rem sleep 1
      CHOICE /N /C YN /T 1 /D Y >NUL
      throughput_mp %endpoint% "SocketUdpSend -multicast -localIP=%local_ip% -groupIP=233.23.33.58 -port=23458" -n=10000 -b=1024 -e
      if ERRORLEVEL 1 ( echo "Failed to run throughput" & GOTO:done )
    )
    if %endpoint% == B (
      throughput_mp %endpoint% "SocketUdpRecv -multicast -localIP=%local_ip% -groupIP=233.23.33.56 -port=23456 -reuse" -n=10000 -b=1024 -v
      if ERRORLEVEL 1 ( echo "Failed to run throughput" & GOTO:done )
      throughput_mp %endpoint% "SocketUdpRecv -multicast -localIP=%local_ip% -groupIP=233.23.33.57 -port=23457 -reuse" -n=10000 -b=1024
      if ERRORLEVEL 1 ( echo "Failed to run throughput" & GOTO:done )
      throughput_mp %endpoint% "SocketUdpRecv -multicast -localIP=%local_ip% -groupIP=233.23.33.58 -port=23458 -reuse" -n=10000 -b=1024 -v -e
      if ERRORLEVEL 1 ( echo "Failed to run throughput" & GOTO:done )
      throughput_mp %endpoint% "SocketUdpRecv -multicast -localIP=%local_ip% -groupIP=233.23.33.58 -port=23458 -reuse" -n=10000 -b=1024 -e
      if ERRORLEVEL 1 ( echo "Failed to run throughput" & GOTO:done )
    )
  )
)

GOTO:done

:done
endlocal
GOTO:eof
