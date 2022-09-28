@echo off

rem Keep variables local to this script
setlocal EnableDelayedExpansion

set debug=no
set less_checking=no
set mmap=no
set cuda=no
set clean=no
set help=no

rem Parse args
for %%a in (%*) do (
  if %%a == debug set debug=yes
  if %%a == less_checking set less_checking=yes
  if %%a == mmap set mmap=yes
  if %%a == cuda set cuda=yes
  if %%a == clean set clean=yes
  if %%a == help set help=yes
)

echo "debug = %debug%"
echo "less_checking = %less_checking%"
echo "mmap  = %mmap%"
echo "cuda  = %cuda%"
echo "clean = %clean%"
echo "help = %help%"

if %help% == yes (
  echo "usage: buildExamples.bat [debug] [mmap] [cuda] [less_checking]"
  echo "       buildExamples.bat debug mmap cuda less_checking"
  echo "       buildExamples.bat clean"
  echo "       buildExamples.bat help"
  GOTO:done
)

if %clean% == yes (
  echo "Cleaning Takyon lib and examples..."

  rem lib
  cd ..\lib
  if ERRORLEVEL 1 ( echo "Failed to cd ..\lib" & GOTO:done )
  nmake -f windows.Makefile clean
  if ERRORLEVEL 1 ( echo "Failed to clean lib" & GOTO:done )

  rem hello-one_sided
  cd ..\examples\hello-one_sided
  if ERRORLEVEL 1 ( echo "Failed to cd ..\examples\hello-one_sided" & GOTO:done )
  nmake -f windows.Makefile clean
  if ERRORLEVEL 1 ( echo "Failed to clean hello-one_sided" & GOTO:done )

  rem hello-two_sided
  cd ..\hello-two_sided
  if ERRORLEVEL 1 ( echo "Failed to cd ..\hello-two_sided" & GOTO:done )
  nmake -f windows.Makefile clean
  if ERRORLEVEL 1 ( echo "Failed to clean hello-two_sided" & GOTO:done )

  rem throughput
  cd ..\throughput
  if ERRORLEVEL 1 ( echo "Failed to cd ..\throughput" & GOTO:done )
  nmake -f windows.Makefile clean
  if ERRORLEVEL 1 ( echo "Failed to clean throughput" & GOTO:done )

  echo "Done cleaning"
  GOTO:done
)

rem Set the make options
set options=
if %debug% == yes (
    set options=%options% DEBUG=Yes
)
if %less_checking% == yes (
    set options=%options% DisableExtraErrorChecking=Yes
)
if %cuda% == yes (
    set options=%options% CUDA=Yes
)
echo extra compile flags = %options%

rem Takyon library
cd ..\lib
if ERRORLEVEL 1 ( echo "Failed to cd ..\lib" & GOTO:done )
echo Cleaning first
nmake -f windows.Makefile clean
if ERRORLEVEL 1 ( echo "Failed to clean lib" & GOTO:done )
set make_command=nmake -f windows.Makefile %options% InterThread=Yes SocketTcp=Yes SocketUdp=Yes
if %mmap% == yes (
    set make_command=%make_command% InterProcess=Yes
)
echo Running: %make_command%
%make_command%
if ERRORLEVEL 1 ( echo "Failed to build lib" & GOTO:done )

rem hello-one_sided
cd ..\examples\hello-one_sided
if ERRORLEVEL 1 ( echo "Failed to cd ..\examples\hello-one_sided" & GOTO:done )
echo Cleaning first
nmake -f windows.Makefile clean
if ERRORLEVEL 1 ( echo "Failed to clean hello-one_sided" & GOTO:done )
set make_command=nmake -f windows.Makefile %options%
if %mmap% == yes (
    set make_command=%make_command% MMAP=Yes
)
echo Running: %make_command%
%make_command%
if ERRORLEVEL 1 ( echo "Failed to build hello-one_sided" & GOTO:done )

rem hello-two_sided
cd ..\hello-two_sided
if ERRORLEVEL 1 ( echo "Failed to cd ..\examples\hello-two_sided" & GOTO:done )
echo Cleaning first
nmake -f windows.Makefile clean
if ERRORLEVEL 1 ( echo "Failed to clean hello-two_sided" & GOTO:done )
set make_command=nmake -f windows.Makefile %options%
if %mmap% == yes (
    set make_command=%make_command% MMAP=Yes
)
echo Running: %make_command%
%make_command%
if ERRORLEVEL 1 ( echo "Failed to build hello-two_sided" & GOTO:done )

rem throughput
cd ..\throughput
if ERRORLEVEL 1 ( echo "Failed to cd ..\examples\throughput" & GOTO:done )
echo Cleaning first
nmake -f windows.Makefile clean
if ERRORLEVEL 1 ( echo "Failed to clean throughput" & GOTO:done )
set make_command=nmake -f windows.Makefile %options%
if %mmap% == yes (
    set make_command=%make_command% MMAP=Yes
)
echo Running: %make_command%
%make_command%
if ERRORLEVEL 1 ( echo "Failed to build throughput" & GOTO:done )

GOTO:done

:done
endlocal
GOTO:eof
