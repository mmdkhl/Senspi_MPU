@echo off
setlocal EnableExtensions DisableDelayedExpansion

set "PUTTY=C:\Program Files\PuTTY"
set "PSCP=%PUTTY%\pscp.exe"
set "PLINK=%PUTTY%\plink.exe"

rem PUBLIC TEMPLATE FILE
rem Copy this to deploy_pi.local.bat and fill in your real machine + Pi values.

set "LOCAL_ROOT=C:\Projects\sense-pi-MPU"
set "PI_USER=pi"
set "PI_HOST=192.168.0.100"
rem The 4-sensor scripts live in their own folder beside any old 3-sensor
rem install (see raspberrypi_scripts_4_sensor/README_rpi.md), never over it.
set "REMOTE_DIR=/home/pi/sensor4"
set "PI_PASS=change-me"
set "AUTH=-batch -noagent -pw %PI_PASS%"

if /I not "%REMOTE_DIR%"=="/home/pi/sensor4" (echo Refusing to wipe %REMOTE_DIR% & exit /b 1)

"%PLINK%" %AUTH% -ssh %PI_USER%@%PI_HOST% "mkdir -p %REMOTE_DIR% && rm -rf %REMOTE_DIR%/* %REMOTE_DIR%/.[!.]* %REMOTE_DIR%/..?* && mkdir -p %REMOTE_DIR%/sensepi/config"
if errorlevel 1 exit /b 1

"%PSCP%" %AUTH% -r "%LOCAL_ROOT%\raspberrypi_scripts_4_sensor\*" %PI_USER%@%PI_HOST%:%REMOTE_DIR%/
if errorlevel 1 exit /b 1

rem Only the Python modules: the Pi imports sensepi.config.log_paths and reads
rem no YAML from here. Copying the whole folder also sent hosts.local.yaml,
rem i.e. the SSH password of every configured Pi, to each Pi.
"%PSCP%" %AUTH% "%LOCAL_ROOT%\src\sensepi\config\*.py" %PI_USER%@%PI_HOST%:%REMOTE_DIR%/sensepi/config/
if errorlevel 1 exit /b 1

"%PSCP%" %AUTH% "%LOCAL_ROOT%\src\sensepi\__init__.py" %PI_USER%@%PI_HOST%:%REMOTE_DIR%/sensepi/__init__.py
if errorlevel 1 exit /b 1

echo DONE
endlocal
