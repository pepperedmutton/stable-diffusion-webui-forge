@echo off

cd /d "%~dp0"

set PYTHON=
if exist "%~dp0venv\Scripts\python.exe" set PYTHON="%~dp0venv\Scripts\python.exe"
set GIT=
set VENV_DIR=
set BASE_COMMANDLINE_ARGS=--xformers --api
set FORGE_PORT=
set "FORGE_PORTS_FILE=%TEMP%\forge_ports_%RANDOM%_%RANDOM%.txt"
netstat -ano -p tcp > "%FORGE_PORTS_FILE%" 2>nul
for /L %%P in (7861,1,7870) do (
    if not defined FORGE_PORT (
        %SystemRoot%\System32\findstr.exe /R /C:":%%P .*LISTENING" "%FORGE_PORTS_FILE%" >nul
        if errorlevel 1 set FORGE_PORT=%%P
    )
)
del "%FORGE_PORTS_FILE%" >nul 2>&1
if not defined FORGE_PORT set FORGE_PORT=7861
set "COMMANDLINE_ARGS=%BASE_COMMANDLINE_ARGS% --port %FORGE_PORT% %*"
echo Using Forge port: %FORGE_PORT%


@REM Uncomment following code to reference an existing A1111 checkout.
@REM set A1111_HOME=Your A1111 checkout dir
@REM
@REM set VENV_DIR=%A1111_HOME%/venv
@REM set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% ^
@REM  --ckpt-dir %A1111_HOME%/models/Stable-diffusion ^
@REM  --hypernetwork-dir %A1111_HOME%/models/hypernetworks ^
@REM  --embeddings-dir %A1111_HOME%/embeddings ^
@REM  --lora-dir %A1111_HOME%/models/Lora

call "%~dp0webui.bat"
