@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set BASE_COMMANDLINE_ARGS=--xformers --api
set FORGE_PORT=
for /f %%P in ('powershell -NoProfile -Command "$ports=7861..7870; foreach ($p in $ports) { if (-not (Get-NetTCPConnection -State Listen -LocalPort $p -ErrorAction SilentlyContinue)) { Write-Output $p; break } }"') do set FORGE_PORT=%%P
if not defined FORGE_PORT set FORGE_PORT=7861
set COMMANDLINE_ARGS=%BASE_COMMANDLINE_ARGS% --port %FORGE_PORT%
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

call webui.bat
