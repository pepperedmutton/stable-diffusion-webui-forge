@echo off
cd /d "%~dp0"
echo 正在启动 Forge...
echo 当前目录: %CD%
echo 端口: 7870
echo API标志: --api
echo 监听标志: --listen
echo.
call "%~dp0webui-user.bat" --api --port 7870 --listen
