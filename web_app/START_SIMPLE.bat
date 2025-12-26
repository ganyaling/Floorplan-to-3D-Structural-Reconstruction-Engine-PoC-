@echo off
REM 简单启动脚本 - 直接使用mmenv Python

echo.
echo ==========================================
echo   户型图3D模型转换系统 - 启动中
echo ==========================================
echo.

REM 使用mmenv的Python直接运行start.py
set PYTHON_EXE=E:\Anaconda\envs\mmenv\python.exe

REM 检查Python是否存在
if not exist "%PYTHON_EXE%" (
    echo 错误: 找不到 mmenv Python
    echo 期望路径: %PYTHON_EXE%
    echo.
    echo 请确保已安装 Anaconda 并创建了 mmenv 环境
    echo 创建环境: conda create -n mmenv python=3.10
    pause
    exit /b 1
)

REM 运行启动脚本
echo ✅ 使用 mmenv Python 启动...
echo.

"%PYTHON_EXE%" "%~dp0start.py"

pause
