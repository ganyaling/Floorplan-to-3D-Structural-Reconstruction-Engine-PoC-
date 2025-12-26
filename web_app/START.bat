@echo off
setlocal enabledelayedexpansion

REM ==========================================
REM   户型图3D模型转换系统 - 启动脚本
REM ==========================================

chcp 65001 >nul
cd /d "%~dp0"

REM 激活conda环境
call conda activate mmenv

REM 运行Python启动脚本
echo.
echo 正在启动Web应用...
echo.

python start.py

pause
    echo.
    echo ==========================================
    echo ✅ 服务启动完成！
    echo ==========================================
    echo.
    echo 🌐 Web应用: http://localhost:8000
    echo 📡 API服务: http://localhost:5000/api
    echo.
    
    start "" http://localhost:8000
    pause
)

