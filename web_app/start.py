#!/usr/bin/env python3
"""
一键启动脚本 - 自动启动后端和前端服务
"""

import subprocess
import time
import os
import sys
from pathlib import Path

def run_command(cmd, cwd=None, name="Task"):
    """运行命令并返回进程"""
    print(f"\n▶ 启动: {name}")
    print(f"  命令: {cmd}")
    if cwd:
        print(f"  目录: {cwd}")
    
    try:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True
        )
        print(f"✅ {name} 已启动 (PID: {process.pid})")
        return process
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return None

def main():
    print("\n" + "="*60)
    print("🏠 户型图3D模型转换系统 - 一键启动")
    print("="*60)
    
    web_app_dir = Path(__file__).parent
    backend_dir = web_app_dir / "backend"
    frontend_dir = web_app_dir / "frontend"
    
    # 第1步：检查环境
    print("\n[1/4] 检查环境...")
    
    # 激活mmenv环境
    activate_cmd = "conda activate mmenv && "
    
    # 检查Python
    result = subprocess.run(
        f"{activate_cmd}python --version",
        shell=True,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"✅ {result.stdout.strip()}")
    else:
        print("❌ Python环境激活失败")
        print("   请确保已安装Anaconda并存在mmenv环境")
        print("   运行: conda activate mmenv")
        input("按Enter退出...")
        return
    
    # 第2步：检查依赖
    print("\n[2/4] 检查和安装依赖...")
    
    deps_cmd = f'{activate_cmd}pip install -q flask flask-cors ultralytics torch opencv-python trimesh mapbox-earcut shapely 2>nul'
    result = subprocess.run(deps_cmd, shell=True)
    
    if result.returncode == 0:
        print("✅ 依赖检查完成")
    else:
        print("⚠️ 部分依赖安装可能失败，继续启动...")
    
    # 第3步：启动后端
    print("\n[3/4] 启动后端服务...")
    
    backend_cmd = f'{activate_cmd}cd "{backend_dir}" && python app.py'
    backend_process = run_command(
        backend_cmd,
        cwd=str(backend_dir),
        name="后端服务 (Flask, 端口5000)"
    )
    
    # 等待后端启动
    time.sleep(4)
    
    # 第4步：启动前端
    print("\n[4/4] 启动前端服务...")
    
    frontend_cmd = f'{activate_cmd}python -m http.server 8000'
    frontend_process = run_command(
        frontend_cmd,
        cwd=str(frontend_dir),
        name="前端服务 (HTTP, 端口8000)"
    )
    
    time.sleep(2)
    
    # 启动浏览器
    print("\n" + "="*60)
    print("✅ 启动完成!")
    print("="*60)
    print("\n📱 访问地址:")
    print("  🌐 Web应用: http://localhost:8000")
    print("  📡 API服务: http://localhost:5000/api")
    print("  🔍 健康检查: http://localhost:5000/api/health")
    print("\n💡 提示:")
    print("  - 后端日志会在第一个窗口显示")
    print("  - 前端日志会在第二个窗口显示")
    print("  - 按 Ctrl+C 停止服务")
    print("  - 浏览器将在3秒后打开...")
    print("\n" + "="*60 + "\n")
    
    # 打开浏览器
    time.sleep(3)
    try:
        import webbrowser
        webbrowser.open('http://localhost:8000')
        print("✅ 浏览器已打开")
    except:
        print("⚠️ 无法自动打开浏览器，请手动访问: http://localhost:8000")
    
    # 保持脚本运行
    try:
        if backend_process:
            backend_process.wait()
        if frontend_process:
            frontend_process.wait()
    except KeyboardInterrupt:
        print("\n\n⏹️ 正在关闭服务...")
        if backend_process:
            backend_process.terminate()
        if frontend_process:
            frontend_process.terminate()
        time.sleep(1)
        print("✅ 服务已关闭")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        input("\n按Enter退出...")
