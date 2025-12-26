# 🚀 Web应用快速开始指南

## Windows用户 - 最简单的方式

### 第1步：进入项目目录
```cmd
cd E:\JOB\attention\web_app
```

### 第2步：双击运行启动脚本
双击 **START.bat** 文件

系统会自动：
- ✅ 检查Python环境
- ✅ 安装必需的依赖包
- ✅ 启动Flask后端服务（端口5000）
- ✅ 启动前端Web服务（端口8000）  
- ✅ 打开浏览器访问应用

### 第3步：在浏览器中使用应用

访问：**http://localhost:8000**

---

## Linux / macOS 用户 - 手动启动

### 第1步：激活Python环境
```bash
conda activate mmenv
```

### 第2步：安装依赖（第一次运行）
```bash
cd web_app
pip install -r requirements.txt
```

### 第3步：启动后端服务
```bash
cd backend
python app.py
```

**输出应该显示：**
```
正在加载模型: E:\JOB\attention\floorplan_ai\v1_cubicasa_base\weights\best.pt
模型类别: {0: 'LivingRoom', 1: 'Bedroom', ...}
✅ 模型加载成功
 * Running on http://0.0.0.0:5000
```

### 第4步：在另一个终端启动前端
```bash
cd frontend
python -m http.server 8000
```

**输出应该显示：**
```
Serving HTTP on 0.0.0.0 port 8000
```

### 第5步：打开浏览器
访问：**http://localhost:8000**

---

## 📋 使用流程

### 1️⃣ 上传图像
- 点击上传区域或拖拽图像到页面
- 支持格式：PNG、JPG、BMP
- 最大大小：50MB

### 2️⃣ 调整参数
- **检测置信度**：0.1-0.9（推荐0.3）
  - 低值检测更多，可能误检
  - 高值检测更精准，可能漏检
- **比例尺**：1.0-5.0 厘米/像素（推荐2.0）
  - 用于准确计算房间尺寸

### 3️⃣ 生成3D模型
- 点击"⚡ 生成3D模型"按钮
- 等待处理完成（通常1-3分钟）
- 右侧3D查看器会显示生成的模型

### 4️⃣ 交互式3D预览
- **旋转**：鼠标左键拖拽
- **缩放**：鼠标滚轮或右键拖拽
- **平移**：鼠标中键或Ctrl+左键拖拽
- 自动旋转显示模型

### 5️⃣ 下载模型
- 查看房间数和尺寸统计
- 点击"⬇️ 下载GLB文件"保存3D模型
- 可在其他3D查看器（Unity、Blender等）中使用

---

## 🔍 检查服务状态

### 检查后端服务
打开浏览器访问：http://localhost:5000/api/health

**成功响应示例：**
```json
{
    "status": "ok",
    "model_loaded": true,
    "model_path": "E:\\JOB\\attention\\floorplan_ai\\v1_cubicasa_base\\weights\\best.pt",
    "model_exists": true
}
```

### 检查前端服务
打开浏览器访问：http://localhost:8000

应该看到漂亮的紫色主题Web界面

---

## ⚠️ 常见问题

### ❌ 问题：执行START.bat后没有反应
**解决方案：**
1. 右键选择"管理员运行"
2. 确保Python已添加到系统PATH
3. 打开cmd手动运行：`python --version` 检查

### ❌ 问题：模型加载失败
**错误信息：** `FileNotFoundError: 模型文件不存在`

**解决方案：**
```bash
# 检查模型文件是否存在
dir "E:\JOB\attention\floorplan_ai\v1_cubicasa_base\weights\"
```

如果不存在，需要先运行训练：
```bash
cd E:\JOB\attention\svg
python train.py --epochs 30
```

### ❌ 问题：无法连接到http://localhost:5000
**解决方案：**
- 确保后端服务正在运行（应该看到"Running on http://0.0.0.0:5000"）
- 检查是否有其他应用占用了5000端口
- 改用替代端口：编辑backend/app.py最后一行

### ❌ 问题：上传文件后没有反应
**解决方案：**
1. 打开浏览器开发者工具（F12）
2. 查看Network标签看请求是否成功
3. 查看Console标签看是否有错误信息
4. 确保后端服务正在运行

### ❌ 问题：生成3D模型失败
**常见原因：**
- 图像中找不到房间（尝试降低置信度到0.2）
- 模型精度不足（考虑重新训练更多epoch）
- GPU内存不足（减小batch size或使用CPU）

### ❌ 问题：3D查看器显示为空
**解决方案：**
- 检查浏览器是否支持WebGL：访问 http://get.webgl.org/
- 更新显卡驱动
- 尝试更换浏览器（Chrome推荐）
- 清除浏览器缓存（Ctrl+Shift+Delete）

---

## 🎯 参数调试建议

### 场景1：检测不到房间
```
当前参数：confidence=0.5, scale=2.0
问题：模型太严格，漏检了房间
解决：降低confidence到0.2-0.3
```

### 场景2：检测结果有误
```
当前参数：confidence=0.1, scale=2.0
问题：检测过于敏感，把墙体当成房间
解决：提高confidence到0.4-0.5
```

### 场景3：尺寸不准确
```
问题：生成的模型尺寸与实际不符
解决：调整scale参数
      - 如果模型太小，增加scale值
      - 如果模型太大，减小scale值
```

---

## 📊 性能参考

### 处理时间
- **模型加载**：首次启动 5-10秒
- **单张图像推理**：1-2秒（GPU）
- **3D模型生成**：0.5-1秒
- **总耗时**：约2-4秒

### 系统要求最低配置
- CPU：Intel i5 或同级
- RAM：8GB
- GPU：GTX 1050 或更好
- 存储：5GB空闲空间

### 推荐配置
- CPU：Intel i7/AMD Ryzen 7
- RAM：16GB
- GPU：RTX 3060 或更好
- 存储：10GB SSD

---

## 🔐 安全性建议

### 部署到局域网
编辑 `backend/app.py` 最后一行：
```python
# 改为绑定本机IP，如192.168.1.100
app.run(host='192.168.1.100', port=5000)
```

### 添加认证（可选）
使用Flask-HTTPAuth添加用户认证：
```bash
pip install flask-httpauth
```

### 限制文件大小和上传目录
已在app.py中配置：
- 最大文件大小：50MB
- 上传目录隔离：web_app/uploads/
- 输出目录隔离：web_app/outputs/

---

## 📞 获取帮助

### 查看日志
后端服务的详细日志会输出到终端，包含：
- 模型加载状态
- 每次预测的详细信息
- 生成的房间数量和类别
- 错误堆栈信息

### 收集诊断信息
如需报告问题，请收集：
1. 上传的图像文件
2. 使用的参数值
3. 完整的错误信息
4. 后端服务的日志输出
5. 浏览器开发者工具的Console输出

---

## ✅ 成功标志

- ✅ START.bat 弹出两个终端窗口（Flask和HTTP服务器）
- ✅ 浏览器自动打开 http://localhost:8000
- ✅ 页面显示紫色主题的Web界面
- ✅ 上传区域可以正常拖拽和点击
- ✅ 能够成功上传图像并看到预览
- ✅ 生成3D模型后在右侧显示

---

**祝你使用愉快！** 🎉
