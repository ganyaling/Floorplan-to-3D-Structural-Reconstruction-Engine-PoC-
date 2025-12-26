# 户型图3D模型转换 Web应用

一个强大的web应用，用于将2D户型图快速转换为交互式3D模型。

## 📋 功能特性

- ✅ **在线上传**：支持拖拽和点击上传户型图
- ✅ **智能检测**：使用YOLOv8深度学习模型自动识别房间
- ✅ **3D可视化**：实时渲染和交互式3D模型查看
- ✅ **参数调整**：灵活的置信度和比例尺设置
- ✅ **批量下载**：生成的GLB模型可下载
- ✅ **响应式设计**：支持桌面和移动设备

## 🚀 快速开始

### 方式1：一键启动（Windows）

双击运行 `START.bat` 文件：

```bash
START.bat
```

系统会自动：
1. ✅ 检查Python环境
2. ✅ 安装必需依赖
3. ✅ 启动Flask后端（端口5000）
4. ✅ 启动HTTP前端服务（端口8000）
5. ✅ 打开浏览器访问应用

### 方式2：手动启动

#### 前提条件

```bash
# 激活conda环境
conda activate mmenv

# 安装依赖
pip install flask flask-cors trimesh mapbox-earcut
```

#### 启动后端服务

```bash
cd backend
python app.py
```

后端服务会在 `http://localhost:5000` 启动

#### 启动前端服务

```bash
cd frontend
python -m http.server 8000
```

访问 `http://localhost:8000` 打开应用

## 📡 API文档

### 1. 健康检查
```
GET /api/health
```
检查服务状态和模型加载情况。

**响应：**
```json
{
    "status": "ok",
    "model_loaded": true,
    "model_path": "...",
    "model_exists": true
}
```

### 2. 上传图像
```
POST /api/upload
```
上传户型图文件。

**参数：**
- `file`: 图像文件（PNG/JPG/BMP，最大50MB）

**响应：**
```json
{
    "status": "success",
    "message": "文件上传成功",
    "filename": "20250101_120000_floor.png",
    "filepath": "..."
}
```

### 3. 生成3D模型
```
POST /api/predict
```
根据上传的图像生成3D模型。

**请求体：**
```json
{
    "filename": "20250101_120000_floor.png",
    "confidence": 0.3,
    "scale": 2.0
}
```

**参数说明：**
- `filename`: 已上传的图像文件名
- `confidence`: 检测置信度（0.1-0.9，默认0.3）
- `scale`: 比例尺厘米/像素（1.0-5.0，默认2.0）

**响应：**
```json
{
    "status": "success",
    "message": "3D模型生成成功",
    "output_filename": "floor.glb",
    "geometry_count": 27,
    "size": {
        "width": 15.76,
        "depth": 9.18,
        "height": 0.00
    },
    "download_url": "/api/download/floor.glb"
}
```

### 4. 下载模型
```
GET /api/download/<filename>
```
下载生成的GLB格式3D模型。

### 5. 获取参数范围
```
GET /api/parameters
```
获取参数的有效范围和默认值。

## 🎛️ 参数说明

### 检测置信度 (Confidence)
- **范围**：0.1 - 0.9
- **默认**：0.3
- **说明**：
  - **低值** (0.1-0.3)：检测更多房间，可能误检
  - **中值** (0.3-0.5)：平衡检测和精准度
  - **高值** (0.5-0.9)：仅检测高置信度房间，可能漏检

### 比例尺 (Scale)
- **范围**：1.0 - 5.0
- **默认**：2.0
- **单位**：厘米/像素
- **说明**：用于准确计算房间的实际物理尺寸

## 🎯 使用流程

1. **上传图像**
   - 点击上传区域或拖拽图像到页面
   - 支持PNG、JPG、BMP格式

2. **调整参数**
   - 根据需要调整置信度和比例尺
   - 实时显示参数值

3. **生成模型**
   - 点击"生成3D模型"按钮
   - 等待处理完成（通常1-5分钟）

4. **查看效果**
   - 3D查看器中预览模型
   - 用鼠标旋转、缩放、平移视图
   - 查看房间数和尺寸统计

5. **下载模型**
   - 点击"下载GLB文件"按钮
   - 保存模型供后续使用

## 📊 模型详情

### YOLOv8m-seg 实例分割模型
- **训练数据**：CubiCasa5K（4000+张户型图）
- **识别类别**：11种房间类型
  - 卧室 (Bedroom)
  - 客厅 (LivingRoom)
  - 厨房 (Kitchen)
  - 卫浴 (Bath)
  - 入户 (Entry)
  - 储物 (Storage)
  - 车库 (Garage)
  - 室外 (Outdoor)
  - 通用房间 (Room)
  - 墙体 (Wall)
  - 栏杆 (Railing)

- **精度指标**：
  - mAP@0.5: 0.762
  - mAP@50-95: 0.673
  - 推理速度：约100ms/张图

### 3D渲染引擎
- **框架**：Three.js (Web GL)
- **支持**：
  - 实时交互式旋转、缩放、平移
  - 自动光照和阴影
  - 高质量模型导出（GLB格式）

## 🔧 系统要求

- **操作系统**：Windows / Linux / macOS
- **Python**：3.8+
- **GPU**：NVIDIA GPU（推荐）或CPU
- **内存**：8GB+
- **浏览器**：Chrome、Firefox、Safari（支持WebGL）

## 📦 依赖包

```
flask>=2.0.0
flask-cors>=3.0.10
ultralytics>=8.0.0
trimesh>=3.20.0
mapbox-earcut>=0.12.0
torch>=1.9.0
opencv-python>=4.5.0
numpy>=1.21.0
shapely>=1.7.0
```

## 🐛 常见问题

### Q: 上传后没有反应
A: 检查浏览器控制台（F12）是否有错误信息，确保后端服务（端口5000）正常运行。

### Q: 生成模型失败
A: 
- 检查模型文件是否存在：`floorplan_ai/v1_cubicasa_base/weights/best.pt`
- 尝试降低置信度（0.2-0.3）
- 检查图像格式和大小

### Q: 3D模型加载不了
A: 
- 确保浏览器支持WebGL（打开http://get.webgl.org/检查）
- 清除浏览器缓存
- 尝试使用Chrome浏览器

### Q: 启动脚本失败
A: 
- 确保Python已添加到系统PATH
- 手动激活mmenv环境后再运行
- 检查权限（右键管理员运行）

## 📝 日志和调试

后端服务的日志会输出到控制台。查看日志可以帮助诊断问题：

```
正在加载模型: ...
模型类别: {0: 'LivingRoom', ...}
正在预测图像: ...
✅ 检测到 22 个对象
正在构建3D模型...
✅ 3D场景构建完成，包含 27 个对象
```

## 📈 性能优化建议

1. **增强模型精度**
   ```bash
   python train.py --epochs 50 --batch 8
   ```
   训练更多epoch获得更高精度（约12-15小时）

2. **并行处理**
   - 使用FastAPI替代Flask获得更好的并发性能
   - 配置Gunicorn实现多进程

3. **GPU加速**
   - 确保CUDA环境正确配置
   - 使用更强大的GPU型号

## 🚀 部署到生产环境

### Docker部署

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000 8000

CMD ["bash", "-c", "python backend/app.py & python -m http.server 8000 --directory frontend"]
```

### 云端部署（AWS/Azure/Google Cloud）

```bash
# 打包应用
zip -r app.zip .

# 上传到云服务
# ... 具体步骤取决于平台
```

## 🤝 反馈和支持

如有问题或建议，请：
1. 检查常见问题部分
2. 查看后端日志输出
3. 联系开发团队

## 📄 许可证

内部使用

---

**最后更新**：2025年1月
