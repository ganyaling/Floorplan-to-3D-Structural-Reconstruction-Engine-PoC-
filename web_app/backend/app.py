"""
户型图到3D模型转换 Web应用 - 后端
Flask API服务
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import sys
from pathlib import Path
import tempfile
import base64
from datetime import datetime

# 添加svg目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'svg'))
from floorplan_to_3d import FloorplanTo3D

app = Flask(__name__)
# 配置CORS允许所有来源
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "max_age": 3600
    }
})

# 配置
UPLOAD_FOLDER = Path(__file__).parent.parent / 'uploads'
OUTPUT_FOLDER = Path(__file__).parent.parent / 'outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# 创建必要的目录
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# 全局模型实例
converter = None
# 正确的模型路径
MODEL_PATH = Path(__file__).parent.parent.parent / 'attention' / 'floorplan_ai' / 'v1_cubicasa_base' / 'weights' / 'best.pt'

# 如果上面的路径不存在，尝试备用路径
if not MODEL_PATH.exists():
    print(f"⚠️ 首选路径不存在: {MODEL_PATH}")
    alt_paths = [
        Path(__file__).parent.parent.parent / 'svg' / 'floorplan_ai' / 'v1_cubicasa_base' / 'weights' / 'best.pt',
        Path('E:/JOB/attention/floorplan_ai/v1_cubicasa_base/weights/best.pt'),
        Path('E:\\JOB\\attention\\floorplan_ai\\v1_cubicasa_base\\weights\\best.pt'),
    ]
    for alt_path in alt_paths:
        if alt_path.exists():
            print(f"✅ 找到备用路径: {alt_path}")
            MODEL_PATH = alt_path
            break
    else:
        print("❌ 警告：未找到任何模型文件!")
        print("   请运行: python train.py --epochs 30 来训练模型")

def init_model():
    """初始化模型"""
    global converter
    if converter is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")
        print(f"正在加载模型: {MODEL_PATH}")
        converter = FloorplanTo3D(str(MODEL_PATH))
    return converter

def allowed_file(filename):
    """检查文件类型"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'model_loaded': converter is not None,
        'model_path': str(MODEL_PATH),
        'model_exists': MODEL_PATH.exists()
    })

@app.route('/api/init_model', methods=['POST'])
def init_model_route():
    """初始化模型"""
    try:
        init_model()
        return jsonify({
            'status': 'success',
            'message': '模型加载成功',
            'class_names': converter.class_names
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'模型加载失败: {str(e)}'
        }), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """上传图像文件"""
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': '没有上传文件'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'status': 'error', 'message': '文件名为空'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'status': 'error',
                'message': f'不支持的文件格式。支持: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # 保存文件
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        filepath = UPLOAD_FOLDER / filename
        
        file.save(str(filepath))
        
        return jsonify({
            'status': 'success',
            'message': '文件上传成功',
            'filename': filename,
            'filepath': str(filepath)
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """生成3D模型"""
    try:
        data = request.get_json()
        
        if not data or 'filename' not in data:
            return jsonify({'status': 'error', 'message': '缺少filename参数'}), 400
        
        filename = data.get('filename')
        conf = float(data.get('confidence', 0.3))
        scale = float(data.get('scale', 2.0))
        straighten = data.get('straighten', True)  # 新增：是否拉直
        
        # 验证文件存在
        filepath = UPLOAD_FOLDER / filename
        if not filepath.exists():
            return jsonify({'status': 'error', 'message': '文件不存在'}), 400
        
        # 初始化模型
        if converter is None:
            init_model()
        
        print(f"\n处理图像: {filename}")
        print(f"参数: conf={conf}, scale={scale}, straighten={straighten}")
        
        # 创建场景
        scene = converter.create_scene_from_image(
            image_path=str(filepath),
            conf=conf,
            straighten=straighten  # 传递拉直选项
        )
        
        if scene is None or len(scene.geometry) == 0:
            return jsonify({
                'status': 'error',
                'message': '未检测到房间或模型生成失败'
            }), 400
        
        # 保存GLB文件
        output_filename = filename.replace(
            filename.split('.')[-1],
            'glb'
        )
        output_path = OUTPUT_FOLDER / output_filename
        
        scene.export(str(output_path))
        
        # 计算统计信息
        bounds = scene.bounds
        size = bounds[1] - bounds[0]
        
        return jsonify({
            'status': 'success',
            'message': '3D模型生成成功',
            'output_filename': output_filename,
            'geometry_count': len(scene.geometry),
            'size': {
                'width': round(float(size[0]), 2),
                'depth': round(float(size[1]), 2),
                'height': round(float(size[2]), 2)
            },
            'download_url': f'/api/download/{output_filename}'
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'处理失败: {str(e)}'
        }), 500

@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """下载3D模型"""
    try:
        filepath = OUTPUT_FOLDER / secure_filename(filename)
        
        if not filepath.exists():
            return jsonify({'status': 'error', 'message': '文件不存在'}), 404
        
        return send_file(
            str(filepath),
            as_attachment=True,
            download_name=filename,
            mimetype='application/octet-stream'
        )
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/preview/<filename>', methods=['GET'])
def preview_image(filename):
    """预览上传的图像"""
    try:
        filepath = UPLOAD_FOLDER / secure_filename(filename)
        
        if not filepath.exists():
            return jsonify({'status': 'error', 'message': '文件不存在'}), 404
        
        return send_file(str(filepath), mimetype='image/jpeg')
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/parameters', methods=['GET'])
def get_parameters():
    """获取参数范围和默认值"""
    return jsonify({
        'confidence': {
            'min': 0.1,
            'max': 0.9,
            'default': 0.3,
            'step': 0.05,
            'description': '检测置信度阈值'
        },
        'scale': {
            'min': 1.0,
            'max': 5.0,
            'default': 2.0,
            'step': 0.5,
            'description': '比例尺 (厘米/像素)'
        }
    })

@app.errorhandler(404)
def not_found(e):
    return jsonify({'status': 'error', 'message': '接口不存在'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'status': 'error', 'message': '服务器错误'}), 500

if __name__ == '__main__':
    print("🚀 户型图3D模型转换 Web应用 - 后端服务")
    print("=" * 50)
    
    # 初始化模型
    try:
        init_model()
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"⚠️ 模型加载失败: {e}")
    
    # 启动Flask服务
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False  # GPU不支持多进程
    )
