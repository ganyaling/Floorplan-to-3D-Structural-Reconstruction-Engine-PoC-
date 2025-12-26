from ultralytics import YOLO
from pathlib import Path

def train_model(epochs=30, batch=8):
    """
    训练YOLOv8实例分割模型
    
    Args:
        epochs: 训练轮数（默认30，建议30-50用于高精度）
        batch: 批处理大小（默认8）
    """
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    yaml_path = script_dir / 'floorplan.yaml'
    
    # 1. 加载预训练模型
    # yolov8n-seg.pt (Nano): 速度最快，精度一般 (适合手机端)
    # yolov8m-seg.pt (Medium): 平衡性能
    # yolov8x-seg.pt (Xlarge): 精度最高，算力要求高 (适合高精度3D模型生成)
    print("正在加载预训练模型...")
    model = YOLO('yolov8m-seg.pt') 

    # 2. 开始训练
    print(f"开始训练... (epochs={epochs}, batch={batch})")
    results = model.train(
        data=str(yaml_path),        # 使用绝对路径
        epochs=epochs,              # 训练轮数 (30+ 用于更高精度)
        imgsz=640,                  # 输入图片大小 (户型图细节多)
        batch=batch,                # 显存不够就调小
        device=0,                   # 使用 GPU (0)
        workers=0,
        project='floorplan_ai',     # 项目名称
        name='v1_cubicasa_base',    # 训练任务名称
        patience=30,                # 早停机制容忍度
        exist_ok=True
    )
    
    print(f"✅ 训练完成！模型保存在 floorplan_ai/v1_cubicasa_base/weights/best.pt")
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='训练YOLOv8实例分割模型')
    parser.add_argument('--epochs', type=int, default=30,
                       help='训练轮数 (推荐 30-50 用于更高精度)')
    parser.add_argument('--batch', type=int, default=8,
                       help='批处理大小 (显存不够调小)')
    
    args = parser.parse_args()
    
    train_model(epochs=args.epochs, batch=args.batch)