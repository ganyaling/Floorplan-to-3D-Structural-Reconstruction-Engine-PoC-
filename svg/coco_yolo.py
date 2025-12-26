import json
import os
import shutil
from pathlib import Path
import numpy as np
from tqdm import tqdm

def convert_coco_json_to_yolo(json_file, output_dir, use_segmentation=True, data_root=None):
    """
    将 COCO JSON 转换为 YOLO 格式 (txt)
    output_dir 结构:
        /images
        /labels
    
    Args:
        json_file: COCO JSON 文件路径
        output_dir: 输出目录
        use_segmentation: 是否使用实例分割格式
        data_root: 数据集根目录（用于定位原始图片）
    """
    print(f"正在加载: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 创建输出目录
    output_path = Path(output_dir)
    images_dir = output_path / 'images'
    labels_dir = output_path / 'labels'
    
    # 如果存在则清理重建 (慎用，防止误删)
    if output_path.exists():
        shutil.rmtree(output_path)
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # 建立 Image ID 到 File Name 的映射
    img_id_map = {img['id']: img for img in data['images']}
    
    # 建立类别 ID 映射 (YOLO 需要从 0 开始连续的 ID)
    # COCO ID 可能是不连续的，或者是从 1 开始的
    # 我们按照 categories 列表的顺序重新映射为 0, 1, 2...
    sorted_cats = sorted(data['categories'], key=lambda x: x['id'])
    cat_id_to_yolo_idx = {cat['id']: i for i, cat in enumerate(sorted_cats)}
    
    print("类别映射关系 (YOLO ID : Name):")
    for cat in sorted_cats:
        print(f"  {cat_id_to_yolo_idx[cat['id']]} : {cat['name']}")

    # 准备复制图片和生成标签
    print("开始转换标签并复制图片...")
    
    for img_info in tqdm(data['images']):
        img_id = img_info['id']
        file_name = img_info['file_name']
        img_w = img_info['width']
        img_h = img_info['height']
        
        # 1. 复制图片到 output/images
        # 清理文件名中的前导 /
        file_name = file_name.lstrip('/')
        
        # 构建源图片路径
        if data_root:
            src_img_path = Path(data_root) / file_name
        else:
            src_img_path = Path(file_name)
        
        if not src_img_path.exists():
            print(f"⚠️  图片不存在: {src_img_path}")
            continue
            
        dst_img_name = f"{img_id:06d}.jpg" # 重命名规范化
        shutil.copy(src_img_path, images_dir / dst_img_name)
        
        # 2. 生成标签文件
        label_txt_path = labels_dir / f"{img_id:06d}.txt"
        
        # 查找该图片的所有标注
        anns = [ann for ann in data['annotations'] if ann['image_id'] == img_id]
        
        with open(label_txt_path, 'w') as f_txt:
            for ann in anns:
                cat_id = ann['category_id']
                if cat_id not in cat_id_to_yolo_idx:
                    continue
                
                class_idx = cat_id_to_yolo_idx[cat_id]
                
                if use_segmentation:
                    # 实例分割格式: class x1 y1 x2 y2 ... (归一化)
                    for seg in ann['segmentation']:
                        # seg 是 [x1, y1, x2, y2...]
                        points = np.array(seg).reshape(-1, 2)
                        # 归一化
                        points[:, 0] /= img_w
                        points[:, 1] /= img_h
                        
                        # 转换为一行字符串
                        line = f"{class_idx} " + " ".join([f"{x:.6f} {y:.6f}" for x, y in points])
                        f_txt.write(line + "\n")
                else:
                    # 目标检测格式: class x_center y_center w h (归一化)
                    bbox = ann['bbox'] # [x, y, w, h]
                    x_center = (bbox[0] + bbox[2]/2) / img_w
                    y_center = (bbox[1] + bbox[3]/2) / img_h
                    w = bbox[2] / img_w
                    h = bbox[3] / img_h
                    f_txt.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    print(f"\n✅ 转换完成！数据保存在: {output_dir}")

# ============ 使用示例 ============
if __name__ == "__main__":
    # 数据集根目录
    DATA_ROOT = r'C:\Users\kawayi_yaling\.cache\kagglehub\datasets\qmarva\cubicasa5k\versions\4\cubicasa5k\cubicasa5k'
    
    # 使用你之前生成的 JSON
    convert_coco_json_to_yolo(
        json_file='./coco_annotation_seg/coco_annotations_train_with_seg.json', 
        output_dir='./cubicasa_yolo/train',
        use_segmentation=True,
        data_root=DATA_ROOT
    )
    # 别忘了转换验证集
    convert_coco_json_to_yolo(
        json_file='./coco_annotation_seg/coco_annotations_val_with_seg.json', 
        output_dir='./cubicasa_yolo/val',
        use_segmentation=True,
        data_root=DATA_ROOT
    )
    # 转换测试集
    convert_coco_json_to_yolo(
        json_file='./coco_annotation_seg/coco_annotations_test_with_seg.json', 
        output_dir='./cubicasa_yolo/test',
        use_segmentation=True,
        data_root=DATA_ROOT
    )