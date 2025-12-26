import os
import json
import torch
from tqdm import tqdm
import sys
from pathlib import Path

from extract import FloorplanSVG

def create_coco_annotations(dataset, output_file, num_images=None):
    """
    从 FloorplanSVG 数据集创建 COCO 格式标注
    
    Args:
        dataset: FloorplanSVG 数据集实例
        output_file: 输出的 JSON 文件路径
        num_images: 处理的图像数量，None 表示处理全部
    """
    
    # 定义类别 - 基于房间类型
    categories = [
        {"id": 1, "name": "LivingRoom", "supercategory": "room"},
        {"id": 2, "name": "Bedroom", "supercategory": "room"},
        {"id": 3, "name": "Kitchen", "supercategory": "room"},
        {"id": 4, "name": "Bath", "supercategory": "room"},
        {"id": 5, "name": "Entry", "supercategory": "room"},
        {"id": 6, "name": "Storage", "supercategory": "room"},
        {"id": 7, "name": "Garage", "supercategory": "room"},
        {"id": 8, "name": "Outdoor", "supercategory": "room"},
        {"id": 9, "name": "Room", "supercategory": "room"},
        {"id": 10, "name": "Wall", "supercategory": "structure"},
        {"id": 11, "name": "Railing", "supercategory": "structure"},
    ]
    
    # 房间类型到类别ID的映射
    room_type_to_category_id = {
        "LivingRoom": 1,
        "Bedroom": 2,
        "Kitchen": 3,
        "Bath": 4,
        "Entry": 5,
        "Storage": 6,
        "Garage": 7,
        "Outdoor": 8,
        "Room": 9,
        "Wall": 10,
        "Railing": 11,
    }
    
    coco_annotations = {
        "images": [],
        "annotations": [],
        "categories": categories
    }
    
    annotation_id = 1
    image_id = 1
    
    num_images = num_images if num_images else len(dataset)
    
    print(f"开始处理 {num_images} 张图像...")
    
    for idx in tqdm(range(num_images), desc="处理图像"):
        try:
            sample = dataset[idx]
            image = sample['image']
            folder = sample['folder']
            svg_rooms = sample['svg_rooms']
            
            # 获取图像尺寸
            if len(image.shape) == 3:  # CxHxW
                height, width = image.shape[1], image.shape[2]
            else:
                height, width = image.shape[0], image.shape[1]
            
            # 构建图像文件路径
            image_file_path = f"{folder}/F1_original.png"
            
            # 添加图像信息
            coco_annotations['images'].append({
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": image_file_path
            })
            
            # 处理房间详细信息
            room_details = svg_rooms.get('room_details', [])
            
            for room in room_details:
                room_type = room['room_type']
                
                # 跳过背景
                if room_type in ['Background']:
                    continue
                
                # 获取类别ID
                category_id = room_type_to_category_id.get(room_type)
                if category_id is None:
                    continue
                
                # 获取边界框
                bbox = room['bbox']
                x_min, y_min, x_max, y_max = bbox
                bbox_width = x_max - x_min
                bbox_height = y_max - y_min
                
                # 确保边界框有效
                if bbox_width > 0 and bbox_height > 0:
                    coco_annotations['annotations'].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x_min, y_min, bbox_width, bbox_height],  # COCO格式: [x, y, width, height]
                        "area": float(room['area_pixels']),
                        "iscrowd": 0,
                        "segmentation": []  # 如果需要分割信息，可以从 label 中提取
                    })
                    annotation_id += 1
            
            image_id += 1
            
        except Exception as e:
            print(f"\n处理图像 {idx} 时出错: {e}")
            continue
    
    # 保存 COCO 标注
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(coco_annotations, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ COCO 标注已保存到: {output_file}")
    print(f"   - 图像数量: {len(coco_annotations['images'])}")
    print(f"   - 标注数量: {len(coco_annotations['annotations'])}")
    print(f"   - 类别数量: {len(coco_annotations['categories'])}")
    
    # 统计每个类别的标注数量
    category_counts = {}
    for ann in coco_annotations['annotations']:
        cat_id = ann['category_id']
        cat_name = next(cat['name'] for cat in categories if cat['id'] == cat_id)
        category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
    
    print("\n类别统计:")
    for cat_name, count in sorted(category_counts.items()):
        print(f"   {cat_name}: {count}")
    
    return coco_annotations


def create_coco_with_segmentation(dataset, output_file, num_images=None):
    """
    创建带有分割掩码的 COCO 格式标注
    
    Args:
        dataset: FloorplanSVG 数据集实例
        output_file: 输出的 JSON 文件路径
        num_images: 处理的图像数量，None 表示处理全部
    """
    from skimage import measure
    import numpy as np
    
    # 定义类别
    categories = [
        {"id": 1, "name": "LivingRoom", "supercategory": "room"},
        {"id": 2, "name": "Bedroom", "supercategory": "room"},
        {"id": 3, "name": "Kitchen", "supercategory": "room"},
        {"id": 4, "name": "Bath", "supercategory": "room"},
        {"id": 5, "name": "Entry", "supercategory": "room"},
        {"id": 6, "name": "Storage", "supercategory": "room"},
        {"id": 7, "name": "Garage", "supercategory": "room"},
        {"id": 8, "name": "Outdoor", "supercategory": "room"},
        {"id": 9, "name": "Room", "supercategory": "room"},
        {"id": 10, "name": "Wall", "supercategory": "structure"},
        {"id": 11, "name": "Railing", "supercategory": "structure"},
    ]
    
    # 房间标签ID到类别ID的映射
    label_id_to_category_id = {
        4: 1,   # LivingRoom
        5: 2,   # Bedroom
        3: 3,   # Kitchen
        6: 4,   # Bath
        7: 5,   # Entry
        9: 6,   # Storage
        10: 7,  # Garage
        1: 8,   # Outdoor
        11: 9,  # Room
        2: 10,  # Wall
        8: 11,  # Railing
    }
    
    coco_annotations = {
        "images": [],
        "annotations": [],
        "categories": categories
    }
    
    annotation_id = 1
    image_id = 1
    
    num_images = num_images if num_images else len(dataset)
    
    print(f"开始处理 {num_images} 张图像（包含分割掩码）...")
    
    for idx in tqdm(range(num_images), desc="处理图像"):
        try:
            sample = dataset[idx]
            image = sample['image']
            label = sample['label']  # 分割标签
            folder = sample['folder']
            
            # 获取图像尺寸
            if len(image.shape) == 3:
                height, width = image.shape[1], image.shape[2]
            else:
                height, width = image.shape[0], image.shape[1]
            
            # 构建图像文件路径
            image_file_path = f"{folder}/F1_original.png"
            
            # 添加图像信息
            coco_annotations['images'].append({
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": image_file_path
            })
            
            # 从 label 中提取分割掩码
            label_array = label.numpy()[0] if len(label.shape) == 3 else label.numpy()
            
            # 对每个房间标签进行处理
            unique_labels = np.unique(label_array)
            
            for label_id in unique_labels:
                # 跳过背景
                if label_id in [0]:
                    continue
                
                # 获取类别ID
                category_id = label_id_to_category_id.get(int(label_id))
                if category_id is None:
                    continue
                
                # 创建该房间的二值掩码
                mask = (label_array == label_id).astype(np.uint8)
                
                # 使用连通组件分析，处理同一类型的多个房间
                labeled_mask, num_features = measure.label(mask, return_num=True, connectivity=2)
                
                for region_id in range(1, num_features + 1):
                    region_mask = (labeled_mask == region_id)
                    
                    # 获取边界框
                    rows, cols = np.where(region_mask)
                    if len(rows) == 0:
                        continue
                    
                    x_min, y_min = int(cols.min()), int(rows.min())
                    x_max, y_max = int(cols.max()), int(rows.max())
                    bbox_width = x_max - x_min + 1
                    bbox_height = y_max - y_min + 1
                    
                    # 计算面积
                    area = float(region_mask.sum())
                    
                    # 提取轮廓作为分割信息
                    contours = measure.find_contours(region_mask, 0.5)
                    segmentation = []
                    
                    for contour in contours:
                        # 将轮廓转换为 COCO 格式 [x1, y1, x2, y2, ...]
                        contour = contour[:, ::-1]  # 从 (row, col) 转换为 (x, y)
                        if len(contour) > 4:  # 至少需要3个点
                            seg = contour.flatten().tolist()
                            segmentation.append(seg)
                    
                    if segmentation and bbox_width > 0 and bbox_height > 0:
                        coco_annotations['annotations'].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_id,
                            "bbox": [x_min, y_min, bbox_width, bbox_height],
                            "area": area,
                            "iscrowd": 0,
                            "segmentation": segmentation
                        })
                        annotation_id += 1
            
            image_id += 1
            
        except Exception as e:
            print(f"\n处理图像 {idx} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存 COCO 标注
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(coco_annotations, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ COCO 标注（含分割）已保存到: {output_file}")
    print(f"   - 图像数量: {len(coco_annotations['images'])}")
    print(f"   - 标注数量: {len(coco_annotations['annotations'])}")
    
    return coco_annotations


# ============ 使用示例 ============
if __name__ == "__main__":
    # 数据集路径
    DATA_PATH = r'C:\Users\kawayi_yaling\.cache\kagglehub\datasets\qmarva\cubicasa5k\versions\4\cubicasa5k\cubicasa5k'
    
    # 创建数据集实例
    print("加载训练集...")
    train_dataset = FloorplanSVG(
        data_folder=DATA_PATH,
        data_file='train.txt',
        is_transform=False,  # 不进行归一化，保持原始数据
        original_size=True
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    
    # 创建 COCO 标注（仅边界框）
    print("\n=== 创建 COCO 标注（边界框） ===")
    create_coco_annotations(
        dataset=train_dataset,
        output_file='coco_annotations_train.json',
        num_images=None  # 先处理100张测试，可以设为None处理全部
    )
    
    # 创建 COCO 标注（含分割掩码）
    print("\n=== 创建 COCO 标注（含分割掩码） ===")
    create_coco_with_segmentation(
        dataset=train_dataset,
        output_file='coco_annotations_train_with_seg.json',
        num_images=None  # 先处理100张测试
    )
    
    # 验证集
    print("\n加载验证集...")
    val_dataset = FloorplanSVG(
        data_folder=DATA_PATH,
        data_file='val.txt',
        is_transform=False,
        original_size=True
    )
    
    create_coco_annotations(
        dataset=val_dataset,
        output_file='coco_annotations_val.json',
        num_images=None
    )
    create_coco_with_segmentation(
        dataset=val_dataset,
        output_file='coco_annotations_val_with_seg.json',
        num_images=None  # 先处理100张测试
    )

    #测试集
    print("\n加载测试集...")
    test_dataset = FloorplanSVG(
        data_folder=DATA_PATH,
        data_file='test.txt',
        is_transform=False,
        original_size=True
    )
    create_coco_annotations(
        dataset=test_dataset,
        output_file='coco_annotations_test.json',
        num_images=None
    )
    create_coco_with_segmentation(
        dataset=test_dataset,
        output_file='coco_annotations_test_with_seg.json',
        num_images=None  # 先处理100张测试
    )
    print("\n✅ 所有标注文件创建完成！")