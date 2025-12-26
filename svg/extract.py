import cv2
import numpy as np
from numpy import genfromtxt
import torch
import sys
from pathlib import Path

# 添加 floortrans 路径
cubicasa_root = Path(r"E:\JOB\CubiCasa5k")
if cubicasa_root.exists():
    sys.path.insert(0, str(cubicasa_root))
else:
    print(f"⚠️  警告: CubiCasa5k 路径不存在: {cubicasa_root}")

try:
    from floortrans.loaders.house import House
except ImportError as e:
    print(f"❌ 错误: 无法导入 floortrans: {e}")
    print(f"请确保 floortrans 在 {cubicasa_root}/floortrans 目录中")

from torch.utils.data import Dataset

class FloorplanSVG(Dataset):
    def __init__(self, data_folder, data_file, is_transform=True,
                 augmentations=None, img_norm=True, format='txt',
                 original_size=False, lmdb_folder='cubi_lmdb/'):
        self.img_norm = img_norm
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.get_data = None
        self.original_size = original_size
        self.image_file_name = '/F1_scaled.png'
        self.org_image_file_name = '/F1_original.png'
        self.svg_file_name = '/model.svg'

        if format == 'txt':
            self.get_data = self.get_txt
        if format == 'lmdb':
            import lmdb
            import pickle, os
            lmdb_path = os.path.join(data_folder, lmdb_folder)
            self.lmdb = lmdb.open(lmdb_path, readonly=True,
                                  max_readers=8, lock=False,
                                  readahead=True, meminit=False)
            self.get_data = self.get_lmdb
            self.is_transform = False

        self.data_folder = data_folder
        import os
        data_file_path = os.path.join(data_folder, data_file)
        self.folders = genfromtxt(data_file_path, dtype='str')
        
        # 房间类型映射（从house.py复制）
        self.room_id_to_name = {
            0: "Background",
            1: "Outdoor", 
            2: "Wall",
            3: "Kitchen",
            4: "LivingRoom",
            5: "Bedroom",
            6: "Bath",
            7: "Entry",
            8: "Railing",
            9: "Storage",
            10: "Garage",
            11: "Room"
        }

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, index):
        sample = self.get_data(index)

        if self.augmentations is not None:
            sample = self.augmentations(sample)
            
        if self.is_transform:
            sample = self.transform(sample)

        return sample
    
    def extract_room_info_from_house(self, house, coef_width=1, coef_height=1):
        """
        从House对象中提取房间信息
        """
        rooms_info = []
        
        # 从representation中获取房间标签信息
        for label_data in house.representation['labels']:
            center_box, room_info = label_data
            room_name = room_info[0]  # 如 "LivingRoom", "Bedroom" 等
            
            # 计算中心点（已经缩放）
            center_x = (center_box[0][1] + center_box[1][1]) / 2 * coef_width
            center_y = (center_box[0][0] + center_box[1][0]) / 2 * coef_height
            
            room_data = {
                'name': room_name,
                'center': [center_x, center_y],
                'center_box': [
                    [center_box[0][1] * coef_width, center_box[0][0] * coef_height],
                    [center_box[1][1] * coef_width, center_box[1][0] * coef_height]
                ]
            }
            
            rooms_info.append(room_data)
        
        # 从walls分割结果中提取每个房间的详细信息
        wall_array = house.walls
        unique_labels = np.unique(wall_array)
        
        room_details = []
        for label in unique_labels:
            if label == 0 or label == 2 or label == 8:  # 跳过背景、墙体、栏杆
                continue
            
            # 获取该房间的所有像素
            mask = (wall_array == label)
            if not mask.any():
                continue
            
            # 获取边界框
            rows, cols = np.where(mask)
            if len(rows) == 0:
                continue
                
            x_min = int(cols.min() * coef_width)
            y_min = int(rows.min() * coef_height)
            x_max = int(cols.max() * coef_width)
            y_max = int(rows.max() * coef_height)
            
            # 计算面积（像素数量）
            area = mask.sum() * coef_width * coef_height
            
            room_detail = {
                'label_id': int(label),
                'room_type': self.room_id_to_name.get(int(label), 'Unknown'),
                'bbox': [x_min, y_min, x_max, y_max],
                'area_pixels': float(area),
                'center': [int((x_min + x_max) / 2), int((y_min + y_max) / 2)]
            }
            
            room_details.append(room_detail)
        
        return {
            'room_labels': rooms_info,      # 房间名称和中心点
            'room_details': room_details    # 详细的房间分割信息
        }

    def get_txt(self, index):
        # 读取图像
        fplan = cv2.imread(self.data_folder + self.folders[index] + self.image_file_name)
        fplan = cv2.cvtColor(fplan, cv2.COLOR_BGR2RGB) 
        height, width, nchannel = fplan.shape
        fplan = np.moveaxis(fplan, -1, 0)

        # 使用House类解析SVG
        house = House(self.data_folder + self.folders[index] + self.svg_file_name, height, width)
        
        # 获取标签和热图
        label = torch.tensor(house.get_segmentation_tensor().astype(np.float32))
        heatmaps = house.get_heatmap_dict()
        
        coef_width = 1
        coef_height = 1
        
        # 处理原始尺寸图像
        if self.original_size:
            fplan = cv2.imread(self.data_folder + self.folders[index] + self.org_image_file_name)
            fplan = cv2.cvtColor(fplan, cv2.COLOR_BGR2RGB)
            height_org, width_org, nchannel = fplan.shape
            fplan = np.moveaxis(fplan, -1, 0)
            
            # 缩放label
            label = label.unsqueeze(0)
            label = torch.nn.functional.interpolate(label,
                                                    size=(height_org, width_org),
                                                    mode='nearest')
            label = label.squeeze(0)

            coef_height = float(height_org) / float(height)
            coef_width = float(width_org) / float(width)
            
            # 缩放热图坐标
            for key, value in heatmaps.items():
                heatmaps[key] = [(int(round(x*coef_width)), int(round(y*coef_height))) for x, y in value]
        
        # 提取房间信息
        svg_rooms = self.extract_room_info_from_house(house, coef_width, coef_height)

        img = torch.tensor(fplan.astype(np.float32))

        sample = {
            'image': img, 
            'label': label, 
            'folder': self.folders[index],
            'heatmaps': heatmaps, 
            'scale': coef_width,
            'svg_rooms': svg_rooms  # 房间信息
        }

        return sample

    def get_lmdb(self, index):
        import pickle
        key = self.folders[index].encode()
        with self.lmdb.begin(write=False) as f:
            data = f.get(key)
        sample = pickle.loads(data)
        return sample

    def transform(self, sample):
        fplan = sample['image']
        fplan = 2 * (fplan / 255.0) - 1
        sample['image'] = fplan
        return sample


# ============ 使用示例 ============
if __name__ == "__main__":
    import sys
    
    # 添加CubiCasa路径
    sys.path.append(r'C:\path\to\CubiCasa5k')
    
    # 数据集路径
    DATA_PATH = r'C:\Users\kawayi_yaling\.cache\kagglehub\datasets\qmarva\cubicasa5k\versions\4\cubicasa5k\cubicasa5k'
    
    # 创建数据集
    dataset = FloorplanSVG(
        data_folder=DATA_PATH,
        data_file='train.txt',
        is_transform=True,
        original_size=True
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    # 加载第一个样本
    sample = dataset[0]
    
    print(f"\n文件夹: {sample['folder']}")
    print(f"图像shape: {sample['image'].shape}")
    print(f"标签shape: {sample['label'].shape}")
    
    # 查看房间信息
    svg_rooms = sample['svg_rooms']
    
    print("\n=== 房间标签信息 ===")
    for i, room in enumerate(svg_rooms['room_labels']):
        print(f"{i+1}. {room['name']}")
        print(f"   中心点: ({room['center'][0]:.1f}, {room['center'][1]:.1f})")
    
    print("\n=== 房间详细信息 ===")
    for i, room in enumerate(svg_rooms['room_details']):
        print(f"{i+1}. {room['room_type']} (ID: {room['label_id']})")
        print(f"   边界框: {room['bbox']}")
        print(f"   面积: {room['area_pixels']:.1f} 像素")
        print(f"   中心: {room['center']}")
    
    # 统计房间类型
    room_types = {}
    for room in svg_rooms['room_details']:
        rtype = room['room_type']
        room_types[rtype] = room_types.get(rtype, 0) + 1
    
    print("\n=== 房间类型统计 ===")
    for rtype, count in room_types.items():
        print(f"{rtype}: {count}")
    
    # 可视化
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # 显示图像
    img = sample['image'].numpy()
    if img.shape[0] == 3:
        img = np.moveaxis(img, 0, -1)
    img = (img + 1) / 2  # 反归一化
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    
    # 绘制房间边界框
    colors = {
        'LivingRoom': 'red',
        'Bedroom': 'blue',
        'Kitchen': 'green',
        'Bath': 'cyan',
        'Entry': 'yellow',
        'Storage': 'orange',
        'Garage': 'purple'
    }
    
    for room in svg_rooms['room_details']:
        if room['room_type'] in ['Background', 'Wall', 'Railing']:
            continue
            
        bbox = room['bbox']
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        
        color = colors.get(room['room_type'], 'white')
        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # 添加标签
        ax.text(
            x_min + 5, y_min + 15,
            room['room_type'],
            color=color,
            fontsize=10,
            weight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7)
        )
    
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('room_visualization.png', dpi=150, bbox_inches='tight')
    print("\n✅ 可视化保存到: room_visualization.png")
    plt.show()