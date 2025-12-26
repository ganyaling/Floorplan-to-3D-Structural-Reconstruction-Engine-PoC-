"""
演示和测试坐标拉直算法
"""

import numpy as np
import matplotlib.pyplot as plt
from floorplan_to_3d import PolygonStraightener


def visualize_polygon_straightening():
    """可视化多边形拉直效果"""
    
    # 创建一个不规则的矩形（模拟AI生成的坐标）
    # 这些点应该是一个矩形，但有轻微的偏差
    original_polygon = [
        (0, 0),
        (100, 2),      # 应该是水平的，但y=2
        (102, 100),    # 应该是垂直的，但x=102
        (2, 98),       # 应该是水平的，但y=98
    ]
    
    # 创建拉直处理器
    straightener = PolygonStraightener(
        rdp_epsilon=0.5,
        angle_threshold=5.0,
        snap_distance=1.0
    )
    
    # 处理
    straightened = straightener.straighten_polygon(original_polygon)
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 原始多边形
    original = np.array(original_polygon + [original_polygon[0]])
    axes[0].plot(original[:, 0], original[:, 1], 'b-o', linewidth=2, markersize=8, label='原始')
    axes[0].set_title('原始多边形（AI生成）')
    axes[0].set_xlabel('X (像素)')
    axes[0].set_ylabel('Y (像素)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_aspect('equal')
    
    # 拉直后的多边形
    straightened_arr = np.array(straightened + [straightened[0]])
    axes[1].plot(straightened_arr[:, 0], straightened_arr[:, 1], 'r-o', linewidth=2, markersize=8, label='拉直后')
    axes[1].set_title('拉直后的多边形（曼哈顿假设）')
    axes[1].set_xlabel('X (像素)')
    axes[1].set_ylabel('Y (像素)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('polygon_straightening.png', dpi=150, bbox_inches='tight')
    print("✅ 可视化已保存为 polygon_straightening.png")
    plt.show()


def test_rdp_algorithm():
    """测试RDP简化算法"""
    
    # 创建一条有噪声的曲线（模拟轮廓提取的结果）
    t = np.linspace(0, 2*np.pi, 100)
    x = 50 * np.cos(t) + np.random.randn(100) * 2
    y = 50 * np.sin(t) + np.random.randn(100) * 2
    
    noisy_points = np.column_stack([x, y])
    
    straightener = PolygonStraightener(rdp_epsilon=2.0)
    simplified = straightener.ramer_douglas_peucker(noisy_points, epsilon=2.0)
    
    print(f"原始点数: {len(noisy_points)}")
    print(f"简化后点数: {len(simplified)}")
    
    # 可视化
    plt.figure(figsize=(10, 8))
    plt.plot(noisy_points[:, 0], noisy_points[:, 1], 'b-', alpha=0.5, linewidth=1, label='原始轮廓')
    plt.plot(simplified[:, 0], simplified[:, 1], 'r-o', linewidth=2, markersize=8, label='RDP简化')
    plt.legend()
    plt.title('Ramer-Douglas-Peucker 算法演示')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.savefig('rdp_simplification.png', dpi=150, bbox_inches='tight')
    print("✅ RDP演示已保存为 rdp_simplification.png")
    plt.show()


def test_angle_detection():
    """测试线段角度检测和Cardinal对齐"""
    
    straightener = PolygonStraightener(angle_threshold=5.0)
    
    test_cases = [
        ("近水平线", np.array([0, 0]), np.array([100, 1])),
        ("完全水平线", np.array([0, 0]), np.array([100, 0])),
        ("近垂直线", np.array([0, 0]), np.array([1, 100])),
        ("完全垂直线", np.array([0, 0]), np.array([0, 100])),
        ("45度线", np.array([0, 0]), np.array([100, 100])),
        ("30度线", np.array([0, 0]), np.array([100, 58])),
    ]
    
    print("\n" + "="*60)
    print("线段角度检测测试")
    print("="*60)
    
    for name, p1, p2 in test_cases:
        angle = straightener.get_line_angle(p1, p2)
        p1_snap, p2_snap = straightener.snap_to_cardinal(p1, p2)
        
        print(f"\n{name}:")
        print(f"  原始端点: {p1} -> {p2}")
        print(f"  角度: {angle:.2f}°")
        print(f"  调整后: {p1_snap} -> {p2_snap}")
        
        if angle < 5:
            print(f"  状态: ✅ 已调整为水平")
        elif angle > 85:
            print(f"  状态: ✅ 已调整为垂直")
        else:
            print(f"  状态: ⚠️ 保持原样（非Cardinal方向）")


def test_complex_polygon():
    """测试复杂多边形的拉直"""
    
    # 创建一个L形的多边形（有轻微的角度偏差）
    l_shaped = [
        (10, 10),
        (110, 11),      # 应该y=10，偏差+1
        (112, 50),      # 应该x=110，偏差+2
        (50, 52),       # 应该y=50，偏差+2
        (48, 110),      # 应该x=50，偏差-2
        (12, 108),      # 应该y=110，偏差-2
    ]
    
    straightener = PolygonStraightener(
        rdp_epsilon=1.0,
        angle_threshold=5.0,
        snap_distance=2.0
    )
    
    straightened = straightener.straighten_polygon(l_shaped)
    
    print("\n" + "="*60)
    print("复杂多边形拉直测试")
    print("="*60)
    print(f"原始点数: {len(l_shaped)}")
    print(f"拉直后点数: {len(straightened)}")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 原始
    orig = np.array(l_shaped + [l_shaped[0]])
    axes[0].plot(orig[:, 0], orig[:, 1], 'b-o', linewidth=2, markersize=8)
    axes[0].set_title('原始L形多边形')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')
    
    # 拉直后
    straight = np.array(straightened + [straightened[0]])
    axes[1].plot(straight[:, 0], straight[:, 1], 'r-o', linewidth=2, markersize=8)
    axes[1].set_title('拉直后的L形多边形（曼哈顿假设）')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('complex_polygon_straightening.png', dpi=150, bbox_inches='tight')
    print("✅ 可视化已保存为 complex_polygon_straightening.png")
    plt.show()


if __name__ == '__main__':
    print("\n" + "="*60)
    print("坐标拉直算法测试套件")
    print("="*60 + "\n")
    
    # 1. 测试角度检测
    test_angle_detection()
    
    # 2. 测试简单矩形拉直
    print("\n运行简单矩形拉直演示...")
    visualize_polygon_straightening()
    
    # 3. 测试RDP算法
    print("\n运行RDP简化测试...")
    test_rdp_algorithm()
    
    # 4. 测试复杂多边形
    print("\n运行复杂多边形拉直测试...")
    test_complex_polygon()
    
    print("\n" + "="*60)
    print("✅ 所有测试完成！")
    print("="*60)
