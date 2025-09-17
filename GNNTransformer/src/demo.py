"""
焊缝点检测推理代码
zhh 20250901
"""
from train import WeldPointRegressionHGTNet, extract_laser_segments, create_topology_graph
import torch
import cv2
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm


# 加载训练好的模型
def load_model(model_path, device):
    """加载训练好的模型"""

    # 初始化模型（使用与训练相同的参数）
    model = WeldPointRegressionHGTNet(
        node_dim=7,  # 节点特征维度，这一项与节点特征直接相关，应该配合节点特征数量进行修改
        edge_dim=4,  # 边特征维度，这一项与节点特征直接相关，应该配合节点特征数量进行修改
        hidden_dim= hyperparams["hidden_dim"],
        num_layers= hyperparams["num_layers"],
        num_heads= hyperparams["num_heads"],
        dropout= hyperparams["dropout"],
        num_kernels=8
    ).to(device)

    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# 可视化函数
def visualize_results(img, segments, center_points, pred_point, true_point=None):
    """
    可视化推理结果（简化版）
    :param img: 原始图像
    :param segments: 提取的激光线段
    :param center_points: 中心点
    :param pred_point: 预测的焊缝点 (归一化坐标)
    :param true_point: 真实的焊缝点 (可选)
    """
    # 创建可视化图像副本
    vis_img = img.copy()
    img_height, img_width = img.shape[:2]

    # 反归一化预测点
    pred_pixel = (pred_point * np.array([img_width, img_height])).astype(int)

    # 1. 绘制不同颜色的线段并添加序号
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0),
              (0, 255, 255), (255, 0, 255), (128, 128, 255), (255, 128, 0)]

    for i, seg in enumerate(segments):
        start = tuple(map(int, seg["start_point"]))
        end = tuple(map(int, seg["end_point"]))

        # 确保坐标在图像范围内
        start = (max(0, min(start[0], img_width - 1)),
                 max(0, min(start[1], img_height - 1)))
        end = (max(0, min(end[0], img_width - 1)),
               max(0, min(end[1], img_height - 1)))

        # 选择颜色（循环使用）
        color = colors[i % len(colors)]

        # 绘制线段
        cv2.line(vis_img, start, end, color, 2)

        # 计算线段中点位置
        mid_x = (start[0] + end[0]) // 2
        mid_y = (start[1] + end[1]) // 2
        mid_point = (mid_x, mid_y)

        # 添加线段序号
        cv2.putText(vis_img, str(i), mid_point,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 2. 绘制预测点（红色五角星）
    # 绘制五角星
    star_points = []
    for i in range(5):
        # 外点
        angle = np.pi / 2 + i * 2 * np.pi / 5
        x = pred_pixel[0] + int(10 * np.cos(angle))
        y = pred_pixel[1] + int(10 * np.sin(angle))
        star_points.append((x, y))

        # 内点
        angle += np.pi / 5
        x = pred_pixel[0] + int(4 * np.cos(angle))
        y = pred_pixel[1] + int(4 * np.sin(angle))
        star_points.append((x, y))

    # 绘制五角星轮廓
    cv2.polylines(vis_img, [np.array(star_points)], True, (0, 0, 255), 2)
    # 填充五角星
    cv2.fillPoly(vis_img, [np.array(star_points)], (0, 0, 255))

    # 添加预测点标签
    cv2.putText(vis_img, "Prediction", (pred_pixel[0] + 15, pred_pixel[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 3. 如果有真实点，绘制真实点（绿色圆）
    if true_point is not None:
        true_pixel = tuple(map(int, true_point))
        cv2.circle(vis_img, true_pixel, 8, (0, 255, 0), -1)  # 绿色点
        cv2.circle(vis_img, true_pixel, 10, (255, 255, 255), 2)  # 白色外圈

        # 添加真实点标签
        cv2.putText(vis_img, "Ground Truth", (true_pixel[0] + 15, true_pixel[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 计算误差
        error = np.linalg.norm(np.array(pred_pixel) - np.array(true_pixel))
        # 显示误差
        cv2.putText(vis_img, f"Error: {error:.1f}px", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 创建Matplotlib可视化
    plt.figure(figsize=(12, 8))

    # 显示最终结果
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.title("Weld Point Detection Result")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("weld_detection_result.png", dpi=300)
    plt.show()

    return vis_img


def load_annotated_data(image_folder):
    """加载标注数据"""
    annotation_file = os.path.join(image_folder, "test_annotations.json")
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # 构建数据集: [(image_path, true_point)]
    dataset = []
    for img_file, points in annotations.items():
        img_path = os.path.join(image_folder, img_file)
        if points:  # 只使用有标注的图像
            # 使用第一个标注点作为焊缝点
            true_point = np.array(points[0])
            dataset.append((img_path, true_point))

    return dataset

def prepare_dataset(image_folder, display=True):
    """准备数据集"""
    # 加载标注数据
    dataset = load_annotated_data(image_folder)
    data = []
    target_size = (896, 400)

    i=0
    for img_path, true_point in tqdm(dataset, desc="Processing images"):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            continue

        orig_h, orig_w = img.shape[:2]
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        scale_x = target_size[0] / orig_w
        scale_y = target_size[1] / orig_h

        true_point = np.array([
            int(true_point[0] * scale_x),
            int(true_point[1] * scale_y)
        ])
        data.append((img, true_point))

        # 可视化部分
        if display:
            # 创建带标注的副本
            img_display = img.copy()
            
            # 在真值点位置绘制标记
            cv2.drawMarker(
                img_display, 
                tuple(true_point), 
                color=(0, 0, 255),  # 红色标记
                markerType=cv2.MARKER_CROSS,
                markerSize=20,
                thickness=2
            )
            
            # 添加坐标文本信息
            text = f"True Point: ({true_point[0]}, {true_point[1]})"
            cv2.putText(
                img_display,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),  # 绿色文字
                2
            )
            
            # 显示图像
            cv2.imshow('Image with True Point', img_display)
            print(i)
            print("/n")
            i=i+1
            cv2.waitKey(0)  # 显示100ms后自动关闭窗口
            cv2.destroyAllWindows()

    return data

def prepare_test_image(img):
    """准备单张测试图像"""
    optimized_segments, _ = extract_laser_segments(img)
    img_h, img_w = img.shape[:2]
    true_point = np.array([0,0])
    graph = create_topology_graph(optimized_segments, true_point, img_w, img_h)
    return graph


# 示例使用
if __name__ == "__main__":
    hyperparams = {
        "batch_size": 8,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "hidden_dim": 256,
        "num_layers": 3,
        "num_heads": 8,
        "dropout": 0,
        "num_epochs": 800,
        "lr_factor": 0.5,
        "lr_patience": 10,
        "min_lr": 1e-6,
        "grad_clip": 1.0,
        "image_folder": "/home/z/seam_tracking_ws/src/paper1_pkg/GNNTransformer/SecondCode/train_data"
    }


    model_path = "GNNTransformer/log/training_logs_20250917-134311/final_model.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = load_model(model_path, device)


    # 测试图像路径
    dataset_path = "GNNTransformer/datasets"
    test_data = prepare_dataset(dataset_path)  # [(img, true_point)] 图片与真实点已经被resize到统一尺寸
    euclidean_distances = []  # 欧式距离（像素误差）
    x_errors = []             # X坐标误差
    y_errors = []             # Y坐标误差
    undetected_count = 0      # 未检测到的图像数量
    for img, true_point in tqdm(test_data, desc="Processing test images"):
        # 生成图结构
        graph_data = prepare_test_image(img)
        if graph_data is None:
            print("Warning: No graph data generated. Skipping this image.")
            undetected_count += 1
            continue

        graph_data = graph_data.to(device)

        with torch.no_grad():
            pred_point = model(graph_data)
        
        img_size = graph_data.img_size
        img_size = img_size.cpu().numpy()
        img_w, img_h = img_size
        pred_point = pred_point.cpu().numpy()
        pred_x = (pred_point[0][0] + 1) * 0.5 * img_w# 反归一化到像素坐标
        pred_y = (pred_point[0][1] + 1) * 0.5 * img_h
        
        euclidean_dist = np.sqrt((pred_x - true_point[0])**2 + (pred_y - true_point[1])**2)
        euclidean_distances.append(euclidean_dist)
        x_error = abs(pred_x - true_point[0])
        x_errors.append(x_error)
        y_error = abs(pred_y - true_point[1])
        y_errors.append(y_error)
    
    if euclidean_distances:
        # 欧式距离统计
        mean_euclidean = np.mean(euclidean_distances)
        median_euclidean = np.median(euclidean_distances)
        min_euclidean = np.min(euclidean_distances)
        max_euclidean = np.max(euclidean_distances)
        # X坐标MAE和RMSE
        x_mae = np.mean(x_errors)
        x_rmse = np.sqrt(np.mean(np.square(x_errors)))
        
        # Y坐标MAE和RMSE
        y_mae = np.mean(y_errors)
        y_rmse = np.sqrt(np.mean(np.square(y_errors)))
        
        # 打印评估结果
        print("\n" + "="*50)
        print("焊缝检测评估结果")
        print("="*50)
        print(f"测试图像总数: {len(test_data)}")
        print(f"成功检测的图像数: {len(euclidean_distances)}")
        print(f"未检测到的图像数: {undetected_count}")
        print("\n欧式距离（像素误差）统计:")
        print(f"  平均值: {mean_euclidean:.2f} 像素")
        print(f"  中位数: {median_euclidean:.2f} 像素")
        print(f"  最小值: {min_euclidean:.2f} 像素")
        print(f"  最大值: {max_euclidean:.2f} 像素")
        print("\nX坐标误差统计:")
        print(f"  MAE: {x_mae:.2f} 像素")
        print(f"  RMSE: {x_rmse:.2f} 像素")
        print("\nY坐标误差统计:")
        print(f"  MAE: {y_mae:.2f} 像素")
        print(f"  RMSE: {y_rmse:.2f} 像素")
        print("="*50)
    else:
        print("警告：没有成功检测到任何焊缝点，无法计算评估指标")