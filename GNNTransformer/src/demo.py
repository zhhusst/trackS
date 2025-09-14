"""
焊缝点检测推理代码
zhh 20250901
"""
from GNNTransformer.SecondCode.里程碑代码.train_topo_0903 import *


# 加载训练好的模型
def load_model(model_path, device):
    """加载训练好的模型"""
    from GNNTransformer.SecondCode.里程碑代码.train_topo import WeldPointRegressionHGTNet  # 确保模型定义可用

    # 初始化模型（使用与训练相同的参数）
    model = WeldPointRegressionHGTNet(
        node_dim=8,  # 节点特征维度，这一项与节点特征直接相关，应该配合节点特征数量进行修改
        edge_dim=4,  # 边特征维度，这一项与节点特征直接相关，应该配合节点特征数量进行修改
        hidden_dim=256,
        num_layers=3,
        num_heads=8,
        dropout=0
    ).to(device)

    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


# 推理函数
def predict_weld_point(model, img, device):
    """
    预测焊缝点位置
    :param model: 加载的模型
    :param img: 输入图像 (BGR格式)
    :param device: 计算设备
    :return:
        pred_point: 预测的焊缝点坐标 (归一化)
        segments: 提取的激光线段
        center_points: 中心点
        graph: 构建的图数据
    """
    # 获取图像尺寸
    img_height, img_width = img.shape[:2]

    # 提取激光线段
    optimized_segments, center_points = extract_laser_segments(img)

    # 如果没有提取到线段，返回默认值
    if not optimized_segments:
        print("Warning: No laser segments detected!")
        return None, optimized_segments, center_points, None

    # 构建图数据
    graph = create_topology_graph(optimized_segments, np.array([0, 0]), img_width, img_height)

    # 转换为PyG数据格式
    data = graph.to(device)

    # 模型推理
    with torch.no_grad():
        node_scores, pred_point = model(data)

    return pred_point.cpu().numpy()[0], optimized_segments, center_points, graph


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


# 主推理函数
def main_inference(image_path, model_path, true_point=None):
    """主推理函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型
    model = load_model(model_path, device)

    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    # 进行预测
    pred_point, segments, center_points, graph = predict_weld_point(model, img, device)

    if pred_point is None:
        print("No prediction made.")
        return

    # 可视化结果
    visualize_results(img, segments, center_points, pred_point, true_point)

    # 返回预测点（像素坐标）
    img_height, img_width = img.shape[:2]
    pred_pixel = (pred_point * np.array([img_width, img_height])).astype(int)
    print(f"Predicted weld point: ({pred_pixel[0]}, {pred_pixel[1]})")

    return pred_pixel


# 以下是训练代码中的辅助函数（需要从训练代码中复制）
# 注意：需要复制以下函数到推理代码中：
# - extract_laser_segments
# - create_topology_graph
# - VirtualNode
# - merge_line_segments
# - can_merge_segments
# - merge_two_segments
# - calculate_confidence

# 示例使用
if __name__ == "__main__":
    # 模型路径
    model_path = "training_logs_20250904-114247/weld_hgt_model.pth"

    # 测试图像路径
    image_path = "/home/z/seam_tracking_ws/src/paper1_pkg/GNNTransformer/SecondCode/train_data/25051928-0061-OFF.png"

    # 真实焊缝点（如果有）
    true_point = [910, 395]  # 示例坐标

    # 运行推理
    main_inference(image_path, model_path, true_point)