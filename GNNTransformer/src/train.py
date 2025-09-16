"""
训练代码
0914
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import linregress
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import TransformerConv
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import json
from torch.utils.data import Dataset

import datetime
import csv

from tqdm import tqdm

# 设置随机种子确保可复现性
torch.manual_seed(42)
np.random.seed(42)


# 加载图像和标注
def load_annotated_data(image_folder):
    """加载标注数据"""
    annotation_file = os.path.join(image_folder, "train_annotations.json")
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


def merge_line_segments(line_segments, center_points):
    """合并和优化线段"""
    # 按斜率分组
    slope_groups = {}
    for segment in line_segments:
        slope_key = round(segment["slope"], 1)
        slope_groups.setdefault(slope_key, []).append(segment)

    # 合并线段
    merged_segments = []
    for slope, segments in slope_groups.items():
        # 按x坐标排序
        if abs(slope) < 10:  # 非垂直线段
            segments.sort(key=lambda s: min(s["start_point"][0], s["end_point"][0]))
        else:  # 垂直线段
            segments.sort(key=lambda s: min(s["start_point"][1], s["end_point"][1]))

        i = 0
        while i < len(segments):
            current = segments[i]
            j = i + 1

            while j < len(segments):
                next_seg = segments[j]
                if can_merge_segments(current, next_seg):
                    current = merge_two_segments(current, next_seg, center_points)
                    segments.pop(j)
                else:
                    j += 1

            merged_segments.append(current)
            i += 1

    # 过滤短线段和低置信度线段
    return [seg for seg in merged_segments if seg["length"] > 5 and seg["confidence"] > 50]


def can_merge_segments(seg1, seg2):
    """判断两个线段是否可以合并"""
    # 检查斜率是否相近
    slope_diff = abs(seg1["slope"] - seg2["slope"])
    if slope_diff > 0.1:
        return False

    # 检查线段是否重叠或邻近
    seg1_min_x = min(seg1["start_point"][0], seg1["end_point"][0])
    seg1_max_x = max(seg1["start_point"][0], seg1["end_point"][0])
    seg2_min_x = min(seg2["start_point"][0], seg2["end_point"][0])
    seg2_max_x = max(seg2["start_point"][0], seg2["end_point"][0])

    overlap = min(seg1_max_x, seg2_max_x) - max(seg1_min_x, seg2_min_x)
    gap = max(seg1_min_x, seg2_min_x) - min(seg1_max_x, seg2_max_x)

    return overlap > 0 or (gap < 0 and abs(gap) < 50)


def merge_two_segments(seg1, seg2, center_points):
    """合并两个线段"""
    all_points = [seg1["start_point"], seg1["end_point"], seg2["start_point"], seg2["end_point"]]
    all_points.sort(key=lambda p: p[0])

    start_point = all_points[0]
    end_point = all_points[-1]

    x1, y1 = start_point
    x2, y2 = end_point
    length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # 计算新线段的置信度
    confidence = calculate_confidence(center_points, start_point, end_point, length)

    return {
        "start_point": start_point,
        "end_point": end_point,
        "slope": (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf'),
        "length": length,
        "confidence": confidence,
        "curvature": calculate_curvature(center_points, x1, y1, x2, y2),  # 新增曲率
        "linearity": calculate_linearity(center_points, x1, y1, x2, y2)  # 新增线性度
    }


def calculate_confidence(center_points, start_point, end_point, length):
    """计算线段的置信度"""
    x1, y1 = start_point
    x2, y2 = end_point
    dx, dy = x2 - x1, y2 - y1

    # 搜索区域宽度
    search_width = 10
    nearby_points = 0

    for point in center_points:
        px, py = point

        # 计算点到线段的距离
        if dx == 0:  # 垂直线
            dist = abs(px - x1)
        else:
            A = -dy / dx
            B = 1
            C = (dy / dx) * x1 - y1
            dist = abs(A * px + B * py + C) / math.sqrt(A ** 2 + B ** 2)

        # 检查点是否在线段附近
        if dist <= search_width:
            # 检查点是否在线段范围内
            t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy) if (dx * dx + dy * dy) != 0 else 0
            if 0 <= t <= 1:
                nearby_points += 1

    # 计算置信度：附近点密度 × 线段长度
    density = nearby_points / (length + 1e-5)
    return density * length


def calculate_curvature(center_points, x1, y1, x2, y2):
    """
    计算线段曲率(半径倒数)
    """
    sample_points = []
    for point in center_points:
        px, py = point
        if min(x1, x2) - 5 <= px <= max(x1, x2) + 5:
            sample_points.append(point)
    if len(sample_points)<3:
        return 0.0
    # 拟合圆计算曲率
    try:
        x = np.array([p[0] for p in sample_points])
        y = np.array([p[1] for p in sample_points])
        a = np.vstack([x, y, np.ones(len(x))]).T
        b = x ** 2 + y ** 2
        c, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
        x0 = c[0] / 2
        y0 = c[1] / 2
        r = np.sqrt(c[2] + x0 ** 2 + y0 ** 2)
        return 1.0 / r  # 曲率 = 1/半径
    except:
        return 0.0


def calculate_linearity(center_points, x1, y1, x2, y2):
    """计算线段线性度（R²值）"""
    # 在线段附近采样点
    sample_x = []
    sample_y = []
    for px, py in center_points:
        if min(x1, x2) - 5 <= px <= max(x1, x2) + 5:
            sample_x.append(px)
            sample_y.append(py)

    if len(sample_x) < 2:
        return 1.0

    # 线性回归计算R²
    slope, intercept, r_value, _, _ = linregress(sample_x, sample_y)
    return r_value ** 2  # 线性度 = R²值


def extract_laser_segments(img):
    """从图像中提取优化的激光线段"""
    # 1. 预处理图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # 2. 二值化和形态学操作
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 3. 使用灰度重心法提取中心线
    rows, cols = blurred.shape
    center_points = []  # 存储中心点坐标

    for col in range(cols):
        col_data = blurred[:, col]
        binary_col = binary[:, col]

        if np.any(binary_col > 0):
            indices = np.where(binary_col > 0)[0]
            if len(indices) > 0:
                start_row, end_row = indices[0], indices[-1]
                segment = col_data[start_row:end_row + 1]

                # 计算重心
                weights = np.arange(start_row, end_row + 1)
                centroid = np.average(weights, weights=segment)
                centroid_int = int(round(centroid))

                if 0 <= centroid_int < rows:
                    center_points.append((col, centroid_int))

    # 4. 创建连接后的中心线图像
    connected_centerline = np.zeros((rows, cols), dtype=np.uint8)
    if len(center_points) > 1:
        center_points.sort(key=lambda x: x[0])
        for i in range(len(center_points) - 1):
            x1, y1 = center_points[i]
            x2, y2 = center_points[i + 1]
            cv2.line(connected_centerline, (x1, y1), (x2, y2), 255, 1)

    # 5. 线段检测
    lines = cv2.HoughLinesP(connected_centerline, 1, np.pi / 180, threshold=20,
                            minLineLength=10, maxLineGap=17.4)

    # 6. 提取线段信息
    line_segments = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # 计算斜率
            if x2 - x1 != 0:
                slope = (y2 - y1) / (x2 - x1)
            else:
                slope = float('inf')

            # 计算置信度
            confidence = calculate_confidence(center_points, (x1, y1), (x2, y2), length)

            segment_info = {
                "start_point": (x1, y1),
                "end_point": (x2, y2),
                "slope": slope,
                "length": length,
                "confidence": confidence,
                "curvature": calculate_curvature(center_points, x1, y1, x2, y2),  # 新增曲率
                "linearity": calculate_linearity(center_points, x1, y1, x2, y2)  # 新增线性度
            }
            line_segments.append(segment_info)

    # 7. 线段合并和优化
    optimized_segments = merge_line_segments(line_segments, center_points)

    # 8. 按照线段的x坐标对线段进行排序
    optimized_segments.sort(key=lambda seg:min(seg["start_point"][0], seg["end_point"][0]))

    return optimized_segments, center_points


def prepare_dataset(image_folder):
    """准备数据集: 返回PyG图数据集列表"""
    # 加载标注数据
    dataset = load_annotated_data(image_folder)
    graph_data = []
    target_size = (896, 400)

    for img_path, true_point in tqdm(dataset, desc="Processing images"):
        img = cv2.imread(img_path)
        if img is None:
            continue

        orig_h, orig_w = img.shape[:2]
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        scale_x = target_size[0] / orig_w
        scale_y = target_size[1] / orig_h

        true_point = np.array([
            int(true_point[0] * scale_x),
            int(true_point[1] * scale_y)
        ])
        # 获取图像尺寸
        # img_height, img_width = img.shape[:2]

        # 使用改进的激光条纹提取方法
        optimized_segments, _ = extract_laser_segments(img)
        if optimized_segments:
            # 构建图数据
            graph = create_topology_graph(optimized_segments, true_point, target_size[0], target_size[1])
            graph_data.append(graph)

    return graph_data


def create_topology_graph(segments, true_point, img_width, img_height):
    """
     构建基于拓扑连接的图结构（添加归一化到[-1,1]）
    """
    norm_x = img_width / 2.0
    norm_y = img_height / 2.0

    def normalize_coord(x, y):
        return (x - norm_x) / norm_x, (y - norm_y) / norm_y

    if len(segments) == 0:
        return Data(
            x=torch.empty((0, 4), dtype=torch.float),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 4), dtype=torch.float),
            y=torch.tensor([[normalize_coord(true_point[0], true_point[1])[0],
                             normalize_coord(true_point[0], true_point[1])[1]]],
                           dtype=torch.float),
            img_size=torch.tensor([img_width, img_height], dtype=torch.float)
        )

    # 1. 对线段按照 x 坐标排序（从左至右）
    segments.sort(key=lambda seg: min(seg["start_point"][0], seg["end_point"][0]))

    # 2. 创建虚拟节点
    nodes = []
    for i in range(len(segments) - 1):
        left_end = segments[i]["end_point"]
        right_start = segments[i + 1]["start_point"]

        lx, ly = normalize_coord(*left_end)
        rx, ry = normalize_coord(*right_start)

        node_feature = [
            lx, ly, rx, ry,
            math.tanh(segments[i]["slope"]),
            segments[i]["confidence"] / 100.0,
            min(1.0, segments[i]["length"] / img_width)
        ]
        nodes.append(node_feature)

    # 3. 首尾虚拟节点
    first_node_left = normalize_coord(*segments[0]["start_point"])
    first_node_right = normalize_coord(*segments[0]["start_point"])
    first_node_feature = [
        first_node_left[0], first_node_left[1],
        first_node_right[0], first_node_right[1],
        math.tanh(segments[0]["slope"]),
        segments[0]["confidence"] / 100.0,
        min(1.0, segments[0]["length"] / img_width)
    ]
    nodes.insert(0, first_node_feature)

    last_node_left = normalize_coord(*segments[-1]["end_point"])
    last_node_right = normalize_coord(*segments[-1]["end_point"])
    last_node_feature = [
        last_node_left[0], last_node_left[1],
        last_node_right[0], last_node_right[1],
        math.tanh(segments[-1]["slope"]),
        segments[-1]["confidence"] / 100.0,
        min(1.0, segments[-1]["length"] / img_width)
    ]
    nodes.insert(-1, last_node_feature)

    # 4. 构建边
    edges = []
    edge_attr = []
    for i, seg in enumerate(segments):
        start_x, start_y = normalize_coord(*seg["start_point"])
        end_x, end_y = normalize_coord(*seg["end_point"])

        edge_feature = [start_x, start_y, end_x, end_y]
        edges.append([i, i + 1])
        edge_attr.append(edge_feature)

    # 5. 真实焊缝点归一化
    true_x, true_y = normalize_coord(true_point[0], true_point[1])

    return Data(
        x=torch.tensor(nodes, dtype=torch.float),
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        y=torch.tensor([[true_x, true_y]], dtype=torch.float),
        img_size=torch.tensor([img_width, img_height], dtype=torch.float)
    )


class WeldGraphDataset(Dataset):
    """焊缝图数据集类"""

    def __init__(self, graph_data):
        self.graph_data = graph_data

    def __len__(self):
        return len(self.graph_data)

    def __getitem__(self, idx):
        return self.graph_data[idx]


class SpatialGaussianEncoder(nn.Module):
    """
    空间高斯核编码器
    """
    def __init__(self, num_kernels=8, hidden_dim=64,dropout=0.1, input_dim=8):
        super().__init__()
        self.num_kernels = num_kernels

        # 可学习高斯核参数
        self.means = nn.Parameter(torch.randn(num_kernels,input_dim))
        self.log_stds = nn.Parameter(torch.zeros(num_kernels,input_dim))

        # 特征变化层
        self.mlp=nn.Sequential(
            nn.Linear(num_kernels,hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    def forward(self, edge_index, node_pos):
        """
        :param edge_index: [2, E] 边连接关系
        :param node_pos: [N, 2] 节点归一化坐标
        :return: [E, hidden_dim] 空间编码特征
        """
        src, dst = edge_index
        src_left = node_pos[src, :2]
        src_right = node_pos[src, 2:4]
        dst_left = node_pos[dst, :2]
        dst_right = node_pos[dst, 2:4]

        # 计算4中可能的相对位置关系
        rel_pos1 = dst_left - src_left  # 左->左
        rel_pos2 = dst_left - src_right  # 右->左
        rel_pos3 = dst_right - src_left  # 左->右
        rel_pos4 = dst_right - src_right  # 右->右

        def to_polar(pos):
            dist = torch.norm(pos, dim=1, keepdim=True)
            angle = torch.atan2(pos[:, 1], pos[:, 0]).unsqueeze(1)
            return torch.cat([dist, angle], dim=1)

        u1 = to_polar(rel_pos1)
        u2 = to_polar(rel_pos2)
        u3 = to_polar(rel_pos3)
        u4 = to_polar(rel_pos4)
        # 合并所有空间关系
        u = torch.cat([u1, u2, u3, u4], dim=1)  # [E, 8]

        # 计算高斯核权重（需要调整高斯核数量）
        weights = []
        for k in range(self.num_kernels):
            diff = u - self.means[k]
            std = torch.exp(self.log_stds[k]) + 1e-6
            z = diff / std
            exp_term = -0.5 * torch.sum(z ** 2, dim=1)
            weights.append(torch.exp(exp_term))

        weights = torch.stack(weights, dim=1)  # [E, K]
        return self.mlp(weights)


class WeldPointRegressionHGTNet(nn.Module):
    """
    焊缝点回归的图Transformer网络
    回归所有节点坐标，焊缝跟踪点是关键节点的中点
    """
    def __init__(self, node_dim=7, edge_dim=4, hidden_dim=64, num_layers=3, num_heads=8, dropout=0.3,num_kernels=4):
        super(WeldPointRegressionHGTNet, self).__init__()

        # 空间编码器
        self.spatial_encoder = SpatialGaussianEncoder(num_kernels, hidden_dim, dropout, input_dim=8)

        # 节点特征嵌入
        self.node_embed = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 边特征嵌入
        self.edge_embed = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 图Transformer层
        self.convs = nn.ModuleList()
        self.edge_updaters = nn.ModuleList()  # 边特征更新器

        for _ in range(num_layers):
            conv = TransformerConv(
                hidden_dim, hidden_dim,
                heads=num_heads,
                edge_dim=hidden_dim,
                concat=False
            )
            self.convs.append(conv)

            edge_updater = nn.Sequential(
                nn.Linear(3 * hidden_dim, hidden_dim),  # 3个特征：边自身+两端节点
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.edge_updaters.append(edge_updater)

        self.weld_point_regressor = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
            nn.Tanh()
        )

    def forward(self, data):
        x, edge_index, edge_attr= data.x, data.edge_index, data.edge_attr

        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        num_graphs = data.num_graphs if hasattr(data, 'num_graphs') else 1

        spatial_feat = self.spatial_encoder(edge_index, x[:, :4])  # 只使用坐标部分

        x = self.node_embed(x)
        edge_attr = self.edge_embed(edge_attr) + spatial_feat  # 特征融合
        # 多层图Transformer处理
        for i, (conv, edge_updater) in enumerate(zip(self.convs, self.edge_updaters)):
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)

            src, dst = edge_index[0], edge_index[1]

            edge_features = torch.cat([
                x[src],  # 源节点特征
                x[dst],  # 目标节点特征
                edge_attr  # 边自身特征
            ], dim=1)

            edge_attr = edge_updater(edge_features)

        graph_features = []
        for i in range(num_graphs):
            node_mask = (batch == i)
            node_feat = x[node_mask]
            node_pool = torch.mean(node_feat, dim=0)

            edge_mask = (batch[edge_index[0]] == i)
            edge_feat = edge_attr[edge_mask]
            edge_pool = torch.mean(edge_feat, dim=0) if edge_feat.numel() > 0 else torch.zeros_like(node_pool)

            combined_feat = torch.cat([node_pool, edge_pool], dim=0)
            graph_features.append(combined_feat)

        graph_features = torch.stack(graph_features)
        weld_points = self.weld_point_regressor(graph_features)
        return weld_points


def train_model():

    # 创建日志目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"GNNTransformer/log/training_logs_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    # 创建日志文件
    log_file = os.path.join(log_dir, "training_log.csv")
    param_file = os.path.join(log_dir, "hyperparameters.json")

    # 定义超参数（可调整部分）
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
        "image_folder": "GNNTransformer/datasets"
    }

    # 保存超参数到JSON文件
    with open(param_file, 'w') as f:
        json.dump(hyperparams, f, indent=2)

    # 创建CSV日志文件并写入表头
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "val_loss", "val_pixel_error","x_mae","y_mae","x_rmse","y_rmse",
            "learning_rate", "time_elapsed"
        ])

    # 训练集地址
    image_folder = hyperparams["image_folder"]

    print("Building graph dataset...")
    graph_data = prepare_dataset(image_folder)

    # 检查数据集是否为空
    if len(graph_data) == 0:
        print("Error: No valid graph data found!")
        return None

    print(f"Loaded {len(graph_data)} valid graphs")

    # 划分训练集和测试集
    split_idx = int(0.8 * len(graph_data))
    train_data = graph_data[:split_idx]
    test_data = graph_data[split_idx:]

    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

    train_dataset = WeldGraphDataset(train_data)
    test_dataset = WeldGraphDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=hyperparams["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=hyperparams["batch_size"], shuffle=False)

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = WeldPointRegressionHGTNet(
        node_dim=7,  # 节点特征维度，这一项与节点特征直接相关，应该配合节点特征数量进行修改
        edge_dim=4,  # 边特征维度，这一项与节点特征直接相关，应该配合节点特征数量进行修改
        hidden_dim= hyperparams["hidden_dim"],
        num_layers= hyperparams["num_layers"],
        num_heads= hyperparams["num_heads"],
        dropout= hyperparams["dropout"],
        num_kernels=8
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"], weight_decay=hyperparams["weight_decay"])

    # 学习率调度器
    max_lr = 0.01  # 设置最大学习率，根据你的经验或 hyperparams["learning_rate"] 调整
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=len(train_loader),
        epochs=hyperparams["num_epochs"],
        pct_start=0.1,  # 前10% epoch逐步升高到最大学习率
        anneal_strategy='cos',  # 余弦下降
        div_factor=25,  # 初始LR = max_lr / div_factor
        final_div_factor=1e2,  # 最终LR = max_lr / final_div_factor
    )

    # 训练循环
    num_epochs = hyperparams["num_epochs"]
    train_losses = []
    val_losses = []

    print("Starting training...")
    for epoch in range(num_epochs):
        epoch_start = datetime.datetime.now()

        # 训练阶段
        model.train()
        epoch_train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            pred_points = model(batch)  # 归一化坐标 [-1,1]
            true_points = batch.y

            # 映射回像素坐标
            if batch.img_size.dim() == 1:
                img_w = batch.img_size[0].repeat(batch.num_graphs)
                img_h = batch.img_size[1].repeat(batch.num_graphs)
            else:
                img_w, img_h = batch.img_size[:, 0], batch.img_size[:, 1]
            pred_x = (pred_points[:, 0] + 1) * 0.5 * img_w
            pred_y = (pred_points[:, 1] + 1) * 0.5 * img_h
            true_x = (true_points[:, 0] + 1) * 0.5 * img_w
            true_y = (true_points[:, 1] + 1) * 0.5 * img_h

            pred_pixels = torch.stack([pred_x, pred_y], dim=1)
            true_pixels = torch.stack([true_x, true_y], dim=1)

            diff = pred_pixels - true_pixels
            pixel_error = torch.mean(torch.sqrt(torch.sum(diff ** 2, dim=1) + 1e-6))
            loss = pixel_error
            # --------------------------------------------------------------------------------------

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyperparams["grad_clip"])  # 梯度裁剪
            optimizer.step()
            scheduler.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)  # 假如batch为8，则8个的平均值
        train_losses.append(avg_train_loss)

        # 训练过程的验证阶段
        model.eval()
        epoch_val_loss = 0
        epoch_pixel_errors = []  # 新增：存储当前epoch的像素误差
        epoch_x_mae_errors = []
        epoch_y_mae_errors = []
        epoch_x_rmse_errors = []
        epoch_y_rmse_errors = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)

                pred_points = model(batch)
                true_points = batch.y  # [x, y]

                if batch.img_size.dim() == 1:
                    img_w = batch.img_size[0].repeat(batch.num_graphs)
                    img_h = batch.img_size[1].repeat(batch.num_graphs)
                else:
                    img_w, img_h = batch.img_size[:, 0], batch.img_size[:, 1]
                pred_x = (pred_points[:, 0] + 1) * 0.5 * img_w
                pred_y = (pred_points[:, 1] + 1) * 0.5 * img_h
                true_x = (true_points[:, 0] + 1) * 0.5 * img_w
                true_y = (true_points[:, 1] + 1) * 0.5 * img_h

                pred_pixels = torch.stack([pred_x, pred_y], dim=1)
                true_pixels = torch.stack([true_x, true_y], dim=1)

                diff = pred_pixels-true_pixels
                pixel_error = torch.mean(torch.sqrt(torch.sum(diff**2, dim=1) + 1e-6))
                loss = pixel_error

                epoch_val_loss += loss.item()
                epoch_pixel_errors.append(pixel_error.cpu().numpy())

                # X / Y 方向误差
                diff_x = pred_x - true_x
                diff_y = pred_y - true_y

                x_mae = torch.mean(torch.abs(diff_x)).cpu().numpy()
                y_mae = torch.mean(torch.abs(diff_y)).cpu().numpy()
                x_rmse = torch.sqrt(torch.mean(diff_x ** 2)).cpu().numpy()
                y_rmse = torch.sqrt(torch.mean(diff_y ** 2)).cpu().numpy()

                epoch_x_mae_errors.append(x_mae)
                epoch_y_mae_errors.append(y_mae)
                epoch_x_rmse_errors.append(x_rmse)
                epoch_y_rmse_errors.append(y_rmse)


        avg_val_loss = epoch_val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

        avg_pixel_error = np.mean(epoch_pixel_errors) if epoch_pixel_errors else 0
        avg_x_mae_errors = np.mean(epoch_x_mae_errors) if epoch_x_mae_errors else  0
        avg_y_mae_errors = np.mean(epoch_y_mae_errors) if epoch_y_mae_errors else 0
        avg_x_rmse_errors = np.mean(epoch_x_rmse_errors) if epoch_x_rmse_errors else 0
        avg_y_rmse_errors = np.mean(epoch_y_rmse_errors) if epoch_y_rmse_errors else 0


        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        # 计算时间消耗
        epoch_time = (datetime.datetime.now() - epoch_start).total_seconds()

        # 打印并记录日志
        log_msg = (f"Epoch {epoch + 1}/{hyperparams['num_epochs']}, "
                   f"Train Loss: {avg_train_loss:.4f}, "
                   f"Val Loss: {avg_val_loss:.4f}, "
                   f"Val Pixel Error: {avg_pixel_error:.2f}px, "
                   f"X MAE: {avg_x_mae_errors:.2f},"
                   f"Y MAE: {avg_y_mae_errors:.2f},"
                   f"X RMSE: {avg_x_rmse_errors:.2f},"
                   f"Y RMSE: {avg_y_rmse_errors:.2f},"
                   f"LR: {current_lr:.2e}, "
                   f"Time: {epoch_time:.1f}s")

        print(log_msg)

        # 写入CSV文件
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                avg_train_loss,
                avg_val_loss,
                avg_pixel_error,
                avg_x_mae_errors,
                avg_y_mae_errors,
                avg_x_rmse_errors,
                avg_y_rmse_errors,
                current_lr,
                epoch_time
            ])

    loss_img_save_file = os.path.join(log_dir, "training_curve.png")
    # 可视化loss变化
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_img_save_file, dpi=300)
    plt.show()

    return model


# 运行训练
if __name__ == "__main__":
    model = train_model()