import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from matplotlib.patches import Arc  # 新增导入语句

def fit_two_lines(points, threshold=1.0):
    """对点集进行两次RANSAC直线拟合"""
    if len(points) < 4:  # 最少需要4个点进行两次拟合
        return None, None

    # 第一次拟合
    k1, b1 = ransac_fit(points, threshold=threshold)
    if k1 is None:
        return None, None

    # 计算第一次拟合的内点
    distances = np.abs(k1 * points[:, 0] - points[:, 1] + b1) / np.sqrt(k1 ** 2 + 1)
    inlier_mask = distances < threshold
    outliers = points[~inlier_mask]

    # 第二次拟合
    k2, b2 = ransac_fit(outliers, threshold=threshold) if len(outliers) >= 2 else (None, None)

    # 斜率排序处理
    lines = []
    if k1 is not None:
        lines.append((k1, b1))
    if k2 is not None:
        lines.append((k2, b2))

    # 按斜率绝对值排序
    lines.sort(key=lambda x: abs(x[0]), reverse=True)
    return lines[0] if len(lines) >= 1 else None, lines[1] if len(lines) >= 2 else None

def ransac_fit(points, max_iters=100, threshold=1.0):
    """手动实现RANSAC直线拟合"""
    if len(points) < 2:
        return None, None

    best_k, best_b = None, None
    best_inliers = []

    for _ in range(max_iters):
        # 随机选取两个点
        idx = np.random.choice(len(points), 2, replace=False)
        p1, p2 = points[idx[0]], points[idx[1]]
        x1, y1 = p1
        x2, y2 = p2

        # 计算直线参数 ax + by + c = 0
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2

        # 计算距离并统计内点
        distances = []
        for (x, y) in points:
            numerator = abs(a * x + b * y + c)
            denominator = np.sqrt(a ** 2 + b ** 2)
            if denominator < 1e-6:
                continue
            distances.append(numerator / denominator)

        if not distances:
            continue

        inliers = np.where(np.array(distances) < threshold)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            # 使用内点重新拟合
            x_inliers = points[inliers, 0]
            y_inliers = points[inliers, 1]
            A = np.vstack([x_inliers, np.ones_like(x_inliers)]).T
            k, b = np.linalg.lstsq(A, y_inliers, rcond=None)[0]
            best_k, best_b = k, b

    return best_k, best_b

def extract_centers(roi_gray, direction='left', initial_cols=4, window=10, ransac_thresh=1.0, show_visualization=True):
    """提取单侧激光条纹中心点"""
    h, w = roi_gray.shape
    points = []
    # 其实这里的ROI区域的一半就是列灰度重心，（因为前面ROI区域就是以灰度重心为原点，上下左右扩充的）
    col_range = range(0,int(w/2)) if direction == 'left' else reversed(range(int(w/2),w))

    # 创建可视化画布
    if show_visualization:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(roi_gray, cmap='gray')
        ax[0].set_title(f"ROI区域 ({direction} side)")

    for local_x in col_range:
        if len(points) < initial_cols:
            # 初始阶段：获取灰度最大值点
            col_data = roi_gray[:, local_x]
            max_val = np.max(col_data)
            if max_val == 0:
                continue
            y_local = np.median(np.where(col_data == max_val)[0])
            points.append((local_x, y_local))
            # 可视化
            if show_visualization:  # 控制初始点可视化
                ax[0].scatter(local_x, y_local, c='red', s=30)
                ax[0].set_title("Initial Max Value Points")
                plt.pause(0.1)
        else:
            # RANSAC拟合
            points_array = np.array(points[-initial_cols:])  # 使用最近的点
            k, b = ransac_fit(points_array, threshold=ransac_thresh)
            if k is None:
                continue

            # 预测当前列位置
            y_pred = k * local_x + b
            y_start = max(0, int(y_pred - window))
            y_end = min(h, int(y_pred + window))


            # 灰度重心计算
            window_data = roi_gray[y_start:y_end, local_x]
            if window_data.sum() == 0:
                continue
            # 生成正确的索引范围
            y_indices = np.arange(y_start, y_end)
            if len(y_indices) != len(window_data):
                continue  # 防止形状不匹配
            y_centroid = np.sum(y_indices * window_data) / window_data.sum()
            points.append((local_x, y_centroid))

            # 绘制预测轨迹
            if show_visualization:  # 控制初始点可视化
                ax[1].clear()
                ax[1].imshow(roi_gray, cmap='gray')
                ax[1].scatter([p[0] for p in points], [p[1] for p in points], c='blue', s=15)
                x_vals = np.array([local_x - 10, local_x + 10])
                y_vals = k * x_vals + b
                ax[1].plot(x_vals, y_vals, 'r--', linewidth=1)
                ax[1].set_title("Real-time Tracking")
                plt.pause(0.01)
    if show_visualization:  # 控制初始点可视化
        plt.close()
    return np.array(points)

def calculate_grayscale_centroid(img):
    """计算行和列方向的灰度重心（对应文档公式）"""
    # 列方向灰度重心 (COL_c)
    col_sum = np.sum(img, axis=0)
    col_indices = np.arange(img.shape[1])
    col_centroid = np.sum(col_indices * col_sum) / (col_sum.sum() + 1e-6)

    # 行方向灰度重心 (ROW_c)
    row_sum = np.sum(img, axis=1)
    row_indices = np.arange(img.shape[0])
    row_centroid = np.sum(row_indices * row_sum) / (row_sum.sum() + 1e-6)

    return int(row_centroid), int(col_centroid)

def median_filter_points(points, window_size=5):
    """
    对二维坐标点进行滑动窗口中值滤波
    该函数主要应用于提取完条纹中心点之后。
    """

    if len(points) < 1:
        return points

    # 转换为numpy数组
    points = np.array(points)
    filtered = np.zeros_like(points)

    half_win = window_size // 2

    for i in range(len(points)):
        # 计算滑动窗口边界
        start = max(0, i - half_win)
        end = min(len(points), i + half_win + 1)

        # 计算窗口内坐标中值
        filtered[i, 0] = np.median(points[start:end, 0])
        filtered[i, 1] = np.median(points[start:end, 1])

    return filtered.tolist()

def recover_endpoints(filtered_points, raw_points, direction='left', img=None, roi=None, debug_plot=False):
    """改进后的端点补偿算法（方向敏感的半圆判断）"""
    if not filtered_points or not raw_points:
        return filtered_points

    # 转换坐标到numpy数组
    filtered = np.array(filtered_points)
    raw = np.array(raw_points)

    # 记录补偿过程用于可视化
    search_steps = []

    # 根据方向确定初始点和半圆方向
    if direction == 'left':
        current_idx = np.argmax(filtered[:, 0])  # 最右侧点作为起点
        semi_direction = 'right'  # 向右搜索
    else:
        current_idx = np.argmin(filtered[:, 0])  # 最左侧点作为起点
        semi_direction = 'left'  # 向左搜索

    current_point = filtered[current_idx]
    recovered = [tuple(current_point)]

    def in_semi_circle(target, center, direction, radius=2):
        dx = target[0] - center[0]
        dy = target[1] - center[1]
        # 根据方向选择半圆条件
        if direction == 'right':
            return (dx ** 2 + dy ** 2 <= radius ** 2) and (dx > 0)
        else:
            return (dx ** 2 + dy ** 2 <= radius ** 2) and (dx < 0)

    while True:
        candidates = [p for p in raw
                      if in_semi_circle(p, current_point, semi_direction)]
        search_steps.append((current_point.copy(), candidates))

        if not candidates:
            break

        candidates = np.array(candidates)
        distances = np.abs(candidates[:, 0] - current_point[0])
        nearest_idx = np.argmin(distances)
        current_point = candidates[nearest_idx]
        recovered.append(tuple(current_point))

    # 可视化调试（新增方向敏感的半圆绘制）
    if debug_plot and img is not None and roi is not None:
        plt.figure(figsize=(12, 6))
        plt.imshow(img, cmap='gray')
        plt.title(f"Endpoint Recovery Process ({direction} side)")

        # 绘制ROI区域
        roi_rect = plt.Rectangle((roi[0], roi[2]), roi[1] - roi[0], roi[3] - roi[2],
                                 edgecolor='yellow', fill=False, linestyle='--')
        plt.gca().add_patch(roi_rect)

        # 绘制原始点和滤波后点
        plt.scatter(raw[:, 0], raw[:, 1], c='blue', s=30, marker='x', label='Raw Points')
        plt.scatter(filtered[:, 0], filtered[:, 1], c='red', s=50, edgecolors='white', label='Filtered Points')

        # 动态绘制搜索过程
        for i, (center, candidates) in enumerate(search_steps):
            # 根据方向选择半圆绘制方式
            theta1 = 270 if semi_direction == 'right' else 90
            theta2 = theta1 + 180
            arc = Arc(center, 4, 4, theta1=theta1, theta2=theta2,
                      edgecolor='orange', linestyle='--', lw=1.5)
            plt.gca().add_patch(arc)

            # 绘制候选点
            if candidates:
                plt.scatter([p[0] for p in candidates], [p[1] for p in candidates],
                            c='lime', s=80, marker='s', edgecolors='black',
                            label='Candidates' if i == 0 else None)

            plt.scatter(center[0], center[1], c='purple', s=100, marker='*',
                        label='Search Center' if i == 0 else None)

        # 绘制恢复点
        plt.scatter([p[0] for p in recovered], [p[1] for p in recovered],
                    c='cyan', s=50, marker='o', edgecolors='black',
                    label='Recovered Points')
        plt.legend(loc='upper right')
        plt.show()

    # 合并结果
    combined = np.unique(np.concatenate([filtered, recovered]), axis=0)
    return sorted(combined.tolist(), key=lambda p: p[0])


def compute_intersection(line1, line2):
    """计算两条直线的交点"""
    k1, b1 = line1
    k2, b2 = line2

    # 处理平行情况
    if np.isclose(k1, k2):
        return None

    # 计算交点坐标
    x = (b2 - b1) / (k1 - k2)
    y = k1 * x + b1
    return (x, y)

def extract_laser_roi(img_path,
                      col1th=50, col2th=50,
                      row1th=20, row2th=20,
                      median_ksize=3,  # 仅保留文档明确要求的预处理
                      thresh_val=200,
                      show_visualization=False):
    """
    ⅰ. 首先要定位焊缝的ROI区域---灰度重心法
    ⅱ. 根据灰度中心法的结果座标，建立ROI区域
    ⅲ. 从ROI区域内部的左边缘开始逐列提取焊缝的中心点，并且在计算的过程中顺带算一下条纹线的曲率分布
    ⅳ. 使用提取到的中心点进行直线拟合
    ⅴ. 去除第一条直线包含的中心点，取剩余的中心点拟合另一条直线。
    ⅵ. 求两条直线的交点作为焊缝特征点
    :param img_path:
    :param col1th:
    :param col2th:
    :param row1th:
    :param row2th:
    :param median_ksize:
    :param thresh_val:
    :param show_visualization:
    :return:
    """


    # 预处理结果可视化
    if show_visualization:
        plt.figure(figsize=(16, 4))

    # 图像读取
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if show_visualization:
        plt.subplot(1, 4, 1), plt.imshow(img, cmap='gray'), plt.title("Original")

    # 技术文档规定的预处理步骤：
    # 1. 中值滤波去噪（对应文档的噪声抑制要求）
    img = cv2.medianBlur(img, ksize=median_ksize)
    if show_visualization:
        plt.subplot(1, 4, 2), plt.imshow(img, cmap='gray'), plt.title("After Median")

    # 2. 二值化处理（利用激光高灰度特性）
    # _, img = cv2.threshold(img, thresh=thresh_val, maxval=255,
    #                        type=cv2.THRESH_BINARY)
    _, img = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if show_visualization:
        plt.subplot(1, 4, 3), plt.imshow(img, cmap='gray'), plt.title("After Threshold")

    # 灰度重心计算---目的是求取ROI区域的中心
    row_c, col_c = calculate_grayscale_centroid(img)

    # 可视化标注（在所有预处理阶段图像上标注）
    if show_visualization:
        # 二值化后标注
        plt.subplot(1, 4, 4)
        plt.imshow(img, cmap='gray')
        plt.scatter(col_c, row_c, c='red', s=80, edgecolors='white', linewidth=1.5, marker='o')
        plt.title("Threshold with Centroid")
    if show_visualization:
        plt.show()

    # ROI区域确定（完全对应公式3.1）
    # 其实ROI的是一个矩型区域，该矩形区域的中心就是灰度中心 列灰度重心col_c  行灰度重心row_c
    roi_col_start = max(0, col_c - col1th)
    roi_col_end = min(img.shape[1], col_c + col2th)
    roi_row_start = max(0, row_c - row1th)
    roi_row_end = min(img.shape[0], row_c + row2th)

    # 提取得到ROI区域
    roi_gray = img[roi_row_start:roi_row_end, roi_col_start:roi_col_end]

    # 基于ROI区域提取左半部分的条纹中心点
    left_centers_local = extract_centers(roi_gray, direction='left',show_visualization=show_visualization)
    left_centers_global = [
        (x + roi_col_start, y + roi_row_start)
        for (x, y) in left_centers_local
    ]

    # 基于ROI区域提取右半部分的条纹中心点
    right_centers_local = extract_centers(roi_gray, direction='right',show_visualization=show_visualization)
    right_centers_global = [
        (x + roi_col_start, y + roi_row_start)
        for (x, y) in right_centers_local
    ]

    #################### 新增离群点处理方法 ####################
    # 保存滤波前的原始点用于可视化对比
    raw_left = left_centers_global.copy()
    raw_right = right_centers_global.copy()

    # 执行中值滤波
    if len(left_centers_global) > 3:
        left_centers_global = median_filter_points(left_centers_global, window_size=9)
    if len(right_centers_global) > 3:
        right_centers_global = median_filter_points(right_centers_global, window_size=9)

    # 新增可视化代码
    if show_visualization:
        plt.figure(figsize=(12, 6))

        # 子图1：左侧中心点对比
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        # 绘制原始点（蓝色）
        if len(raw_left) > 0:
            x_raw = [p[0] for p in raw_left]
            y_raw = [p[1] for p in raw_left]
            plt.scatter(x_raw, y_raw, c='blue', s=30, marker='x', label='Raw Points')
        # 绘制滤波后点（红色）
        if len(left_centers_global) > 0:
            x_filt = [p[0] for p in left_centers_global]
            y_filt = [p[1] for p in left_centers_global]
            plt.scatter(x_filt, y_filt, c='red', s=50, edgecolors='white', label='Filtered')
        plt.title("Left Centers Comparison")
        plt.legend()

        # 子图2：右侧中心点对比
        plt.subplot(1, 2, 2)
        plt.imshow(img, cmap='gray')
        # 绘制原始点（蓝色）
        if len(raw_right) > 0:
            x_raw = [p[0] for p in raw_right]
            y_raw = [p[1] for p in raw_right]
            plt.scatter(x_raw, y_raw, c='blue', s=30, marker='x', label='Raw Points')
        # 绘制滤波后点（红色）
        if len(right_centers_global) > 0:
            x_filt = [p[0] for p in right_centers_global]
            y_filt = [p[1] for p in right_centers_global]
            plt.scatter(x_filt, y_filt, c='red', s=50, edgecolors='white', label='Filtered')
        plt.title("Right Centers Comparison")
        plt.legend()

        plt.tight_layout()
        plt.show()

    # 使用中值滤波对中心点进行过滤之后，会将部分严重的离群点进行滤出，但是随之而来的负面影响就是对条纹线端点处的中心点也滤掉了
    # 为此我们添加补救措施，以左侧点为例，我们拿出过滤之后的中心点中的最右侧点filtered_r_current_point，以这个点为圆心，绘制一个半圆范围，该半圆只有右半部分，
    # 半圆的半径为2，因为激光条纹是连续的，也就是说基本上每一列都会有一个中心点，如果这个激光条纹右侧还有中心点的话，那么这个中心点应该是落在
    # 半圆内的。从而我们可以以此方法来判断filtered_r_current_point是否还有中心点存在，如果有的话，那么filtered_r_current_point被更新成落在
    # 半圆之内的点。再次以新的filtered_r_current_point建立右半圆判断右侧是否还有中心点存在。直到一个filtered_r_current_point的右半圆中没有中心点落在内。
    # 这个时候就证明此时的filtered_r_current_point是条纹激光线真正的右端点。

    # 左侧点补偿（向右搜索）
    left_centers_global = recover_endpoints(
        left_centers_global, raw_left, 'left',
        img=img,  # 传入原始图像
        roi=(roi_col_start, roi_col_end, roi_row_start, roi_row_end),
        debug_plot=show_visualization
    )
    # 右侧点补偿 (向左搜索)
    right_centers_global = recover_endpoints(
        right_centers_global, raw_right, 'right',
        img=img,
        roi=(roi_col_start, roi_col_end, roi_row_start, roi_row_end),
        debug_plot=show_visualization
    )

    # 拼接修复后的中心点
    all_centers = []
    if len(left_centers_global) > 0:
        all_centers.extend(left_centers_global)
    if len(right_centers_global) > 0:
        all_centers.extend(right_centers_global)
    all_centers = np.array(all_centers)

    # 拟合两条直线
    line1, line2 = fit_two_lines(all_centers)
    intersection_point = None

    # 计算交点
    if line1 is not None and line2 is not None:
        intersection_point = compute_intersection(line1, line2)

    # 最终可视化
    if show_visualization:
        plt.figure(figsize=(12, 6))
        plt.imshow(img, cmap='gray')
        plt.title("焊缝特征点检测结果")

        # 绘制ROI区域
        roi_rect = plt.Rectangle((roi_col_start, roi_row_start),
                                 roi_col_end - roi_col_start,
                                 roi_row_end - roi_row_start,
                                 edgecolor='yellow', fill=False, linestyle='--')
        plt.gca().add_patch(roi_rect)

        # 绘制中心点
        if len(left_centers_global) > 0:
            left_points = np.array(left_centers_global)
            plt.scatter(left_points[:, 0], left_points[:, 1], c='blue', s=20, label='左侧点')
        if len(right_centers_global) > 0:
            right_points = np.array(right_centers_global)
            plt.scatter(right_points[:, 0], right_points[:, 1], c='green', s=20, label='右侧点')

        # 绘制拟合直线
        x_min, x_max = roi_col_start, roi_col_end
        if line1 is not None:
            x_vals = np.linspace(x_min, x_max, 100)
            y_vals = line1[0] * x_vals + line1[1]
            plt.plot(x_vals, y_vals, 'r--', linewidth=2, label='左侧拟合线')
        if line2 is not None:
            x_vals = np.linspace(x_min, x_max, 100)
            y_vals = line2[0] * x_vals + line2[1]
            plt.plot(x_vals, y_vals, 'm--', linewidth=2, label='右侧拟合线')

        # 绘制交点
        if intersection_point is not None:
            plt.scatter(intersection_point[0], intersection_point[1],
                        c='yellow', s=200, marker='*',
                        edgecolors='red', linewidth=2,
                        label=f'特征点 ({intersection_point[0]:.1f}, {intersection_point[1]:.1f})')

        plt.legend()
        plt.show()

    # 返回ROI坐标和交点坐标
    roi_coords = (roi_row_start, roi_row_end, roi_col_start, roi_col_end)

    return roi_coords, intersection_point

def calculate_errors(pred_point, true_point):
    """计算各种误差指标"""
    # 像素误差（欧氏距离）
    pixel_error = np.sqrt((pred_point[0] - true_point[0])**2 + (pred_point[1] - true_point[1])**2)
    
    # X/Y方向绝对误差
    x_mae = abs(pred_point[0] - true_point[0])
    y_mae = abs(pred_point[1] - true_point[1])
    
    # X/Y方向均方根误差
    x_rmse = np.sqrt((pred_point[0] - true_point[0])**2)
    y_rmse = np.sqrt((pred_point[1] - true_point[1])**2)
    
    return pixel_error, x_mae, y_mae, x_rmse, y_rmse

def load_annotations(image_folder):
    """加载标注数据"""
    annotation_file = os.path.join(image_folder, "annotations.json")
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    return annotations

def main():
    # 图像路径和标注路径
    image_path = '/home/z/seam_tracking_ws/src/paper1_pkg/GNNTransformer/SecondCode/train_data/25051928-0001-OFF.png'
    image_folder = os.path.dirname(image_path)
    # 加载标注数据
    annotations = load_annotations(image_folder)
    image_name = os.path.basename(image_path)
    true_point = np.array(annotations[image_name][0]) if image_name in annotations else None
    if true_point is None:
        print(f"Warning: No annotation found for {image_name}")
        return
    # 记录开始时间
    start_time = time.perf_counter()

    # 调用对比方法2的核心函数
    roi_coords, intersection_point = extract_laser_roi(
        image_path,
        col1th=80, col2th=80,
        row1th=60, row2th=60,
        median_ksize=3,
        thresh_val=120,
        show_visualization=False  # 关闭可视化以提高速度
    )
    # 记录结束时间
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    if intersection_point is None:
        print("Error: Failed to detect weld point")
        return
    # 计算误差指标
    pixel_error, x_mae, y_mae, x_rmse, y_rmse = calculate_errors(intersection_point, true_point)
    # 输出结果（与您的方法格式一致）
    print(f"Val Pixel Error: {pixel_error:.2f}px, "
          f"X MAE: {x_mae:.2f}, Y MAE: {y_mae:.2f}, "
          f"X RMSE: {x_rmse:.2f}, Y RMSE: {y_rmse:.2f}, "
          f"Time: {elapsed:.4f}s")


    return 0


if __name__ == "__main__":
    main()
