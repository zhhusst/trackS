import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import json
import os

# 加载标注数据
def load_annotations(image_folder):
    annotation_file = os.path.join(image_folder, "annotations.json")
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    return annotations

# 计算误差指标
def calculate_errors(pred_point, true_point):
    # 像素误差（欧氏距离）
    pixel_error = np.sqrt((pred_point[0] - true_point[0])**2 + (pred_point[1] - true_point[1])**2)
    
    # X/Y方向绝对误差
    x_mae = abs(pred_point[0] - true_point[0])
    y_mae = abs(pred_point[1] - true_point[1])
    
    # X/Y方向均方根误差
    x_rmse = np.sqrt((pred_point[0] - true_point[0])**2)
    y_rmse = np.sqrt((pred_point[1] - true_point[1])**2)
    
    return pixel_error, x_mae, y_mae, x_rmse, y_rmse

# 主函数
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
    
    # 1. 图像读取
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 2. 中值滤波 - 去除椒盐噪声
    filtered_image = cv2.medianBlur(image, 5)  # 5是滤波器的大小
    
    # 3. Otsu 阈值分割
    ret, otsu_image = cv2.threshold(filtered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. 提取ROI区域
    x1, y1, x2, y2 = 0, 0, 1792, 800
    roi_image = otsu_image[y1:y2, x1:x2]
    
    # ---------------------方向模板匹配+脊线跟踪------------------------------
    # 定义方向模板
    K1 = np.array([[0, 0, 1, 1, 1, 0, 0],
                   [0, 0, 1, 1, 1, 0, 0],
                   [0, 0, 1, 1, 1, 0, 0],
                   [0, 0, 1, 1, 1, 0, 0],
                   [0, 0, 1, 1, 1, 0, 0]])

    K2 = np.array([[0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 1, 1, 1, 1, 0],
                   [0, 1, 1, 1, 1, 1, 0],
                   [0, 1, 1, 1, 1, 1, 0],
                   [0, 0, 0, 0, 0, 0, 0]])

    K3 = np.array([[1, 1, 1, 0, 0, 0, 0],
                   [0, 1, 1, 1, 0, 0, 0],
                   [0, 0, 1, 1, 1, 0, 0],
                   [0, 0, 0, 1, 1, 1, 0],
                   [0, 0, 0, 0, 1, 1, 1]])

    K4 = np.array([[0, 0, 0, 0, 1, 1, 1],
                   [0, 0, 0, 1, 1, 1, 0],
                   [0, 0, 1, 1, 1, 0, 0],
                   [0, 1, 1, 1, 0, 0, 0],
                   [1, 1, 1, 0, 0, 0, 0]])

    templates = [K1, K2, K3, K4]

    # 优化后的计算模板响应函数
    def compute_template_response_optimized(image, templates):
        responses = []
        for template in templates:
            response = convolve2d(image, template, mode='same', boundary='fill')
            responses.append(response)
        responses = np.stack(responses, axis=-1)
        return np.max(responses, axis=-1)

    # 优化后的方向模板法
    def direction_template_method_optimized(image, roi, templates):
        y_min, y_max, x_min, x_max = roi
        roi_image = image[y_min:y_max, x_min:x_max]
        Hk_image_roi = compute_template_response_optimized(roi_image, templates)
        Hk_image_roi = cv2.normalize(Hk_image_roi, None, 0, 255, cv2.NORM_MINMAX)
        Hk_image = np.zeros_like(image)
        Hk_image[y_min:y_max, x_min:x_max] = Hk_image_roi
        return Hk_image

    # 找到ROI区域内每一列的最大值位置作为条纹中心
    def find_ridge_centers_in_roi(Hk_image, roi):
        y_min, y_max, x_min, x_max = roi
        ridge_centers = []
        for x in range(x_min, x_max):
            y_max_in_col = np.argmax(Hk_image[y_min:y_max, x]) + y_min
            ridge_centers.append((y_max_in_col, x))
        return ridge_centers

    # 计算方向模板响应（仅在ROI区域进行）
    start = time.time()
    Hk_image = direction_template_method_optimized(otsu_image, (y1, y2, x1, x2), templates)
    end = time.time()
    usetime = end - start
    print(f"计算整张图像ROI区域的Hk_image时间: {usetime:.4f}秒")

    # 仅在ROI区域计算条纹光中心
    ridge_centers_in_roi = find_ridge_centers_in_roi(Hk_image, (y1, y2, x1, x2))

    # ------------计算中心线上的特征点--------------------------------
    # 条纹光中心线的坐标数据（y, x）
    coordinates = ridge_centers_in_roi

    # 从坐标中提取y坐标
    y_values = np.array([coord[0] for coord in coordinates])

    # 计算斜率Ki的函数
    def compute_slope(y_values):
        slope = np.zeros(len(y_values))
        for i in range(4, len(y_values) - 4):
            slope[i] = ((y_values[i + 1] - y_values[i - 1]) / 2 + 
                        (y_values[i + 2] - y_values[i - 2]) / 4 + 
                        (y_values[i + 3] - y_values[i - 3]) / 6 + 
                        (y_values[i + 4] - y_values[i - 4]) / 8) / 4
        return slope

    # 计算斜率
    slope_values = compute_slope(y_values)

    # 提取特征点a
    def extract_feature_points(slope_values, threshold=0.7):
        optimal_point = None
        max_change = 0

        for i in range(10, len(slope_values) - 10):
            slope_change = abs(slope_values[i] - slope_values[i - 10]) + abs(slope_values[i] - slope_values[i + 10])
            if slope_change > threshold and slope_change > max_change:
                optimal_point = i
                max_change = slope_change
        return optimal_point

    a = extract_feature_points(slope_values)

    if a is None:
        print("特征点未找到，无法继续")
        return

    feature_point = coordinates[a]

    # ----------------------焊缝特征点的修正
    from sklearn.linear_model import RANSACRegressor

    # 使用RANSAC拟合直线的函数
    def fit_line_ransac(points):
        points = np.array(points)
        X = points[:, 1].reshape(-1, 1)
        y = points[:, 0]
        ransac = RANSACRegressor(residual_threshold=2.0)
        ransac.fit(X, y)
        slope = ransac.estimator_.coef_[0]
        intercept = ransac.estimator_.intercept_
        return slope, intercept

    # 计算两直线交点的函数
    def compute_intersection(line1, line2):
        a1, b1 = line1
        a2, b2 = line2
        if b1 == b2:
            raise ValueError("Lines are parallel and do not intersect.")
        x = (a2 - a1) / (b1 - b2)
        y = a1 + b1 * x
        return x, y

    # 划分左右区域并拟合直线
    feature_point_coord = coordinates[a]
    left_points = [p for p in coordinates if p[1] < feature_point_coord[1] - 3]
    right_points = [p for p in coordinates if p[1] > feature_point_coord[1] + 3]

    # 使用RANSAC拟合左右两侧的直线
    line_left = fit_line_ransac(left_points)
    line_right = fit_line_ransac(right_points)

    # 计算交点
    try:
        intersection = compute_intersection(line_left, line_right)
    except ValueError:
        print("直线平行，无法计算交点")
        return
    
    # 预测点坐标 (x, y)
    pred_point = np.array([intersection[0], intersection[1]])
    
    # 计算误差指标
    pixel_error, x_mae, y_mae, x_rmse, y_rmse = calculate_errors(pred_point, true_point)
    
    # 输出结果
    print(f"Val Pixel Error: {pixel_error:.2f}px, "
          f"X MAE: {x_mae:.2f}, Y MAE: {y_mae:.2f}, "
          f"X RMSE: {x_rmse:.2f}, Y RMSE: {y_rmse:.2f}")

if __name__ == "__main__":
    main()