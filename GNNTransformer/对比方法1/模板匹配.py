import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# 1. 图像读取
image = cv2.imread('/home/z/seam_tracking_ws/src/paper1_pkg/GNNTransformer/SecondCode/train_data/25051928-0001-OFF.png', cv2.IMREAD_GRAYSCALE)

# 2. 中值滤波 - 去除椒盐噪声
filtered_image = cv2.medianBlur(image, 5)  # 5是滤波器的大小

# 3. Otsu 阈值分割
# 使用Otsu方法自动计算最佳阈值并进行图像分割
ret, otsu_image = cv2.threshold(filtered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 4. 提取ROI区域 - 假设我们知道感兴趣区域 (ROI) 的坐标，进行裁剪
# 假设 ROI 的坐标为 (x1, y1) 到 (x2, y2)
x1, y1, x2, y2 = 0, 0, 1792, 800  # 示例ROI坐标
# x1, y1, x2, y2 = 0, 0, 944, 706  # 示例ROI坐标
roi_image = otsu_image[y1:y2, x1:x2]

# 5. 显示图像
plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image')

plt.subplot(2, 2, 3)
plt.imshow(otsu_image, cmap='gray')
plt.title('Otsu Thresholding')

plt.subplot(2, 2, 4)
plt.imshow(roi_image, cmap='gray')
plt.title('ROI Extracted Image')

plt.show()

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
    """
    对每个模板执行卷积操作，返回响应图像
    """
    responses = []
    for template in templates:
        response = convolve2d(image, template, mode='same', boundary='fill')  # 使用卷积进行响应计算
        responses.append(response)

    # 将每个模板的响应堆叠在一起，沿着最后一个维度（即模板维度）合并
    responses = np.stack(responses, axis=-1)

    # 返回每个像素的最大响应
    return np.max(responses, axis=-1)  # 对应每个像素取最大响应

# 优化后的方向模板法
def direction_template_method_optimized(image, roi, templates):
    # 只在ROI区域内进行操作
    y_min, y_max, x_min, x_max = roi

    # 提取ROI区域
    roi_image = image[y_min:y_max, x_min:x_max]

    # 计算模板响应（卷积操作）
    Hk_image_roi = compute_template_response_optimized(roi_image, templates)
    # 对ROI响应图像进行归一化处理，避免过大的值或过小的值影响
    Hk_image_roi = cv2.normalize(Hk_image_roi, None, 0, 255, cv2.NORM_MINMAX)

    # 将ROI区域的响应图像填回原图的相应位置
    Hk_image = np.zeros_like(image)
    Hk_image[y_min:y_max, x_min:x_max] = Hk_image_roi

    return Hk_image

# 找到ROI区域内每一列的最大值位置作为条纹中心
def find_ridge_centers_in_roi(Hk_image, roi):
    y_min, y_max, x_min, x_max = roi
    ridge_centers = []
    # 仅在 ROI 范围内操作
    for x in range(x_min, x_max):
        # 找到该列最大值的位置，仅考虑 y_min 到 y_max 的范围
        y_max_in_col = np.argmax(Hk_image[y_min:y_max, x]) + y_min  # 相对 y_min 的偏移
        ridge_centers.append((y_max_in_col, x))  # (行坐标, 列坐标)为条纹中心
    return ridge_centers


# 将条纹中心标注到ROI区域的图像上
def mark_ridge_centers_on_roi(image, ridge_centers):
    # 将灰度图像转换为BGR图像
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for center in ridge_centers:
        y, x = center
        # 在条纹中心处标记一个红色圆圈 (BGR格式)
        cv2.circle(image_bgr, (x, y), 1, (0, 0, 255), -1)  # 圆心为条纹中心，颜色为红色

    return image_bgr

# 计算方向模板响应（仅在ROI区域进行）---
start = time.time()
Hk_image = direction_template_method_optimized(otsu_image, (y1, y2, x1, x2), templates)
end = time.time()
usetime =  end-start
print("计算整张图像ROI区域的Hk_image包含ROI区域中每个像素的模板相似度计算时间和模板选取时间：",usetime,"秒")

# 仅在ROI区域计算条纹光中心
ridge_centers_in_roi = find_ridge_centers_in_roi(Hk_image, (y1, y2, x1, x2))

# 将条纹中心标注到原图上（仅标注ROI区域）
image_with_ridges_in_roi = mark_ridge_centers_on_roi(image, ridge_centers_in_roi)

# 显示结果
plt.figure(figsize=(10, 10))
plt.imshow(image_with_ridges_in_roi, cmap='gray')
plt.title('Image with Ridge Centers in ROI')
plt.show()

# ------------计算中心线上的特征点--------------------------------
import numpy as np
import matplotlib.pyplot as plt

# 条纹光中心线的坐标数据（y, x）
coordinates = ridge_centers_in_roi

# 从坐标中提取y坐标
y_values = np.array([coord[0] for coord in coordinates])


# 计算斜率Ki的函数
def compute_slope(y_values):
    slope = np.zeros(len(y_values))  # 用于存储每个点的斜率
    for i in range(4, len(y_values) - 4):
        # 计算该点的斜率（参考公式4）
        slope[i] = ((y_values[i + 1] - y_values[i - 1]) / 2 + (y_values[i + 2] - y_values[i - 2]) / 4 + (
                    y_values[i + 3] - y_values[i - 3]) / 6 + (y_values[i + 4] - y_values[i - 4]) / 8)/4
    return slope

# 计算斜率
slope_values = compute_slope(y_values)

# 绘制斜率曲线
plt.figure(figsize=(10, 6))
plt.plot(slope_values, label="Slope Curve")
plt.title("Slope Curve of Laser Stripe Center Line")
plt.xlabel("Index")
plt.ylabel("Slope")
plt.legend()
plt.show()

# 提取特征点a和b
def extract_feature_points(slope_values, threshold=0.7):
    """
        提取全局最优的特征点
        :param slope_values: 斜率值列表
        :param threshold: 判定特征点的斜率变化阈值
        :return: 全局最优特征点的索引
        """
    optimal_point = None
    max_change = 0  # 用于记录最大斜率变化

    for i in range(10, len(slope_values) - 10):  # 确保索引不越界
        # 计算当前点的斜率变化
        slope_change = abs(slope_values[i] - slope_values[i - 10]) + abs(slope_values[i] - slope_values[i + 10])

        # 如果斜率变化超过阈值且是最大的变化
        if slope_change > threshold and slope_change > max_change:
            optimal_point = i
            max_change = slope_change  # 更新最大变化

    return optimal_point

# a是ROI中的第i个点作为斜率的极值点
a = extract_feature_points(slope_values)

# 如果找到特征点a和b
if a is not None:
    print(f"Feature points: a = {a}")
    feature_points = coordinates[a]

    # 可视化特征点
    plt.figure(figsize=(10, 6))
    plt.plot(slope_values, label="Slope Curve")
    plt.scatter(a, slope_values[a], color='red', label='Feature Point a')
    plt.title("Feature Points on Slope Curve")
    plt.xlabel("Index")
    plt.ylabel("Slope")
    plt.legend()
    plt.show()
else:
    print("Feature points a could not be found.")


# 在原图上标注特征点
def mark_feature_point_on_image(image, feature_point):
    """
    在原图上标注特征点
    :param image: 原图像 (灰度图)
    :param feature_point: 特征点 (y, x) 坐标
    :return: 带标注的图像
    """
    # 将灰度图像转换为BGR彩色图像，便于标注
    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # 在特征点处标记一个绿色圆圈
    y, x = feature_point
    cv2.circle(image_bgr, (x, y), radius=5, color=(0, 255, 0), thickness=-1)  # (B, G, R)格式

    return image_bgr


# 标注特征点
if a is not None:
    marked_image = mark_feature_point_on_image(image, feature_points)

    # 显示结果
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))  # 转换为RGB格式以适配 Matplotlib
    plt.title('Original Image with Feature Point')
    plt.axis('off')
    plt.show()
else:
    print("Feature points could not be marked because they were not found.")


# ----------------------焊缝特征点的修正
"""
基于得到的feature_point，以计算出来的特征点为界，将图像的行坐标划分为2个区域，分别对feature_point的左右两侧的点进行直线拟合（可以使用最小二乘拟合算法）。
然后得到两直线，计算两直线的交点坐标。该交点就是最终的焊缝点
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor


# 使用RANSAC拟合直线的函数
def fit_line_ransac(points):
    """
    使用RANSAC算法拟合直线
    :param points: 点的坐标列表 [(y1, x1), (y2, x2), ...]
    :return: 直线参数 (slope, intercept)，对应 y = slope * x + intercept
    """
    # 将点转换为 numpy 数组
    points = np.array(points)

    # RANSAC需要输入的是 (x, y) 格式，所以我们交换点的顺序
    X = points[:, 1].reshape(-1, 1)  # x 坐标
    y = points[:, 0]  # y 坐标

    # 初始化RANSAC回归模型
    ransac = RANSACRegressor(residual_threshold=2.0)

    # 拟合数据
    ransac.fit(X, y)

    # 获取拟合的参数（斜率和截距）
    slope = ransac.estimator_.coef_[0]  # 斜率
    intercept = ransac.estimator_.intercept_  # 截距

    return slope, intercept

# 计算两直线交点的函数
def compute_intersection(line1, line2):
    """
    计算两直线的交点
    :param line1: 直线1的参数 (a1, b1)，对应 y = a1 + b1*x
    :param line2: 直线2的参数 (a2, b2)，对应 y = a2 + b2*x
    :return: 交点坐标 (x, y)
    """
    a1, b1 = line1
    a2, b2 = line2
    if b1 == b2:  # 避免平行的情况
        raise ValueError("Lines are parallel and do not intersect.")
    x = (a2 - a1) / (b1 - b2)  # 交点的x坐标
    y = a1 + b1 * x  # 交点的y坐标
    return x, y

# 划分左右区域并拟合直线
if a is not None:
    feature_point = coordinates[a]  # 特征点 (y, x)
    left_points = [p for p in coordinates if p[1] < feature_point[1]-3]  # 左侧点
    right_points = [p for p in coordinates if p[1] > feature_point[1]+3]  # 右侧点

    # 使用RANSAC拟合左右两侧的直线
    line_left = fit_line_ransac(left_points)
    line_right = fit_line_ransac(right_points)

    # 计算交点
    intersection = compute_intersection(line_left, line_right)

    # 打印直线和交点信息
    print(f"Left line (RANSAC): y = {line_left[0]} + {line_left[1]}x")
    print(f"Right line (RANSAC): y = {line_right[0]} + {line_right[1]}x")
    print(f"Intersection point: {intersection}")

    # 可视化结果
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))  # 转换为RGB格式

    # 绘制拟合直线
    x_vals = np.linspace(0, image.shape[1], 100)
    plt.plot(x_vals, line_left[0] + line_left[1] * x_vals, color='blue', label='Left Line (RANSAC)')
    plt.plot(x_vals, line_right[0] + line_right[1] * x_vals, color='green', label='Right Line (RANSAC)')

    # 标注特征点和交点
    plt.scatter([feature_point[1]], [feature_point[0]], color='red', label='Feature Point')
    plt.scatter([intersection[0]], [intersection[1]], color='yellow', label='Intersection')

    plt.legend()
    plt.title('RANSAC Linear Fit and Intersection Point')
    plt.axis('off')
    plt.show()
else:
    print("Feature point was not found, cannot proceed with line fitting and intersection computation.")
