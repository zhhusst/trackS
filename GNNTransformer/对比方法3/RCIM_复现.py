import cv2
import numpy as np
import math

from imageio import imread
from scipy.ndimage import maximum_filter
from scipy.signal import convolve2d
import matplotlib.pyplot as plt


class WeldDetector:
    def __init__(self, R=50, theta=5, threshold=0.6, angle_range=(110, 170), Ts=100):
        """
        初始化焊缝检测器
        参数:
            R: 卷积核半径 (默认50)
            theta: 角度步长 (默认5度)
            threshold: 似然图阈值 (默认0.6)
            angle_range: 焊缝角度范围 (默认110-170度)
            Ts: 强度累积阈值 (默认100)
        """
        self.R = R
        self.theta = theta
        self.threshold = threshold
        self.angle_range = angle_range
        self.Ts = Ts

        # 生成卷积核
        self.kernels = self._generate_kernels()

    def _generate_kernels(self):
        """生成多角度卷积核组"""
        kernels = []
        start_angle, end_angle = self.angle_range
        n_angles = int((end_angle - start_angle) / self.theta) + 1

        for i in range(n_angles):
            angle = start_angle + i * self.theta
            kernel = self._create_single_kernel(angle)
            if kernel is not None:  # 忽略无效核
                kernels.append(kernel)

        return kernels

    def _create_single_kernel(self, alpha):
        """
        创建单个角度的卷积核
        修复了除零错误并添加了数值稳定性
        """
        size = 2 * self.R + 1
        center = (self.R, self.R)
        kernel = np.zeros((size, size))
        tan_half_theta = math.tan(math.radians(self.theta / 2))
        EPSILON = 1e-5  # 避免除零的小值

        # 计算中心线参数 (Ax + By + C = 0)
        alpha_rad = math.radians(alpha)
        A = -1
        B = -math.tan(alpha_rad)
        C = (math.tan(alpha_rad) - 1) * (self.R + 1)
        denom = max(math.sqrt(A ** 2 + B ** 2), EPSILON)

        for y in range(size):
            for x in range(size):
                # 计算到中心点的距离
                if y < self.R + 1:  # 使用严格小于避免边界问题
                    D = max(tan_half_theta * (self.R + 1 - y), EPSILON)
                    d_val = abs(x - center[0])
                else:
                    # 计算边界点
                    alpha_min = math.radians(alpha - self.theta / 2)
                    x0 = math.tan(alpha_min) * y + (1 - math.tan(alpha_min)) * (self.R + 1)
                    d_val = abs(A * x + B * y + C) / denom
                    D = max(abs(A * x0 + B * y + C) / denom, EPSILON)

                # 避免除零错误
                D = max(D, EPSILON)

                # 计算卷积核值
                if d_val <= D:
                    kernel[y, x] = 2 * (D - d_val) / D
                elif D < d_val <= 2 * D:
                    kernel[y, x] = (D - d_val) / D
                else:
                    kernel[y, x] = 0

        # 归一化核，保持正负权重平衡
        pos_mask = kernel > 0
        neg_mask = kernel < 0
        pos_sum = np.sum(kernel[pos_mask])
        neg_sum = np.sum(kernel[neg_mask])

        if pos_sum > 0 and neg_sum < 0:
            pos_sum = max(pos_sum, EPSILON)
            neg_sum = min(neg_sum, -EPSILON)  # 确保为负数
            total_sum = pos_sum + abs(neg_sum)
            kernel = kernel / total_sum

        return kernel

    def compute_likelihood_map(self, img):
        """
        计算焊缝似然图
        优化了卷积速度
        """
        # 转换为浮点型以便计算
        if img.dtype != np.float32:
            img_float = img.astype(np.float32) / 255.0
        else:
            img_float = img.copy()

        # 初始化响应图
        response_map = np.zeros_like(img_float)

        # 多角度卷积 - 使用OpenCV加速
        for kernel in self.kernels:
            # 使用OpenCV的filter2D优化卷积速度
            conv_result = cv2.filter2D(img_float, -1, kernel, borderType=cv2.BORDER_REPLICATE)
            # 取最大响应
            response_map = np.maximum(response_map, conv_result)

        # 归一化到[0,1]
        min_val = np.min(response_map)
        max_val = np.max(response_map)
        if max_val > min_val:
            likelihood_map = (response_map - min_val) / (max_val - min_val)
        else:
            likelihood_map = response_map

        return likelihood_map

    def non_maxima_suppression(self, likelihood_map):
        """
        非极大值抑制提取候选点
        添加了边界检查
        """
        # 3x3最大值滤波
        dilated = maximum_filter(likelihood_map, size=3)

        # 找出局部极大值点
        local_max = (likelihood_map == dilated)

        # 阈值筛选
        thresholded = (likelihood_map > self.threshold) & local_max

        # 获取候选点坐标
        candidate_points = np.column_stack(np.where(thresholded))

        # 按响应值排序 (高响应优先)
        if candidate_points.size > 0:
            response_values = likelihood_map[tuple(candidate_points[:, 0]), tuple(candidate_points[:, 1])]
            sorted_indices = np.argsort(-response_values)  # 降序排序
            candidate_points = candidate_points[sorted_indices]

        return candidate_points

    def structural_verification(self, img, candidate_points, R_scan=50):
        """
        候选点结构验证
        优化了极坐标扫描速度
        """
        # 特征增强滤波 (垂直方向)
        kernel_vertical = np.array([[-2], [1], [2], [1], [-2]])  # 5x1 垂直滤波器
        enhanced_img = cv2.filter2D(img, -1, kernel_vertical)
        h, w = img.shape

        # 存储通过验证的点
        weld_points = []

        # 预计算所有角度值
        angles = np.deg2rad(np.arange(181))
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)

        for point in candidate_points:
            y, x = point  # 注意: numpy索引是(y, x)

            # 角度累积函数 (0-180度)
            Cf = np.zeros(181)

            # 优化扫描 - 避免内部循环角度计算
            for i, (cos_phi, sin_phi) in enumerate(zip(cos_angles, sin_angles)):
                # 极坐标扫描
                for r in range(1, R_scan + 1):
                    xp = int(x + r * cos_phi)
                    yp = int(y + r * sin_phi)

                    # 边界检查
                    if 0 <= xp < w and 0 <= yp < h:
                        Cf[i] += enhanced_img[yp, xp]

            # 寻找前两个峰值 - 使用简单梯度方法
            peaks = []
            for angle in range(1, 180):
                if Cf[angle] > Cf[angle - 1] and Cf[angle] > Cf[angle + 1]:
                    peaks.append(angle)

            # 按峰值强度排序
            if peaks:
                peak_values = [Cf[peak] for peak in peaks]
                sorted_indices = np.argsort(peak_values)[::-1]  # 降序排序
                sorted_peaks = [peaks[i] for i in sorted_indices[:2]]

                # 验证特征条件
                if len(sorted_peaks) == 2:
                    phi1, phi2 = sorted_peaks
                    angle_diff = abs(phi2 - phi1)

                    # 计算最小角度差（考虑周期性）
                    min_angle_diff = min(angle_diff, 180 - angle_diff)

                    # 计算最近垂直距离
                    min_vert_dist = min(min(phi1, 180 - phi1), min(phi2, 180 - phi2))

                    # 验证条件
                    if (min_vert_dist < 10 and
                            110 <= min_angle_diff <= 170 and
                            Cf[phi1] > self.Ts and
                            Cf[phi2] > self.Ts):
                        weld_points.append((x, y))  # 返回(x,y)格式
                        continue  # 避免重复添加

        return weld_points

    def detect_welds(self, img):
        """
        完整焊缝检测流程
        参数:
            img: 输入BGR图像或灰度图
        返回:
            weld_points: 检测到的焊缝点 (x,y)
            likelihood_map: 焊缝概率图
        """
        # 如果输入是灰度图直接使用
        if len(img.shape) == 2:
            gray = img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 计算焊缝似然图
        likelihood_map = self.compute_likelihood_map(gray)

        # 非极大值抑制获取候选点
        candidate_points = self.non_maxima_suppression(likelihood_map)

        # 结构验证
        weld_points = self.structural_verification(gray, candidate_points)

        return weld_points, likelihood_map


# =============== 示例用法 ===============
if __name__ == "__main__":
    # 参数设置 (与论文一致)
    R = 40  # 卷积核半径
    theta = 5  # 角度步长 (度)
    threshold = 0.6 # 概率阈值
    angle_range = (90, 180)  # 角度范围

    # 初始化检测器
    detector = WeldDetector(R, theta, threshold, angle_range)

    img = imread("/home/z/seam_tracking_ws/src/paper1_pkg/视频数据/RCIM/1.jpg")

    # 创建测试图像 - 清晰的V型焊缝
    # img = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # # 创建模拟焊缝结构
    # cv2.line(img, (100, 150), (200, 250), (255, 255, 255), 3)  # 左侧线
    # cv2.line(img, (300, 150), (200, 250), (255, 255, 255), 3)  # 右侧线

    # 添加焊缝点标记
    # cv2.circle(img, (200, 250), 5, (0, 255, 0), -1)  # 绿色表示真实焊缝位置

    # 添加噪声 (模拟工业干扰)
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)

    # 添加高斯模糊模拟实际图像
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # 检测焊缝
    weld_points, likelihood_map = detector.detect_welds(img)

    # 可视化结果
    plt.figure(figsize=(15, 10))

    # 原始图像
    plt.subplot(221)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image with Simulated Weld")

    # 焊缝概率图
    plt.subplot(222)
    plt.imshow(likelihood_map, cmap='hot')
    plt.title("Weld Likelihood Map")
    plt.colorbar()

    # 检测结果
    result_img = img.copy()
    for x, y in weld_points:
        cv2.circle(result_img, (x, y), 5, (0, 0, 255), -1)  # 红色标记检测到的焊缝点

    plt.subplot(212)
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected Weld Points: {len(weld_points)}")

    plt.tight_layout()
    plt.savefig("weld_detection_result.png", dpi=300)
    plt.show()

    # 打印检测结果
    print(f"Detected {len(weld_points)} weld points")
    for i, pt in enumerate(weld_points):
        print(f"Point {i + 1}: ({pt[0]}, {pt[1]})")