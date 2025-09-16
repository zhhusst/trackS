import cv2
import numpy as np
import math

from imageio import imread
from scipy.ndimage import maximum_filter
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


class WeldDetector:
    def __init__(self, R=50, theta=5, threshold=0.6, angle_range=(110, 170), Ts=100, n=10,epsilon=80, angleA_min=180,angleA_max = 270,angleB_min = 270,angleB_max = 360):
        """
        初始化焊缝检测器
        参数:
            R: 卷积核半径 (默认50)
            theta: 角度步长 (默认5度)
            threshold: 似然图阈值 (默认0.6)
            angle_range: 焊缝角度范围 (默认110-170度)
            Ts: 强度累积阈值 (默认100)
        """
        self.R = R  # 卷积核半径
        self.theta = theta  # 卷积核中每条线的宽度夹角
        self.threshold = threshold  # 对图像进行卷积之后的筛选时的阈值
        self.angle_range = angle_range  # 焊缝图像中两条中心线的夹角
        self.Ts = Ts  # 最后一步按照夹角进行筛选时Cf的阈值
        self.n = n  # 将图像分成的矩形块的边长
        self.epsilon = epsilon
        self.angleA_min = angleA_min
        self.angleA_max = angleA_max
        self.angleB_min = angleB_min
        self.angleB_max = angleB_max

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
        center = (self.R+1, self.R+1)
        kernel = np.zeros((size, size))
        tan_half_theta = math.tan(math.radians(self.theta / 2))
        EPSILON = 1e-5  # 避免除零的小值

        # 计算中心线参数 (Ax + By + C = 0)
        alpha_rad = math.radians(alpha)
        A = -1
        B = math.tan(alpha_rad)
        C = (1-math.tan(alpha_rad)) * (self.R + 1)
        denom = max(math.sqrt(A ** 2 + B ** 2), EPSILON)

        for x in range(size):  # 第x行
            for y in range(size):  # 第y列
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
                    kernel[x, y] = 2 * (D - d_val) / D
                elif D < d_val <= 2 * D:
                    kernel[x, y] = (D - d_val) / D
                else:
                    kernel[x, y] = 0
        # 可视化卷积核
        # plt.figure(figsize=(10, 8))
        # # 显示核的热力图
        # plt.subplot(1, 2, 1)
        # plt.imshow(kernel, cmap='coolwarm', vmin=-1, vmax=1)
        # plt.colorbar(label='核值')
        # plt.title(f'角度 {alpha}° 的卷积核 (归一化前)')
        # # 显示核的3D表面图
        # ax = plt.subplot(1, 2, 2, projection='3d')
        # X, Y = np.meshgrid(np.arange(size), np.arange(size))
        # ax.plot_surface(X, Y, kernel, cmap='viridis')
        # ax.set_title(f'3D视图 (角度 {alpha}°)')
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('核值')
        # plt.tight_layout()
        # plt.show()

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
        严格按照流程图实现
        """
        # 获取参数
        n = self.n  # 邻域半径，需要在类初始化时设置
        T = self.threshold  # 阈值
        
        candidates = []
        rows, cols = likelihood_map.shape
        
        # 1. 将地图分割成(n+1)×(n+1)大小的块
        block_size = n + 1
        
        for i in range(0, rows, block_size):
            for j in range(0, cols, block_size):
                # 获取当前块
                block = likelihood_map[i:i+block_size, j:j+block_size]
                if block.size == 0:
                    continue
                    
                # 2. 在每个块内搜索最大值S(u₀,v₀)及其位置(u₀,v₀)
                max_val = block.max()
                max_pos_in_block = np.unravel_index(block.argmax(), block.shape)
                u0 = i + max_pos_in_block[0]  # 全局坐标
                v0 = j + max_pos_in_block[1]  # 全局坐标
                
                # 3. 比较S(u₀,v₀)与其邻域内的最大元素
                # 计算邻域范围（考虑边界）
                u_min = max(0, u0 - n)
                u_max = min(rows, u0 + n + 1)
                v_min = max(0, v0 - n)
                v_max = min(cols, v0 + n + 1)
                
                # 获取邻域
                neighborhood = likelihood_map[u_min:u_max, v_min:v_max]  # u是行
                
                # 找到邻域内的最大值
                neighborhood_max = neighborhood.max()
                
                # 4. 判断S(u₀,v₀)是否等于邻域最大值
                if max_val == neighborhood_max:
                    # 5. 判断S(u₀,v₀)是否≥阈值T
                    if max_val >= T:
                        # S(u₀,v₀)是局部极大值，记录其位置
                        candidates.append((u0, v0))
                    else:
                        # 不满足阈值条件，移动到下一个块
                        continue
                else:
                    # 该块内无局部极大值，移动到下一个块
                    continue
        
        return candidates


    def enhance_linear_features(self,img):
        filter_kernel = np.array([-2, 1, 2, 1, -2]).reshape(5, 1)
        ImgE = cv2.filter2D(img, -1, filter_kernel, borderType=cv2.BORDER_REPLICATE)
        return ImgE

    def cumulative_intensity(self, ImgE, center_y, center_x):
        h, w = ImgE.shape
        angles = np.arange(0,360,20)
        Cf = np.zeros_like(angles,dtype=np.float32)
        angles_rads = np.deg2rad(angles)

        for i, angle_rad in enumerate(angles_rads):
            for r in range(1, 50):
                dx = int(r * np.cos(angle_rad))  # 列
                dy = int(r * np.sin(angle_rad))  # 行
                x=center_x+dx
                y=center_y+dy
                if 0 <= x < w and 0 <= y < h:  # x为列，y为行
                    Cf[i] += ImgE[y, x]

        return angles, Cf

    def intercept_neighborhood(self, image, center_y, center_x):
        """
        截取以候选点为中心的邻域图像

        参数:
            image: 原始图像
            center_x: 候选点x坐标
            center_y: 候选点y坐标

        返回:
            Img0: 截取的邻域图像
        """
        h, w = image.shape
        x_min = max(0, center_x - self.R)
        x_max = min(w, center_x + self.R + 1)
        y_min = max(0, center_y - self.R)
        y_max = min(h, center_y + self.R + 1)

        Img0 = image[y_min:y_max, x_min:x_max]

        # 如果截取的图像尺寸不足，进行填充
        if Img0.shape[0] != 2 * self.R + 1 or Img0.shape[1] != 2 * self.R + 1:
            padded_img = np.zeros((2 * self.R + 1, 2 * self.R + 1), dtype=image.dtype)
            start_y = (2 * self.R + 1 - Img0.shape[0]) // 2
            start_x = (2 * self.R + 1 - Img0.shape[1]) // 2
            padded_img[start_y:start_y + Img0.shape[0], start_x:start_x + Img0.shape[1]] = Img0
            Img0 = padded_img

        return Img0

    def find_two_largest_peaks(self, Cf, angles):
        """
        找到累积强度中的两个最大峰值及其角度

        参数:
            Cf: 累积强度数组
            angles: 角度数组

        返回:
            Cf1, Cf2: 两个最大峰值的强度
            Angle1, Angle2: 两个最大峰值对应的角度
        """
        # 使用scipy的find_peaks函数找到所有峰值
        # peaks, _ = find_peaks(Cf, prominence=self.Ts / 2)
        peaks, _ = find_peaks(Cf)

        if len(peaks) < 2:
            return None, None, None, None

        # 获取两个最大峰值
        peak_values = Cf[peaks]
        sorted_indices = np.argsort(peak_values)[::-1]
        top_two_indices = sorted_indices[:2]

        Cf1, Cf2 = peak_values[top_two_indices[0]], peak_values[top_two_indices[1]]
        Angle1, Angle2 = angles[peaks[top_two_indices[0]]], angles[peaks[top_two_indices[1]]]

        # 确保Angle1 <= Angle2
        if Angle1 > Angle2:
            Angle1, Angle2 = Angle2, Angle1
            Cf1, Cf2 = Cf2, Cf1

        return Cf1, Cf2, Angle1, Angle2

    def visualize_angles(self, Angle1, Angle2, Cf1, Cf2, condition1, condition2, candidate_coords=None):
        plt.figure(figsize=(12, 6))

        # 创建极坐标图
        ax = plt.subplot(121, projection='polar')

        # 绘制角度射线
        ax.plot([0, np.deg2rad(Angle1)], [0, 1], 'r-', linewidth=2, label=f'Angle1: {Angle1:.1f}°')
        ax.plot([0, np.deg2rad(Angle2)], [0, 1], 'b-', linewidth=2, label=f'Angle2: {Angle2:.1f}°')

        # 标记角度范围
        theta1_min = np.deg2rad(180)
        theta1_max = np.deg2rad(270)
        theta1_range = np.linspace(theta1_min, theta1_max, 50)
        ax.fill_between(theta1_range, 0, 1, color='red', alpha=0.2, label='Angle1 Range')

        theta2_min = np.deg2rad(270)
        theta2_max = np.deg2rad(360)
        theta2_range = np.linspace(theta2_min, theta2_max, 50)
        ax.fill_between(theta2_range, 0, 1, color='blue', alpha=0.2, label='Angle2 Range')

        # 设置极坐标图属性
        ax.set_theta_zero_location('E')  # 0度在右侧
        ax.set_theta_direction(-1)  # 角度顺时针增加
        ax.set_rlim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.set_title('Angle Visualization')

        # 添加条件判断信息
        plt.subplot(122)
        plt.axis('off')

        info_text = "Angle Judgment Criteria\n\n"
        if candidate_coords:
            info_text += f"Candidate: {candidate_coords}\n\n"

        info_text += f"Angle1: {Angle1:.1f}°\n"
        info_text += f"Cf1: {Cf1:.1f}\n"
        info_text += f"Condition 1: {self.angleA_max}-{self.angleA_max}° & Cf1 > {self.Ts}\n"
        info_text += f"Status: {'PASS' if condition1 else 'FAIL'}\n\n"

        info_text += f"Angle2: {Angle2:.1f}°\n"
        info_text += f"Cf2: {Cf2:.1f}\n"
        info_text += f"Condition 2: {self.angleB_min}-{self.angleB_max}° & Cf2 > {self.Ts}\n"
        info_text += f"Status: {'PASS' if condition2 else 'FAIL'}\n\n"

        info_text += f"Overall Result: {'FILET WELD' if (condition1 and condition2) else 'NOT A WELD'}"

        plt.text(0.1, 0.5, info_text, fontsize=12, va='center')
        plt.title('Judgment Criteria')

        # 添加整体标题
        if candidate_coords:
            plt.suptitle(f"Candidate at {candidate_coords} - Angle Analysis", fontsize=14)
        else:
            plt.suptitle("Angle Analysis for Candidate Point", fontsize=14)

        plt.tight_layout()
        plt.savefig("angle_visualization.png", dpi=150, bbox_inches='tight')
        plt.show()

    def judge_candidate(self, Cf1, Cf2, Angle1, Angle2, candidate_coords=None):
        # 根据流程图的条件进行判断
        condition1 = (self.angleA_min< Angle1 < self.angleA_max) and (Cf1 > self.Ts)
        condition2 = (self.angleB_min < Angle2 < self.angleB_max) and (Cf2 > self.Ts)

        self.visualize_angles(Angle1, Angle2, Cf1, Cf2, condition1, condition2, candidate_coords)

        # 两个条件都必须满足
        return condition1 and condition2

    def reexamine_candidates(self, image, candidates, R=20, TS=100):
        """
        Reexamine candidates based on local structural features.
        
        :param image: Original image
        :param candidates: List of (u,v)
        :param R: Neighborhood radius
        :param TS: Cumulative intensity threshold
        :return: List of true fillet weld joints
        """
        true_joints = []
        results = []
        for i, (y, x) in enumerate(candidates):  # y是行 x是列
            Img0 = self.intercept_neighborhood(image, y, x)
            ImgE = self.enhance_linear_features(Img0)
            angles, Cf = self.cumulative_intensity(ImgE, R, R)
            Cf1,Cf2,Angle1,Angle2 = self.find_two_largest_peaks(Cf, angles)
            if Cf1 is None or Cf2 is None:
                # 没有找到足够的峰值，不是焊缝点
                results.append({
                    'candidate': (y, x),  # y是行  x是列
                    'is_weld': False,
                    'reason': 'Insufficient peaks',
                    'Cf1': Cf1,
                    'Cf2': Cf2,
                    'Angle1': Angle1,
                    'Angle2': Angle2
                })
                continue
            is_weld =self.judge_candidate(Cf1, Cf2, Angle1, Angle2,
                                       candidate_coords=(y, x))
            # is_weld = True
            if is_weld:
                true_joints.append((y, x))
            # 保存结果用于分析
            results.append({
                'candidate': (x, y),
                'is_weld': is_weld,
                'reason': 'Judgment criteria' if is_weld else 'Does not meet criteria',
                'Cf1': Cf1,
                'Cf2': Cf2,
                'Angle1': Angle1,
                'Angle2': Angle2,
                'Img0': Img0,
                'ImgE': ImgE,
                'Cf': Cf,
                'angles': angles
            })
        return true_joints, results

    def visualize_reexamination(self, result):
        """
        可视化重新审查过程

        参数:
            result: 单个候选点的审查结果
            save_path: 保存路径（可选）
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 原始邻域图像
        axes[0, 0].imshow(result['Img0'], cmap='gray')
        axes[0, 0].set_title('Neighborhood Image (Img0)')
        axes[0, 0].axis('off')

        # 增强后的图像
        axes[0, 1].imshow(result['ImgE'], cmap='gray')
        axes[0, 1].set_title('Enhanced Image (ImgE)')
        axes[0, 1].axis('off')

        # 累积强度曲线
        axes[0, 2].plot(result['angles'], result['Cf'])
        axes[0, 2].set_xlabel('Angle (degrees)')
        axes[0, 2].set_ylabel('Cumulative Intensity (Cf)')
        axes[0, 2].set_title('Cumulative Intensity vs Angle')

        # 标记峰值
        peaks, _ = find_peaks(result['Cf'], prominence=self.Ts / 2)
        if len(peaks) >= 2:
            peak_values = result['Cf'][peaks]
            sorted_indices = np.argsort(peak_values)[::-1]
            top_two_indices = sorted_indices[:2]

            axes[0, 2].plot(result['angles'][peaks[top_two_indices[0]]],
                            result['Cf'][peaks[top_two_indices[0]]], 'ro')
            axes[0, 2].plot(result['angles'][peaks[top_two_indices[1]]],
                            result['Cf'][peaks[top_two_indices[1]]], 'ro')

            axes[0, 2].text(result['angles'][peaks[top_two_indices[0]]],
                            result['Cf'][peaks[top_two_indices[0]]] + 5,
                            f"Angle: {result['angles'][peaks[top_two_indices[0]]]:.1f}°\nCf: {result['Cf'][peaks[top_two_indices[0]]]:.1f}",
                            ha='center')

            axes[0, 2].text(result['angles'][peaks[top_two_indices[1]]],
                            result['Cf'][peaks[top_two_indices[1]]] + 5,
                            f"Angle: {result['angles'][peaks[top_two_indices[1]]]:.1f}°\nCf: {result['Cf'][peaks[top_two_indices[1]]]:.1f}",
                            ha='center')

        # 判断条件说明
        axes[1, 0].axis('off')
        text = f"Candidate: {result['candidate']}\n"
        text += f"Is Weld: {result['is_weld']}\n"
        text += f"Reason: {result['reason']}\n"
        text += f"Angle1: {result['Angle1']:.1f}°, Cf1: {result['Cf1']:.1f}\n"
        text += f"Angle2: {result['Angle2']:.1f}°, Cf2: {result['Cf2']:.1f}\n"
        text += f"Condition 1: Angle1 near 0° (ε={self.epsilon}°) & Cf1 > TS ({self.Ts})\n"
        text += f"Condition 2: Angle2 in [110°, 170°] & Cf2 > TS ({self.Ts})"
        axes[1, 0].text(0.1, 0.5, text, va='center', fontsize=12)

        # 角度范围可视化
        axes[1, 1].axis('off')
        axes[1, 1].set_xlim(0, 360)
        axes[1, 1].set_ylim(0, 1)

        # 绘制角度范围
        axes[1, 1].axvspan(0 - self.epsilon, 0 + self.epsilon, alpha=0.3, color='green', label='Angle1 Range')
        axes[1, 1].axvspan(110, 170, alpha=0.3, color='blue', label='Angle2 Range')

        # 标记实际角度
        if result['Angle1'] is not None:
            axes[1, 1].axvline(result['Angle1'], color='red', linestyle='--', label=f'Angle1: {result["Angle1"]:.1f}°')
        if result['Angle2'] is not None:
            axes[1, 1].axvline(result['Angle2'], color='purple', linestyle='--',
                               label=f'Angle2: {result["Angle2"]:.1f}°')

        axes[1, 1].legend()
        axes[1, 1].set_title('Angle Criteria Visualization')
        axes[1, 1].set_xlabel('Angle (degrees)')

        # 判断结果
        axes[1, 2].axis('off')
        result_text = "REEXAMINATION RESULT:\n\n"
        result_text += "VALID FILET WELD" if result['is_weld'] else "NOT A WELD JOINT"
        axes[1, 2].text(0.5, 0.5, result_text, ha='center', va='center', fontsize=16,
                        color='green' if result['is_weld'] else 'red', weight='bold')

        plt.tight_layout()

        # if save_path:
        #     plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

    def detect_welds(self, img):
        # 如果输入是灰度图直接使用
        if len(img.shape) == 2:
            gray = img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 计算焊缝似然图
        likelihood_map = self.compute_likelihood_map(gray)
        # =============== 添加似然图可视化 ===============
        # plt.figure(figsize=(12, 6))

        # # 原始灰度图像
        # plt.subplot(121)
        # plt.imshow(gray, cmap='gray')
        # plt.title("Original Grayscale Image")
        # plt.axis('off')
        #
        # # 焊缝似然图
        # plt.subplot(122)
        # plt.imshow(likelihood_map, cmap='jet')
        # plt.colorbar(label='Probability')
        # plt.title("Weld Likelihood Map")
        # plt.axis('off')
        #
        # plt.tight_layout()
        # # plt.savefig("weld_likelihood_map.png", dpi=150)
        # plt.show()
        # =============== 结束可视化 ===============
        # 非极大值抑制获取候选点
        candidate_points = self.non_maxima_suppression(likelihood_map)
        joints, detailed_results = self.reexamine_candidates(gray, candidate_points, 80, self.Ts)

        return joints, likelihood_map


# =============== 示例用法 ===============
if __name__ == "__main__":
    # 参数设置
    R = 50  # 卷积核半径
    theta = 2  # 角度步长 (度)
    threshold = 0.6 # 概率阈值
    angle_range = (110, 180)  # 角度范围
    Ts = 2*R  # 最后一步按照夹角进行筛选时Cf的阈值
    n = 20  # 将图像分成的矩形块的边长
    epsilon = 10
    angleA_min = 180
    angleA_max = 270
    angleB_min = 270
    angleB_max = 360

    # 初始化检测器
    detector = WeldDetector(R, theta, threshold, angle_range,Ts,n,epsilon,angleA_min,angleA_max,angleB_min,angleB_max)

    img = imread("/home/z/trackS/trackS/GNNTransformer/对比方法3/微信图片_20250915184816_281.jpg")

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
    for y, x in weld_points:
        cv2.circle(result_img, (x, y), 5, (0, 0, 255), -1)  # 红色标记检测到的焊缝点

    plt.subplot(212)
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected Weld Points: {len(weld_points)}")

    plt.tight_layout()
    # plt.savefig("weld_detection_result.png", dpi=300)
    plt.show()

    # 打印检测结果
    print(f"Detected {len(weld_points)} weld points")
    for i, pt in enumerate(weld_points):
        print(f"Point {i + 1}: ({pt[0]}, {pt[1]})")