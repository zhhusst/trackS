import cv2
import numpy as np
import math
from scipy.ndimage import rotate
from imageio import imread
from scipy.ndimage import maximum_filter
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import json
import os
from tqdm import tqdm


class WeldDetector:
    def __init__(self, R=50, theta=5, threshold=0.6, angle_range=(110, 170), Ts=100, n=10,epsilon=80, angleA_min=180,angleA_max = 270,angleB_min = 270,angleB_max = 360):
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

    def visualize_kernels(self, base_kernels, rotation_angles):
            """
            可视化基础卷积核及其旋转版本
            
            参数:
                base_kernels: 基础卷积核列表
                rotation_angles: 旋转角度列表
            """
            # 选择前几个基础核进行可视化
            num_base_to_show = min(3, len(base_kernels))
            num_rotations_to_show = min(4, len(rotation_angles))
            
            # 创建大图
            fig, axes = plt.subplots(num_base_to_show, num_rotations_to_show + 1, 
                                    figsize=(15, 10),
                                    squeeze=False)
            fig.suptitle("卷积核及其旋转版本可视化", fontsize=16)
            
            for i in range(num_base_to_show):
                kernel = base_kernels[i]
                
                # 显示原始核
                ax = axes[i, 0]
                ax.imshow(kernel, cmap='coolwarm', vmin=-np.max(np.abs(kernel)), vmax=np.max(np.abs(kernel)))
                ax.set_title(f"yuanshihe {i+1}")
                ax.axis('off')
                
                # 显示旋转版本
                for j, rot_angle in enumerate(rotation_angles[:num_rotations_to_show]):
                    # 旋转卷积核（顺时针旋转）
                    rotated_kernel = rotate(kernel, rot_angle, reshape=False, mode='constant', cval=0)
                    
                    # 裁剪回原始尺寸
                    center = np.array(kernel.shape) // 2
                    start = center - self.R
                    end = center + self.R + 1
                    rotated_kernel = rotated_kernel[start[0]:end[0], start[1]:end[1]]
                    
                    # 归一化旋转后的核
                    # rotated_kernel = rotated_kernel / np.max(np.abs(rotated_kernel))
                    
                    ax = axes[i, j+1]
                    ax.imshow(rotated_kernel, cmap='coolwarm', 
                            vmin=-np.max(np.abs(rotated_kernel)), 
                            vmax=np.max(np.abs(rotated_kernel)))
                    ax.set_title(f"旋转 {rot_angle}°")
                    ax.axis('off')
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            # plt.savefig("kernel_visualization.png", dpi=150, bbox_inches='tight')
            plt.show()
    def visualize_rotated_kernel(self, original_kernel, rotated_kernel, rotation_angle):
        """可视化旋转后的卷积核"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 原始卷积核
        axes[0].imshow(original_kernel, cmap='coolwarm', 
                      vmin=-np.max(np.abs(original_kernel)), 
                      vmax=np.max(np.abs(original_kernel)))
        axes[0].set_title("ori")
        axes[0].axis('off')
        
        # 旋转后的卷积核
        axes[1].imshow(rotated_kernel, cmap='coolwarm', 
                      vmin=-np.max(np.abs(rotated_kernel)), 
                      vmax=np.max(np.abs(rotated_kernel)))
        axes[1].set_title(f"rotated {rotation_angle}° conv")
        axes[1].axis('off')
        
        plt.suptitle(f"conv {rotation_angle}° visual", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.savefig(f"rotated_kernel_{rotation_angle}.png", dpi=150, bbox_inches='tight')
        plt.show()

    def _generate_kernels(self):
        kernels = []
        start_angle, end_angle = self.angle_range
        n_angles = int((end_angle - start_angle) / self.theta) + 1
        
        base_kernels = []
        for i in range(n_angles):
            angle = start_angle+i*self.theta
            kernel = self._create_single_kernel(angle)
            if kernel is not None:  # 忽略无效核
                base_kernels.append(kernel)
        rotation_angles = np.arange(0, -50, -10)
        
        # 可视化卷积核
        # self.visualize_kernels(base_kernels, rotation_angles)

        for kernel in base_kernels:
            kernels.append(kernel)  # 添加原始核
            
            # 添加旋转版本
            for rot_angle in rotation_angles:
                if rot_angle == 0:  # 跳过0度旋转（已经是原始核）
                    continue
                    
                # 旋转卷积核（顺时针旋转）
                rotated_kernel = rotate(kernel, rot_angle, reshape=False, mode='constant', cval=0)
                
                # 裁剪回原始尺寸（旋转可能会改变尺寸）
                center = np.array(kernel.shape) // 2
                start = center - self.R
                end = center + self.R + 1
                rotated_kernel = rotated_kernel[start[0]:end[0], start[1]:end[1]]
                kernels.append(rotated_kernel)
                # self.visualize_rotated_kernel(kernel, rotated_kernel, rot_angle)


        # for i in range(n_angles):
        #     angle = start_angle + i * self.theta
        #     kernel = self._create_single_kernel(angle)
        #     if kernel is not None:  # 忽略无效核
        #         kernels.append(kernel)

        return kernels

    def _create_single_kernel(self, alpha):
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
        angles = np.arange(0,360,5)
        Cf = np.zeros_like(angles,dtype=np.float32)
        angles_rads = np.deg2rad(angles)

        for i, angle_rad in enumerate(angles_rads):
            for r in range(1, int(h/2)):
                dx = int(r * np.cos(angle_rad))  # 列
                dy = int(r * np.sin(angle_rad))  # 行
                x=center_x+dx
                y=center_y+dy
                if 0 <= x < w and 0 <= y < h:  # x为列，y为行
                    Cf[i] += ImgE[y, x]

        return angles, Cf

    def intercept_neighborhood(self, image, center_y, center_x):
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
        # plt.savefig("angle_visualization.png", dpi=150, bbox_inches='tight')
        plt.show()

    def judge_candidate(self, Cf1, Cf2, Angle1, Angle2, candidate_coords=None):
        # 根据流程图的条件进行判断
        condition1 = (self.angleA_min< Angle1 < self.angleA_max) and (Cf1 > self.Ts)
        condition2 = (self.angleB_min < Angle2 < self.angleB_max) and (Cf2 > self.Ts)

        # self.visualize_angles(Angle1, Angle2, Cf1, Cf2, condition1, condition2, candidate_coords)

        # 两个条件都必须满足
        return condition1 and condition2

    def reexamine_candidates(self, image, candidates):
        true_joints = []
        results = []
        for i, (y, x) in enumerate(candidates):  # y是行 x是列
            Img0 = self.intercept_neighborhood(image, y, x)
            ImgE = self.enhance_linear_features(Img0)
            angles, Cf = self.cumulative_intensity(ImgE, int(ImgE.shape[0]/2), int(ImgE.shape[1]/2))
            Cf1,Cf2,Angle1,Angle2 = self.find_two_largest_peaks(Cf, angles)
            if Cf1 is None or Cf2 is None:
                continue
            is_weld =self.judge_candidate(Cf1, Cf2, Angle1, Angle2,
                                       candidate_coords=(y, x))
            # is_weld = True
            if is_weld:
                true_joints.append((x, y))

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

        
        likelihood_map = self.compute_likelihood_map(gray)
        # =============== 添加似然图可视化 ===============
        # plt.figure(figsize=(12, 6))

        # # 原始灰度图像
        # plt.subplot(121)
        # plt.imshow(gray, cmap='gray')
        # plt.title("Original Grayscale Image")
        # plt.axis('off')
        
        # # 焊缝似然图
        # plt.subplot(122)
        # plt.imshow(likelihood_map, cmap='jet')
        # plt.colorbar(label='Probability')
        # plt.title("Weld Likelihood Map")
        # plt.axis('off')
        
        # plt.tight_layout()
        # plt.show()
        # # =============== 结束可视化 ===============
        # 非极大值抑制获取候选点
        candidate_points = self.non_maxima_suppression(likelihood_map)
        joints, _ = self.reexamine_candidates(gray, candidate_points)

        return joints, likelihood_map


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

def prepare_dataset(image_folder):
    """准备数据集"""
    # 加载标注数据
    dataset = load_annotated_data(image_folder)
    data = []
    target_size = (896, 400)

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

    return data


def main():
    R = 50  # 卷积核半径
    theta = 10  # 角度步长 (度)
    threshold = 0.6 # 概率阈值
    angle_range = (110, 180)  # 角度范围
    Ts = 2*R  # 最后一步按照夹角进行筛选时Cf的阈值
    n = 10  # 将图像分成的矩形块的边长,该矩形框是用于非极大值抑制的。
    epsilon = 10
    angleA_min = 190-10
    angleA_max = 190+10
    angleB_min = 350-10
    angleB_max = 350+10

    # 读取测试数据集(测试数据集和标注数据放在一个文件夹中)
    dataset_path = "GNNTransformer/datasets"
    test_data = prepare_dataset(dataset_path)

    # 初始化检测器
    detector = WeldDetector(R, theta, threshold, angle_range,Ts,n,epsilon,angleA_min,angleA_max,angleB_min,angleB_max)

    euclidean_distances = []  # 欧式距离（像素误差）
    x_errors = []             # X坐标误差
    y_errors = []             # Y坐标误差
    undetected_count = 0      # 未检测到的图像数量
    for img, true_point in test_data:
        # 检测焊缝
        weld_points, likelihood_map = detector.detect_welds(img)
        # # 可视化结果
        # plt.figure(figsize=(15, 10))

        # # 原始图像
        # plt.subplot(221)
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # plt.title("Original Image with Simulated Weld")

        # # 焊缝概率图
        # plt.subplot(222)
        # plt.imshow(likelihood_map, cmap='hot')
        # plt.title("Weld Likelihood Map")
        # plt.colorbar()

        # # 检测结果
        # result_img = img.copy()
        # for y, x in weld_points:
        #     cv2.circle(result_img, (y, x), 5, (0, 0, 255), -1)  # 红色标记检测到的焊缝点

        # plt.subplot(212)
        # plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        # plt.title(f"Detected Weld Points: {len(weld_points)}")

        # plt.tight_layout()
        # # plt.savefig("weld_detection_result.png", dpi=300)
        # plt.show()

        # 打印检测结果
        print(f"Detected {len(weld_points)} weld points")
        for i, pt in enumerate(weld_points):
            print(f"Point {i + 1}: ({pt[0]}, {pt[1]})")

        if weld_points:
            pred_point = weld_points[0]
            # pred_point(x,y)
            euclidean_dist = np.sqrt((pred_point[0] - true_point[0])**2 + (pred_point[1] - true_point[1])**2)
            euclidean_distances.append(euclidean_dist)

            # 计算X坐标误差
            x_error = abs(pred_point[0] - true_point[0])
            x_errors.append(x_error)
            # 计算Y坐标误差
            y_error = abs(pred_point[1] - true_point[1])
            y_errors.append(y_error)
        else:
            undetected_count+=1
            print(f"Warning: No weld points detected in image with true point at {true_point}")

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




        



if __name__ == "__main__":
    main()
    