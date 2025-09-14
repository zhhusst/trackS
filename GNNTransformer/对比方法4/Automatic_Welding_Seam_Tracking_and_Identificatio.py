import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import euclidean
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.path import Path
import matplotlib.patches as patches
# ====================== 1. 数据结构定义 ======================
class LineSegment:
    """线段元素类 (表I)"""

    def __init__(self, start, end):
        self.start = np.array(start)  # 起点坐标 [x, y]
        self.end = np.array(end)  # 终点坐标 [x, y]
        self.length = euclidean(start, end)
        self.slope = self._calculate_slope()
        self.type = self._classify_segment()

    def _calculate_slope(self):
        """计算线段斜率 (弧度)"""
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return np.arctan2(dy, dx) if dx != 0 else np.pi / 2

    def _classify_segment(self):
        """根据斜率分类线段 (表I)"""
        angle = np.degrees(self.slope)  # 拟合线段后会得到每条线段的起点、终点、斜率，根据斜率的范围可以定义线段的类型。  比如说线段的斜率很小就是水平直线段
        if -10 < angle < 10:
            return 'h'  # 水平
        elif 10 <= angle < 80:
            return 'u'  # 上斜
        elif -80 < angle <= -10:
            return 'd'  # 下斜
        elif abs(angle) >= 80:
            return 'v'  # 垂直
        return 'u'  # 默认


class Junction:
    """连接关系类 (表II)"""

    def __init__(self, prev_seg, next_seg):
        self.prev = prev_seg
        self.next = next_seg
        self.angle = self._calculate_angle()
        self.type = self._classify_junction()

    def _calculate_angle(self):
        """计算连接角度差 (弧度)"""
        return self.next.slope - self.prev.slope

    def _classify_junction(self):
        """分类连接关系 (表II)"""
        angle_deg = np.degrees(self.angle)
        if -5 < angle_deg < 5:  # 平滑过渡
            return 'c1'
        elif 5 <= angle_deg < 175:  # 向上
            return 'c2'
        elif -175 < angle_deg <= -5:  # 向下
            return 'c3'
        # 断裂关系需根据位置判断 (gh/gv/gu/gd)
        return self._classify_break()

    def _classify_break(self):
        """分类断裂关系 (表II)"""
        vec = self.next.start - self.prev.end
        angle = np.degrees(np.arctan2(vec[1], vec[0]))
        if -10 < angle < 10:
            return 'gh'
        elif 10 <= angle < 80:
            return 'gu'
        elif -80 < angle <= -10:
            return 'gd'
        return 'gv'


# ====================== 2. 焊缝轮廓处理 ======================
def extract_centerline(image_window):
    """
    提取激光条纹中心线 (重心法)
    参数: 预处理后的图像窗口 (二值化/去噪)
    返回: 中心线点集 [[x,y], ...]
    """
    points = []
    for x in range(image_window.shape[1]):
        col = image_window[:, x]
        y_vals = np.where(col > 0)[0]
        if len(y_vals) > 0:
            y_center = np.mean(y_vals)
            points.append([x, y_center])
    return np.array(points)


def fit_line_segments(points, max_error=3.0):
    """
    最小二乘法拟合线段 (分段线性拟合)
    参数:
        points - 中心线点集
        max_error - 最大拟合误差阈值
    返回: LineSegment对象列表
    """
    segments = []
    start_idx = 0

    while start_idx < len(points) - 1:
        end_idx = start_idx + 2
        best_error = float('inf')
        best_model = None

        # 动态扩展线段直到误差超标
        while end_idx <= len(points):
            segment_points = points[start_idx:end_idx]
            X = segment_points[:, 0].reshape(-1, 1)
            y = segment_points[:, 1]

            model = LinearRegression()
            model.fit(X, y)
            pred_y = model.predict(X)
            error = np.mean(np.abs(pred_y - y))

            if error <= best_error:
                best_error = error
                best_model = model
                best_end = end_idx

            if error > max_error or end_idx == len(points):
                if best_model:  # 保存最优线段
                    start_pt = [points[start_idx][0], best_model.predict([[points[start_idx][0]]])[0]]
                    end_pt = [points[best_end - 1][0], best_model.predict([[points[best_end - 1][0]]])[0]]
                    segments.append(LineSegment(start_pt, end_pt))
                    start_idx = best_end - 1
                break

            end_idx += 1
    return segments


def generate_profile_string(segments):
    """
    生成焊缝轮廓定性描述字符串
    参数: LineSegment对象列表
    返回: 描述字符串 (如"h c3 d c2 u")
    """
    profile_str = ""

    # 首元素直接添加
    if segments:
        profile_str += segments[0].type

    # 遍历生成连接关系+下一线段
    for i in range(len(segments) - 1):
        junction = Junction(segments[i], segments[i + 1])
        profile_str += f" {junction.type} {segments[i + 1].type}"

    return profile_str.strip()


# ====================== 3. 模型匹配算法 ======================
class WeldingSeamMatcher:
    def __init__(self, model_segments):
        """
        重构初始化方法：直接接收线段对象列表作为模型
        参数:
            model_segments - 模型焊缝的LineSegment对象列表
        """
        self.model_segments = model_segments
        self.model_string = self._generate_model_string()

    def _generate_model_string(self):
        """从线段对象生成模型字符串"""
        return generate_profile_string(self.model_segments)

    def match(self, object_segments, collinearity_thresh=0.9):
        """
        修改后的匹配方法：直接处理线段对象
        参数:
            object_segments - 对象轮廓的LineSegment对象列表
            collinearity_thresh - 共线度阈值
        返回: (匹配结果, 修正后的线段列表)
        """

        # 2. 线段元素整合 (解决断裂干扰)
        integrated_segments = self._integrate_segments(object_segments, collinearity_thresh)
        integrated_string = generate_profile_string(integrated_segments)

        # 3. 执行序列三重组匹配
        model_tokens = self.model_string.split()
        object_tokens = integrated_string.split()
        match_result = self._stgm_match(model_tokens, object_tokens)

        return match_result, integrated_segments

    def _integrate_segments(self, segments, threshold):
        """
        线段元素整合 (基于几何信息)
        参数:
            segments - LineSegment对象列表
            threshold - 共线度阈值
        返回: 整合后的LineSegment列表
        """
        if len(segments) <= 1:
            return segments

        integrated = []
        i = 0
        while i < len(segments):
            current = segments[i]
            # 尝试合并后续线段
            j = i + 1
            while j < len(segments):
                next_seg = segments[j]
                # 计算共线度 (公式11)
                Dc = self._calculate_collinearity(current, next_seg)

                if Dc > threshold and self._can_merge(current, next_seg):
                    # 合并线段：创建新线段连接起点和终点
                    merged_seg = LineSegment(current.start, next_seg.end)
                    current = merged_seg
                    j += 1
                else:
                    break

            integrated.append(current)
            i = j if j > i else i + 1

        return integrated

    def _calculate_collinearity(self, seg1, seg2):
        """
        计算共线度 Dc = Dis(ρ₁,ρ₂)/[L(l₁)+L(g)+L(l₂)]
        参数: 两个相邻线段
        返回: 共线度值 (0~1)
        """
        direct_distance = euclidean(seg1.start, seg2.end)
        total_length = seg1.length + euclidean(seg1.end, seg2.start) + seg2.length
        return direct_distance / total_length

    def _can_merge(self, seg1, seg2):
        """检查两个线段是否可以合并 (同向且角度差小)"""
        angle_diff = abs(seg1.slope - seg2.slope)
        return angle_diff < np.radians(15)  # 角度差小于15度

    def _stgm_match(self, model_tokens, object_tokens):
        """
        实现序列三重组匹配 (STGM) 算法
        参数:
            model_tokens - 模型分割后的token列表
            object_tokens - 对象分割后的token列表
        返回: 匹配结果字典 {模型索引: 对象索引}
        """
        matches = {}
        model_idx = 0
        object_idx = 0
        matched_triples = 0

        while model_idx < len(model_tokens) - 2 and object_idx < len(object_tokens) - 2:
            # 获取当前三重组 (线段-连接关系-线段)
            model_triple = model_tokens[model_idx:model_idx + 3]
            object_triple = object_tokens[object_idx:object_idx + 3]

            # 检查三重组是否匹配
            if self._triple_match(model_triple, object_triple):
                # 记录匹配位置
                matches[model_idx] = object_idx
                matches[model_idx + 1] = object_idx + 1
                matches[model_idx + 2] = object_idx + 2

                # 移动索引
                model_idx += 3
                object_idx += 3
                matched_triples += 1
            else:
                # 尝试在对象中跳过1个元素继续匹配
                model_idx += 1

        # 处理剩余元素（如果模型长度不是3的倍数）
        while model_idx < len(model_tokens) and object_idx < len(object_tokens):
            if model_tokens[model_idx] == object_tokens[object_idx]:
                matches[model_idx] = object_idx
                model_idx += 1
                object_idx += 1
            else:
                object_idx += 1

        return matches

    def _triple_match(self, model_triple, object_triple):
        """
        检查三重组匹配（允许一定的灵活性）
        参数:
            model_triple - 模型的三重组 [线段, 连接, 线段]
            object_triple - 对象的三重组 [线段, 连接, 线段]
        返回: 是否匹配
        """
        # 线段类型必须精确匹配
        if model_triple[0] != object_triple[0] or model_triple[2] != object_triple[2]:
            return False

        # 连接关系允许一定灵活性
        model_conn = model_triple[1]
        object_conn = object_triple[1]

        # 如果模型是连接关系，对象可以是连接或断裂
        if model_conn.startswith('c'):
            return object_conn.startswith('c') or object_conn.startswith('g')

        # 如果模型是断裂关系，对象可以是断裂或连接
        if model_conn.startswith('g'):
            return object_conn.startswith('g') or object_conn.startswith('c')

        return False
# ====================== 4. 焊缝定位 ======================
def determine_weld_position(segments, weld_type):
    """
    根据匹配结果确定焊缝位置 (图6)
    参数:
        segments - 匹配修正后的线段列表
        weld_type - 焊缝类型 ('v_groove', 'i_groove', 'fillet')
    返回: 焊缝位置坐标 [x, y]
    """
    if weld_type == 'v_groove':
        # 查找d和u线段 (图6b)
        d_segment = next((s for s in segments if s.type == 'd'), None)
        u_segment = next((s for s in segments if s.type == 'u'), None)

        if d_segment and u_segment:
            # 计算端点中点
            ep1 = d_segment.end if d_segment.slope < 0 else d_segment.start
            ep2 = u_segment.start if u_segment.slope > 0 else u_segment.end
            return [(ep1[0] + ep2[0]) / 2, (ep1[1] + ep2[1]) / 2]

    elif weld_type == 'fillet':
        # 查找d和u/h的交点 (图6c)
        d_segment = next((s for s in segments if s.type == 'd'), None)
        u_segment = next((s for s in segments if s.type in ['u', 'h']), None)

        if d_segment and u_segment:
            # 计算线段交点 (实际实现需几何计算)
            return [d_segment.end[0], d_segment.end[1]]

    # 默认返回第一线段中点
    return segments[0].start if segments else [0, 0]


# ====================== 5. 可视化工具 ======================
def plot_weld_profile(segments, position=None):
    """可视化焊缝轮廓及定位点"""
    plt.figure(figsize=(10, 6))

    # 绘制线段
    for i, seg in enumerate(segments):
        plt.plot([seg.start[0], seg.end[0]],
                 [seg.start[1], seg.end[1]],
                 'o-', label=f'Seg {i + 1} ({seg.type})')

        # 标注连接关系
        if i < len(segments) - 1:
            mid_x = (seg.end[0] + segments[i + 1].start[0]) / 2
            mid_y = (seg.end[1] + segments[i + 1].start[1]) / 2
            junction = Junction(seg, segments[i + 1])
            plt.text(mid_x, mid_y, junction.type,
                     fontsize=9, bbox=dict(facecolor='yellow', alpha=0.5))

    # 标定焊缝位置
    if position:
        plt.plot(position[0], position[1], 'rs', markersize=10, label='Weld Position')

    plt.legend()
    plt.title('Welding Seam Profile & Position')
    plt.grid(True)
    plt.axis('equal')
    plt.show()


# ====================== 6. 完整流程示例 ======================
if __name__ == "__main__":
    # 1. 创建模拟点集 (V型坡口)
    points = np.array([
        [0, 10], [5, 9], [10, 8], [15, 7],  # 水平段
        [20, 5], [25, 3], [30, 1],  # 下斜坡段
        [35, 1], [40, 3], [45, 5],  # 上斜坡段
        [50, 7], [55, 8], [60, 9], [65, 10]  # 水平段
    ])

    # ============================================================

    # 创建专业可视化
    plt.figure(figsize=(14, 8), dpi=120)

    # ====================== 1. 基础焊缝轮廓可视化 ======================
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax1.set_title('seam profile wanzhengshitu', fontsize=14, weight='bold')
    ax1.plot(points[:, 0], points[:, 1], 'o-', color='#1f77b4', linewidth=2.5, markersize=8, label='hanfengzhongxinxian')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # 添加区域标签
    ax1.fill_betweenx([0, 12], 0, 20, color='#f0f9e8', alpha=0.3, label='shuipingquyu')
    ax1.fill_betweenx([0, 6], 20, 35, color='#fff7bc', alpha=0.3, label='xiaxiequyu')
    ax1.fill_betweenx([0, 6], 35, 50, color='#fee0d2', alpha=0.3, label='shangxiequyu')
    ax1.fill_betweenx([6, 12], 50, 65, color='#f0f9e8', alpha=0.3)

    # 添加工件示意
    ax1.add_patch(Rectangle((0, 0), 65, 3, facecolor='#a1a1a1', edgecolor='k', alpha=0.7, label='gongjianjiti'))
    ax1.add_patch(Polygon([[20, 3], [30, 3], [35, 3], [45, 3], [50, 3]], closed=True,
                          facecolor='#bdbdbd', edgecolor='k', alpha=0.8, label='hanfnegquyu'))

    # ====================== 2. 焊缝细节放大视图 ======================
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    ax2.set_title('Vxingpokouxijiefangda', fontsize=12, weight='bold')
    ax2.plot(points[4:10, 0], points[4:10, 1], 'o-', color='#ff7f0e', linewidth=2.5, markersize=8)
    ax2.set_xlim(18, 52)
    ax2.set_ylim(0, 6)
    ax2.grid(True, linestyle='--', alpha=0.7)

    # 添加几何标注
    for i in range(4, 10):
        ax2.annotate(f'P{i}', (points[i, 0] + 0.5, points[i, 1] + 0.2), fontsize=10)

    # 添加角度标注
    ax2.annotate('', xy=(30, 1), xytext=(20, 5),
                 arrowprops=dict(arrowstyle='<->', color='#d62728', lw=1.5))
    ax2.text(24, 3, 'α=60°', fontsize=10, color='#d62728', bbox=dict(facecolor='white', alpha=0.8))

    ax2.annotate('', xy=(45, 5), xytext=(35, 1),
                 arrowprops=dict(arrowstyle='<->', color='#2ca02c', lw=1.5))
    ax2.text(38, 3, 'β=60°', fontsize=10, color='#2ca02c', bbox=dict(facecolor='white', alpha=0.8))

    # 添加激光扫描示意
    laser_x = np.linspace(15, 50, 20)
    laser_y = np.abs(np.sin(laser_x / 5)) * 0.5 + 6.5
    ax2.plot(laser_x, laser_y, '--', color='#e377c2', alpha=0.7, label='jiguangsaomiaolujing')

    # ====================== 3. 三维焊缝剖面视图 ======================
    ax3 = plt.subplot2grid((2, 2), (1, 1), projection='3d')
    ax3.set_title('hanfeng3dpomianshitu', fontsize=12, weight='bold')

    # 创建3D剖面
    X = points[:, 0]
    Y = np.linspace(-3, 3, 20)
    X, Y = np.meshgrid(X, Y)
    Z = np.interp(X, points[:, 0], points[:, 1])

    # 添加渐变颜色映射
    surf = ax3.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, rstride=1, cstride=1,
                            linewidth=0, antialiased=True)

    # 添加中心线
    ax3.plot(points[:, 0], np.zeros_like(points[:, 0]), points[:, 1],
             'o-', color='#ff7f0e', linewidth=2.5, markersize=6, label='hanfengzhongxinxian')

    # 设置视角
    ax3.view_init(elev=25, azim=-45)
    ax3.set_zlim(0, 12)

    # ====================== 全局设置 ======================
    plt.tight_layout(pad=3.0)
    plt.figtext(0.5, 0.01, '1: hanfenglunkuo - Vxingjiegou', ha='center', fontsize=12)
    plt.subplots_adjust(bottom=0.1)

    # 添加图例
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, framealpha=0.9)
    ax2.legend(loc='best')

    plt.savefig('welding_seam_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ==========================================================================================================================================



    # 2. 线段拟合
    segments = fit_line_segments(points)
    print("拟合线段数量:", len(segments))
    print("各线段类型:", [seg.type for seg in segments])

    # 3. 生成轮廓字符串
    profile_str = generate_profile_string(segments)
    print("轮廓描述字符串:", profile_str)

    # 4. 创建V型坡口模型 (使用线段对象)
    # 典型V型坡口模型: 水平-下斜-上斜-水平
    model_points = np.array([
        [0, 10], [15, 10],  # 水平
        [20, 10], [35, 5],   # 下斜
        [40, 5], [55, 10],   # 上斜
        [60, 10], [75, 10]   # 水平
    ])

    # 创建专业可视化
    plt.figure(figsize=(12, 8), dpi=120)
    ax = plt.gca()

    # ====================== 1. 模板焊缝基础轮廓 ======================
    # 绘制完整焊缝轮廓
    ax.plot(model_points[:, 0], model_points[:, 1],
            'o-', color='#1f77b4', linewidth=3, markersize=10,
            label='Weld contour centerline', zorder=5)

    # 添加基准坐标系
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    # ====================== 2. 焊缝区域填充 ======================
    # 创建V型坡口闭合路径
    vertsp = [
        (model_points[0][0], 0),  # 左下点
        (model_points[0][0], model_points[0][1]),  # 起点
        *[(x, y) for x, y in model_points[1:-1]],  # 轮廓点
        (model_points[-1][0], model_points[-1][1]),  # 终点
        (model_points[-1][0], 0),  # 右下点
        (model_points[0][0], 0)  # 闭合点
    ]
    codes = [Path.MOVETO] + [Path.LINETO] * (len(vertsp) - 2) + [Path.CLOSEPOLY]

    path = Path(vertsp, codes)
    patch = patches.PathPatch(path, facecolor='#ffebcd', alpha=0.6, edgecolor='#8b4513', lw=1.5)
    ax.add_patch(patch)

    # ====================== 3. 区域划分与标注 ======================
    # 水平段1
    plt.fill_between([0, 15], 0, [10, 10], color='#e6f2ff', alpha=0.4)
    plt.text(7.5, 5, 'Horizontal segment 1', ha='center', fontsize=11, weight='bold', bbox=dict(alpha=0.7))

    # 下斜坡段
    points_down = np.array([[20, 10], [35, 5]])
    plt.fill_between(points_down[:, 0], 0, points_down[:, 1],
                     color='#e6ffe6', alpha=0.4)
    plt.text(27.5, 3, 'Downward sloping section\nslope:' + r'$\theta$' + '=35°',
             ha='center', fontsize=11, bbox=dict(alpha=0.7))

    # 上斜坡段
    points_up = np.array([[40, 5], [55, 10]])
    plt.fill_between(points_up[:, 0], 0, points_up[:, 1],
                     color='#ffe6e6', alpha=0.4)
    plt.text(47.5, 3, 'Upper inclined section\nslope:' + r'$\theta$' + '=35°',
             ha='center', fontsize=11, bbox=dict(alpha=0.7))

    # 水平段2
    plt.fill_between([60, 75], 0, [10, 10], color='#e6f2ff', alpha=0.4)
    plt.text(67.5, 5, 'Horizontal segment 2', ha='center', fontsize=11, weight='bold', bbox=dict(alpha=0.7))

    # 标定关键点
    for i, point in enumerate(model_points):
        ax.annotate(f'P{i}', (point[0] + 1, point[1] + 0.5), fontsize=10, weight='bold')
        # 添加点坐标标签
        ax.text(point[0] - 2 if i in [0, 6] else point[0] + 1,
                point[1] - 1.5 if i == 0 else point[1] - 1,
                f'({point[0]},{point[1]})', fontsize=9)

    # ====================== 4. 焊缝几何特征标注 ======================
    # 标注V型坡口深度
    ax.annotate('', xy=(37.5, 5), xytext=(37.5, 10),
                arrowprops=dict(arrowstyle='<->', color='#d62728', lw=1.5))
    ax.text(39, 7.5, 'groove depth:5mm', rotation=90, fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8))

    # 标注V型坡口角度
    ax.annotate('', xy=(23, 8), xytext=(35, 5),
                arrowprops=dict(arrowstyle='-', color='#d62728', lw=1.5))
    ax.annotate('', xy=(52, 8), xytext=(40, 5),
                arrowprops=dict(arrowstyle='-', color='#d62728', lw=1.5))
    ax.text(30, 10.5, 'Groove angle:70°', fontsize=10, color='#d62728',
            bbox=dict(facecolor='white', alpha=0.8))

    # 标注焊缝全长
    ax.annotate('', xy=(0, 13), xytext=(75, 13),
                arrowprops=dict(arrowstyle='<|-|>', color='#2ca02c', lw=1.5))
    ax.text(37.5, 13.5, 'Total length of weld seam:75mm', ha='center', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8))

    # ====================== 5. 工业应用示意 ======================
    # 添加工件基体
    plt.fill_between([-5, 80], 0, -3, color='#a1a1a1', alpha=0.7, label='Workpiece substrate')

    # 添加激光扫描示意
    laser_x = np.linspace(0, 75, 20)
    laser_y = np.abs(np.sin(laser_x / 5)) * 1.5 + 14
    ax.plot(laser_x, laser_y, '--', color='#e377c2', lw=1.5, alpha=0.7, label='Laser scanning path')

    # 添加焊枪示意图
    ax.plot(37.5, 17, '^', markersize=20, color='#e74c3c', label='Welding gun position')
    ax.text(37.5, 18, '焊枪', ha='center', fontsize=10, weight='bold')

    # ====================== 全局设置 ======================
    ax.set_title('Visualization of V-shaped groove weld template', fontsize=16, weight='bold')
    ax.set_xlabel('Horizontal position (mm)', fontsize=12)
    ax.set_ylabel('Vertical position (mm)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(-5, 80)
    ax.set_ylim(-4, 20)
    ax.set_aspect('equal')
    plt.legend(loc='upper right', framealpha=0.8)

    plt.text(40, -3.5, 'Figure 1: Template structure of V-shaped groove weld seam', ha='center', fontsize=12)
    plt.tight_layout()
    plt.savefig('welding_seam_template_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()


    # ====================================================================================================================
    model_segments = fit_line_segments(model_points)

    model_profile_str = generate_profile_string(model_segments)
    print("模型轮廓描述字符串:", model_profile_str)

    # 5. 模型匹配
    matcher = WeldingSeamMatcher(model_segments)
    matches, corrected_segments = matcher.match(segments)

    print("匹配结果:", matches)
    print("修正后线段:", [seg.type for seg in corrected_segments])

    # 6. 焊缝定位
    weld_pos = determine_weld_position(corrected_segments, 'v_groove')
    print("焊缝位置坐标:", weld_pos)

    # 7. 可视化
    plot_weld_profile(corrected_segments, weld_pos)