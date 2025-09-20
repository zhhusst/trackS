import os
import cv2
import numpy as np
from scipy import stats
from scipy.ndimage import median_filter
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

# ---------------------------
# Config / hyper-params
# ---------------------------
IMG_W = 640
IMG_H = 480

# Kalman (use paper defaults)
A = np.array([[1,0,1,0],
              [0,1,0,1],
              [0,0,1,0],
              [0,0,0,1]], dtype=float)
P0 = np.eye(4) * 10.0
H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
R = np.array([[0.2845,0.0045],[0.0045,0.0455]])
Q = np.eye(4) * 0.01

# Search window proportional factor
ALPHA = 1  # proportional factor alpha for vertical padding (paper uses α)
STRIPE_WIDTH_EST = 5  # initial stripe width estimate in pixels Ψ

# Similarity measure weights (eq.10)
Ww = 0.1
Wg = 0.9


def load_images_from_folder(folder):
    imgs = []
    if not os.path.isdir(folder):
        return imgs
    for f in sorted(os.listdir(folder)):
        p = os.path.join(folder, f)
        if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tiff')):
            imgs.append(cv2.imread(p, cv2.IMREAD_GRAYSCALE))
    return imgs

class SimpleKalman:
    def __init__(self, A, P, H, R, Q):
        self.A = A
        self.P = P.copy()
        self.H = H
        self.R = R
        self.Q = Q
        self.x = np.zeros((4,1), dtype=float)
        self.initialized = False

    def init_state(self, cx, cy):
        self.x = np.array([[cx],[cy],[0.0],[0.0]], dtype=float)
        self.initialized = True

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x.copy()

    def update(self, z):
        # z is 2x1
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = z - (self.H @ self.x)
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P
        return self.x.copy()

def preprocess_image(img):
    col_med = img.copy()
    col_med = median_filter(col_med, size=(3,1))
    med5 = cv2.medianBlur(col_med, 5)
    return med5

def extract_center_of_gravity(img, col_range=None):
    H, W = img.shape
    xs = []
    ys = []
    if col_range is None:
        col_range = range(W)
    for i in col_range:
        col = img[:, i].astype(np.float32)
        # consider pixels above a background threshold
        thr = col.mean() + 0.5*col.std()
        inds = np.where(col > thr)[0]
        if inds.size == 0:
            continue
        j, k = inds.min(), inds.max()
        weights = col[j:k+1]
        ys_val = (weights * np.arange(j, k+1)).sum() / (weights.sum() + 1e-9)
        xs.append(i)
        ys.append(ys_val)
    return np.array(xs), np.array(ys)

def detect_center_in_window(img_win):
    """
    在窗口中提取中心线点：
    1. 找到候选峰（局部极大值）
    2. 计算每个候选峰的宽度和灰度
    3. 用相似度 S_alpha 筛选最优峰
    4. 在该峰的左右半高宽范围内用重心法计算精确中心点
    """
    h, w = img_win.shape
    centers = []

    std_width = STRIPE_WIDTH_EST   # 标准条纹宽度
    std_gray = img_win.max()       # 标准灰度（最大值）

    for i in range(w):  # 遍历窗口中的每一列
        col = img_win[:, i].astype(np.float32)
        col_s = cv2.GaussianBlur(col, (3, 1), 0).flatten()

        # 1. 找局部峰值
        peaks = np.where((col_s[1:-1] > col_s[:-2]) & (col_s[1:-1] > col_s[2:]))[0] + 1
        if peaks.size == 0:
            centers.append(None)
            continue

        best_peak = None
        best_score = -1
        best_range = None

        # 2. 遍历每个候选峰，计算相似度
        for p in peaks:
            peak_val = col_s[p]
            half = peak_val / 2.0

            # 半高宽范围
            left = p
            while left > 0 and col_s[left] > half:
                left -= 1
            right = p
            while right < h - 1 and col_s[right] > half:
                right += 1
            width = max(1, right - left)

            # 计算相似度 S_alpha
            fw, fg = std_width, std_gray
            fcw, fcg = width, peak_val
            numerator = Ww * abs(fw - fcw) + Wg * abs(fg - fcg)
            denom = max(abs(fw - fcw), abs(fg - fcg)) + 1e-9
            S_alpha = 1 - (numerator / denom) if denom > 1e-9 else 1.0

            # 选最优峰
            if S_alpha > best_score:
                best_score = S_alpha
                best_peak = p
                best_range = (left, right)

        # 3. 对最优峰范围做重心法计算精确中心
        if best_peak is not None and best_range is not None:
            j, k = best_range
            j = max(0, j-5)
            k = min(k+5, len(col))
            weights = col[j:k+1]
            ys_val = (weights * np.arange(j, k+1)).sum() / (weights.sum() + 1e-9)
            centers.append(ys_val)
        else:
            centers.append(None)

    # 4. 转换为点坐标 (x_rel, y)
    pts = []
    for x_rel, y in enumerate(centers):
        if y is not None:
            pts.append((x_rel, float(y)))

    return pts

class LineSegment:
    def __init__(self, start, end):
        self.start = np.array(start)  # 起点坐标 [x, y]
        self.end = np.array(end)  # 终点坐标 [x, y]
        self.length = euclidean(start, end)
        self.slope = self._calculate_slope()
        self.type = self._classify_segment()

    def _calculate_slope(self):
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return np.arctan2(dy, dx) if dx != 0 else np.pi / 2

    def _classify_segment(self):
        angle = np.degrees(self.slope)  # 拟合线段后会得到每条线段的起点、终点、斜率，根据斜率的范围可以定义线段的类型。  比如说线段的斜率很小就是水平直线段
        if -5 < angle < 5:
            return 'h'  # 水平
        elif -85 <= angle < -5:
            return 'u'  # 上斜
        elif 5 < angle <= 85:
            return 'd'  # 下斜
        elif abs(angle) >= 85:
            return 'v'  # 垂直
        return 'u'  # 默认

class Junction:
    def __init__(self, prev_seg, next_seg):
        self.prev = prev_seg
        self.next = next_seg
        self.angle = self._calculate_angle()
        self.type = self._classify_junction()

    def _calculate_angle(self):
        return self.next.slope - self.prev.slope

    def _classify_junction(self):
        angle_deg = np.degrees(self.angle)
        if -5 < angle_deg < 5:  # 平滑过渡
            return 'c1'
        elif -175 <= angle_deg < 5:  # 向上
            return 'c2'
        elif 5 < angle_deg <= 175:  # 向下
            return 'c3'
        # 断裂关系需根据位置判断 (gh/gv/gu/gd)
        return self._classify_break()

    def _classify_break(self):
        vec = self.next.start - self.prev.end
        angle = np.degrees(np.arctan2(vec[1], vec[0]))
        if -10 < angle < 10:
            return 'gh'
        elif -80 <= angle < -10: # 从左至右向上断裂
            return 'gu'
        elif 10< angle <= 80:  # 从左至右向下断裂
            return 'gd'
        return 'gv'

class WeldingSeamMatcher:
    def __init__(self, model_string="h c3 d c2 h c2 u c3 h"):
        self.model_string = model_string

    def match(self, object_segments, collinearity_thresh=0.9):
        integrated_segments = self._integrate_segments(object_segments, collinearity_thresh)
        integrated_string = generate_profile_string(integrated_segments)
        model_tokens = self.model_string.split()
        object_tokens = integrated_string.split()
        match_result = self._stgm_match(model_tokens, object_tokens)

        return match_result, integrated_segments

    def _integrate_segments(self, segments, threshold):
        if len(segments) <= 1:
            return segments

        integrated = []
        i = 0
        while i < len(segments):
            current = segments[i]
            j = i + 1
            while j < len(segments):
                next_seg = segments[j]
                Dc = self._calculate_collinearity(current, next_seg)

                if Dc > threshold and self._can_merge(current, next_seg):                    # 合并线段：创建新线段连接起点和终点
                    merged_seg = LineSegment(current.start, next_seg.end)
                    current = merged_seg
                    j += 1
                else:
                    break

            integrated.append(current)
            i = j if j > i else i + 1

        return integrated

    def _calculate_collinearity(self, seg1, seg2):
        direct_distance = euclidean(seg1.start, seg2.end)
        total_length = seg1.length + euclidean(seg1.end, seg2.start) + seg2.length
        return direct_distance / total_length

    def _can_merge(self, seg1, seg2):
        angle_diff = abs(seg1.slope - seg2.slope)
        return angle_diff < np.radians(5)  # 角度差小于15度

    def _stgm_match(self, model_tokens, object_tokens):
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

def generate_profile_string(segments):
    profile_str = ""

    # 首元素直接添加
    if segments:
        profile_str += segments[0].type

    # 遍历生成连接关系+下一线段
    for i in range(len(segments) - 1):
        junction = Junction(segments[i], segments[i + 1])
        profile_str += f" {junction.type} {segments[i + 1].type}"

    return profile_str.strip()

def fit_line_segments(points, max_error=0.5, min_len=8):

    segments = []
    start_idx = 0
    N = len(points)

    while start_idx < N - 1:
        end_idx = start_idx + min_len
        if end_idx > N:  # 剩余点不够
            break

        best_model = None
        best_rmse = float("inf")
        best_end = end_idx

        while end_idx <= N:
            segment_points = points[start_idx:end_idx]
            X = segment_points[:, 0].reshape(-1, 1)
            y = segment_points[:, 1]

            model = LinearRegression()
            model.fit(X, y)
            pred_y = model.predict(X)
            rmse = np.sqrt(np.mean((pred_y - y) ** 2))

            if rmse <= max_error:
                best_rmse = rmse
                best_model = model
                best_end = end_idx
                end_idx += 1  # 继续尝试扩展
            else:
                break

        # 保存拟合段
        if best_model:
            x0 = points[start_idx][0]
            x1 = points[best_end - 1][0]
            y0 = best_model.predict([[x0]])[0]
            y1 = best_model.predict([[x1]])[0]
            segments.append(LineSegment([x0, y0], [x1, y1]))

        start_idx = best_end - 1  # 从上次结束点继续

    return segments

def determine_weld_position(segments, weld_type):
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

def process_frame(matcher, img_gray, kf=None, prev_window=None,frame_idx=0, show_vis=True):
    H_img, W_img = img_gray.shape
    img_pre = preprocess_image(img_gray)

    if show_vis:
        vis_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        vis_img_pre = cv2.cvtColor(img_pre, cv2.COLOR_GRAY2BGR)
    
    if kf is None or not kf.initialized:
        xs, ys = extract_center_of_gravity(img_pre)
        if len(xs) < 10:
            cx, cy = W_img//2, H_img//2
        else:
            cx = int(xs.mean())
            cy = int(ys.mean())

        Xl = max(0, int(xs.min()) - 2)
        Xr = min(W_img-1, int(xs.max()) + 2)
        Yt = max(0, int(np.min(ys)) - int(ALPHA*STRIPE_WIDTH_EST))
        Yb = min(H_img-1, int(np.max(ys)) + int(ALPHA*STRIPE_WIDTH_EST))
        if kf is None:
            kf = SimpleKalman(A, P0, H, R, Q)
        kf.init_state(cx, cy)
        window = (Xl, Xr, Yt, Yb)

        # 可视化初始状态
        if show_vis:
            # 绘制初始中心点
            cv2.circle(vis_img, (cx, cy), 5, (0, 255, 0), -1)
            # 绘制初始窗口
            cv2.rectangle(vis_img, (Xl, Yt), (Xr, Yb), (0, 255, 255), 2)
            # 显示重心点
            for x, y in zip(xs, ys):
                cv2.circle(vis_img, (int(x), int(y)), 2, (0, 0, 255), -1)
            
            # 显示图像
            cv2.imshow('Initial Detection', vis_img)
            cv2.waitKey(0)

        return kf, window, None

    pred = kf.predict()
    pred_x = int(pred[0,0])
    pred_y = int(pred[1,0])

    if prev_window is None:
        Xl, Xr, Yt, Yb = max(0, pred_x-80), min(W_img-1, pred_x+80), max(0, pred_y-60), min(H_img-1, pred_y+60)
    else:
        Xl, Xr, Yt, Yb = prev_window
        w = Xr - Xl
        h = Yb - Yt
        Xl = max(0, pred_x - w//2); Xr = min(W_img-1, pred_x + w//2)
        Yt = max(0, pred_y - h//2); Yb = min(H_img-1, pred_y + h//2)
    
    if show_vis:
        cv2.rectangle(vis_img, (Xl, Yt), (Xr, Yb), (0, 255, 255), 2)
        cv2.circle(vis_img, (pred_x, pred_y), 5, (255, 0, 0), -1)

    win = img_pre[Yt:Yb+1, Xl:Xr+1]
    pts = detect_center_in_window(win)

    if len(pts) == 0:
        return kf, (Xl,Xr,Yt,Yb), []
    
    points = np.array([[Xl + p[0], Yt + p[1]] for p in pts], dtype=float)

    if show_vis:
        for pt in points:
            cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)

    z = np.array([[points[:,0].mean()],[points[:,1].mean()]])
    kf.update(z)

    # 线段拟合
    segments = fit_line_segments(points)    
    matches, corrected_segments = matcher.match(segments)

    print("匹配结果:", matches)
    print("修正后线段:", [seg.type for seg in corrected_segments])

    # 可视化线段拟合结果
    if show_vis:
        # 绘制修正后的线段
        for seg in corrected_segments:
            cv2.line(vis_img, 
                     (int(seg.start[0]), int(seg.start[1])),
                     (int(seg.end[0]), int(seg.end[1])),
                     (255, 0, 0), 2)
            # 标记线段类型
            mid_x = int((seg.start[0] + seg.end[0]) / 2)
            mid_y = int((seg.start[1] + seg.end[1]) / 2)
            cv2.putText(vis_img, seg.type, (mid_x, mid_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # 6. 焊缝定位
    weld_pos = determine_weld_position(corrected_segments, 'fillet')
    print("焊缝位置坐标:", weld_pos)

     # 可视化焊缝位置 - 修复错误
    if show_vis and weld_pos is not None and len(weld_pos) == 2:
        cv2.circle(vis_img, (int(weld_pos[0]), int(weld_pos[1])), 8, (0, 0, 255), -1)
        cv2.putText(vis_img, "Weld Point", (int(weld_pos[0])+10, int(weld_pos[1])), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # 显示所有可视化结果
    if show_vis:
        # 显示预处理图像
        cv2.imshow('Preprocessed Image', vis_img_pre)
        
        # 显示检测窗口
        win_vis = cv2.cvtColor(win, cv2.COLOR_GRAY2BGR)
        for pt in pts:
            cv2.circle(win_vis, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)
        cv2.imshow('Detection Window', win_vis)
        
        # 显示最终结果
        cv2.imshow('Welding Seam Detection', vis_img)
        cv2.waitKey(0)

    return kf, (Xl,Xr,Yt,Yb), weld_pos

def main():
    data_folder = 'GNNTransformer/test'
    imgs = load_images_from_folder(data_folder)

    kf = None
    prev_win = None

    # 初始化焊缝检测器
    matcher = WeldingSeamMatcher("d c2 u")

    for idx, img in enumerate(imgs):
        if img.shape != (IMG_H, IMG_W):
            img = cv2.resize(img, (IMG_W, IMG_H))
        kf, prev_win, pred_weld = process_frame(matcher,img, kf=kf, prev_window=prev_win, frame_idx=idx, show_vis=True)
        prev_win = prev_win
    

if __name__ == '__main__':
    main()
