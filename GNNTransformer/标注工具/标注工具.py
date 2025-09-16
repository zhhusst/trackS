"""
标注工具
zhh 20250901 修正版 + 记忆上次进度 + 进度条
"""

import cv2
import numpy as np
import os
import json
import copy
from skimage.morphology import skeletonize


class WeldAnnotationTool:
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.annotation_file = os.path.join(image_folder, "annotations.json")

        # 按顺序读取图片
        self.image_files = sorted(
            [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        )

        self.annotations = {}
        self.current_index = 0
        self.points = []
        self.current_image = None
        self.win_name = "Weld Annotation Tool"

        # 尝试加载现有标注
        if os.path.exists(self.annotation_file):
            with open(self.annotation_file, 'r') as f:
                self.annotations = json.load(f)

            # 如果有保存的上次退出位置
            if "_last_index" in self.annotations:
                self.current_index = min(self.annotations["_last_index"], len(self.image_files) - 1)
                print(f"恢复到上次进度: 第 {self.current_index+1}/{len(self.image_files)} 张 ({self.image_files[self.current_index]})")

    def extract_laser_stripe(self, image):
        """提取激光条纹中心线"""
        red_channel = image[:, :, 2]
        _, binary = cv2.threshold(red_channel, 150, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        skeleton = skeletonize((binary // 255).astype(bool))
        skeleton = (skeleton * 255).astype(np.uint8)

        contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        points = []
        for cnt in contours:
            for point in cnt:
                points.append(point[0].tolist())
        return points

    def auto_annotate(self):
        """自动标注辅助函数"""
        filename = self.image_files[self.current_index]
        img = cv2.imread(os.path.join(self.image_folder, filename))

        if filename in self.annotations:
            self.points = self.annotations[filename]
        else:
            stripe_points = self.extract_laser_stripe(img)
            if stripe_points:
                lowest_point = max(stripe_points, key=lambda p: p[1])
                self.points = [lowest_point]
            else:
                self.points = []

    def draw_progress_bar(self, img):
        """绘制进度条 (显示在顶部)"""
        total = len(self.image_files)
        current = self.current_index + 1
        bar_x, bar_y, bar_w, bar_h = 50, 10, 400, 15

        progress = int((current / total) * bar_w)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), -1)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + progress, bar_y + bar_h), (0, 200, 0), -1)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (0, 0, 0), 1)

        text = f"{current}/{total}"
        cv2.putText(img, text, (bar_x + bar_w + 15, bar_y + bar_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    def run(self):
        """运行标注工具"""
        cv2.namedWindow(self.win_name)
        cv2.setMouseCallback(self.win_name, self.mouse_callback)

        while self.current_index < len(self.image_files):
            img_path = os.path.join(self.image_folder, self.image_files[self.current_index])
            img = cv2.imread(img_path)
            self.current_image = img.copy()

            self.auto_annotate()

            display_img = img.copy()

            # 绘制已有标注点
            for point in self.points:
                cv2.circle(display_img, tuple(point), 8, (0, 0, 255), -1)

            # 绘制文件名和提示
            text = f"{self.image_files[self.current_index]} - Points: {len(self.points)}"
            cv2.putText(display_img, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
            cv2.putText(display_img,
                        "Click:add | d:delete | s:save | n:next | p:prev | q:quit",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 0), 1)

            # 绘制进度条
            self.draw_progress_bar(display_img)

            cv2.imshow(self.win_name, display_img)

            key = cv2.waitKey(0) & 0xFF

            if key == ord('s'):
                self.save_annotation()
                print(f"Saved annotation for {self.image_files[self.current_index]}")
            elif key == ord('d'):
                if self.points:
                    self.points.pop()
            elif key == ord('n'):
                self.save_annotation()
                self.current_index = min(self.current_index + 1, len(self.image_files) - 1)
            elif key == ord('p'):
                self.save_annotation()
                self.current_index = max(self.current_index - 1, 0)
            elif key == ord('q'):
                self.save_annotation()
                self.annotations["_last_index"] = self.current_index
                with open(self.annotation_file, 'w') as f:
                    json.dump(self.annotations, f, indent=4)
                print(f"退出并保存进度: 第 {self.current_index+1}/{len(self.image_files)} 张")
                break

        cv2.destroyAllWindows()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append([x, y])
            print(f"Added point at ({x}, {y})")
            display_img = self.current_image.copy()
            for point in self.points:
                cv2.circle(display_img, tuple(point), 8, (0, 0, 255), -1)
            self.draw_progress_bar(display_img)
            cv2.imshow(self.win_name, display_img)

    def save_annotation(self):
        filename = self.image_files[self.current_index]
        self.annotations[filename] = copy.deepcopy(self.points)
        self.annotations["_last_index"] = self.current_index
        with open(self.annotation_file, 'w') as f:
            json.dump(self.annotations, f, indent=4)


if __name__ == "__main__":
    image_folder = ""  # 修改为你的路径
    tool = WeldAnnotationTool(image_folder)
    tool.run()
