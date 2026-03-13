import os
import cv2
import numpy as np
from pathlib import Path

def convert_image(src_image, dst_dir, target_size=(192, 192), to_bgr=False):
    # 1. 读取图像（OpenCV 默认 BGR）
    img = cv2.imread(src_image, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"警告：无法读取图片 {src_image}，跳过")
        return

    # 2. 统一转为 3 通道
    if len(img.shape) == 2:  # 灰度图
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:  # RGBA 图，去掉 Alpha 通道
        img = img[:, :, :3]

    # 3. 调整尺寸
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

    # 4. 转换颜色空间（False 表示输出 RGB）
    if not to_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 5. 归一化到 [0, 1] 并转为 float32
    img = img.astype(np.float32)

    # 6. 添加 batch 维度并保存为 raw 二进制
    save_path = os.path.join(dst_dir, Path(src_image).stem + '.raw')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img_batch = img[np.newaxis, ...]  # shape (1, 192, 192, 3)
    img_batch.tofile(save_path)       # 纯二进制，无头部
    print(f"已保存 raw: {save_path}，形状 {img_batch.shape}")

def list_images(folder):
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    return [os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(valid_exts)]

if __name__ == '__main__':
    dst_dir = 'calibration_data_rgb_x5/input_1'
    image_dir = 'test_images'
    target_size = (192, 192)
    to_bgr = False   # 输出 RGB

    images = list_images(image_dir)
    if not images:
        print(f"在 {image_dir} 中未找到图片")
        exit(1)

    print(f"找到 {len(images)} 张图片，开始处理...")
    for img_path in images:
        convert_image(img_path, dst_dir, target_size, to_bgr)
    print("所有图片处理完成！")