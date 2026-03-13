#!/usr/bin/env python3
"""
预处理校准数据集脚本
按照 insightface.md 文档中的预处理流程处理图片
并将处理后的数据保存为float32格式的二进制文件
"""

import os
import cv2
import numpy as np
from pathlib import Path


def preprocess_image_for_insightface(image_path):
    """
    按照insightface.md文档中的流程预处理图像

    Args:
        image_path: 输入图像路径

    Returns:
        预处理后的numpy数组，形状为[1, 3, 112, 112]，dtype=float32
    """
    # 1. 读取图像
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # 2. 调整尺寸到112x112
    img = cv2.resize(img, (112, 112))

    # 3. 颜色空间转换: BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 4. 数值预处理: 只转换为float32，不做归一化，保持[0, 255]范围
    img = img.astype(np.float32)

    # 5. 添加批次维度并调整为NCHW格式
    if len(img.shape) == 3:  # HWC
        img = np.expand_dims(img, axis=0)  # 添加批次维度 -> BHWC
        img = np.transpose(img, (0, 3, 1, 2))  # 转换为NCHW

    return img


def main():
    """主函数"""
    # 设置输入输出路径
    calibration_dir = Path("/home/x5_ptq/insightface/calibration_data")
    output_dir = Path("/home/x5_ptq/insightface/calibration_processed")

    # 创建输出目录
    output_dir.mkdir(exist_ok=True)

    # 获取所有图片文件
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(calibration_dir.glob(f"*{ext}"))
        image_files.extend(calibration_dir.glob(f"*{ext.upper()}"))

    if not image_files:
        print(f"在 {calibration_dir} 中未找到图像文件")
        return

    print(f"找到 {len(image_files)} 个图像文件")

    # 处理每个图像文件
    for i, image_path in enumerate(image_files):
        try:
            print(f"正在处理 {i + 1}/{len(image_files)}: {image_path.name}")

            # 预处理图像
            processed_img = preprocess_image_for_insightface(image_path)

            # 生成输出文件名（替换原扩展名为.bin）
            output_filename = image_path.stem + ".bin"
            output_path = output_dir / output_filename

            # 保存为float32格式的二进制文件
            processed_img.tofile(output_path)

            print(f"  保存到: {output_path}")
            print(f"  形状: {processed_img.shape}, 数据类型: {processed_img.dtype}")
            print(f"  数值范围: [{processed_img.min():.3f}, {processed_img.max():.3f}]")

        except Exception as e:
            print(f"  错误处理 {image_path.name}: {str(e)}")
            continue

    print(f"\n预处理完成! 处理后的文件保存在: {output_dir}")
    print(f"共处理了 {len(list(output_dir.glob('*.bin')))} 个二进制文件")


if __name__ == "__main__":
    main()
