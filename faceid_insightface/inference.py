#!/usr/bin/env python3
"""
InsightFace HB_ONNXRuntime 推理代码
用于在RDK X5等Horizon平台上进行人脸特征提取
"""

import os
import argparse
import numpy as np
import cv2
from pathlib import Path

# Horizon工具库
try:
    from horizon_tc_ui import HB_ONNXRuntime
except ImportError:
    print("错误: 未安装 horizon_tc_ui")
    print("请确认在Horizon开发环境中运行")
    raise


def normalize_vector(v):
    """L2归一化向量"""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def cosine_similarity(a, b):
    """计算两个向量的余弦相似度"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def preprocess_image(image_path, model_type="float"):
    """
    预处理图像用于InsightFace模型推理

    参数:
        image_path: 图像文件路径
        model_type: 模型类型，"float" 或 "quantized"

    返回:
        预处理后的图像数组
    """
    # 读取图像
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # BGR转RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 调整大小为112x112
    img = cv2.resize(img, (112, 112))

    if model_type == "quantized":
        # 量化模型期望原始UINT8像素值[0, 255]
        img = img.astype(np.uint8)
    elif model_type == "float":
        # Float模型归一化到[-1, 1]
        img = img.astype(np.float32)
        img = (img - 127.5) / 127.5

    # 添加batch维度并转换为NCHW
    img = np.expand_dims(img, axis=0)  # NHWC
    img = np.transpose(img, (0, 3, 1, 2))  # NCHW

    return img


class InsightFaceInference:
    """InsightFace HB_ONNXRuntime推理类"""

    def __init__(self, model_path):
        """
        初始化推理会话

        参数:
            model_path: ONNX模型文件路径
        """
        self.model_path = model_path

        # 创建HB_ONNXRuntime会话
        self.sess = HB_ONNXRuntime(model_file=model_path)
        self.sess.set_dim_param(0, 0, "?")

        # 判断模型类型
        self.model_type = "quantized" if "quantized" in model_path.lower() else "float"

        # 获取输入布局
        self.input_layout = (
            self.sess.layout[0] if hasattr(self.sess, "layout") else "NCHW"
        )

        print(f"模型加载成功: {model_path}")
        print(f"模型类型: {self.model_type}")
        print(f"输入布局: {self.input_layout}")
        print(f"输入名称: {self.sess.input_names}")
        print(f"输出名称: {self.sess.output_names}")

    def extract_feature(self, image_path):
        """
        从图像中提取人脸特征向量

        参数:
            image_path: 图像文件路径

        返回:
            归一化后的特征向量
        """
        # 预处理图像
        image_data = preprocess_image(image_path, self.model_type)

        # 运行推理
        input_name = self.sess.input_names[0]
        output_names = self.sess.output_names
        output = self.sess.run(output_names, {input_name: image_data})

        # 提取并归一化特征向量
        feature_vector = output[0].flatten()
        normalized_feature = normalize_vector(feature_vector)

        return normalized_feature

    def compare_faces(self, image_path1, image_path2):
        """
        比较两张人脸图像的相似度

        参数:
            image_path1: 第一张图像路径
            image_path2: 第二张图像路径

        返回:
            相似度分数
        """
        feature1 = self.extract_feature(image_path1)
        feature2 = self.extract_feature(image_path2)

        similarity = cosine_similarity(feature1, feature2)
        return similarity


def main():
    parser = argparse.ArgumentParser(description="InsightFace HB_ONNXRuntime推理")
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="ONNX模型文件路径"
    )
    parser.add_argument("-i", "--image", type=str, required=True, help="输入图像路径")
    parser.add_argument(
        "-r", "--reference", type=str, default=None, help="参考图像路径（用于人脸比对）"
    )
    parser.add_argument(
        "--save-feature", type=str, default=None, help="保存特征向量到.npy文件"
    )

    args = parser.parse_args()

    # 初始化模型
    model = InsightFaceInference(args.model)

    # 提取特征
    print(f"\n处理图像: {args.image}")
    feature = model.extract_feature(args.image)
    print(f"特征向量维度: {feature.shape}")
    print(f"特征向量前10个值: {feature[:10]}")

    # 如果提供了参考图像，进行比对
    if args.reference:
        print(f"\n比对图像: {args.reference}")
        similarity = model.compare_faces(args.image, args.reference)
        print(f"相似度: {similarity:.4f}")

        # 判断是否同一人
        threshold = 0.5
        if similarity > threshold:
            print(f"判断结果: 同一人 (相似度 > {threshold})")
        else:
            print(f"判断结果: 不同人 (相似度 <= {threshold})")

    # 保存特征向量
    if args.save_feature:
        np.save(args.save_feature, feature)
        print(f"\n特征向量已保存到: {args.save_feature}")


if __name__ == "__main__":
    main()
