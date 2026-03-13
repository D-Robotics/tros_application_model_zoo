#!/usr/bin/env python3
"""
InsightFace ONNX 推理代码
用于加载ONNX模型并进行人脸特征提取
"""

import os
import argparse
import numpy as np
import cv2
from pathlib import Path

# 尝试导入onnxruntime，如果失败则提供安装提示
try:
    import onnxruntime as ort
except ImportError:
    print("错误: 未安装 onnxruntime")
    print("请安装: pip install onnxruntime")
    raise


def preprocess_image(image_path, target_size=(112, 112)):
    """
    预处理图像用于InsightFace模型推理
    
    参数:
        image_path: 图像文件路径
        target_size: 目标尺寸，默认为(112, 112)
    
    返回:
        预处理后的图像数组，形状为(1, 3, 112, 112)
    """
    # 读取图像
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # BGR转RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 调整大小为112x112
    img = cv2.resize(img, target_size)
    
    # 归一化到[-1, 1]范围
    img = img.astype(np.float32)
    img = (img - 127.5) / 127.5
    
    # 转换为NCHW格式 (batch, channel, height, width)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    
    return img


def normalize_feature(feature):
    """
    L2归一化特征向量
    
    参数:
        feature: 输入特征向量
    
    返回:
        归一化后的特征向量
    """
    norm = np.linalg.norm(feature)
    if norm == 0:
        return feature
    return feature / norm


def cosine_similarity(feature1, feature2):
    """
    计算两个特征向量之间的余弦相似度
    
    参数:
        feature1: 第一个特征向量
        feature2: 第二个特征向量
    
    返回:
        余弦相似度值，范围[-1, 1]
    """
    return np.dot(feature1, feature2)


class InsightFaceONNX:
    """
    InsightFace ONNX模型推理类
    """
    
    def __init__(self, model_path, device='cpu'):
        """
        初始化ONNX推理会话
        
        参数:
            model_path: ONNX模型文件路径
            device: 运行设备，'cpu' 或 'cuda'
        """
        self.model_path = model_path
        
        # 设置运行提供者
        if device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
            self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            self.providers = ['CPUExecutionProvider']
        
        # 创建推理会话
        self.session = ort.InferenceSession(model_path, providers=self.providers)
        
        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # 获取输入形状
        input_shape = self.session.get_inputs()[0].shape
        self.input_shape = input_shape
        
        print(f"模型加载成功: {model_path}")
        print(f"输入名称: {self.input_name}")
        print(f"输入形状: {input_shape}")
        print(f"输出名称: {self.output_names}")
        print(f"运行设备: {self.providers[0]}")
    
    def extract_feature(self, image_path):
        """
        从图像中提取人脸特征向量
        
        参数:
            image_path: 图像文件路径
        
        返回:
            归一化后的特征向量
        """
        # 预处理图像
        input_data = preprocess_image(image_path)
        
        # 运行推理
        outputs = self.session.run(self.output_names, {self.input_name: input_data})
        
        # 提取特征向量（通常是第一个输出）
        feature = outputs[0].flatten()
        
        # L2归一化
        feature = normalize_feature(feature)
        
        return feature
    
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
    parser = argparse.ArgumentParser(description='InsightFace ONNX模型推理')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='ONNX模型文件路径')
    parser.add_argument('-i', '--image', type=str, required=True,
                        help='输入图像路径')
    parser.add_argument('-r', '--reference', type=str, default=None,
                        help='参考图像路径（用于人脸比对）')
    parser.add_argument('-d', '--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='运行设备: cpu 或 cuda')
    parser.add_argument('--save-feature', type=str, default=None,
                        help='保存特征向量到文件')
    
    args = parser.parse_args()
    
    # 初始化模型
    model = InsightFaceONNX(args.model, device=args.device)
    
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
        threshold = 0.5  # 通常使用0.5作为阈值
        if similarity > threshold:
            print(f"判断结果: 同一人 (相似度 > {threshold})")
        else:
            print(f"判断结果: 不同人 (相似度 <= {threshold})")
    
    # 保存特征向量
    if args.save_feature:
        np.save(args.save_feature, feature)
        print(f"\n特征向量已保存到: {args.save_feature}")


if __name__ == '__main__':
    main()
