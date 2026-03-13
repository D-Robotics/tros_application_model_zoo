#!/usr/bin/env python3
"""
InsightFace 推理测试脚本
测试新的图片命名: image_a_1.jpg, image_a_2.jpg (同一人), image_b.jpg (不同人)
"""

import os
import sys
from inference import InsightFaceInference

def test_inference(model_path="insightface.onnx"):
    """运行推理测试"""
    
    # 测试图片路径
    image_a_1 = "image_a_1.jpg"  # Person A - 图片1
    image_a_2 = "image_a_2.jpg"  # Person A - 图片2 (同一人)
    image_b = "image_b.jpg"      # Person B (不同人)
    
    # 检查文件是否存在
    for img in [image_a_1, image_a_2, image_b]:
        if not os.path.exists(img):
            print(f"错误: 找不到图片 {img}")
            return
    
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型 {model_path}")
        return
    
    print("="*60)
    print("InsightFace 推理测试")
    print("="*60)
    
    # 初始化模型
    print(f"\n[1] 加载模型: {model_path}")
    model = InsightFaceInference(model_path)
    
    # 测试1: 同一人 (image_a_1 vs image_a_2)
    print("\n" + "="*60)
    print("[测试1] 同一人比对")
    print("="*60)
    print(f"图片1: {image_a_1}")
    print(f"图片2: {image_a_2}")
    print("-"*60)
    
    similarity_same = model.compare_faces(image_a_1, image_a_2)
    print(f"相似度: {similarity_same:.4f}")
    
    threshold = 0.5
    if similarity_same > threshold:
        print(f"判断结果: 同一人 (相似度 > {threshold})")
    else:
        print(f"判断结果: 不同人 (相似度 <= {threshold})")
    
    # 测试2: 不同人 (image_a_1 vs image_b)
    print("\n" + "="*60)
    print("[测试2] 不同人比对")
    print("="*60)
    print(f"图片1: {image_a_1}")
    print(f"图片2: {image_b}")
    print("-"*60)
    
    similarity_diff = model.compare_faces(image_a_1, image_b)
    print(f"相似度: {similarity_diff:.4f}")
    
    if similarity_diff > threshold:
        print(f"判断结果: 同一人 (相似度 > {threshold})")
    else:
        print(f"判断结果: 不同人 (相似度 <= {threshold})")
    
    # 汇总结果
    print("\n" + "="*60)
    print("[测试结果汇总]")
    print("="*60)
    print(f"同一人比对 (A_1 vs A_2): 相似度 = {similarity_same:.4f}")
    print(f"不同人比对 (A_1 vs B):   相似度 = {similarity_diff:.4f}")
    print("-"*60)
    
    # 验证结果合理性
    print("\n[结果验证]")
    if similarity_same > threshold and similarity_diff < threshold:
        print("✓ 测试通过: 同一人相似度高，不同人相似度低")
    elif similarity_same > threshold and similarity_diff > threshold:
        print("✗ 测试异常: 不同人相似度过高")
    elif similarity_same < threshold and similarity_diff < threshold:
        print("✗ 测试异常: 同一人相似度过低")
    else:
        print("✗ 测试失败: 结果不符合预期")
    
    print("="*60)

if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "insightface.onnx"
    
    test_inference(model_path)
