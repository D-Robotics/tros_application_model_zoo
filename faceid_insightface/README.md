# InsightFace 模型推理

本目录包含InsightFace模型的ONNX推理代码和相关配置文件，用于在D-Robotics RDK X5平台上进行人脸特征提取。

## 目录结构

```
faceid_insightface/
├── insightface.onnx          # Float ONNX模型
├── insightface.yaml          # 量化配置文件
├── insightface.bin           # 量化后的模型 (hbm格式)
├── inference.py              # HB_ONNXRuntime推理代码
├── test_inference.py         # 推理测试脚本
├── preprocess_calibration_data.py  # 数据预处理脚本
├── image_a_1.jpg             # 测试图片: Person A - 图片1
├── image_a_2.jpg             # 测试图片: Person A - 图片2 (同一人)
├── image_b.jpg               # 测试图片: Person B (不同人)
└── README.md                 # 本说明文档
```

## 环境要求

- D-Robotics RDK X5 开发板
- horizon_tc_ui 工具库
- Python 3.8+
- OpenCV
- NumPy

## 图像预处理逻辑

### Float模型预处理流程

Float模型(`insightface.onnx`)的正确预处理步骤：

1. **读取图像**：使用OpenCV读取，格式为BGR (默认)
2. **转换颜色空间**：BGR → RGB
3. **调整尺寸**：Resize到112x112像素
4. **数据类型转换**：转换为float32
5. **归一化**：`(pixel - 127.5) / 127.5`，将像素值从[0, 255]映射到[-1, 1]
6. **维度转换**：添加batch维度并转换为NCHW格式
   - 原始: HWC (112x112x3)
   - 目标: NCHW (1x3x112x112)

**代码实现**（见`inference.py:preprocess_image`）：
```python
img = cv2.imread(image_path)                    # 1. 读取BGR图像
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     # 2. BGR转RGB
img = cv2.resize(img, (112, 112))              # 3. Resize到112x112
img = img.astype(np.float32)                   # 4. 转float32
img = (img - 127.5) / 127.5                    # 5. 归一化到[-1, 1]
img = np.expand_dims(img, axis=0)              # 6. 添加batch维度
img = np.transpose(img, (0, 3, 1, 2))          # 7. NCHW: NHWC -> NCHW
```

### 量化模型预处理流程

量化模型(`insightface.bin`)的预处理步骤：

1. **读取图像**：BGR格式
2. **转换颜色空间**：BGR → RGB
3. **调整尺寸**：Resize到112x112像素
4. **保持UINT8格式**：[0, 255]范围，不做归一化
5. **维度转换**：NCHW格式 (1x3x112x112)
6. **内部处理**：推理时自动减去127.0

**注意**：量化模型在校准阶段不进行归一化，而是在推理阶段内部处理。

### 预处理差异说明

| 步骤 | Float模型 | 量化模型 |
|------|-----------|----------|
| 颜色空间 | BGR→RGB | BGR→RGB |
| 归一化 | (x-127.5)/127.5 | 无(内部处理) |
| 数据类型 | float32 | uint8 |
| 数值范围 | [-1, 1] | [0, 255] |

## 快速开始

### 1. 模型检查

```bash
hb_mapper checker --model-type onnx \
                  --model insightface.onnx \
                  --march bayes-e
```

### 2. 模型量化

```bash
hb_mapper makertbin --config insightface.yaml \
                    --model-type onnx
```

### 3. 运行推理

#### 使用测试脚本（推荐）

运行自动测试脚本，测试同一人和不同人的比对：

```bash
cd faceid_insightface
python3 test_inference.py
```

#### 单张图像特征提取

```bash
cd faceid_insightface
python3 inference.py \
    --model insightface.onnx \
    --image image_a_1.jpg
```

#### 人脸比对（两张图像）

同一人脸比对：
```bash
cd faceid_insightface
python3 inference.py \
    --model insightface.onnx \
    --image image_a_1.jpg \
    --reference image_a_2.jpg
```

不同人脸比对：
```bash
cd faceid_insightface
python3 inference.py \
    --model insightface.onnx \
    --image image_a_1.jpg \
    --reference image_b.jpg
```

## 测试步骤与结果

### 测试图片说明

| 图片 | 说明 |
|------|------|
| `image_a_1.jpg` | Person A - 正面照片 |
| `image_a_2.jpg` | Person A - 同一人，不同角度/表情 |
| `image_b.jpg` | Person B - 不同人 |

### 测试步骤

1. **初始化模型**：加载`insightface.onnx`模型
2. **提取特征**：对每张图片提取512维特征向量
3. **计算相似度**：使用余弦相似度计算两张图片的相似度
4. **判断身份**：相似度 > 0.5 判定为同一人

### 测试结果

使用Float模型(`insightface.onnx`)的测试结果：

| 测试场景 | 图片对 | 相似度 | 判断结果 |
|---------|--------|--------|----------|
| 同一人比对 | image_a_1.jpg vs image_a_2.jpg | **0.7611** | ✓ 同一人 |
| 不同人比对 | image_a_1.jpg vs image_b.jpg | **-0.0261** | ✓ 不同人 |

**测试结论**：
- 同一人相似度：0.7611（明显高于阈值0.5）
- 不同人相似度：-0.0261（明显低于阈值0.5）
- 模型能够有效区分同一人与不同人

### 预期结果范围

- **同一人**：相似度通常在 0.60 ~ 0.85 之间
- **不同人**：相似度通常在 -0.10 ~ 0.30 之间
- **判定阈值**：0.5（可根据实际场景调整）

## 详细使用方法

### inference.py 参数说明

| 参数 | 简写 | 必填 | 说明 |
|------|------|------|------|
| `--model` | `-m` | 是 | ONNX模型文件路径 |
| `--image` | `-i` | 是 | 输入图像路径 |
| `--reference` | `-r` | 否 | 参考图像路径（用于比对） |
| `--save-feature` | 无 | 否 | 保存特征向量的文件路径 |

### Python API使用示例

```python
from inference import InsightFaceInference

# 初始化模型
model = InsightFaceInference("insightface.onnx")

# 提取特征
feature = model.extract_feature("image_a_1.jpg")
print(f"特征维度: {feature.shape}")  # 输出: (512,)

# 人脸比对
similarity = model.compare_faces("image_a_1.jpg", "image_a_2.jpg")
print(f"相似度: {similarity:.4f}")

# 判断是否同一人
if similarity > 0.5:
    print("同一人")
else:
    print("不同人")
```

## 相似度对比详解

### 余弦相似度原理

InsightFace使用**余弦相似度**来衡量两个人脸特征向量的相似程度：

```
similarity = cos(θ) = (A · B) / (||A|| × ||B||)
```

其中：
- **A · B**：两个向量的点积
- **||A||, ||B||**：两个向量的L2范数（模长）
- **θ**：两个向量之间的夹角

### 特征向量预处理

模型输出的512维特征向量已经过**L2归一化**，即每个特征向量的模长为1：

```python
# L2归一化
norm = sqrt(x₁² + x₂² + ... + x₅₁₂²) = 1
normalized_vector = vector / norm
```

由于特征向量已归一化，余弦相似度计算简化为：

```python
similarity = dot(feature1, feature2)  # 因为 ||A|| = ||B|| = 1
```

### 相似度取值范围

余弦相似度的取值范围为 **[-1, 1]**：

| 相似度值 | 含义 | 人脸关系 |
|---------|------|---------|
| 1.0 | 完全相同 | 理论上完全相同的两张脸 |
| 0.7 ~ 0.9 | 非常相似 | 同一人，角度/表情略有不同 |
| 0.5 ~ 0.7 | 相似 | 同一人，变化较大 |
| 0.0 ~ 0.5 | 一般相似 | 可能是同一人（边界情况） |
| -0.5 ~ 0.0 | 不太相似 | 通常不同人 |
| -1.0 ~ -0.5 | 不相似 | 明显不同人 |

### 判定阈值选择

**推荐阈值：0.5**

阈值选择需要在**准确率**和**召回率**之间权衡：

| 阈值 | 特点 | 适用场景 |
|------|------|---------|
| 0.6 | 更严格，误识别率低 | 高安全性场景（如金融支付） |
| **0.5** | **平衡，推荐值** | **通用场景** |
| 0.4 | 更宽松，漏识别率低 | 需要高召回率的场景 |
| 0.3 | 非常宽松 | 可能产生较多误识别 |

### 实际测试数据解读

本次测试的实际结果：

| 对比 | 相似度 | 解读 |
|------|--------|------|
| Person A (图1 vs 图2) | 0.7611 | 高相似度，确信同一人 |
| Person A vs Person B | -0.0261 | 接近0，确信不同人 |

**距离感**：
- 同一人相似度 **0.7611** 意味着两张人脸特征向量的夹角很小（约40度）
- 不同人相似度 **-0.0261** 意味着两张人脸特征向量几乎垂直（约91.5度）

### 代码实现细节

**L2归一化**（`inference.py:22-27`）：
```python
def normalize_vector(v):
    """L2归一化向量"""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm
```

**余弦相似度计算**（`inference.py:30-32`）：
```python
def cosine_similarity(a, b):
    """计算两个向量的余弦相似度"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

**特征提取与归一化**（`inference.py:120-122`）：
```python
# 提取并归一化特征向量
feature_vector = output[0].flatten()
normalized_feature = normalize_vector(feature_vector)
```

### 相似度分布示意图

```
相似度值:    -1.0      -0.5       0.0       0.5       1.0
              |         |         |         |         |
不同人脸 <----|----->    |         |         |<--------> 同一人
                           阈值0.5 ↑
```

### 注意事项

1. **为什么相似度不是1.0？**
   - 即使是同一人，不同照片的角度、表情、光照、年龄都会导致特征有差异
   - 只要相似度显著高于阈值即可判定为同一人

2. **负相似度的含义**
   - 负值表示两个向量方向相反（夹角>90度）
   - 在实际人脸比对中，负值通常明确表示不同人

3. **阈值调优建议**
   - 收集100+对同一人样本和100+对不同人样本
   - 统计各自的相似度分布
   - 选择使错误率最低的阈值作为判定标准

## 输出说明

- **输出特征向量维度**：512维
- **输出已进行L2归一化**：特征向量模长为1
- **特征值范围**：[-1, 1]
- **余弦相似度阈值建议**：0.5（同一人判断）
- **相似度计算**：`similarity = dot(feature1, feature2) / (norm(feature1) * norm(feature2))`

## 性能优化

对于量化模型（`.bin`文件），推理速度更快且占用更少内存：

```bash
# 使用量化模型
python3 inference.py \
    --model insightface.bin \
    --image image_a_1.jpg \
    --reference image_a_2.jpg
```

量化模型会自动识别并应用正确的预处理流程。

## 注意事项

1. **输入图像**：应该包含清晰的人脸，建议使用人脸检测模型（如SCRFD）先检测人脸区域
2. **图像质量**：确保图像质量良好，光照均匀，避免过度曝光或欠曝光
3. **预处理一致性**：Float模型和Float模型的预处理方式不同，代码会自动识别模型类型并应用对应的预处理
4. **校准数据**：`preprocess_calibration_data.py`用于量化校准数据预处理，与推理预处理不同

## 常见问题

**Q: 为什么同一人相似度不是1.0？**  
A: 即使是同一人，不同照片的角度、表情、光照等因素都会导致特征有差异。只要相似度高于阈值（0.5）即可判定为同一人。

**Q: 量化模型和Float模型结果一致吗？**  
A: 量化模型会有少量精度损失，但通常相似度差异在±0.05以内，不影响身份判断。

**Q: 如何处理多人脸图片？**  
A: 本代码假设输入图像已包含裁剪好的人脸区域。如需处理完整图片，请先使用人脸检测模型（如SCRFD）提取人脸区域。

## 参考资料

- [InsightFace GitHub](https://github.com/deepinsight/insightface)
- [Horizon RDK X5 文档](https://developer.horizon.ai/)
