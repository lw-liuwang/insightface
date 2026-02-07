# 模型选择
 Face Recognition models
 List of models by various depth IResNet and training datasets:中的 R100

# ONNX 运行时推理说明

## 概述

`onnx_runtime/onnx_inference_all.py` 脚本用于人脸检测和特征提取，使用 InsightFace 框架和 ONNX 模型进行推理。

## 使用的模型

### 1. 人脸检测模型
- **模型类型**: SCRFD (Sample Convolutional Receptive Field Detection)
- **模型来源**: FaceAnalysis 自动加载的 buffalo_l 模型包
- **模型路径**: `~/.insightface/models/buffalo_l/det_10g.onnx`
- **功能**: 检测图像中的人脸位置和关键点

### 2. 人脸识别模型
- **模型类型**: ArcFace (Additive Angular Margin Loss)
- **模型路径**: `/home/vision/insightface/onnx_runtime/model.onnx`
- **输入尺寸**: 112x112
- **输入均值**: 127.5
- **输入标准差**: 127.5
- **输出维度**: 512
- **功能**: 提取人脸特征向量（embedding）

## 模型输入输出说明

### 模型输入

#### 1. 对齐人脸图像
- **文件格式**: `face_embedding_face{N}_aligned.jpg`
- **文件说明**: 经过对齐和裁剪的人脸图像，直接作为模型的视觉输入
- **图像尺寸**: 112x112 像素
- **颜色空间**: BGR（OpenCV 默认格式）
- **预处理**:
  - 使用人脸关键点进行相似变换对齐
  - 裁剪到固定尺寸 112x112

#### 2. 模型输入张量
- **文件格式**: `face_embedding_face{N}_input.npy`
- **文件说明**: 经过预处理后的数值张量，直接输入到 ONNX 模型
- **数据形状**: `(1, 3, 112, 112)`
- **数据类型**: `float32`
- **预处理步骤**:
  ```python
  blob = cv2.dnn.blobFromImages(
      [aimg],                          # 对齐后的人脸图像
      1.0 / 127.5,                    # 缩放因子 (1/std)
      (112, 112),                      # 目标尺寸
      (127.5, 127.5, 127.5),          # 减去的均值 (mean)
      swapRB=True                      # BGR 转 RGB
  )
  ```
- **归一化公式**: `output = (input - 127.5) / 127.5`

### 模型输出

#### 1. 模型原始输出
- **文件格式**: `face_embedding_face{N}_output.npy`
- **文件说明**: ONNX 模型的原始推理输出
- **数据形状**: `(1, 512)`
- **数据类型**: `float32`
- **数值范围**: 通常在 -1 到 1 之间（经过 L2 归一化）

#### 2. 人脸特征向量
- **文件格式**: `face_embedding_face{N}.npy`
- **文件说明**: 展平后的人脸特征向量，用于人脸识别和比对
- **数据形状**: `(512,)`
- **数据类型**: `float32`
- **用途**:
  - 人脸识别：与数据库中的特征向量比对
  - 人脸验证：计算两张图片的相似度
  - 人脸搜索：在海量人脸库中查找相似人脸

## 其他输出文件

### 标记后的图像
- **文件格式**: `face_with_detections.jpg`
- **文件说明**: 在原图上标记人脸检测结果的图像
- **标记内容**:
  - 绿色边界框：人脸位置
  - 蓝色圆点：人脸关键点（5个点）
  - 文本标签：Face 1, Face 2, ...

## 使用示例

### 加载和使用特征向量

```python
import numpy as np

# 加载特征向量
emb1 = np.load('face_embedding_face1.npy')
emb2 = np.load('face_embedding_face2.npy')

# 计算余弦相似度
similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
print(f"相似度: {similarity:.4f}")

# 判断是否为同一个人（阈值通常为 0.5-0.7）
threshold = 0.65
is_same_person = similarity > threshold
print(f"是否为同一人: {'是' if is_same_person else '否'}")
```

### 加载模型输入张量

```python
import numpy as np

# 加载模型输入
input_blob = np.load('face_embedding_face1_input.npy')
print(f"输入形状: {input_blob.shape}")  # (1, 3, 112, 112)
print(f"输入范围: [{input_blob.min():.3f}, {input_blob.max():.3f}]")
```

## 文件命名规则

所有输出文件都基于 `face_embedding.npy` 基础路径生成，按人脸编号（face{N}）区分：

- `face_embedding_face1_aligned.jpg` - 第1个人脸的对齐图像
- `face_embedding_face1_input.npy` - 第1个人脸的模型输入张量
- `face_embedding_face1_output.npy` - 第1个人脸的模型输出
- `face_embedding_face1.npy` - 第1个人脸的特征向量

## 注意事项

1. **模型输入**: 对齐人脸图像（`_aligned.jpg`）是模型的视觉输入，不包含任何标记
2. **预处理**: 模型输入张量已经过归一化处理，值域约为 [-1, 1]
3. **特征向量**: 输出的特征向量已经过 L2 归一化，可以直接用于余弦相似度计算
4. **多人脸**: 如果图像中检测到多个人脸，会为每个人脸生成一组对应的文件
