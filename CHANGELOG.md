# 变更日志 (CHANGELOG)

本文档记录项目的所有重要变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

## [2026.02.07]

### 新增 (Added)
- 新增 `onnx_runtime/onnx_inference_all.py` 脚本，实现人脸检测和特征提取功能
- 支持检测图像中的多个人脸并标记
- 支持提取每个人脸的特征向量（embedding）
- 保存对齐后的人脸图像作为模型输入
- 保存模型输入张量（预处理后的数值数据）
- 保存模型原始输出
- 在原图上绘制人脸边界框、关键点和标签

### 改进 (Changed)
- 使用 InsightFace 的 FaceAnalysis 进行人脸检测
- 使用指定的 ONNX 模型（ArcFace）进行特征提取
- 遵循 Google 代码规范编写代码

### 文档 (Documentation)
- 新增 `README.md` 文档，详细说明：
  - 使用的模型（SCRFD 检测模型和 ArcFace 识别模型）
  - 模型输入输出文件说明
  - 数据形状、类型和预处理步骤
  - 使用示例代码
  - 文件命名规则和注意事项

### 技术细节 (Technical Details)

#### 模型配置
- **检测模型**: SCRFD (buffalo_l/det_10g.onnx)
- **识别模型**: ArcFace (model.onnx)
- **输入尺寸**: 112x112
- **输入均值**: 127.5
- **输入标准差**: 127.5
- **输出维度**: 512

#### 输出文件
- `face_embedding_face{N}_aligned.jpg` - 对齐后的人脸图像（模型视觉输入）
- `face_embedding_face{N}_input.npy` - 模型输入张量 `(1, 3, 112, 112)`
- `face_embedding_face{N}_output.npy` - 模型原始输出 `(1, 512)`
- `face_embedding_face{N}.npy` - 人脸特征向量 `(512,)`
- `face_with_detections.jpg` - 标记后的图像

#### 预处理流程
1. 使用 FaceAnalysis 检测人脸位置和关键点
2. 使用 `face_align.norm_crop()` 对齐人脸到 112x112
3. 使用 `cv2.dnn.blobFromImages()` 进行归一化：`(input - 127.5) / 127.5`
4. 将预处理后的张量输入 ONNX 模型
5. 获取 512 维特征向量