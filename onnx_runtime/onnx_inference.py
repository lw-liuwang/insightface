import cv2
import numpy as np
import insightface

# 1. 加载人脸识别模型
handler = insightface.model_zoo.get_model('/home/vision/insightface/onnx_runtime/model.onnx')
handler.prepare(ctx_id=0)

# 2. 加载图像
img = cv2.imread('/home/vision/insightface/onnx_runtime/face.png')

# 3. 直接提取人脸特征
# 注意：get_feat 方法会自动处理输入图像
# 包括调整大小、归一化等预处理步骤
embedding = handler.get_feat(img)
print(f"Face embedding shape: {embedding.shape}")

# 4. 保存模型输出（特征向量）
output_path = f"/home/vision/insightface/onnx_runtime/model_output_direct.npy"
np.save(output_path, embedding)
print(f"Saved model output (embedding) to: {output_path}")

# 5. 在图像上绘制特征信息
# 由于没有人脸检测，我们在图像左上角绘制信息
feat_dim = embedding.shape[-1]
cv2.putText(img, f'Feature Dimension: {feat_dim}', (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.putText(img, 'Direct Feature Extraction', (10, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# 6. 保存带有特征信息的图像
output_img_path = "/home/vision/insightface/onnx_runtime/face_with_features.jpg"
cv2.imwrite(output_img_path, img)
print(f"Saved image with feature info to: {output_img_path}")

print("Direct face feature extraction completed!")