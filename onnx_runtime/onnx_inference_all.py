import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align


def detect_and_mark_faces(image_path, model_path, output_img_path, output_feat_path, ctx_id=-1, det_size=640):
    """Detect faces in image and mark them with bounding boxes and landmarks.
    
    Args:
        image_path: Path to input image file.
        model_path: Path to ONNX recognition model file.
        output_img_path: Path to save marked image.
        output_feat_path: Path to save face embeddings.
        ctx_id: Context ID for execution (-1 for CPU, 0 for GPU).
        det_size: Detection size.
    
    Returns:
        Number of faces detected.
    """
    app = FaceAnalysis(providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
    
    recognition_model = insightface.model_zoo.get_model(model_path)
    recognition_model.prepare(ctx_id=ctx_id)
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot load image from {image_path}")
        return 0
    
    faces = app.get(img)
    num_faces = len(faces)
    print(f"Detected {num_faces} face(s)")
    
    for idx, face in enumerate(faces):
        bbox = face.bbox.astype(int)
        landmarks = face.kps
        
        aimg = face_align.norm_crop(img, landmark=face.kps, image_size=recognition_model.input_size[0])
        
        aimg_path = output_feat_path.replace('.npy', f'_face{idx + 1}_aligned.jpg')
        cv2.imwrite(aimg_path, aimg)
        print(f"Saved aligned face {idx + 1} to: {aimg_path}")
        
        input_size = recognition_model.input_size
        blob = cv2.dnn.blobFromImages([aimg], 1.0 / recognition_model.input_std, input_size,
                                      (recognition_model.input_mean, recognition_model.input_mean, recognition_model.input_mean), swapRB=True)
        net_out = recognition_model.session.run(recognition_model.output_names, {recognition_model.input_name: blob})[0]
        embedding = net_out
        
        input_blob_path = output_feat_path.replace('.npy', f'_face{idx + 1}_input.npy')
        np.save(input_blob_path, blob)
        print(f"Saved model input blob {idx + 1} to: {input_blob_path}")
        
        output_blob_path = output_feat_path.replace('.npy', f'_face{idx + 1}_output.npy')
        np.save(output_blob_path, embedding)
        print(f"Saved model output {idx + 1} to: {output_blob_path}")
        
        if embedding is not None:
            print(f"Face {idx + 1} embedding shape: {embedding.shape}")
            feat_path = output_feat_path.replace('.npy', f'_face{idx + 1}.npy')
            np.save(feat_path, embedding)
            print(f"Saved face {idx + 1} embedding to: {feat_path}")
        
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        if landmarks is not None:
            for point in landmarks:
                cv2.circle(img, (int(point[0]), int(point[1])), 2, (255, 0, 0), -1)
        
        cv2.putText(img, f'Face {idx + 1}', (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imwrite(output_img_path, img)
    print(f"Saved marked image to: {output_img_path}")
    
    return num_faces


def main():
    image_path = '/home/vision/insightface/onnx_runtime/face.png'
    model_path = '/home/vision/insightface/onnx_runtime/model.onnx'
    output_img_path = '/home/vision/insightface/onnx_runtime/face_with_detections.jpg'
    output_feat_path = '/home/vision/insightface/onnx_runtime/face_embedding.npy'
    
    num_faces = detect_and_mark_faces(image_path, model_path, output_img_path, output_feat_path)
    print(f"Face detection completed! Total faces: {num_faces}")


if __name__ == '__main__':
    main()