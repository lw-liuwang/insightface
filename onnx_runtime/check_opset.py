import onnx


def check_onnx_opset(model_path):
    """Check the opset version of an ONNX model.
    
    Args:
        model_path: Path to the ONNX model file.
    
    Returns:
        opset_version: The opset version of the model.
    """
    model = onnx.load(model_path)
    opset_version = model.opset_import[0].version
    print(f"Model: {model_path}")
    print(f"Opset Version: {opset_version}")
    return opset_version


if __name__ == '__main__':
    model_path = '/home/vision/insightface/onnx_runtime/model.onnx'
    check_onnx_opset(model_path)