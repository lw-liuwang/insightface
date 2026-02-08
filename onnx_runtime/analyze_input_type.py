import onnx
import numpy as np


def analyze_input_data_type(model_path):
    """Analyze input data type of ONNX model.
    
    Args:
        model_path: Path to ONNX model file.
    
    Returns:
        data_type: The input data type ('rgb', 'bgr', 'yuv444', 'gray', 'featuremap')
    """
    model = onnx.load(model_path)
    
    print(f"Model: {model_path}")
    print(f"\n=== Model Information ===")
    print(f"Producer: {model.producer_name} {model.producer_version}")
    print(f"Opset Version: {[opset.version for opset in model.opset_import]}")
    
    print(f"\n=== Input Information ===")
    for inp in model.graph.input:
        print(f"\nInput Name: {inp.name}")
        
        shape = inp.type.tensor_type.shape.dim
        shape_list = [d.dim_value if d.dim_value > 0 else 'dynamic' for d in shape]
        print(f"Shape: {shape_list}")
        
        dtype = inp.type.tensor_type.elem_type
        dtype_str = onnx.helper.tensor_dtype_to_np_dtype(dtype)
        print(f"Data Type: {dtype_str}")
        
        data_type = infer_data_type(shape_list, dtype_str)
        print(f"Inferred Data Type: {data_type}")
        
        return data_type
    
    return None


def infer_data_type(shape, dtype):
    """Infer input data type from shape and dtype.
    
    Args:
        shape: List of dimension values or 'dynamic'
        dtype: ONNX data type string
    
    Returns:
        data_type: 'rgb', 'bgr', 'yuv444', 'gray', 'featuremap'
    """
    if len(shape) != 4:
        if len(shape) == 3:
            return 'featuremap'
        elif len(shape) == 2:
            return 'featuremap'
        return 'unknown'
    
    channels = shape[1]
    
    if channels == 1:
        return 'gray'
    elif channels == 3:
        return 'rgb'
    elif channels == 4:
        return 'yuv444'
    else:
        return 'unknown'


if __name__ == '__main__':
    model_path = '/home/vision/insightface/onnx_runtime/model_fixed.onnx'
    data_type = analyze_input_data_type(model_path)
    
    if data_type:
        print(f"\n=== Conclusion ===")
        print(f"Input Data Type: {data_type}")
        print(f"\nSupported types: 'rgb', 'bgr', 'yuv444', 'gray', 'featuremap'")