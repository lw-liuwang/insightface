import onnx
from onnx import shape_inference


def fix_dynamic_input_shape(model_path, output_path, fixed_batch_size=1):
    """Fix dynamic input shape in ONNX model by setting batch size to fixed value.
    
    Args:
        model_path: Path to input ONNX model file.
        output_path: Path to save fixed ONNX model file.
        fixed_batch_size: Fixed batch size to set (default: 1).
    """
    model = onnx.load(model_path)
    
    print(f"Original model inputs:")
    for inp in model.graph.input:
        print(f"  Name: {inp.name}, Shape: {[d.dim_value if d.dim_value > 0 else 'None' for d in inp.type.tensor_type.shape.dim]}")
    
    for inp in model.graph.input:
        if inp.type.tensor_type.shape.dim[0].dim_value == 0:
            inp.type.tensor_type.shape.dim[0].dim_value = fixed_batch_size
            print(f"Fixed input '{inp.name}' batch size to {fixed_batch_size}")
    
    model = shape_inference.infer_shapes(model)
    
    onnx.save(model, output_path)
    print(f"\nFixed model saved to: {output_path}")
    
    print(f"\nFixed model inputs:")
    for inp in model.graph.input:
        print(f"  Name: {inp.name}, Shape: {[d.dim_value for d in inp.type.tensor_type.shape.dim]}")


if __name__ == '__main__':
    model_path = '/home/vision/insightface/onnx_runtime/model.onnx'
    output_path = '/home/vision/insightface/onnx_runtime/model_fixed.onnx'
    fix_dynamic_input_shape(model_path, output_path, fixed_batch_size=1)