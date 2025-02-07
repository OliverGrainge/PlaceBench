import onnxruntime as ort 
import onnx 
import numpy as np 
import torch.nn as nn 
import torch 
from onnx import numpy_helper
from onnxruntime.quantization import quantize_dynamic, QuantType
from PIL import Image
from torchvision import transforms



model = torch.hub.load(
        "gmberton/eigenplaces",
        "get_trained_model",
        backbone="ResNet50",
        fc_output_dim=2048,
    )

print(model)

# Set the model to evaluation mode
model.eval()

# Create dummy input (adjust size according to your model's input requirements)
dummy_input = torch.randn(1, 3, 512, 512)  # Batch size 1, 3 channels, 224x224 image

# Export to ONNX
onnx_path = "eigenplaces_model.onnx"
torch.onnx.export(model,               # model being run
                 dummy_input,          # model input (or a tuple for multiple inputs)
                 onnx_path,            # where to save the model
                 export_params=True,   # store the trained parameter weights inside the model file
                 opset_version=11,     # the ONNX version to export the model to
                 do_constant_folding=True)  # whether to execute constant folding for optimization

# Quantize the model
quantized_model_path = "eigenplaces_model_quantized.onnx"
quantize_dynamic(
    model_input=onnx_path,
    model_output=quantized_model_path,
    weight_type=QuantType.QUInt8
)

print(f"Quantized model saved to: {quantized_model_path}")

# Initialize inference session with the quantized model
session = ort.InferenceSession("eigenplaces_model_quantized.onnx")

# Define image preprocessing (similar to what you'd use with PyTorch)

def get_embedding():
    # Load and preprocess the image
    image_tensor = torch.randn(10, 3, 512, 512)
    
    # Convert to numpy array - no need for expand_dims since image_tensor already has batch dimension
    input_tensor = image_tensor.numpy()
    
    # Get input name
    input_name = session.get_inputs()[0].name
    
    # Run inference
    embedding = session.run(None, {input_name: input_tensor})[0]
    
    return embedding


# Example usage
if __name__ == "__main__":
    # Replace with your image path
    embedding = get_embedding()
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding: {embedding}")