import torch
import torchvision.models as models
import onnx

# Pre trained model: ResNet-18
print("Loading ResNet-18...")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

# Dummy input to match model input shape
dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)

# Export directly to ONNX
output_path = "resnet18.onnx"
print(f"Exporting model to {output_path}...")

torch.onnx.export(
    model,
    dummy_input,
    output_path,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"]
)

# Clean out  metadata (clean local paths)
onnx_model = onnx.load(output_path)
onnx_model.doc_string = ""
onnx_model.graph.doc_string = ""
while len(onnx_model.metadata_props) > 0:
    onnx_model.metadata_props.pop()
for node in onnx_model.graph.node:
    node.doc_string = ""

onnx.save(onnx_model, output_path)
print("Export complete and metadata cleaned.")