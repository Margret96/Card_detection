import onnx

onnx_model = onnx.load("models/best.onnx")

onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")