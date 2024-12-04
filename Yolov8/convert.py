from ultralytics import YOLO

model = YOLO('models/best.pt')

model.export(format='onnx', imgsz=640, dynamic=False, simplify=True)