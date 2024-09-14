from ultralytics import YOLO
import torch
# Load a model
torch.cuda.set_device(1)
model = YOLO('models/yolov8m.pt')  # load an official model
torch.cuda.set_device(1)
model.to('cuda:1')
# Export the model
results = model.train(data='conf/forgery1.yaml', epochs=7, imgsz=1024)
