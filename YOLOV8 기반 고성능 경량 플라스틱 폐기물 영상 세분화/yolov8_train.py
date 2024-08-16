import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

from ultralytics import YOLO
import os

#os.environ['TORCH_USE_CUDA_DSA'] = "1"
# Load a model
model = YOLO("yolov8n-seg.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="yolo/dataset.yaml", 
                      epochs=1000, 
                      imgsz=512,
                      batch=10, 
                      save=True,
                      device = 0, 
                      project='train_yolo',
                      pretrained = 'yolov8n.pt',
                      optimizer = 'auto',
                      verbose = True, 
                      val = True, 
                      half = True,
                      cache = 'disk'
                      )