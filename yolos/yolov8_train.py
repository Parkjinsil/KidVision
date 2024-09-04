import os
import numpy as np
import pandas as pd

from ultralytics import YOLO

selected_model = input("pt file path : ")
data_path = input("data.yaml path : ")

model = YOLO(selected_model)
results = model.train(data=data_path , epochs=200, imgsz=512, device=0, save_period=50, dropout=0.1)

