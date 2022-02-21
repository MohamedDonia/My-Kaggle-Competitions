"""
Created on Mon Jan 17 13:36:54 2022

@author: MohamedDonia
"""

import os
import sys
sys.path.insert(0, "/home/umbra/Work/Kaggle/tensorflow-barrier-reef/sahi")
from sahi.predict import get_prediction, get_sliced_prediction, predict
from sahi.model import Yolov5DetectionModel
import warnings
warnings.filterwarnings("ignore")



detection_model = Yolov5DetectionModel(
    repo_path='sahi/yolov2-lib-ds',
    model_path='yolov5/runs/train/cots_f1/weights/best.pt',
    confidence_threshold=0.3,
    device="cpu",
    image_size=640)
result1 = get_prediction('/home/umbra/Work/Kaggle/tensorflow-barrier-reef/sahi/demo/demo_data/small-vehicles1.jpeg', 
                        detection_model)

result2 = get_sliced_prediction(
    '/home/umbra/Work/Kaggle/tensorflow-barrier-reef/sahi/demo/demo_data/small-vehicles1.jpeg',
    detection_model,
    slice_height = 256,
    slice_width = 256,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2
)
