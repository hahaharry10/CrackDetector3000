"""
Python class for the Crack detection CNN
"""

import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from torch import nn
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from collections import deque

class CrackCNN:
    def __init__(self, model_path):
        self.model = YOLO(model_path).to("cpu")

    def predict(self, frame):
        self.prediction = self.model.predict(frame)
    
    def frameHasCrack(self) -> bool:
        if self.prediction is None:
            print("Error: Prediction not made.")
            return False
        return len(self.prediction[0].boxes.cls) > 0
