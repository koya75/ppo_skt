import torch
from torch import nn
import cv2
import numpy as np


class MyBinarization(nn.Module):

    def __init__(self, image, device):
        super().__init__()

        img = cv2.imread(image)
        img = cv2.resize(img, (128, 128))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        o_lower = np.array([15,64,190])
        o_upper = np.array([25,255,255])
        g_lower = np.array([60,70,0])
        g_upper = np.array([90,255,255])
        y_lower = np.array([25,64,195])
        y_upper = np.array([80,255,255])
        o_frame_mask = cv2.inRange(hsv, o_lower, o_upper)
        g_frame_mask = cv2.inRange(hsv, g_lower, g_upper)
        y_frame_mask = cv2.inRange(hsv, y_lower, y_upper)
        o_skeleton = cv2.ximgproc.thinning(o_frame_mask, thinningType=cv2.ximgproc.THINNING_GUOHALL)
        g_skeleton = cv2.ximgproc.thinning(g_frame_mask, thinningType=cv2.ximgproc.THINNING_GUOHALL)
        y_skeleton = cv2.ximgproc.thinning(y_frame_mask, thinningType=cv2.ximgproc.THINNING_GUOHALL)
        self.o_points = torch.from_numpy(np.column_stack(np.where(o_skeleton == 255))).clone().to(device).to(torch.float32)
        self.g_points = torch.from_numpy(np.column_stack(np.where(g_skeleton == 255))).clone().to(device).to(torch.float32)
        self.y_points = torch.from_numpy(np.column_stack(np.where(y_skeleton == 255))).clone().to(device).to(torch.float32)

    def forward(self):
        return self.o_points, self.g_points, self.y_points