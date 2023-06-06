import torch
from torch import nn
import cv2
import numpy as np


class MyBinarization2(nn.Module):

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
        o_points = torch.from_numpy(np.column_stack(np.where(o_skeleton == 255))).clone().to(device).to(torch.float32)
        g_points = torch.from_numpy(np.column_stack(np.where(g_skeleton == 255))).clone().to(device).to(torch.float32)
        y_points = torch.from_numpy(np.column_stack(np.where(y_skeleton == 255))).clone().to(device).to(torch.float32)
        max_lenge = max(len(o_points), len(g_points), len(y_points))
        o_zeros = torch.zeros([max_lenge, 2], dtype=torch.float32, device=device)
        g_zeros = torch.zeros([max_lenge, 2], dtype=torch.float32, device=device)
        y_zeros = torch.zeros([max_lenge, 2], dtype=torch.float32, device=device)
        for i in range(len(o_points)):
            o_zeros[i] = o_points[i]
        for i in range(len(g_points)):
            g_zeros[i] = g_points[i]
        for i in range(len(y_points)):
            y_zeros[i] = y_points[i]
        self.all_zeros = torch.stack([o_zeros.flatten(), g_zeros.flatten(), y_zeros.flatten()], dim=0)

    def forward(self):
        return self.all_zeros