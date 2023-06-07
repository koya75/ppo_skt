import torch
from torch import nn
import cv2
import numpy as np


class MyRandomization(nn.Module):

    def __init__(self, device):
        super().__init__()

        pattern1 = torch.from_numpy(np.loadtxt('image/pattern1.csv', delimiter=",")).clone().to(device).to(torch.float32)
        self.pattern1 = pattern1.reshape([10, 5, 2],-1).flatten(1,2)
        pattern2 = torch.from_numpy(np.loadtxt('image/pattern2.csv', delimiter=",")).clone().to(device).to(torch.float32)
        self.pattern2 = pattern2.reshape([10, 5, 2],-1).flatten(1,2)
        pattern3 = torch.from_numpy(np.loadtxt('image/pattern3.csv', delimiter=",")).clone().to(device).to(torch.float32)
        self.pattern3 = pattern3.reshape([10, 5, 2],-1).flatten(1,2)
        pattern4 = torch.from_numpy(np.loadtxt('image/pattern4.csv', delimiter=",")).clone().to(device).to(torch.float32)
        self.pattern4 = pattern4.reshape([10, 5, 2],-1).flatten(1,2)


    def forward(self, rand):
        print(rand)
        if rand==0:
            return self.pattern1
        elif rand==1:
            return self.pattern2
        elif rand==2:
            return self.pattern3
        elif rand==3:
            return self.pattern4