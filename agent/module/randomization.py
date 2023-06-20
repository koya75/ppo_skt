import torch
from torch import nn
import cv2
import numpy as np


class MyRandomization(nn.Module):

    def __init__(self, device, task):
        super().__init__()

        self.device=device
        if task == "Franka":
            pattern1 = torch.from_numpy(np.loadtxt('image/pattern1.csv', delimiter=",")).clone().to(device).to(torch.float32)
            self.pattern1 = pattern1.reshape([10, 5, 2],-1).flatten(1,2)
            pattern2 = torch.from_numpy(np.loadtxt('image/pattern2.csv', delimiter=",")).clone().to(device).to(torch.float32)
            self.pattern2 = pattern2.reshape([10, 5, 2],-1).flatten(1,2)
            pattern3 = torch.from_numpy(np.loadtxt('image/pattern3.csv', delimiter=",")).clone().to(device).to(torch.float32)
            self.pattern3 = pattern3.reshape([10, 5, 2],-1).flatten(1,2)
            pattern4 = torch.from_numpy(np.loadtxt('image/pattern4.csv', delimiter=",")).clone().to(device).to(torch.float32)
            self.pattern4 = pattern4.reshape([10, 5, 2],-1).flatten(1,2)
        elif task == "HSR":
            pattern1 = torch.from_numpy(np.loadtxt('image/hsr1.csv', delimiter=",")).clone().to(device).to(torch.float32)
            self.pattern1 = pattern1.reshape([10, 5, 2],-1).flatten(1,2)


    def select(self, rand):
        pattern = []
        for i in range(len(rand)):
            if rand[i]==0:
                pattern.append(self.pattern1)
            elif rand[i]==1:
                pattern.append(self.pattern2)
            elif rand[i]==2:
                pattern.append(self.pattern3)
            elif rand[i]==3:
                pattern.append(self.pattern4)
        pattern = torch.stack(pattern, dim=0).detach().to(self.device)
        return pattern
        
    def batch_select(self, rand):
        pattern = []
        for i in range(len(rand)):
            if rand[i]==0:
                pattern.append(self.pattern1)
            elif rand[i]==1:
                pattern.append(self.pattern2)
            elif rand[i]==2:
                pattern.append(self.pattern3)
            elif rand[i]==3:
                pattern.append(self.pattern4)
        pattern = torch.stack(pattern, dim=0).detach().to(self.device)
        return pattern