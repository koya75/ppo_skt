from pathlib import Path

from isaacgym import gymapi
from isaacgym import gymtorch
import math
import numpy as np

from isaacgym.torch_utils import *


class Camera:
    def create(self, gym, sim, width, height, image_type):
        self.gym = gym
        self.sim = sim
        self.cam_props = gymapi.CameraProperties()
        self.cam_props.width = width
        self.cam_props.height = height
        self.cam_props.enable_tensors = True
        self.cam_props.horizontal_fov = 70

        #self.camera_pos = gymapi.Vec3(0.0, 0.2, 0.8)
        #self.camera_target = gymapi.Vec3(0.0, 0.2, 0.0)
        self.transform = gymapi.Transform()
        self.transform.p = gymapi.Vec3(1.5, 0.0, 4.0)
        self.transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.radians(90.0))#*gymapi.Quat.from_axis_angle(gymapi.Vec3(1,0,0), np.radians(180.0))

        self.image_type =image_type
        intensity = gymapi.Vec3(0.2, 0.2, 0.2)
        ambient = gymapi.Vec3(1.0, 1.0, 1.0)
        direction = gymapi.Vec3(1.0, 1.0, 0.5)
        for i in range(4):
            self.gym.set_light_parameters(self.sim, i, intensity, ambient, direction)
        
    def add(self, env):
        # add camera
        cam_handle = self.gym.create_camera_sensor(env, self.cam_props)
        self.gym.set_camera_transform(cam_handle, env, self.transform)

        # obtain camera tensor
        cam_tensor = self.gym.get_camera_image_gpu_tensor(
            self.sim, env, cam_handle, self.image_type
        )
        cam_tensor = gymtorch.wrap_tensor(cam_tensor)
        return cam_tensor
